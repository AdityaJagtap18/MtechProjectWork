"""Leakage-safe feature construction (plan §11-26).

Three things happen here, deliberately kept separate:

1. **Static node features** (`build_static_features`): one row per graph
   node, read straight off `graph/nodes.csv`. These are generation-time
   constants (plan §5, feature_audit.csv's blanket `graph/nodes.csv,*,...
   True` rule) -- identical in every split, so "fit on train" doesn't
   apply to *which values exist*, only to how any scaler/encoder over them
   is later fit (see `preprocessing.py`).

2. **Dynamic node features** (`build_dynamic_panels`): rolling-window
   aggregates ending at `t`, one row per (entity_id, t). Built from a
   *full* entity x time grid so a period with zero activity is a real,
   known zero -- not a missing observation -- and so every window is
   causal by construction (it only ever reads rows at or before its own
   `t`; there is no code path that could reach forward).

3. **Prediction examples and target** (`build_prediction_examples`):
   `Y(supplier, t) = max(supplier_disrupted[t+1 .. t+H])`, read directly
   from the existing `labels/supplier_labels.csv` (plan §5/§20/§21) --
   never recomputed from events or re-derived independently.

`audit_feature_sources` is the automated leakage gate (plan §6/§22):
every (table, column) this module reads dynamically is declared in
`FEATURE_SOURCE_COLUMNS` below, and checked against the benchmark's own
`feature_audit.csv` before any feature is built. An unknown or
`allowed=False` source raises immediately -- the pipeline fails loudly
rather than silently building a leaky feature.

**`supplier_risk_score` is deliberately never read by this module** (plan
§7/§72). `feature_audit.csv` already marks the entire `labels/
supplier_labels.csv` table `allowed=False` ("only ever a prediction
TARGET"), and `labels/risk_labels.py` confirms why: it's a realized
fulfillment shortfall computed from the same simulation outcomes as the
target itself, not an independently-observable input. The target column
is read from that same table under a separate, explicit code path
(`build_prediction_examples`) that is never merged into the feature frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..schema.graph import SupplyChainGraph
from ..schema.nodes import NodeType

# ---------------------------------------------------------------------------
# Static features
# ---------------------------------------------------------------------------

# Per node type: (numeric fields, categorical fields), reading plan §12-17's
# field lists against the actual dataclasses in schema/nodes.py. Identifier
# / foreign-key fields (region_id, plant_id, supplier_id, ...) and
# `country_or_region` are intentionally excluded: they name a specific
# *other node*, and that relationship already reaches the model structurally
# through the graph's own edges (SUPPLIER_REGION, PLANT_REGION, ...) rather
# than needing to be duplicated as a feature value.
STATIC_NUMERIC_FIELDS: dict[NodeType, list[str]] = {
    NodeType.SUPPLIER: [
        "tier", "capacity", "capacity_utilization", "reliability", "financial_health",
        "lead_time_mean", "lead_time_variability", "quality_score", "inventory_buffer",
        "substitution_availability", "geopolitical_exposure", "disaster_exposure",
        "cyber_exposure", "criticality",
    ],
    NodeType.MATERIAL: [
        "criticality", "substitutability", "demand", "unit_cost", "inventory_level",
        "safety_stock", "supplier_count", "concentration", "required_quantity_per_product",
    ],
    NodeType.PLANT: [
        "production_capacity", "utilization", "operating_cost", "inventory_capacity",
        "resilience_score", "downtime_cost", "recovery_rate",
    ],
    NodeType.PRODUCT: [
        "demand", "revenue_per_unit", "margin", "material_dependency", "criticality",
        "substitution_score", "backlog",
    ],
    NodeType.REGION: [
        "geopolitical_risk", "natural_disaster_risk", "infrastructure_risk", "trade_risk",
        "cyber_risk", "transport_reliability",
    ],
    NodeType.PROCUREMENT: [
        "order_quantity", "order_value", "order_frequency", "promised_lead_time",
        "actual_lead_time", "urgency", "contract_duration", "priority",
    ],
}

STATIC_CATEGORICAL_FIELDS: dict[NodeType, list[str]] = {
    NodeType.SUPPLIER: ["industry"],
    NodeType.MATERIAL: ["material_category", "procurement_type"],
    NodeType.PLANT: [],
    NodeType.PRODUCT: ["product_category"],
    NodeType.REGION: [],
    NodeType.PROCUREMENT: [],
}

ID_FIELD: dict[NodeType, str] = {
    NodeType.SUPPLIER: "supplier_id",
    NodeType.MATERIAL: "material_id",
    NodeType.PLANT: "plant_id",
    NodeType.PRODUCT: "product_id",
    NodeType.REGION: "region_id",
    NodeType.PROCUREMENT: "procurement_id",
}


def build_static_features(graph: SupplyChainGraph) -> dict[NodeType, pd.DataFrame]:
    """One DataFrame per node type, indexed by that type's id field, columns
    = numeric fields + categorical fields (plan §11-17). Values are read
    verbatim off the dataclasses -- no scaling/encoding here, that's
    preprocessing.py's job once train/val/test membership is known."""
    tables: dict[NodeType, pd.DataFrame] = {}
    for node_type in NodeType:
        id_field = ID_FIELD[node_type]
        numeric = STATIC_NUMERIC_FIELDS[node_type]
        categorical = STATIC_CATEGORICAL_FIELDS[node_type]
        rows = []
        for node in graph.nodes_of_type(node_type):
            row = {id_field: getattr(node, id_field)}
            for f in numeric:
                row[f] = getattr(node, f)
            for f in categorical:
                val = getattr(node, f)
                row[f] = val if val is not None else "UNKNOWN"
            rows.append(row)
        df = pd.DataFrame(rows).set_index(id_field)
        tables[node_type] = df
    return tables


# ---------------------------------------------------------------------------
# Dynamic features
# ---------------------------------------------------------------------------


@dataclass
class DynamicSource:
    table: str  # operations/<file> as named in feature_audit.csv
    columns: tuple[str, ...]


# Registry of every (table, column) this module reads to build a *dynamic*
# (time-varying) feature -- cross-checked against feature_audit.csv by
# `audit_feature_sources` before any feature is actually built (plan §6/§22).
FEATURE_SOURCE_COLUMNS: list[DynamicSource] = [
    DynamicSource("operations/procurement.csv", ("quantity_ordered", "quantity_fulfilled")),
    DynamicSource("operations/deliveries.csv", ("quantity",)),
    DynamicSource("operations/inventory.csv", ("inventory",)),
    DynamicSource("operations/production.csv", ("production",)),
    DynamicSource("operations/demand.csv", ("demand",)),
    DynamicSource("operations/backlog.csv", ("backlog",)),
]


def audit_feature_sources(feature_audit: pd.DataFrame) -> None:
    """Fails loudly (plan §6) if any column this module reads dynamically
    is not present in feature_audit.csv, or is present but marked
    `allowed=False`. This is checked against the benchmark's own audit
    file rather than re-derived, per plan's "use it as the authoritative
    feature-leakage reference." """
    allowed_lookup: dict[tuple[str, str], bool] = {}
    wildcard_allowed: dict[str, bool] = {}
    for _, row in feature_audit.iterrows():
        if row["column"] == "*":
            wildcard_allowed[row["table"]] = bool(row["allowed"])
        else:
            # A single audit row may cover several columns at once, e.g.
            # "quantity_ordered, quantity_fulfilled" (see feature_audit.csv's
            # operations/procurement.csv row) -- split rather than matching
            # the joined string literally.
            for col in str(row["column"]).split(","):
                allowed_lookup[(row["table"], col.strip())] = bool(row["allowed"])

    violations: list[str] = []
    for source in FEATURE_SOURCE_COLUMNS:
        for col in source.columns:
            if (source.table, col) in allowed_lookup:
                allowed = allowed_lookup[(source.table, col)]
            elif source.table in wildcard_allowed:
                allowed = wildcard_allowed[source.table]
            else:
                violations.append(f"{source.table}:{col} not found in feature_audit.csv -- refusing to use an unaudited source")
                continue
            if not allowed:
                violations.append(f"{source.table}:{col} is marked allowed=False in feature_audit.csv")

    if violations:
        raise ValueError("Leakage audit failed for the following dynamic feature source(s):\n" + "\n".join(violations))


def _full_grid(entity_ids: list[str], horizon_periods: int, entity_col: str) -> pd.DataFrame:
    return pd.DataFrame(
        [(eid, t) for eid in entity_ids for t in range(horizon_periods)], columns=[entity_col, "time"]
    )


def _rolling(df: pd.DataFrame, entity_col: str, value_col: str, windows: list[int]) -> pd.DataFrame:
    """Adds `{value_col}_mean_{w}`, `_std_{w}`, `_trend_{w}` for each window
    `w`, causal by construction: each row's window only ever looks back
    from its own `time` (rolling `min_periods=1`, ascending-time order)."""
    df = df.sort_values([entity_col, "time"]).reset_index(drop=True)
    grouped = df.groupby(entity_col)[value_col]
    for w in windows:
        df[f"{value_col}_mean_{w}"] = grouped.transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
        df[f"{value_col}_std_{w}"] = grouped.transform(lambda s, w=w: s.rolling(w, min_periods=1).std(ddof=0)).fillna(0.0)
        first = grouped.transform(lambda s, w=w: s.shift(w - 1))
        denom = max(w - 1, 1)
        df[f"{value_col}_trend_{w}"] = (df[value_col] - first) / denom
        df[f"{value_col}_trend_{w}"] = df[f"{value_col}_trend_{w}"].fillna(0.0)
    return df


def _rolling_sum(df: pd.DataFrame, entity_col: str, value_col: str, windows: list[int]) -> pd.DataFrame:
    df = df.sort_values([entity_col, "time"]).reset_index(drop=True)
    grouped = df.groupby(entity_col)[value_col]
    for w in windows:
        df[f"{value_col}_sum_{w}"] = grouped.transform(lambda s, w=w: s.rolling(w, min_periods=1).sum())
    return df


def _supplier_panel(operations: dict[str, pd.DataFrame], supplier_ids: list[str], horizon_periods: int, windows: list[int]) -> pd.DataFrame:
    proc = operations["procurement.csv"]
    proc_agg = (
        proc.groupby(["supplier_id", "time_ordered"])
        .agg(
            quantity_ordered=("quantity_ordered", "sum"),
            quantity_fulfilled=("quantity_fulfilled", "sum"),
            orders_count=("quantity_ordered", "size"),
        )
        .reset_index()
        .rename(columns={"time_ordered": "time"})
    )

    deliv = operations["deliveries.csv"]
    deliv_agg = deliv.groupby(["supplier_id", "time"])["quantity"].sum().reset_index().rename(columns={"quantity": "delivery_quantity"})

    grid = _full_grid(supplier_ids, horizon_periods, "supplier_id")
    panel = grid.merge(proc_agg, on=["supplier_id", "time"], how="left").merge(
        deliv_agg, on=["supplier_id", "time"], how="left"
    )
    for col in ("quantity_ordered", "quantity_fulfilled", "orders_count", "delivery_quantity"):
        panel[col] = panel[col].fillna(0.0)

    panel = _rolling_sum(panel, "supplier_id", "quantity_ordered", windows)
    panel = _rolling_sum(panel, "supplier_id", "quantity_fulfilled", windows)
    panel = _rolling_sum(panel, "supplier_id", "orders_count", windows)
    panel = _rolling_sum(panel, "supplier_id", "delivery_quantity", windows)

    feature_cols = ["supplier_id", "time"]
    for w in windows:
        ordered_sum = panel[f"quantity_ordered_sum_{w}"]
        fulfilled_sum = panel[f"quantity_fulfilled_sum_{w}"]
        ratio = np.where(ordered_sum > 0, fulfilled_sum / ordered_sum.replace(0, np.nan), np.nan)
        panel[f"fulfillment_ratio_{w}"] = ratio
        panel = panel.rename(columns={f"quantity_ordered_sum_{w}": f"order_volume_{w}", f"orders_count_sum_{w}": f"orders_count_{w}", f"delivery_quantity_sum_{w}": f"delivery_volume_{w}"})
        feature_cols += [f"order_volume_{w}", f"orders_count_{w}", f"delivery_volume_{w}", f"fulfillment_ratio_{w}"]

    return panel[feature_cols].set_index(["supplier_id", "time"])


def _material_panel(operations: dict[str, pd.DataFrame], material_ids: list[str], horizon_periods: int, windows: list[int]) -> pd.DataFrame:
    inv = operations["inventory.csv"]
    inv_agg = inv.groupby(["material_id", "time"])["inventory"].sum().reset_index()
    grid = _full_grid(material_ids, horizon_periods, "material_id")
    panel = grid.merge(inv_agg, on=["material_id", "time"], how="left")
    panel["inventory"] = panel["inventory"].fillna(0.0)
    panel = _rolling(panel, "material_id", "inventory", windows)
    feature_cols = ["material_id", "time"] + [c for c in panel.columns if c.startswith("inventory_") and c != "inventory"]
    return panel[feature_cols].set_index(["material_id", "time"])


def _plant_panel(operations: dict[str, pd.DataFrame], plant_ids: list[str], horizon_periods: int, windows: list[int]) -> pd.DataFrame:
    prod = operations["production.csv"]
    prod_agg = prod.groupby(["plant_id", "time"])["production"].sum().reset_index()
    inv = operations["inventory.csv"]
    inv_agg = inv.groupby(["plant_id", "time"])["inventory"].sum().reset_index()

    grid = _full_grid(plant_ids, horizon_periods, "plant_id")
    panel = grid.merge(prod_agg, on=["plant_id", "time"], how="left").merge(inv_agg, on=["plant_id", "time"], how="left")
    panel["production"] = panel["production"].fillna(0.0)
    panel["inventory"] = panel["inventory"].fillna(0.0)

    panel = _rolling(panel, "plant_id", "production", windows)
    for w in windows:
        panel[f"inventory_mean_{w}"] = panel.groupby("plant_id")["inventory"].transform(
            lambda s, w=w: s.rolling(w, min_periods=1).mean()
        )

    feature_cols = ["plant_id", "time"] + [c for c in panel.columns if c.startswith("production_") and c != "production"]
    feature_cols += [f"inventory_mean_{w}" for w in windows]
    return panel[feature_cols].set_index(["plant_id", "time"])


def _product_panel(operations: dict[str, pd.DataFrame], product_ids: list[str], horizon_periods: int, windows: list[int]) -> pd.DataFrame:
    demand = operations["demand.csv"]
    demand_agg = demand.groupby(["product_id", "time"])["demand"].sum().reset_index()
    backlog = operations["backlog.csv"]
    backlog_agg = backlog.groupby(["product_id", "time"])["backlog"].sum().reset_index()

    grid = _full_grid(product_ids, horizon_periods, "product_id")
    panel = grid.merge(demand_agg, on=["product_id", "time"], how="left").merge(backlog_agg, on=["product_id", "time"], how="left")
    panel["demand"] = panel["demand"].fillna(0.0)
    panel["backlog"] = panel["backlog"].fillna(0.0)

    panel = _rolling(panel, "product_id", "demand", windows)
    for w in windows:
        panel[f"backlog_mean_{w}"] = panel.groupby("product_id")["backlog"].transform(
            lambda s, w=w: s.rolling(w, min_periods=1).mean()
        )

    feature_cols = ["product_id", "time"] + [c for c in panel.columns if c.startswith("demand_") and c != "demand"]
    feature_cols += [f"backlog_mean_{w}" for w in windows]
    return panel[feature_cols].set_index(["product_id", "time"])


def build_dynamic_panels(
    graph: SupplyChainGraph, operations: dict[str, pd.DataFrame], horizon_periods: int, windows: list[int]
) -> dict[NodeType, pd.DataFrame]:
    """Per node type with an operational time series (supplier, material,
    plant, product -- region and procurement are static-only, plan §16/§17
    docstrings above), a DataFrame indexed by (entity_id, time) of rolling
    aggregates ending at `time`. Missing raw activity in a period is a real
    zero (full grid, fillna(0)) except `fulfillment_ratio_*`, which is
    genuinely undefined with zero orders in the window and is left NaN for
    `preprocessing.py`'s missing-value handling (plan §16/§26)."""
    supplier_ids = [n.supplier_id for n in graph.nodes_of_type(NodeType.SUPPLIER)]
    material_ids = [n.material_id for n in graph.nodes_of_type(NodeType.MATERIAL)]
    plant_ids = [n.plant_id for n in graph.nodes_of_type(NodeType.PLANT)]
    product_ids = [n.product_id for n in graph.nodes_of_type(NodeType.PRODUCT)]

    return {
        NodeType.SUPPLIER: _supplier_panel(operations, supplier_ids, horizon_periods, windows),
        NodeType.MATERIAL: _material_panel(operations, material_ids, horizon_periods, windows),
        NodeType.PLANT: _plant_panel(operations, plant_ids, horizon_periods, windows),
        NodeType.PRODUCT: _product_panel(operations, product_ids, horizon_periods, windows),
    }


# ---------------------------------------------------------------------------
# Prediction examples / target
# ---------------------------------------------------------------------------


@dataclass
class NodeFeatureFrames:
    """Per node type: a combined (static + dynamic) raw feature DataFrame,
    plus the column-name split preprocessing.py needs to know how to treat
    each column. Supplier/material/plant/product frames are indexed by
    (id, time) -- one row per graph snapshot they'll appear in; region and
    procurement frames are indexed by id alone (static-only, plan §16/§17:
    no operational time series exists for either in this benchmark)."""

    frames: dict[NodeType, pd.DataFrame]
    numeric_columns: dict[NodeType, list[str]]
    categorical_columns: dict[NodeType, list[str]]


def build_feature_frames(
    graph: SupplyChainGraph, operations: dict[str, pd.DataFrame], horizon_periods: int, windows: list[int]
) -> NodeFeatureFrames:
    static = build_static_features(graph)
    dynamic = build_dynamic_panels(graph, operations, horizon_periods, windows)

    frames: dict[NodeType, pd.DataFrame] = {}
    numeric_columns: dict[NodeType, list[str]] = {}
    categorical_columns: dict[NodeType, list[str]] = {}

    for node_type in (NodeType.SUPPLIER, NodeType.MATERIAL, NodeType.PLANT, NodeType.PRODUCT):
        id_field = ID_FIELD[node_type]
        dyn = dynamic[node_type].reset_index()
        combined = dyn.merge(static[node_type].reset_index(), on=id_field, how="left")
        combined = combined.set_index([id_field, "time"])
        frames[node_type] = combined
        categorical_columns[node_type] = list(STATIC_CATEGORICAL_FIELDS[node_type])
        numeric_columns[node_type] = [c for c in combined.columns if c not in categorical_columns[node_type]]

    for node_type in (NodeType.REGION, NodeType.PROCUREMENT):
        frames[node_type] = static[node_type]
        categorical_columns[node_type] = list(STATIC_CATEGORICAL_FIELDS[node_type])
        numeric_columns[node_type] = [c for c in frames[node_type].columns if c not in categorical_columns[node_type]]

    return NodeFeatureFrames(frames=frames, numeric_columns=numeric_columns, categorical_columns=categorical_columns)


def build_prediction_examples(
    supplier_labels: pd.DataFrame, horizon_periods: int, min_history_periods: int, prediction_horizon: int
) -> pd.DataFrame:
    """plan §18-21: one row per (supplier_id, prediction_time), target
    `Y(t) = max(supplier_disrupted[t+1 .. t+H])`, read directly from
    `labels/supplier_labels.csv`. Usable prediction times are restricted to
    `[min_history_periods - 1, horizon_periods - 1 - H]` so every dynamic
    feature window is fully populated and every target window fits inside
    the recorded horizon (plan §18's "never accidentally align X(t)/Y(t)")."""
    wide = supplier_labels.pivot(index="supplier_id", columns="time", values="supplier_disrupted")
    wide = wide.reindex(columns=range(horizon_periods), fill_value=0)
    supplier_ids = wide.index.tolist()
    values = wide.values  # [n_suppliers, horizon_periods]

    t_min = min_history_periods - 1
    t_max = horizon_periods - 1 - prediction_horizon
    if t_max < t_min:
        raise ValueError(
            f"horizon_periods={horizon_periods} too short for min_history_periods={min_history_periods} "
            f"and prediction_horizon={prediction_horizon}: no usable prediction times remain"
        )

    records = []
    for t in range(t_min, t_max + 1):
        target_window = values[:, t + 1 : t + 1 + prediction_horizon]
        y = (target_window.max(axis=1) > 0).astype(int)
        for i, supplier_id in enumerate(supplier_ids):
            records.append({"supplier_id": supplier_id, "time": t, "target": int(y[i])})
    return pd.DataFrame(records)
