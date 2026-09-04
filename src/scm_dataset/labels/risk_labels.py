"""Ground-truth cascade labels (plan §22, §42 Phase 6).

Labels are derived from SIMULATED consequences recorded by Phase 3/5's
engine (production, inventory, fulfillment, backlog) — never copied from
the static input risk/criticality attributes generated in Phase 2 (plan
§13, §43.2). Where an event's own capacity/lead-time multiplier is the
most direct and unambiguous signal that an entity was being acted on by a
disruption (`supplier_disrupted`), that mechanism is used directly rather
than an indirect threshold — it's still a *label*, not something a model
would see as an input feature (plan §27).

Most of these are direct reads of quantities the simulator already tracks.
The one place this module adds genuinely new logic is
`compute_event_impact_labels`: tracing a *specific event's* downstream
reach through the graph (Supplier -> Material -> Plant -> Product), since
the simulator itself only tracks the union of everything happening at
once, not which event caused which consequence. No counterfactual
re-simulation is performed to isolate one event's marginal effect from
concurrent events or ordinary demand noise — this is a documented
imprecision, not a hidden one.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ..generator.config import GeneratorConfig
from ..schema.edges import EdgeType
from ..schema.graph import SupplyChainGraph
from ..schema.nodes import NodeType
from ..simulation.engine import OperationsResult, build_indices, compute_flow_scales

# A plant counts as "disrupted" once it loses more than this fraction of
# what it could have produced given capacity and demand alone (i.e. ignoring
# the material-availability constraint) -- not merely "produced less because
# demand was lower," which would make every quiet period look disrupted.
PLANT_DISRUPTION_THRESHOLD = 0.1


def _entity_disrupted_windows(events: list, list_field: str) -> dict[str, set[int]]:
    """For each entity_id referenced in `list_field` of any event (e.g.
    'affected_suppliers'), the set of periods where that event was active."""
    windows: dict[str, set[int]] = defaultdict(set)
    for event in events:
        entity_ids = getattr(event, list_field)
        if not entity_ids:
            continue
        window_end = event.start_time + event.duration + event.recovery_delay + event.recovery_periods
        periods = set(range(event.start_time, window_end))
        for entity_id in entity_ids:
            windows[entity_id] |= periods
    return windows


def compute_supplier_labels(graph: SupplyChainGraph, result: OperationsResult) -> pd.DataFrame:
    """Per (supplier, period): `supplier_disrupted` is 1 exactly when an
    active event's effect covered this supplier at this period. `supplier_
    risk_score` is the realized fulfillment shortfall (1 - fulfilled/
    ordered) for periods with at least one order — a simulated consequence,
    not the static `criticality`/exposure input features."""
    suppliers = graph.nodes_of_type(NodeType.SUPPLIER)
    disrupted_windows = _entity_disrupted_windows(result.events, "affected_suppliers")

    fulfillment_ratio: dict[tuple[str, int], float] = {}
    if result.procurement_records:
        procurement_df = pd.DataFrame(result.procurement_records)
        agg = procurement_df.groupby(["supplier_id", "time_ordered"])[["quantity_ordered", "quantity_fulfilled"]].sum()
        for (supplier_id, t), row in agg.iterrows():
            if row.quantity_ordered > 0:
                fulfillment_ratio[(supplier_id, t)] = row.quantity_fulfilled / row.quantity_ordered

    horizon = max((r["time"] for r in result.demand_records), default=-1) + 1
    records = []
    for supplier in suppliers:
        windows = disrupted_windows.get(supplier.supplier_id, set())
        for t in range(horizon):
            ratio = fulfillment_ratio.get((supplier.supplier_id, t))
            records.append(
                {
                    "supplier_id": supplier.supplier_id,
                    "time": t,
                    "supplier_disrupted": int(t in windows),
                    "supplier_risk_score": float(1.0 - ratio) if ratio is not None else None,
                }
            )
    return pd.DataFrame(records)


def compute_material_labels(graph: SupplyChainGraph, result: OperationsResult, config: GeneratorConfig) -> pd.DataFrame:
    """Per (plant, material, period): `material_shortage` (binary) and
    `material_risk_score` (continuous, [0, 1]) compare simulated inventory
    against that pair's own reorder threshold — derived from
    `plant.production_capacity * material.required_quantity_per_product`
    (see `simulation.engine.compute_flow_scales`), not the Material node's
    standalone `safety_stock` (same units-mismatch reason as Phase 3)."""
    plants = graph.nodes_of_type(NodeType.PLANT)
    materials_by_id = {m.material_id: m for m in graph.nodes_of_type(NodeType.MATERIAL)}
    suppliers_by_id = {s.supplier_id: s for s in graph.nodes_of_type(NodeType.SUPPLIER)}
    plant_materials, _plant_products, primary_supplier = build_indices(graph)
    _initial, reorder_threshold, _target, _capacity = compute_flow_scales(
        plants, materials_by_id, plant_materials, primary_supplier, suppliers_by_id, config.simulation
    )

    records = []
    for row in result.inventory_records:
        key = (row["plant_id"], row["material_id"])
        threshold = reorder_threshold.get(key)
        if not threshold:
            shortage, risk = None, None
        else:
            risk = float(np.clip(1.0 - row["inventory"] / threshold, 0.0, 1.0))
            shortage = int(row["inventory"] < threshold)
        records.append(
            {
                "plant_id": row["plant_id"],
                "material_id": row["material_id"],
                "time": row["time"],
                "material_shortage": shortage,
                "material_risk_score": risk,
            }
        )
    return pd.DataFrame(records)


def compute_plant_labels(graph: SupplyChainGraph, result: OperationsResult) -> pd.DataFrame:
    """Per (plant, period): `production_loss`/`production_loss_fraction`
    measure output actually lost to a binding constraint (capacity or
    material availability) — not lost to low demand, which would make
    "disruption" indistinguishable from "nobody wanted more of this right
    now." The reference is `min(plant.production_capacity, demand+backlog)`
    — what the plant could have produced ignoring the material constraint
    — against what it actually produced."""
    plants = graph.nodes_of_type(NodeType.PLANT)
    plant_products: dict[str, list[str]] = defaultdict(list)
    for e in graph.edges_of_type(EdgeType.PLANT_PRODUCT):
        plant_products[e.source_id].append(e.target_id)

    demand_by_key = {(r["product_id"], r["time"]): r["demand"] for r in result.demand_records}
    backlog_by_key = {(r["product_id"], r["time"]): r["backlog"] for r in result.backlog_records}

    production_by_plant_time: dict[tuple[str, int], float] = defaultdict(float)
    for r in result.production_records:
        production_by_plant_time[(r["plant_id"], r["time"])] += r["production"]

    horizon = max((r["time"] for r in result.demand_records), default=-1) + 1
    capacity_by_plant = {p.plant_id: p.production_capacity for p in plants}

    records = []
    for plant in plants:
        product_ids = plant_products.get(plant.plant_id, [])
        for t in range(horizon):
            demand_cap = sum(demand_by_key.get((pid, t), 0.0) + backlog_by_key.get((pid, t), 0.0) for pid in product_ids)
            unconstrained = min(capacity_by_plant[plant.plant_id], demand_cap)
            actual = production_by_plant_time.get((plant.plant_id, t), 0.0)
            loss = max(0.0, unconstrained - actual)
            loss_fraction = (loss / unconstrained) if unconstrained > 0 else 0.0
            records.append(
                {
                    "plant_id": plant.plant_id,
                    "time": t,
                    "production_loss": loss,
                    "production_loss_fraction": loss_fraction,
                    "plant_disruption": int(loss_fraction > PLANT_DISRUPTION_THRESHOLD),
                }
            )
    return pd.DataFrame(records)


def compute_product_labels(graph: SupplyChainGraph, result: OperationsResult) -> pd.DataFrame:
    """Per (product, period): `product_shortage` (binary) and
    `revenue_impact` = this period's unmet demand * the product's (static,
    price-like) `revenue_per_unit`. Combining a simulated consequence with
    a price attribute is a standard "lost sales" costing, not a copy of an
    input risk score (plan §13, §43.2)."""
    products = graph.nodes_of_type(NodeType.PRODUCT)
    revenue_per_unit = {p.product_id: p.revenue_per_unit for p in products}

    production_by_key: dict[tuple[str, int], float] = defaultdict(float)
    for r in result.production_records:
        production_by_key[(r["product_id"], r["time"])] += r["production"]

    records = []
    for r in result.demand_records:
        key = (r["product_id"], r["time"])
        shortfall = max(0.0, r["demand"] - production_by_key.get(key, 0.0))
        records.append(
            {
                "product_id": r["product_id"],
                "time": r["time"],
                "product_shortage": int(shortfall > 0),
                "revenue_impact": float(shortfall * revenue_per_unit[r["product_id"]]),
            }
        )
    return pd.DataFrame(records)


def _downstream_entities(
    graph: SupplyChainGraph, affected_suppliers: set[str], affected_plants: set[str]
) -> tuple[set[str], set[str], set[str]]:
    """Materials, plants, and products reachable via the graph (Supplier ->
    Material -> Plant -> Product) from an event's directly affected
    entities — how far a disruption's consequences *could* propagate
    structurally (plan §19), independent of whether they actually did."""
    materials = {e.target_id for e in graph.edges_of_type(EdgeType.SUPPLIER_MATERIAL) if e.source_id in affected_suppliers}
    plants = {e.target_id for e in graph.edges_of_type(EdgeType.MATERIAL_PLANT) if e.source_id in materials}
    plants |= affected_plants
    products = {e.target_id for e in graph.edges_of_type(EdgeType.PLANT_PRODUCT) if e.source_id in plants}
    return materials, plants, products


def compute_event_impact_labels(
    graph: SupplyChainGraph, events: list, plant_labels: pd.DataFrame, product_labels: pd.DataFrame
) -> pd.DataFrame:
    """Per event: `cascade_severity`, `total_affected_nodes`, `time_to_impact`,
    `recovery_time` (plan §22 "Overall cascade").

    `time_to_impact` only counts *purely downstream* plants (reached via a
    material dependency, not directly targeted by the event) — this isolates
    the cascade-propagation delay rather than trivially reporting ~0 for
    event types that hit plants directly. `recovery_time` is the last period
    (relative to the event's start) at which any directly or indirectly
    affected plant/product still shows a disruption signal — 0 if none ever
    did. `cascade_severity` is the mean plant-level loss fraction across all
    affected plants during the event's own active window.
    """
    plant_disrupted: dict[str, set[int]] = defaultdict(set)
    for row in plant_labels.itertuples():
        if row.plant_disruption:
            plant_disrupted[row.plant_id].add(row.time)

    product_shortage: dict[str, set[int]] = defaultdict(set)
    for row in product_labels.itertuples():
        if row.product_shortage:
            product_shortage[row.product_id].add(row.time)

    records = []
    for event in events:
        materials, all_plants, all_products = _downstream_entities(graph, set(event.affected_suppliers), set(event.affected_plants))
        pure_downstream_plants = all_plants - set(event.affected_plants)
        window_start = event.start_time
        window_end = event.start_time + event.duration + event.recovery_delay + event.recovery_periods

        total_affected_nodes = len(event.affected_suppliers) + len(materials) + len(all_plants) + len(all_products)

        impact_times = [
            t for plant_id in pure_downstream_plants for t in plant_disrupted.get(plant_id, set()) if window_start <= t < window_end
        ]
        time_to_impact = (min(impact_times) - window_start) if impact_times else None

        disruption_times = [t for plant_id in all_plants for t in plant_disrupted.get(plant_id, set())]
        disruption_times += [t for product_id in all_products for t in product_shortage.get(product_id, set())]
        recovery_time = (max(disruption_times) - window_start) if disruption_times else 0

        window_losses = plant_labels[
            plant_labels.plant_id.isin(all_plants) & plant_labels.time.between(window_start, window_end - 1)
        ].production_loss_fraction
        cascade_severity = float(window_losses.mean()) if len(window_losses) else 0.0

        records.append(
            {
                "event_id": event.event_id,
                "cascade_severity": cascade_severity,
                "total_affected_nodes": total_affected_nodes,
                "time_to_impact": time_to_impact,
                "recovery_time": recovery_time,
            }
        )
    return pd.DataFrame(records)
