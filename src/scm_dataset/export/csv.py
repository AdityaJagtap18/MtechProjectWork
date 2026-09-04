"""CSV serialization (plan §24: graph/nodes.csv, graph/edges.csv,
graph/edge_features.csv, operations/{demand,production,inventory,
deliveries,procurement,backlog}.csv, events/{events,event_impact}.csv, and
labels/{supplier,material,plant,product}_labels.csv).

`export_graph` writes a wide nodes.csv (one row per node, columns are the
union of all six node types' attributes, blank where not applicable) plus
edges.csv and edge_features.csv. `load_graph` reverses this, reconstructing
typed dataclasses and coercing pandas' read-back types (which collapse to
float64 wherever a column has NaNs elsewhere) back to each field's declared
type. `export_operations` writes the Phase 3 simulation output,
`export_events`/`export_event_impact` the Phase 5 event log and its Phase 6
cascade summary, and `export_labels` the Phase 6 ground-truth labels — all
plain flat tables, since there's no schema/validation layer over these
records the way there is for graph nodes/edges.
"""

from __future__ import annotations

import os
import typing
from dataclasses import asdict, fields
from typing import TYPE_CHECKING

import pandas as pd

from ..schema.edges import Edge, EdgeType
from ..schema.events import Event
from ..schema.graph import NODE_CLASS_BY_TYPE, SupplyChainGraph, node_id
from ..schema.nodes import NodeType

if TYPE_CHECKING:
    from ..simulation.engine import OperationsResult


def export_graph(graph: SupplyChainGraph, output_dir: str) -> None:
    graph_dir = os.path.join(output_dir, "graph")
    os.makedirs(graph_dir, exist_ok=True)

    node_records = []
    for node in graph.nodes.values():
        row = asdict(node)
        row["node_id"] = node_id(node)
        row["node_type"] = node.node_type.value
        node_records.append(row)
    nodes_df = pd.DataFrame(node_records)
    lead_cols = ["node_id", "node_type"]
    nodes_df = nodes_df[lead_cols + [c for c in nodes_df.columns if c not in lead_cols]]
    nodes_df.to_csv(os.path.join(graph_dir, "nodes.csv"), index=False)

    edge_records = []
    feature_records = []
    for edge_id, edge in enumerate(graph.edges):
        edge_records.append(
            {
                "edge_id": edge_id,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "edge_type": edge.edge_type.value,
            }
        )
        if edge.attributes:
            feature_records.append({"edge_id": edge_id, **edge.attributes})
    pd.DataFrame(edge_records).to_csv(os.path.join(graph_dir, "edges.csv"), index=False)
    features_df = pd.DataFrame(feature_records) if feature_records else pd.DataFrame(columns=["edge_id"])
    features_df.to_csv(os.path.join(graph_dir, "edge_features.csv"), index=False)


def _unwrap_optional(tp: type) -> type:
    origin = typing.get_origin(tp)
    if origin is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if args:
            return args[0]
    return tp


def _coerce(value, tp: type):
    base = _unwrap_optional(tp)
    if base is int:
        return int(value)
    if base is float:
        return float(value)
    if base is str:
        return str(value)
    return value


def load_graph(input_dir: str) -> SupplyChainGraph:
    graph_dir = os.path.join(input_dir, "graph")
    nodes_df = pd.read_csv(os.path.join(graph_dir, "nodes.csv"))
    edges_df = pd.read_csv(os.path.join(graph_dir, "edges.csv"))
    features_path = os.path.join(graph_dir, "edge_features.csv")
    features_df = pd.read_csv(features_path) if os.path.exists(features_path) else pd.DataFrame()

    graph = SupplyChainGraph()
    for node_type, cls in NODE_CLASS_BY_TYPE.items():
        hints = typing.get_type_hints(cls)
        field_names = {f.name for f in fields(cls)}
        subset = nodes_df[nodes_df["node_type"] == node_type.value]
        for _, row in subset.iterrows():
            kwargs = {}
            for name in field_names:
                if name in row and pd.notna(row[name]):
                    kwargs[name] = _coerce(row[name], hints[name])
            graph.add_node(cls(**kwargs))

    features_by_edge_id: dict[int, dict] = {}
    if not features_df.empty:
        for _, row in features_df.iterrows():
            edge_id = int(row["edge_id"])
            attrs = {k: v for k, v in row.items() if k != "edge_id" and pd.notna(v)}
            features_by_edge_id[edge_id] = attrs

    for _, row in edges_df.iterrows():
        edge_id = int(row["edge_id"])
        graph.add_edge(
            Edge(
                source_id=str(row["source_id"]),
                target_id=str(row["target_id"]),
                edge_type=EdgeType(row["edge_type"]),
                attributes=features_by_edge_id.get(edge_id, {}),
            )
        )
    return graph


_OPERATIONS_FILES = {
    "demand_records": "demand.csv",
    "production_records": "production.csv",
    "inventory_records": "inventory.csv",
    "deliveries_records": "deliveries.csv",
    "procurement_records": "procurement.csv",
    "backlog_records": "backlog.csv",
}


def export_operations(result: "OperationsResult", output_dir: str) -> None:
    ops_dir = os.path.join(output_dir, "operations")
    os.makedirs(ops_dir, exist_ok=True)
    for attr_name, filename in _OPERATIONS_FILES.items():
        records = getattr(result, attr_name)
        pd.DataFrame(records).to_csv(os.path.join(ops_dir, filename), index=False)


def load_operations(input_dir: str) -> dict[str, pd.DataFrame]:
    ops_dir = os.path.join(input_dir, "operations")
    return {
        filename: pd.read_csv(os.path.join(ops_dir, filename))
        for filename in _OPERATIONS_FILES.values()
        if os.path.exists(os.path.join(ops_dir, filename))
    }


_EVENT_LIST_FIELDS = ("affected_regions", "affected_suppliers", "affected_plants")


def export_events(events: list[Event], output_dir: str) -> None:
    events_dir = os.path.join(output_dir, "events")
    os.makedirs(events_dir, exist_ok=True)
    records = []
    for event in events:
        row = asdict(event)
        row["event_type"] = event.event_type.value
        for name in _EVENT_LIST_FIELDS:
            row[name] = ";".join(row[name])  # mirrors the multi-supplier convention observed in NIST's SGE_products.csv
        records.append(row)
    columns = [f.name for f in fields(Event)]
    df = pd.DataFrame(records, columns=columns) if records else pd.DataFrame(columns=columns)
    df.to_csv(os.path.join(events_dir, "events.csv"), index=False)


def load_events(input_dir: str) -> list[Event]:
    from ..schema.events import EventType

    path = os.path.join(input_dir, "events", "events.csv")
    if not os.path.exists(path):
        return []

    df = pd.read_csv(path)
    hints = typing.get_type_hints(Event)
    events = []
    for _, row in df.iterrows():
        kwargs = {}
        for f in fields(Event):
            value = row[f.name]
            if f.name in _EVENT_LIST_FIELDS:
                kwargs[f.name] = [x for x in str(value).split(";") if x] if pd.notna(value) else []
            elif f.name == "event_type":
                kwargs[f.name] = EventType(value)
            elif pd.notna(value):
                kwargs[f.name] = _coerce(value, hints[f.name])
        events.append(Event(**kwargs))
    return events


_LABEL_FILES = {
    "supplier": "supplier_labels.csv",
    "material": "material_labels.csv",
    "plant": "plant_labels.csv",
    "product": "product_labels.csv",
}


def export_labels(
    output_dir: str,
    supplier_labels: pd.DataFrame,
    material_labels: pd.DataFrame,
    plant_labels: pd.DataFrame,
    product_labels: pd.DataFrame,
) -> None:
    """Writes plan §24's labels/{supplier,material,plant,product}_labels.csv."""
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    frames = {"supplier": supplier_labels, "material": material_labels, "plant": plant_labels, "product": product_labels}
    for name, filename in _LABEL_FILES.items():
        frames[name].to_csv(os.path.join(labels_dir, filename), index=False)


def load_labels(input_dir: str) -> dict[str, pd.DataFrame]:
    labels_dir = os.path.join(input_dir, "labels")
    return {
        name: pd.read_csv(os.path.join(labels_dir, filename))
        for name, filename in _LABEL_FILES.items()
        if os.path.exists(os.path.join(labels_dir, filename))
    }


def export_event_impact(event_impact: pd.DataFrame, output_dir: str) -> None:
    """Writes plan §24's events/event_impact.csv (the per-event cascade
    summary — cascade_severity, total_affected_nodes, time_to_impact,
    recovery_time)."""
    events_dir = os.path.join(output_dir, "events")
    os.makedirs(events_dir, exist_ok=True)
    event_impact.to_csv(os.path.join(events_dir, "event_impact.csv"), index=False)


def load_event_impact(input_dir: str) -> pd.DataFrame | None:
    path = os.path.join(input_dir, "events", "event_impact.csv")
    return pd.read_csv(path) if os.path.exists(path) else None
