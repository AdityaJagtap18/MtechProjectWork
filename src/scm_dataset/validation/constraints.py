"""Operational constraint / sanity checks (plan §35 "Operational validation",
§36 "Sanity Tests").

These validate the CSV files an exported dataset actually ships — not just
the simulator's in-memory state, which Phases 3-6's own unit tests already
cover extensively. Re-checking the *exported artifact* also catches any
export/round-trip regression those tests would miss (e.g. a serialization
bug that corrupts a value on the way to disk). Each function returns a
list of human-readable violation strings; an empty list means the check
passed.
"""

from __future__ import annotations

import pandas as pd

from ..schema.events import Event
from ..schema.graph import SupplyChainGraph
from ..schema.nodes import NodeType


def check_negative_inventory(inventory_df: pd.DataFrame | None, tolerance: float = 1e-6) -> list[str]:
    if inventory_df is None or inventory_df.empty:
        return []
    bad = inventory_df[inventory_df.inventory < -tolerance]
    return [f"{row.plant_id}/{row.material_id} at t={row.time}: inventory={row.inventory:.4f} < 0" for row in bad.itertuples()]


def check_production_within_capacity(production_df: pd.DataFrame | None, graph: SupplyChainGraph, tolerance: float = 1e-6) -> list[str]:
    if production_df is None or production_df.empty:
        return []
    capacity_by_plant = {p.plant_id: p.production_capacity for p in graph.nodes_of_type(NodeType.PLANT)}
    totals = production_df.groupby(["plant_id", "time"]).production.sum()
    errors = []
    for (plant_id, t), total in totals.items():
        cap = capacity_by_plant.get(plant_id)
        if cap is not None and total > cap + tolerance:
            errors.append(f"{plant_id} at t={t}: production={total:.2f} exceeds capacity={cap:.2f}")
    return errors


def check_deliveries_follow_orders(deliveries_df: pd.DataFrame | None, procurement_df: pd.DataFrame | None) -> list[str]:
    """Every delivered (plant, material, supplier) triple must have at
    least one procurement order for that triple placed strictly before the
    delivery (plan §11.4/§36: "no delivery before lead time")."""
    if deliveries_df is None or deliveries_df.empty:
        return []
    if procurement_df is None or procurement_df.empty:
        return [f"{len(deliveries_df)} delivery record(s) exist but no procurement orders were found"]

    earliest_order = procurement_df.groupby(["plant_id", "material_id", "supplier_id"]).time_ordered.min()
    errors = []
    for row in deliveries_df.itertuples():
        key = (row.plant_id, row.material_id, row.supplier_id)
        earliest = earliest_order.get(key)
        if earliest is None:
            errors.append(f"delivery {key} at t={row.time} has no matching procurement order for that triple")
        elif earliest >= row.time:
            errors.append(f"delivery {key} at t={row.time} arrived at or before its earliest order (t={earliest})")
    return errors


def check_event_references(graph: SupplyChainGraph, events: list[Event]) -> list[str]:
    """Every event references valid nodes/regions (plan §36)."""
    supplier_ids = {s.supplier_id for s in graph.nodes_of_type(NodeType.SUPPLIER)}
    plant_ids = {p.plant_id for p in graph.nodes_of_type(NodeType.PLANT)}
    region_ids = {r.region_id for r in graph.nodes_of_type(NodeType.REGION)}

    errors = []
    for event in events:
        for supplier_id in event.affected_suppliers:
            if supplier_id not in supplier_ids:
                errors.append(f"{event.event_id}: affected_suppliers references unknown supplier {supplier_id!r}")
        for plant_id in event.affected_plants:
            if plant_id not in plant_ids:
                errors.append(f"{event.event_id}: affected_plants references unknown plant {plant_id!r}")
        for region_id in event.affected_regions:
            if region_id not in region_ids:
                errors.append(f"{event.event_id}: affected_regions references unknown region {region_id!r}")
    return errors


def run_constraint_checks(
    graph: SupplyChainGraph,
    operations: dict[str, pd.DataFrame],
    events: list[Event],
) -> dict[str, list[str]]:
    """Runs every check above. Note "every procurement references a valid
    supplier/material/plant" and "no impossible edge types" are already
    covered by `graph.validate()` (SUPPLIER_PROCUREMENT/PROCUREMENT_MATERIAL/
    PROCUREMENT_PLANT edges are derived directly from each ProcurementOrder's
    own fields at generation time, so a dangling reference there would
    already fail edge-type/referential-integrity validation)."""
    return {
        "graph_validity": graph.validate(),
        "negative_inventory": check_negative_inventory(operations.get("inventory.csv")),
        "production_within_capacity": check_production_within_capacity(operations.get("production.csv"), graph),
        "deliveries_follow_orders": check_deliveries_follow_orders(operations.get("deliveries.csv"), operations.get("procurement.csv")),
        "event_references_valid": check_event_references(graph, events),
    }
