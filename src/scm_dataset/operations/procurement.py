"""Procurement order decisions during simulation (plan §10, §11.3, §11.4).

Two steps. `decide_orders` triggers a replenishment order whenever a
(plant, material) inventory dips below that pair's `reorder_threshold`
(an order-up-to policy targeting `reorder_target`) — both precomputed by
`simulation.engine.compute_flow_scales` from `plant.production_capacity *
material.required_quantity_per_product`, not from the Material node's own
`safety_stock` (see that module's docstring for why). Orders go to that
pair's *primary* supplier — the supplier already established for it by
Phase 2's static procurement orders, or the lowest-id supplier among the
material's suppliers as a deterministic fallback. Reordering from the same
supplier every time (rather than re-choosing) is a deliberate Phase 3
simplification: "normal operations" has no reason to force substitution —
that only becomes interesting once Phase 5 disrupts the primary supplier.

`fulfill_orders` enforces plan §11.3 (a supplier can't deliver more than
its available capacity in one period): when a supplier's total requested
quantity in a period exceeds its precomputed `supplier_effective_capacity`
(times that period's `supplier_capacity_multiplier`, plan §42 Phase 5 — how
a Supplier Failure/Cyberattack/etc. event cuts a supplier's capacity),
every order to that supplier is scaled down proportionally, not on a
first-come-first-served basis. Delivery is scheduled `lead_periods` (from
`supplier.lead_time_mean`, similarly stretched by that period's
`supplier_lead_time_multiplier`) plus a reliability-linked slip after the
order (plan §11.4: delivery only occurs after the modeled lead time has
passed).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..generator.config import SimulationConfig
from ..schema.nodes import Supplier

Order = tuple[str, str, str, float]  # (supplier_id, material_id, plant_id, quantity)


def decide_orders(
    plant_id: str,
    material_ids: list[str],
    inventory: dict[tuple[str, str], float],
    reorder_threshold: dict[tuple[str, str], float],
    reorder_target: dict[tuple[str, str], float],
    primary_supplier: dict[tuple[str, str], str | None],
    rng: np.random.Generator,
) -> list[Order]:
    orders: list[Order] = []
    for material_id in material_ids:
        key = (plant_id, material_id)
        level = inventory.get(key, 0.0)
        if level >= reorder_threshold.get(key, 0.0):
            continue
        supplier_id = primary_supplier.get(key)
        if supplier_id is None:
            continue
        target = reorder_target.get(key, 0.0)
        qty = max(0.0, target - level) * float(rng.uniform(0.9, 1.1))
        if qty > 0:
            orders.append((supplier_id, material_id, plant_id, qty))
    return orders


def fulfill_orders(
    orders: list[Order],
    suppliers_by_id: dict[str, Supplier],
    supplier_effective_capacity: dict[str, float],
    sim: SimulationConfig,
    t: int,
    pending: dict[tuple[str, str], list[tuple[int, float, str]]],
    rng: np.random.Generator,
    supplier_capacity_multiplier: dict[str, float] | None = None,
    supplier_lead_time_multiplier: dict[str, float] | None = None,
) -> list[dict]:
    supplier_capacity_multiplier = supplier_capacity_multiplier or {}
    supplier_lead_time_multiplier = supplier_lead_time_multiplier or {}

    by_supplier: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for supplier_id, material_id, plant_id, qty in orders:
        by_supplier[supplier_id].append((material_id, plant_id, qty))

    records = []
    for supplier_id, items in by_supplier.items():
        supplier = suppliers_by_id[supplier_id]
        capacity = supplier_effective_capacity.get(supplier_id, 0.0) * supplier_capacity_multiplier.get(supplier_id, 1.0)
        total_requested = sum(qty for _, _, qty in items)
        scale = min(1.0, capacity / total_requested) if total_requested > 0 else 1.0
        lead_time_mean = supplier.lead_time_mean * supplier_lead_time_multiplier.get(supplier_id, 1.0)
        lead_periods = max(1, round(lead_time_mean / sim.days_per_period))

        for material_id, plant_id, qty in items:
            fulfilled = qty * scale
            slip = max(0.0, float(rng.normal((1 - supplier.reliability) * lead_periods * 0.3, 0.5)))
            arrival = t + lead_periods + int(round(slip))

            pending[(plant_id, material_id)].append((arrival, fulfilled, supplier_id))
            records.append(
                {
                    "plant_id": plant_id,
                    "material_id": material_id,
                    "supplier_id": supplier_id,
                    "time_ordered": t,
                    "quantity_ordered": qty,
                    "quantity_fulfilled": fulfilled,
                    "expected_delivery_time": arrival,
                }
            )
    return records
