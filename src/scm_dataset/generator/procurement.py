"""Procurement order generation (plan §5.2).

Each order is anchored to an existing (supplier, material) edge and an
existing (material, plant) edge already produced by the topology generator,
so orders never reference a relationship that doesn't exist in the graph
(plan §36's "every procurement references valid supplier/material/plant").
Order fields are dependency functions of the referenced supplier/material
(plan §13), not independent draws: `order_value` derives from
`order_quantity * unit_cost`, `promised_lead_time` derives from the
supplier's own `lead_time_mean`, `actual_lead_time` slips past the promise
in proportion to `1 - supplier.reliability` (unreliable suppliers slip
more, on average), and `urgency`/`priority` track `material.criticality`.
"""

from __future__ import annotations

import numpy as np

from ..schema.nodes import Material, ProcurementOrder, Supplier


def generate_procurement_orders(
    num_orders: int,
    suppliers_by_id: dict[str, Supplier],
    materials_by_id: dict[str, Material],
    supplier_material_pairs: list[tuple[str, str]],
    material_plant_map: dict[str, list[str]],
    rng: np.random.Generator,
) -> list[ProcurementOrder]:
    valid_pairs = [(s, m) for (s, m) in supplier_material_pairs if material_plant_map.get(m)]
    if not valid_pairs:
        return []

    orders = []
    for i in range(num_orders):
        supplier_id, material_id = valid_pairs[rng.integers(0, len(valid_pairs))]
        supplier = suppliers_by_id[supplier_id]
        material = materials_by_id[material_id]
        plant_choices = material_plant_map[material_id]
        plant_id = plant_choices[rng.integers(0, len(plant_choices))]

        order_quantity = float(rng.gamma(shape=2.0, scale=max(material.required_quantity_per_product * 10, 1e-3)))
        order_value = float(order_quantity * material.unit_cost * rng.uniform(0.9, 1.1))
        order_frequency = float(rng.gamma(shape=2.0, scale=2.0))

        promised_lead_time = float(
            max(0.0, supplier.lead_time_mean + rng.normal(0, supplier.lead_time_variability * 0.5))
        )
        expected_slip = (1 - supplier.reliability) * promised_lead_time * 0.3
        slip = float(max(0.0, rng.normal(expected_slip, supplier.lead_time_variability * 0.3 + 1e-6)))
        actual_lead_time = promised_lead_time + slip

        urgency = float(np.clip(material.criticality + rng.normal(0, 0.1), 0.0, 1.0))
        priority = float(np.clip(material.criticality + rng.normal(0, 0.1), 0.0, 1.0))
        contract_duration = float(rng.gamma(shape=3.0, scale=90.0))

        orders.append(
            ProcurementOrder(
                procurement_id=f"procurement_{i}",
                supplier_id=supplier_id,
                material_id=material_id,
                plant_id=plant_id,
                order_quantity=order_quantity,
                order_value=order_value,
                order_frequency=order_frequency,
                promised_lead_time=promised_lead_time,
                actual_lead_time=actual_lead_time,
                urgency=urgency,
                contract_duration=contract_duration,
                priority=priority,
            )
        )
    return orders
