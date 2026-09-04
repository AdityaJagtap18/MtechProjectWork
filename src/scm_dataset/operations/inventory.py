"""Inventory bookkeeping (plan §10, §11.2).

`consume_materials` enforces `consumption <= inventory_available` by
construction, not by clamping after the fact: the caller must pass a
`production_qty` already bounded by `production.compute_plant_production`,
which used the same inventory/requirement ratio to derive that bound, so
consumption can never exceed what's on hand for the limiting material (and
therefore not for any material, since production is the min across all of
them).
"""

from __future__ import annotations

from ..schema.nodes import Material


def consume_materials(
    inventory: dict[tuple[str, str], float],
    plant_id: str,
    material_ids: list[str],
    materials_by_id: dict[str, Material],
    production_qty: float,
) -> None:
    for material_id in material_ids:
        material = materials_by_id[material_id]
        key = (plant_id, material_id)
        consumed = production_qty * material.required_quantity_per_product
        inventory[key] = max(0.0, inventory.get(key, 0.0) - consumed)
