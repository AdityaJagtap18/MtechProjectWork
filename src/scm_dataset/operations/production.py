"""Plant production (plan §5.4, §10, §11.1/§11.2/§11.5, §20).

production(t) = min(plant_capacity * capacity_multiplier, min over materials m
consumed by the plant of inventory[m,t] / m.required_quantity_per_product,
demand_cap) — the multi-input generalization of plan §20's production
formula (which shows the single-material case), with an added `demand_cap`
term so production actually responds to demand/backlog (§11.5) instead of
always running at whatever capacity/materials allow regardless of whether
there's demand for it. `production_capacity` (set in Phase 2) is treated as
the plant's per-simulation-period throughput limit; the plan doesn't pin
down its units, so this is the simplest defensible reading.

`capacity_multiplier` (plan §42 Phase 5) is how a Natural Disaster event
directly reduces a plant's own output — see `events/base.py`'s
`capacity_multiplier`. Every other event type only affects plants
indirectly, through this same function's material-availability term once a
disrupted supplier's deliveries shrink (plan §19's cascade).
"""

from __future__ import annotations

from ..schema.nodes import Material, Plant


def compute_plant_production(
    plant: Plant,
    material_ids: list[str],
    materials_by_id: dict[str, Material],
    inventory: dict[tuple[str, str], float],
    demand_cap: float | None = None,
    capacity_multiplier: float = 1.0,
) -> float:
    limit = plant.production_capacity * capacity_multiplier
    for material_id in material_ids:
        material = materials_by_id[material_id]
        available = inventory.get((plant.plant_id, material_id), 0.0)
        required = max(material.required_quantity_per_product, 1e-6)
        limit = min(limit, available / required)
    if demand_cap is not None:
        limit = min(limit, demand_cap)
    return max(0.0, limit)
