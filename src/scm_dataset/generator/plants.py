"""Plant generation (plan §5.4).

`resilience_score` is discounted by the assigned region's `infrastructure_risk`
rather than sampled independently, per plan §13.
"""

from __future__ import annotations

import numpy as np

from ..schema.nodes import Plant, Region


def generate_plants(num_plants: int, regions: list[Region], rng: np.random.Generator) -> list[Plant]:
    plants = []
    for i in range(num_plants):
        region = regions[rng.integers(0, len(regions))]
        production_capacity = float(rng.lognormal(mean=np.log(2000), sigma=0.5))
        utilization = float(np.clip(rng.beta(5, 3), 0.0, 1.0))
        operating_cost = float(production_capacity * rng.uniform(2.0, 6.0))
        inventory_capacity = float(production_capacity * rng.uniform(0.5, 1.5))
        resilience_score = float(np.clip(rng.beta(4, 2) - 0.3 * region.infrastructure_risk, 0.0, 1.0))
        downtime_cost = float(operating_cost * rng.uniform(0.5, 2.0))
        recovery_rate = float(rng.uniform(0.05, 0.3))

        plants.append(
            Plant(
                plant_id=f"plant_{i}",
                region_id=region.region_id,
                production_capacity=production_capacity,
                utilization=utilization,
                operating_cost=operating_cost,
                inventory_capacity=inventory_capacity,
                resilience_score=resilience_score,
                downtime_cost=downtime_cost,
                recovery_rate=recovery_rate,
            )
        )
    return plants
