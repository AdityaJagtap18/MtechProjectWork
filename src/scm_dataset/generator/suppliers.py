"""Supplier generation (plan §5.1).

Attributes are generated via dependency functions per plan §13 rather than
independent sampling: `financial_health`/`reliability`/`quality_score` share
a common latent "operational health" factor instead of being drawn as three
separate independent variables; `reliability` is further discounted when
`capacity_utilization` is high (less slack to absorb variance); and the
three exposure scores are inherited from the assigned region's own risk
profile plus idiosyncratic noise, not resampled from scratch.
"""

from __future__ import annotations

import numpy as np

from ..schema.nodes import Region, Supplier

INDUSTRIES = ["electronics", "metals", "chemicals", "textiles", "machinery", "software", "logistics"]


def _beta_from_mean(rng: np.random.Generator, mean: float, concentration: float = 8.0) -> float:
    mean = min(max(mean, 1e-3), 1 - 1e-3)
    return float(rng.beta(mean * concentration, (1 - mean) * concentration))


def generate_suppliers(num_suppliers: int, regions: list[Region], rng: np.random.Generator) -> list[Supplier]:
    suppliers = []
    for i in range(num_suppliers):
        region = regions[rng.integers(0, len(regions))]
        industry = INDUSTRIES[rng.integers(0, len(INDUSTRIES))]
        tier = int(rng.integers(1, 4))  # tiers 1-3

        latent_health = np.clip(rng.normal(0.6, 0.15), 0.0, 1.0)
        financial_health = _beta_from_mean(rng, float(np.clip(latent_health + rng.normal(0, 0.05), 0, 1)))
        reliability = _beta_from_mean(rng, float(np.clip(latent_health + rng.normal(0, 0.05), 0, 1)))
        quality_score = _beta_from_mean(rng, float(np.clip(latent_health + rng.normal(0, 0.05), 0, 1)))

        capacity = float(rng.lognormal(mean=np.log(1000) - (tier - 1) * 0.3, sigma=0.5))
        capacity_utilization = _beta_from_mean(rng, 0.55)
        reliability = float(np.clip(reliability - 0.15 * max(0.0, capacity_utilization - 0.8), 0.0, 1.0))

        lead_time_mean = float(rng.gamma(shape=4.0, scale=2.5))
        lead_time_variability = float(lead_time_mean * rng.uniform(0.1, 0.4))
        inventory_buffer = float(rng.gamma(shape=2.0, scale=max(capacity * 0.05, 1e-3)))
        substitution_availability = _beta_from_mean(rng, 0.5)

        geopolitical_exposure = float(np.clip(region.geopolitical_risk + rng.normal(0, 0.05), 0, 1))
        disaster_exposure = float(np.clip(region.natural_disaster_risk + rng.normal(0, 0.05), 0, 1))
        cyber_exposure = float(np.clip(region.cyber_risk + rng.normal(0, 0.05), 0, 1))

        criticality = _beta_from_mean(rng, 0.4)

        suppliers.append(
            Supplier(
                supplier_id=f"supplier_{i}",
                tier=tier,
                region_id=region.region_id,
                industry=industry,
                capacity=capacity,
                capacity_utilization=capacity_utilization,
                reliability=reliability,
                financial_health=financial_health,
                lead_time_mean=lead_time_mean,
                lead_time_variability=lead_time_variability,
                quality_score=quality_score,
                inventory_buffer=inventory_buffer,
                substitution_availability=substitution_availability,
                geopolitical_exposure=geopolitical_exposure,
                disaster_exposure=disaster_exposure,
                cyber_exposure=cyber_exposure,
                criticality=criticality,
            )
        )
    return suppliers
