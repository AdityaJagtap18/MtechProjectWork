"""Demand generation (plan §10).

Demand is not i.i.d. per period: each product's mean demand follows a slow
random walk (drift) around its Phase 2 baseline (`product.demand`),
modulated by a per-product seasonal cycle — a resolution-agnostic
generalization of SupplySim's weekday/weekend pattern (see
DATASET_DESIGN_REVIEW.md §2.1: reused as an idea, not as code) — and then
Gamma-sampled around that mean each period so realized demand still varies
period to period without going negative.
"""

from __future__ import annotations

import numpy as np

from ..generator.config import SimulationConfig
from ..schema.nodes import Product


def generate_demand_schedule(
    products: list[Product],
    horizon_periods: int,
    periods_per_year: int,
    sim: SimulationConfig,
    rng: np.random.Generator,
) -> dict[tuple[str, int], float]:
    schedule: dict[tuple[str, int], float] = {}
    base = {p.product_id: max(p.demand, 1e-3) for p in products}
    phase = {p.product_id: float(rng.uniform(0, 2 * np.pi)) for p in products}

    for t in range(horizon_periods):
        for product in products:
            pid = product.product_id
            base[pid] = max(1e-3, base[pid] + float(rng.normal(0, base[pid] * sim.demand_drift_std)))
            seasonal = 1 + sim.demand_seasonal_amplitude * np.sin(2 * np.pi * t / periods_per_year + phase[pid])
            mean_demand = max(1e-6, base[pid] * seasonal)
            schedule[(pid, t)] = float(rng.gamma(shape=4.0, scale=mean_demand / 4.0))

    return schedule
