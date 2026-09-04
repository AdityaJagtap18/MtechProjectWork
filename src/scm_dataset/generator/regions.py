"""Region generation (plan §5.6).

Two modes, selected by whether a calibration table is passed in:

- **Synthetic (default, Phase 2 behavior)**: constraint-based archetypes.
  Each region is drawn from one of a small number of risk archetypes rather
  than i.i.d. per-field sampling (plan §13): all five risk dimensions are
  centered on one shared per-region archetype mean instead of drawn
  independently uniform on [0, 1].
- **Calibrated (Phase 4, opt-in via `real_data.use_real_region_calibration`
  in config — see configs/real_calibration.yaml)**: each region's risk
  profile is a real country's WGI/INFORM-derived values, via
  `real_data.calibrator.sample_calibrated_region_risk` — see
  DATASET_DESIGN_REVIEW.md decision 3 for why only 4 of the 6 risk
  dimensions have a real source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..real_data.calibrator import sample_calibrated_region_risk
from ..schema.nodes import Region

# (name, base_risk, spread): base_risk centers all risk dimensions for a
# region drawn from this archetype; transport_reliability is centered on
# (1 - base_risk), since well-run regions tend to have both lower risk and
# more reliable transport.
_ARCHETYPES = [
    ("stable", 0.15, 0.05),
    ("moderate", 0.40, 0.10),
    ("volatile", 0.70, 0.10),
]


def generate_regions(
    num_regions: int, rng: np.random.Generator, region_risk_table: pd.DataFrame | None = None
) -> list[Region]:
    if region_risk_table is not None and len(region_risk_table) > 0:
        return _generate_calibrated_regions(num_regions, rng, region_risk_table)
    return _generate_synthetic_regions(num_regions, rng)


def _generate_synthetic_regions(num_regions: int, rng: np.random.Generator) -> list[Region]:
    regions = []
    for i in range(num_regions):
        name, base_risk, spread = _ARCHETYPES[rng.integers(0, len(_ARCHETYPES))]

        def clipped(mean: float = base_risk, spread: float = spread) -> float:
            return float(np.clip(rng.normal(mean, spread), 0.0, 1.0))

        regions.append(
            Region(
                region_id=f"region_{i}",
                country_or_region=f"{name}_{i}",
                geopolitical_risk=clipped(),
                natural_disaster_risk=clipped(),
                infrastructure_risk=clipped(),
                trade_risk=clipped(),
                cyber_risk=clipped(),
                transport_reliability=float(np.clip(rng.normal(1 - base_risk, spread), 0.0, 1.0)),
            )
        )
    return regions


def _generate_calibrated_regions(num_regions: int, rng: np.random.Generator, region_risk_table: pd.DataFrame) -> list[Region]:
    regions = []
    for i in range(num_regions):
        risk = sample_calibrated_region_risk(region_risk_table, rng)
        regions.append(
            Region(
                region_id=f"region_{i}",
                country_or_region=risk["country_or_region"],
                geopolitical_risk=risk["geopolitical_risk"],
                natural_disaster_risk=risk["natural_disaster_risk"],
                infrastructure_risk=risk["infrastructure_risk"],
                trade_risk=risk["trade_risk"],
                cyber_risk=risk["cyber_risk"],
                transport_reliability=risk["transport_reliability"],
            )
        )
    return regions
