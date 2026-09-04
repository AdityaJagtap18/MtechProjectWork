"""Statistical validation of calibrated attributes against real data (plan
§12's validation-report template, §35 "Statistical validation").

Region risk (from Phase 4's WGI/INFORM calibration) is the only Phase 4
target with both a real reference distribution and a synthetic generation
path to compare it against — DATASET_DESIGN_REVIEW.md decision 1 means no
other attribute has a real distribution available. This compares the real
country table against a fresh sample from the *default* synthetic
archetype generator (`generator.regions`, uncalibrated) — a standalone
check of the generator's realism, independent of any one exported dataset,
which is why it takes a region count and RNG rather than a specific graph.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from ..generator.regions import generate_regions

RISK_DIMENSIONS = ("geopolitical_risk", "natural_disaster_risk", "infrastructure_risk", "trade_risk")


def compare_region_risk_distributions(
    real_table: pd.DataFrame, rng: np.random.Generator, n_synthetic: int = 200
) -> pd.DataFrame:
    """Plan §12's per-feature template: feature, real_mean, synthetic_mean,
    real_std, synthetic_std, KS_statistic, wasserstein_distance."""
    synthetic_regions = generate_regions(n_synthetic, rng)

    records = []
    for dim in RISK_DIMENSIONS:
        real_values = real_table[dim].to_numpy(dtype=float)
        synthetic_values = np.array([getattr(r, dim) for r in synthetic_regions])
        ks_result = scipy_stats.ks_2samp(real_values, synthetic_values)
        records.append(
            {
                "feature": dim,
                "real_mean": float(np.mean(real_values)),
                "synthetic_mean": float(np.mean(synthetic_values)),
                "real_std": float(np.std(real_values)),
                "synthetic_std": float(np.std(synthetic_values)),
                "ks_statistic": float(ks_result.statistic),
                "wasserstein_distance": float(scipy_stats.wasserstein_distance(real_values, synthetic_values)),
            }
        )
    return pd.DataFrame(records)


def build_statistical_report(region_risk_table_path: str, seed: int = 0, n_synthetic: int = 200) -> dict | None:
    import os

    if not os.path.exists(region_risk_table_path):
        return None

    real_table = pd.read_csv(region_risk_table_path)
    rng = np.random.default_rng(seed)
    comparison = compare_region_risk_distributions(real_table, rng, n_synthetic)
    return {
        "region_risk_vs_real": comparison.to_dict(orient="records"),
        "note": (
            "Compares real WGI/INFORM country data against the DEFAULT synthetic "
            "archetype generator (not real-calibrated) -- a large KS/Wasserstein gap "
            "here is expected and is exactly why configs/real_calibration.yaml exists; "
            "it does not mean a real-calibrated dataset is uncalibrated."
        ),
    }
