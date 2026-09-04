"""Region risk calibration from real public data (plan §42 Phase 4,
DATASET_DESIGN_REVIEW.md decision 3).

Builds a country-level table of (geopolitical_risk, trade_risk,
natural_disaster_risk, infrastructure_risk) normalized to [0, 1] from WGI +
INFORM, restricted to countries present in both sources. `cyber_risk` and
`transport_reliability` have no clean public country-level source we found
(DATASET_DESIGN_REVIEW.md decision 3) and stay synthetic — but correlated
with the real risk dimensions rather than drawn independently (plan §13).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import load_inform_risk, load_wgi_indicator


def _wgi_to_risk(value: float) -> float:
    """WGI estimates run roughly [-2.5, 2.5]; higher = more stable = lower risk."""
    return float(np.clip((2.5 - value) / 5.0, 0.0, 1.0))


def _inform_to_risk(value: float) -> float:
    """INFORM sub-scores run [0, 10]; higher already means more risk."""
    return float(np.clip(value / 10.0, 0.0, 1.0))


def build_region_risk_table(raw_dir: str) -> pd.DataFrame:
    political_stability = load_wgi_indicator(raw_dir, "political_stability.json")
    regulatory_quality = load_wgi_indicator(raw_dir, "regulatory_quality.json")
    inform_df = load_inform_risk(raw_dir)

    records = []
    for _, row in inform_df.iterrows():
        iso3 = row["iso3"]
        if iso3 not in political_stability or iso3 not in regulatory_quality:
            continue
        if pd.isna(row["natural_hazard"]) or pd.isna(row["infrastructure"]):
            continue
        records.append(
            {
                "country": row["country"],
                "iso3": iso3,
                "geopolitical_risk": _wgi_to_risk(political_stability[iso3]),
                "trade_risk": _wgi_to_risk(regulatory_quality[iso3]),
                "natural_disaster_risk": _inform_to_risk(row["natural_hazard"]),
                "infrastructure_risk": _inform_to_risk(row["infrastructure"]),
            }
        )
    return pd.DataFrame(records)


def load_region_risk_table(processed_path: str) -> pd.DataFrame:
    return pd.read_csv(processed_path)


def sample_calibrated_region_risk(table: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Pick one real country's row and return its 4 calibrated risk values
    plus synthetic-but-correlated cyber_risk/transport_reliability — these
    move with the country's overall real risk level (plan §13) rather than
    being resampled from scratch."""
    row = table.iloc[int(rng.integers(0, len(table)))]
    avg_risk = float(np.mean([row.geopolitical_risk, row.natural_disaster_risk, row.infrastructure_risk, row.trade_risk]))
    cyber_risk = float(np.clip(rng.normal(avg_risk, 0.15), 0.0, 1.0))
    transport_reliability = float(np.clip(rng.normal(1 - avg_risk, 0.15), 0.0, 1.0))
    return {
        "country_or_region": str(row.country),
        "geopolitical_risk": float(row.geopolitical_risk),
        "natural_disaster_risk": float(row.natural_disaster_risk),
        "infrastructure_risk": float(row.infrastructure_risk),
        "trade_risk": float(row.trade_risk),
        "cyber_risk": cyber_risk,
        "transport_reliability": transport_reliability,
    }
