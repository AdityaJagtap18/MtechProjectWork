#!/usr/bin/env python3
"""A5-D2 -- Feature distribution shift
(GRAPHSAGE_PHASE_A5_ROOT_CAUSE_INVESTIGATION.md §6).

Read-only diagnostic. Compares seed43 vs seed44's own model-input feature
distributions (each dataset's full, independently-built feature frame --
no preprocessing is fit here, on either dataset alone or combined; this is
purely descriptive statistics over already-generated data).

Usage:
    python scripts/analyze_feature_shift.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from scm_dataset.modeling.data import load_benchmark
from scm_dataset.modeling.features import build_feature_frames, STATIC_CATEGORICAL_FIELDS
from scm_dataset.schema.nodes import NodeType

BENCHMARK_ROOT = "data/benchmark"
SEED43 = "scm_v1_black_swan_seed43"
SEED44 = "scm_v1_black_swan_seed44"
ROLLING_WINDOWS = [4, 8, 12]
OUT_DIR = "experiments/analysis/feature_shift"
PERCENTILES = [1, 5, 25, 50, 75, 95, 99]


def _numeric_stats(series: pd.Series) -> dict:
    s = series.dropna()
    stats = {
        "dtype": str(series.dtype),
        "missing_rate": float(series.isna().mean()),
        "mean": float(s.mean()) if len(s) else None,
        "std": float(s.std()) if len(s) else None,
        "min": float(s.min()) if len(s) else None,
        "max": float(s.max()) if len(s) else None,
    }
    for p in PERCENTILES:
        stats[f"p{p}"] = float(np.percentile(s, p)) if len(s) else None
    return stats


def _compare_numeric_column(node_type: str, column: str, a: pd.Series, b: pd.Series) -> dict:
    row = {"node_type": node_type, "column": column, "kind": "numeric"}
    stats_a = _numeric_stats(a)
    stats_b = _numeric_stats(b)
    row.update({f"seed43_{k}": v for k, v in stats_a.items()})
    row.update({f"seed44_{k}": v for k, v in stats_b.items()})

    a_clean, b_clean = a.dropna(), b.dropna()
    if len(a_clean) and len(b_clean):
        pooled_std = np.sqrt((a_clean.std() ** 2 + b_clean.std() ** 2) / 2.0)
        row["standardized_mean_diff"] = float((b_clean.mean() - a_clean.mean()) / pooled_std) if pooled_std > 0 else (0.0 if b_clean.mean() == a_clean.mean() else None)
        ks_stat, ks_p = ks_2samp(a_clean, b_clean)
        row["ks_statistic"] = float(ks_stat)
        row["ks_pvalue"] = float(ks_p)
        row["wasserstein_distance"] = float(wasserstein_distance(a_clean, b_clean))
    else:
        row["standardized_mean_diff"] = None
        row["ks_statistic"] = None
        row["ks_pvalue"] = None
        row["wasserstein_distance"] = None
    return row


def _compare_categorical_column(node_type: str, column: str, a: pd.Series, b: pd.Series) -> dict:
    prop_a = a.value_counts(normalize=True, dropna=False)
    prop_b = b.value_counts(normalize=True, dropna=False)
    all_categories = sorted(set(prop_a.index) | set(prop_b.index), key=str)
    diffs = {cat: abs(float(prop_a.get(cat, 0.0)) - float(prop_b.get(cat, 0.0))) for cat in all_categories}
    return {
        "node_type": node_type,
        "column": column,
        "kind": "categorical",
        "seed43_categories": dict(prop_a.round(4)),
        "seed44_categories": dict(prop_b.round(4)),
        "max_abs_proportion_diff": max(diffs.values()) if diffs else 0.0,
        "total_variation_distance": 0.5 * sum(diffs.values()),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    b43 = load_benchmark(BENCHMARK_ROOT, SEED43)
    b44 = load_benchmark(BENCHMARK_ROOT, SEED44)

    frames43 = build_feature_frames(b43.graph, b43.operations, b43.horizon_periods, ROLLING_WINDOWS)
    frames44 = build_feature_frames(b44.graph, b44.operations, b44.horizon_periods, ROLLING_WINDOWS)

    rows = []
    for node_type in (NodeType.SUPPLIER, NodeType.REGION):
        df43 = frames43.frames[node_type]
        df44 = frames44.frames[node_type]
        categorical_cols = set(STATIC_CATEGORICAL_FIELDS.get(node_type, []))
        for column in df43.columns:
            if column not in df44.columns:
                continue
            if column in categorical_cols:
                rows.append(_compare_categorical_column(node_type.value, column, df43[column], df44[column]))
            else:
                rows.append(_compare_numeric_column(node_type.value, column, df43[column], df44[column]))

    result = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "feature_distribution_comparison.csv")
    result.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(result)} feature rows)")

    numeric = result[result["kind"] == "numeric"].copy()
    numeric_ranked = numeric.reindex(numeric["standardized_mean_diff"].abs().sort_values(ascending=False).index)

    md = []
    md.append("# A5-D2 -- Feature Distribution Shift\n")
    md.append(f"Full comparison: `{csv_path}` ({len(result)} features across supplier and region node types, full feature mode, {ROLLING_WINDOWS} rolling windows).\n")
    md.append("No preprocessing was fit anywhere in this analysis, on either dataset alone or combined -- this is descriptive statistics only over each dataset's own independently-built feature frame.\n")
    md.append("## Largest standardized mean differences (|SMD|), numeric features\n")
    md.append("```\n" + numeric_ranked[["node_type", "column", "standardized_mean_diff", "ks_statistic", "wasserstein_distance"]].head(15).to_string(index=False) + "\n```\n")
    md.append("## Smallest standardized mean differences (|SMD|), numeric features\n")
    md.append("```\n" + numeric_ranked[["node_type", "column", "standardized_mean_diff", "ks_statistic", "wasserstein_distance"]].tail(10).to_string(index=False) + "\n```\n")

    categorical = result[result["kind"] == "categorical"]
    if len(categorical):
        md.append("## Categorical features\n")
        for _, row in categorical.iterrows():
            md.append(f"- `{row['node_type']}.{row['column']}`: max proportion diff = {row['max_abs_proportion_diff']:.4f}, total variation distance = {row['total_variation_distance']:.4f}")
            md.append(f"  - seed43: {row['seed43_categories']}")
            md.append(f"  - seed44: {row['seed44_categories']}")
        md.append("")

    n_large_shift = int((numeric["standardized_mean_diff"].abs() > 0.5).sum())
    md.append("## Interpretation\n")
    md.append(f"{n_large_shift} of {len(numeric)} numeric features have |standardized mean difference| > 0.5 (a conventional \"medium effect size\" threshold, not a formal significance test).")
    md.append("A large distribution shift on a feature the model relies on heavily is *consistent with* covariate shift contributing to the transfer failure (H2), but does not by itself establish that this specific shift is the mechanism -- that requires relating shifted features to the regions/suppliers implicated in the D1 and D3 analyses, and to the feature-mode ablation in D4.\n")
    md.append("## Limitations\n")
    md.append("- Standardized mean difference and Wasserstein distance are on each feature's raw (unstandardized) scale; a large absolute shift on a feature with naturally high variance is not directly comparable to the same absolute shift on a low-variance feature. Use KS statistic (scale-free, in [0,1]) for cross-feature comparison of shift magnitude.")
    md.append("- Dynamic (rolling-window) features are compared pooling all (supplier, time) rows across the full 104-period horizon in each dataset, not restricted to any particular split.")

    md_path = os.path.join(OUT_DIR, "FEATURE_SHIFT_ANALYSIS.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
