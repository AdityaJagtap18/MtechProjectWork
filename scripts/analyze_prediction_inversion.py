#!/usr/bin/env python3
"""A5-D3 -- Prediction inversion analysis
(GRAPHSAGE_PHASE_A5_ROOT_CAUSE_INVESTIGATION.md §7).

Read-only diagnostic over the D2 cross-dataset predictions already on
disk. For each direction, ranks suppliers by mean predicted risk
(averaged over the 5 model seeds and all of that supplier's test rows)
and inspects the top/bottom of that ranking against ground truth, region,
static risk-exposure features, and graph degree.

Usage:
    python scripts/analyze_prediction_inversion.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scm_dataset.modeling.data import load_benchmark
from scm_dataset.schema.edges import EdgeType
from scm_dataset.schema.nodes import NodeType

from _analysis_common import load_cross_dataset_predictions

BENCHMARK_ROOT = "data/benchmark"
EXPERIMENTS_ROOT = "experiments/classical_gnn"
SEED43 = "scm_v1_black_swan_seed43"
SEED44 = "scm_v1_black_swan_seed44"
MODEL_SEEDS = [42, 43, 44, 45, 46]
OUT_DIR = "experiments/analysis/prediction_inversion"
TOP_N = [10, 20, 50]


def _supplier_attrs(benchmark) -> pd.DataFrame:
    material_degree: dict[str, int] = {}
    for e in benchmark.graph.edges_of_type(EdgeType.SUPPLIER_MATERIAL):
        material_degree[e.source_id] = material_degree.get(e.source_id, 0) + 1
    procurement_degree: dict[str, int] = {}
    for e in benchmark.graph.edges_of_type(EdgeType.SUPPLIER_PROCUREMENT):
        procurement_degree[e.source_id] = procurement_degree.get(e.source_id, 0) + 1

    rows = []
    for s in benchmark.graph.nodes_of_type(NodeType.SUPPLIER):
        rows.append({
            "supplier_id": s.supplier_id,
            "region_id": s.region_id,
            "geopolitical_exposure": s.geopolitical_exposure,
            "disaster_exposure": s.disaster_exposure,
            "cyber_exposure": s.cyber_exposure,
            "criticality": s.criticality,
            "supplier_material_degree": material_degree.get(s.supplier_id, 0),
            "supplier_procurement_degree": procurement_degree.get(s.supplier_id, 0),
        })
    return pd.DataFrame(rows).set_index("supplier_id")


def _per_supplier_summary(predictions: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    per_supplier = predictions.groupby("supplier_id").agg(
        mean_risk_probability=("risk_probability", "mean"),
        actual_disruption_rate=("actual_disruption", "mean"),
        n_rows=("risk_probability", "size"),
    )
    return per_supplier.join(attrs, how="left")


def _analyze_direction(name: str, target_benchmark, predictions: pd.DataFrame, out_dir: str) -> tuple[pd.DataFrame, list[str]]:
    attrs = _supplier_attrs(target_benchmark)
    summary = _per_supplier_summary(predictions, attrs).sort_values("mean_risk_probability", ascending=False)
    summary = summary.reset_index()

    tables = []
    for n in TOP_N:
        top = summary.head(n).copy()
        top["rank_group"] = f"top_{n}"
        bottom = summary.tail(n).copy()
        bottom["rank_group"] = f"bottom_{n}"
        tables.append(top)
        tables.append(bottom)
    combined = pd.concat(tables, ignore_index=True)
    combined.insert(0, "direction", name)

    rho_pred_actual, p_pred_actual = spearmanr(summary["mean_risk_probability"], summary["actual_disruption_rate"])
    rho_pred_degree_mat, _ = spearmanr(summary["mean_risk_probability"], summary["supplier_material_degree"])
    rho_pred_degree_proc, _ = spearmanr(summary["mean_risk_probability"], summary["supplier_procurement_degree"])
    rho_pred_exposure, _ = spearmanr(summary["mean_risk_probability"], (summary["geopolitical_exposure"] + summary["disaster_exposure"] + summary["cyber_exposure"]) / 3.0)

    top10 = summary.head(10)
    bottom10 = summary.tail(10)
    top10_region_counts = top10["region_id"].value_counts().to_dict()
    bottom10_region_counts = bottom10["region_id"].value_counts().to_dict()

    lines = []
    lines.append(f"### {name}\n")
    lines.append(f"- Spearman(predicted risk, actual disruption rate) across all {len(summary)} suppliers: **{rho_pred_actual:.3f}** (p={p_pred_actual:.3g})")
    lines.append(f"- Top-10 mean actual disruption rate: **{top10['actual_disruption_rate'].mean():.4f}**; bottom-10 mean actual disruption rate: **{bottom10['actual_disruption_rate'].mean():.4f}**")
    lines.append(f"- Spearman(predicted risk, supplier_material degree): {rho_pred_degree_mat:.3f}")
    lines.append(f"- Spearman(predicted risk, supplier_procurement degree): {rho_pred_degree_proc:.3f}")
    lines.append(f"- Spearman(predicted risk, mean static exposure score): {rho_pred_exposure:.3f}")
    lines.append(f"- Top-10 region concentration: {top10_region_counts}")
    lines.append(f"- Bottom-10 region concentration: {bottom10_region_counts}")
    lines.append("")
    lines.append("Top 10 by predicted risk:\n")
    lines.append("```\n" + top10[["supplier_id", "mean_risk_probability", "actual_disruption_rate", "region_id", "supplier_material_degree", "supplier_procurement_degree"]].to_string(index=False) + "\n```\n")
    lines.append("Bottom 10 by predicted risk:\n")
    lines.append("```\n" + bottom10[["supplier_id", "mean_risk_probability", "actual_disruption_rate", "region_id", "supplier_material_degree", "supplier_procurement_degree"]].to_string(index=False) + "\n```\n")

    return combined, lines


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    b43 = load_benchmark(BENCHMARK_ROOT, SEED43)
    b44 = load_benchmark(BENCHMARK_ROOT, SEED44)

    preds_43_to_44 = load_cross_dataset_predictions(EXPERIMENTS_ROOT, SEED43, SEED44, MODEL_SEEDS)
    preds_44_to_43 = load_cross_dataset_predictions(EXPERIMENTS_ROOT, SEED44, SEED43, MODEL_SEEDS)

    table_43_44, md_43_44 = _analyze_direction("seed43_to_seed44", b44, preds_43_to_44, OUT_DIR)
    table_44_43, md_44_43 = _analyze_direction("seed44_to_seed43", b43, preds_44_to_43, OUT_DIR)

    combined = pd.concat([table_43_44, table_44_43], ignore_index=True)
    csv_path = os.path.join(OUT_DIR, "prediction_inversion.csv")
    combined.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(combined)} rows)")

    md = ["# A5-D3 -- Prediction Inversion Analysis\n", f"Full top/bottom tables: `{csv_path}`.\n"]
    md += md_43_44
    md += md_44_43
    md.append("## Answers to the required questions\n")
    md.append("1. **Are highest-ranked suppliers actually safer than average?** See top-N actual disruption rate above per direction, and the full Spearman correlation (negative or near-zero correlation would indicate yes for seed43->seed44, per the D1 region-level finding of the same sign).")
    md.append("2. **Are lowest-ranked suppliers more disrupted?** Compare bottom-N actual disruption rate to the dataset base rate above.")
    md.append("3. **Is inversion concentrated in specific regions?** See the top/bottom-10 region concentration counts -- compare against the disjoint disrupted-region sets found in A5-D1.")
    md.append("4. **Is it associated with static risk features?** See Spearman(predicted risk, mean static exposure score) -- note A5-D2 found these features are numerically identical between seed43 and seed44, so any association here reflects the model's learned weighting, not an input distribution difference.")
    md.append("5. **Is it associated with graph degree/connectivity?** See Spearman(predicted risk, supplier_material / supplier_procurement degree) above.")
    md.append("6. **Is it associated with particular structural paths?** Not directly tested here (would require A5-D6, run only if D1-D5 leave the asymmetry unexplained).\n")
    md.append("## Limitations\n")
    md.append("- Correlational only; per rule #13, none of the above establishes causality, only association consistent or inconsistent with each hypothesis.")
    md.append("- Predictions are averaged over 5 model seeds and all of a supplier's test-period rows, which smooths over within-supplier time variation.")

    md_path = os.path.join(OUT_DIR, "PREDICTION_INVERSION_ANALYSIS.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
