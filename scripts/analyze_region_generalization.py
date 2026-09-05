#!/usr/bin/env python3
"""A5-D1 -- Regional distribution analysis
(GRAPHSAGE_PHASE_A5_ROOT_CAUSE_INVESTIGATION.md §5).

Read-only diagnostic: does not train anything, does not touch
data/benchmark/, does not write into any existing experiments/classical_gnn
run directory. Uses the D2 cross-dataset predictions already on disk
(scripts/run_cross_dataset_experiment.py) plus each dataset's own graph
and labels.

For every supplier region, reports (per H1, "regional concentration"):
  - supplier count, disrupted-supplier count, disruption rate, onset count
  - mean/median static risk-exposure score (geopolitical/disaster/cyber
    exposure -- genuinely time-invariant supplier fields) and mean/median
    realized risk score (the supplier_risk_score LABEL, i.e. a ground-truth
    outcome measure, not a model input -- reported for descriptive contrast
    only)
  - for the two cross-dataset directions: mean/median model risk_probability
    and PR-AUC/recall at the frozen 0.5 threshold, computed only where the
    region has enough positive examples to be meaningful (else N/A(n=...))

Usage:
    python scripts/analyze_region_generalization.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score

from scm_dataset.modeling.data import load_benchmark
from scm_dataset.schema.nodes import NodeType

from _analysis_common import load_cross_dataset_predictions, short_dataset_id


def _md_table(df: pd.DataFrame) -> str:
    """Renders a DataFrame as a fenced plain-text table (no `tabulate`
    dependency needed for a pipe-table)."""
    return "```\n" + df.to_string() + "\n```"

BENCHMARK_ROOT = "data/benchmark"
EXPERIMENTS_ROOT = "experiments/classical_gnn"
SEED43 = "scm_v1_black_swan_seed43"
SEED44 = "scm_v1_black_swan_seed44"
MODEL_SEEDS = [42, 43, 44, 45, 46]
MIN_POSITIVES_FOR_RANKING_METRIC = 5  # below this, PR-AUC/ROC-AUC is too noisy to report
OUT_DIR = "experiments/analysis/region_generalization"


def _region_ground_truth(benchmark) -> pd.DataFrame:
    supplier_region = {s.supplier_id: s.region_id for s in benchmark.graph.nodes_of_type(NodeType.SUPPLIER)}
    exposure = {
        s.supplier_id: (s.geopolitical_exposure + s.disaster_exposure + s.cyber_exposure) / 3.0
        for s in benchmark.graph.nodes_of_type(NodeType.SUPPLIER)
    }

    labels = benchmark.labels["supplier"].copy()
    labels["region_id"] = labels["supplier_id"].map(supplier_region)
    labels = labels.sort_values(["supplier_id", "time"])
    labels["prev_disrupted"] = labels.groupby("supplier_id")["supplier_disrupted"].shift(1).fillna(0)
    labels["is_onset"] = (labels["supplier_disrupted"] == 1) & (labels["prev_disrupted"] == 0)

    per_supplier = labels.groupby("supplier_id").agg(
        region_id=("region_id", "first"),
        ever_disrupted=("supplier_disrupted", "max"),
        mean_realized_risk_score=("supplier_risk_score", "mean"),
        onset_count=("is_onset", "sum"),
    )
    per_supplier["static_exposure_score"] = per_supplier.index.map(exposure)

    rows = []
    for region_id, group in per_supplier.groupby("region_id"):
        rows.append({
            "region_id": region_id,
            "supplier_count": len(group),
            "disrupted_supplier_count": int(group["ever_disrupted"].sum()),
            "disruption_rate": float(group["ever_disrupted"].mean()),
            "onset_count": int(group["onset_count"].sum()),
            "mean_static_exposure_score": float(group["static_exposure_score"].mean()),
            "median_static_exposure_score": float(group["static_exposure_score"].median()),
            "mean_realized_risk_score": float(group["mean_realized_risk_score"].mean()),
            "median_realized_risk_score": float(group["mean_realized_risk_score"].median()),
        })
    return pd.DataFrame(rows)


def _region_predictions(benchmark, predictions: pd.DataFrame) -> pd.DataFrame:
    supplier_region = {s.supplier_id: s.region_id for s in benchmark.graph.nodes_of_type(NodeType.SUPPLIER)}
    predictions = predictions.copy()
    predictions["region_id"] = predictions["supplier_id"].map(supplier_region)

    rows = []
    for region_id, group in predictions.groupby("region_id"):
        n_pos = int(group["actual_disruption"].sum())
        if n_pos >= MIN_POSITIVES_FOR_RANKING_METRIC and group["actual_disruption"].nunique() > 1:
            pr_auc = float(average_precision_score(group["actual_disruption"], group["risk_probability"]))
            roc_auc = float(roc_auc_score(group["actual_disruption"], group["risk_probability"]))
            recall = float(recall_score(group["actual_disruption"], (group["risk_probability"] >= 0.5).astype(int), zero_division=0))
        else:
            pr_auc = None
            roc_auc = None
            recall = None
        rows.append({
            "region_id": region_id,
            "n_examples": len(group),
            "n_positive": n_pos,
            "mean_model_prediction": float(group["risk_probability"].mean()),
            "median_model_prediction": float(group["risk_probability"].median()),
            "pr_auc": pr_auc if pr_auc is not None else f"N/A (n={n_pos})",
            "roc_auc": roc_auc if roc_auc is not None else f"N/A (n={n_pos})",
            "recall_at_0.5": recall if recall is not None else f"N/A (n={n_pos})",
        })
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    b43 = load_benchmark(BENCHMARK_ROOT, SEED43)
    b44 = load_benchmark(BENCHMARK_ROOT, SEED44)

    gt43 = _region_ground_truth(b43)
    gt43.insert(0, "source", "seed43_ground_truth")
    gt44 = _region_ground_truth(b44)
    gt44.insert(0, "source", "seed44_ground_truth")

    preds_43_to_44 = load_cross_dataset_predictions(EXPERIMENTS_ROOT, SEED43, SEED44, MODEL_SEEDS)
    preds_44_to_43 = load_cross_dataset_predictions(EXPERIMENTS_ROOT, SEED44, SEED43, MODEL_SEEDS)

    pred43to44 = _region_predictions(b44, preds_43_to_44)  # target graph is seed44
    pred43to44.insert(0, "source", "seed43_to_seed44_predictions")
    pred44to43 = _region_predictions(b43, preds_44_to_43)  # target graph is seed43
    pred44to43.insert(0, "source", "seed44_to_seed43_predictions")

    combined = pd.concat([gt43, gt44, pred43to44, pred44to43], ignore_index=True, sort=False)
    csv_path = os.path.join(OUT_DIR, "region_breakdown.csv")
    combined.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(combined)} rows)")

    # ---- interpretive comparisons (no causal claims -- rule #13) ----
    gt43_i = gt43.set_index("region_id")
    gt44_i = gt44.set_index("region_id")
    pred43to44_i = pred43to44.set_index("region_id")
    pred44to43_i = pred44to43.set_index("region_id")

    geography_shift = pd.DataFrame({
        "seed43_disruption_rate": gt43_i["disruption_rate"],
        "seed44_disruption_rate": gt44_i["disruption_rate"],
    })
    geography_shift["rate_delta_44_minus_43"] = geography_shift["seed44_disruption_rate"] - geography_shift["seed43_disruption_rate"]
    geography_shift = geography_shift.sort_values("rate_delta_44_minus_43")

    over_ranked_43_to_44 = pred43to44_i[["mean_model_prediction"]].join(gt44_i[["disruption_rate"]], how="left")
    over_ranked_43_to_44["gap_prediction_minus_actual"] = over_ranked_43_to_44["mean_model_prediction"] - over_ranked_43_to_44["disruption_rate"]
    over_ranked_43_to_44 = over_ranked_43_to_44.sort_values("gap_prediction_minus_actual", ascending=False)

    corr_43to44 = pred43to44_i["mean_model_prediction"].corr(gt44_i["disruption_rate"], method="spearman")
    corr_44to43 = pred44to43_i["mean_model_prediction"].corr(gt43_i["disruption_rate"], method="spearman")
    corr_43_within = gt43_i["mean_static_exposure_score"].corr(gt43_i["disruption_rate"], method="spearman")
    corr_44_within = gt44_i["mean_static_exposure_score"].corr(gt44_i["disruption_rate"], method="spearman")

    n_regions_seed43_only_disrupted = int(((gt43_i["disruption_rate"] > 0) & (gt44_i["disruption_rate"] == 0)).sum())
    n_regions_seed44_only_disrupted = int(((gt44_i["disruption_rate"] > 0) & (gt43_i["disruption_rate"] == 0)).sum())
    n_regions_both = int(((gt43_i["disruption_rate"] > 0) & (gt44_i["disruption_rate"] > 0)).sum())
    n_regions_neither = int(((gt43_i["disruption_rate"] == 0) & (gt44_i["disruption_rate"] == 0)).sum())

    md = []
    md.append("# A5-D1 -- Regional Distribution Analysis\n")
    md.append(f"Full per-region data: `{csv_path}` ({combined['region_id'].nunique()} regions).\n")
    md.append("## Regional disruption geography: seed43 vs seed44\n")
    md.append(f"- Regions disrupted in seed43 but not seed44: **{n_regions_seed43_only_disrupted}**")
    md.append(f"- Regions disrupted in seed44 but not seed43: **{n_regions_seed44_only_disrupted}**")
    md.append(f"- Regions disrupted in both: **{n_regions_both}**")
    md.append(f"- Regions disrupted in neither: **{n_regions_neither}**\n")
    md.append("Per-region disruption rate, sorted by (seed44 - seed43) delta (most negative first -- regions that mattered in seed43 but not seed44):\n")
    md.append(_md_table(geography_shift.round(4)))
    md.append("")
    md.append("## Does the model's cross-dataset ranking track the target's actual regional geography?\n")
    md.append(f"- Spearman correlation, seed43-trained model's mean predicted risk per region vs seed44's actual regional disruption rate: **{corr_43to44:.3f}**")
    md.append(f"- Spearman correlation, seed44-trained model's mean predicted risk per region vs seed43's actual regional disruption rate: **{corr_44to43:.3f}**")
    md.append(f"- (Within-dataset reference) Spearman correlation, seed43's own static exposure score vs seed43's own disruption rate: **{corr_43_within:.3f}**")
    md.append(f"- (Within-dataset reference) Spearman correlation, seed44's own static exposure score vs seed44's own disruption rate: **{corr_44_within:.3f}**\n")
    md.append("A near-zero or negative cross-dataset correlation, contrasted with a positive within-dataset correlation, would be *consistent with* (not proof of) H1: the model ranking regions by something specific to the training dataset's realized geography rather than a property that transfers.\n")
    md.append("## Regions most over-ranked by the seed43-trained model on seed44 (predicted risk far above seed44's actual regional disruption rate)\n")
    md.append(_md_table(over_ranked_43_to_44.round(4).head(10)))
    md.append("\nSee the full CSV for the seed44-trained-on-seed43 direction and per-region PR-AUC/ROC-AUC/recall where statistically meaningful (`N/A (n=...)` otherwise).\n")
    md.append("## Limitations\n")
    md.append(f"- Region-level PR-AUC/ROC-AUC/recall required at least {MIN_POSITIVES_FOR_RANKING_METRIC} positive examples in that region's rows to be reported; many regions fall short given seed43's overall 46 onsets and seed44's 93, spread across 20 regions. See `N/A (n=...)` entries in the CSV.")
    md.append("- This is descriptive/correlational analysis over two dataset realizations. It cannot establish that regional concentration *causes* the transfer asymmetry -- only whether the pattern is consistent with that hypothesis.")

    md_path = os.path.join(OUT_DIR, "REGION_GENERALIZATION_ANALYSIS.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
