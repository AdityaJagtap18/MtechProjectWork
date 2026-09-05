#!/usr/bin/env python3
"""A5-D4 -- Cross-dataset feature-mode ablation aggregation
(GRAPHSAGE_PHASE_A5_ROOT_CAUSE_INVESTIGATION.md §8).

Read-only: aggregates the 8 already-completed
`scripts/run_cross_dataset_experiment.py --feature-mode ...` runs (2
directions x 4 modes, 5 model seeds each) into the required summary
table. Does not train anything and does not write into any existing run
directory.

Usage:
    python scripts/analyze_crossdataset_feature_ablation.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from _analysis_common import find_latest_cross_dataset_runs_by_feature_mode

EXPERIMENTS_ROOT = "experiments/classical_gnn"
SEED43 = "scm_v1_black_swan_seed43"
SEED44 = "scm_v1_black_swan_seed44"
MODEL_SEEDS = [42, 43, 44, 45, 46]
FEATURE_MODES = ["dynamic_only", "static_only", "region_risk_only", "full"]
DIRECTIONS = [(SEED43, SEED44), (SEED44, SEED43)]


def _aggregate(run_dirs: dict[int, str], prefix: str) -> dict:
    prs, rocs, f1s, precs, recs, briers, eces = [], [], [], [], [], [], []
    fresh_recalls, ongoing_recalls, n_fresh, n_ongoing = [], [], [], []
    for run_dir in run_dirs.values():
        m = json.load(open(os.path.join(run_dir, f"metrics_{prefix}.json")))["by_split"]["test"]
        c = json.load(open(os.path.join(run_dir, f"calibration_{prefix}.json")))["test"]
        onset = json.load(open(os.path.join(run_dir, f"onset_breakdown_{prefix}.json")))["test"]
        prs.append(m["pr_auc"]); rocs.append(m["roc_auc"]); f1s.append(m["f1"])
        precs.append(m["precision"]); recs.append(m["recall"])
        briers.append(c["brier_score"]); eces.append(c["expected_calibration_error"])
        fresh_recalls.append(onset["recall_fresh_onset"])
        ongoing_recalls.append(onset["recall_already_ongoing"])
        n_fresh.append(onset["n_fresh_onset"])
        n_ongoing.append(onset["n_already_ongoing"])

    def s(x):
        vals = [v for v in x if v is not None]
        return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}" if vals else "N/A"

    return {
        "pr_auc": s(prs), "roc_auc": s(rocs), "f1": s(f1s), "precision": s(precs), "recall": s(recs),
        "brier": s(briers), "ece": s(eces),
        "fresh_onset_recall": s(fresh_recalls) if all(n > 0 for n in n_fresh) else f"N/A (n_fresh={n_fresh[0]})",
        "already_ongoing_recall": s(ongoing_recalls) if all(n > 0 for n in n_ongoing) else f"N/A (n_ongoing={n_ongoing[0]})",
    }


def main() -> None:
    rows = []
    for source, target in DIRECTIONS:
        for mode in FEATURE_MODES:
            run_dirs = find_latest_cross_dataset_runs_by_feature_mode(EXPERIMENTS_ROOT, source, target, MODEL_SEEDS, mode)
            for prefix, label in (("within_seed", "within_seed"), ("cross_dataset", "cross_dataset")):
                stats = _aggregate(run_dirs, prefix)
                rows.append({
                    "source_dataset": source, "target_dataset": target, "feature_mode": mode, "condition": label,
                    **stats,
                })

    result = pd.DataFrame(rows)
    csv_path = os.path.join(EXPERIMENTS_ROOT, "crossdataset_feature_ablation_summary.csv")
    result.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(result)} rows)")

    cross_only = result[result["condition"] == "cross_dataset"]

    md = ["# A5-D4 -- Cross-Dataset Feature-Mode Ablation\n"]
    md.append(f"Full table (within-seed and cross-dataset, both directions, all 4 feature modes): `{csv_path}`.\n")
    md.append("## Cross-dataset PR-AUC / ROC-AUC by feature mode\n")
    md.append("```\n" + cross_only[["source_dataset", "target_dataset", "feature_mode", "pr_auc", "roc_auc", "fresh_onset_recall", "already_ongoing_recall"]].to_string(index=False) + "\n```\n")

    md.append("## Interpretation\n")
    for source, target in DIRECTIONS:
        sub = cross_only[(cross_only["source_dataset"] == source) & (cross_only["target_dataset"] == target)]
        pr_aucs = sub.set_index("feature_mode")["pr_auc"].to_dict()
        md.append(f"**{source} -> {target}**: " + ", ".join(f"{mode}={pr_aucs.get(mode)}" for mode in FEATURE_MODES))
    md.append("")
    md.append("Per the plan's decision matrix (GRAPHSAGE_PHASE_A5_ROOT_CAUSE_INVESTIGATION.md §13): if every feature mode fails to transfer to a comparable degree in the same direction, that points to the transfer problem being a property of the label/region structure itself (consistent with A5-D1/D2/D3/D5's findings on disjoint disruption geography), not specific to any one feature group -- i.e. \"static-only also transfers poorly\" / \"not GraphSAGE-only\" rather than \"dynamic-only transfers better\" or \"full model uniquely fails\". Read the numbers above directly rather than assuming this in advance.\n")
    md.append("## Limitations\n")
    md.append("- Same two-dataset-realization caveat as the rest of Phase A.5: this is evidence from one seed43/seed44 pair, not five independent synthetic worlds.")

    md_path = os.path.join(EXPERIMENTS_ROOT, "CROSSDATASET_FEATURE_ABLATION.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
