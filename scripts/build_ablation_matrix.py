#!/usr/bin/env python3
"""Builds the Phase J ablation matrix (GRAPH_SAGE_IMPROVEMENT_PLAN.md §14)
from already-saved run directories -- pure aggregation over each run's
`metrics.json` and `onset_breakdown.json`, no training involved. Multiple
run directories under one `--row` are averaged (e.g. one per seed).

Usage:
    python scripts/build_ablation_matrix.py \\
        --row "Majority" experiments/classical_gnn/<majority_run> \\
        --row "Logistic Regression" experiments/classical_gnn/<logreg_run> \\
        --row "Dynamic-only GraphSAGE" experiments/classical_gnn/<run_seed42> experiments/classical_gnn/<run_seed43> ... \\
        --row "Static-only GraphSAGE" experiments/classical_gnn/<run>... \\
        --row "Region/Risk-only GraphSAGE" experiments/classical_gnn/<run>... \\
        --row "Static + Graph Structure" experiments/classical_gnn/<run>... \\
        --row "Full GraphSAGE" experiments/classical_gnn/<run>... \\
        --output experiments/classical_gnn/ablation_matrix.md
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

COLUMNS = ["Model/Input", "n_runs", "Overall PR-AUC", "Overall ROC-AUC", "Fresh-Onset PR-AUC", "Fresh-Onset ROC-AUC", "Fresh-Onset Recall", "Already-Ongoing Recall"]


def _load_run(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "metrics.json")) as f:
        metrics = json.load(f)
    test_metrics = metrics.get("by_split", metrics).get("test", {})

    onset_path = os.path.join(run_dir, "onset_breakdown.json")
    onset_test = {}
    if os.path.exists(onset_path):
        with open(onset_path) as f:
            onset = json.load(f)
        onset_test = onset.get("test", {})

    return {
        "pr_auc": test_metrics.get("pr_auc"),
        "roc_auc": test_metrics.get("roc_auc"),
        "pr_auc_fresh_onset": onset_test.get("pr_auc_fresh_onset"),
        "roc_auc_fresh_onset": onset_test.get("roc_auc_fresh_onset"),
        "recall_fresh_onset": onset_test.get("recall_fresh_onset"),
        "recall_already_ongoing": onset_test.get("recall_already_ongoing"),
    }


def _mean_or_none(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def build_matrix(rows: list[tuple[str, list[str]]]) -> pd.DataFrame:
    records = []
    for label, run_dirs in rows:
        per_run = [_load_run(d) for d in run_dirs]
        records.append(
            {
                "Model/Input": label,
                "n_runs": len(run_dirs),
                "Overall PR-AUC": _mean_or_none([r["pr_auc"] for r in per_run]),
                "Overall ROC-AUC": _mean_or_none([r["roc_auc"] for r in per_run]),
                "Fresh-Onset PR-AUC": _mean_or_none([r["pr_auc_fresh_onset"] for r in per_run]),
                "Fresh-Onset ROC-AUC": _mean_or_none([r["roc_auc_fresh_onset"] for r in per_run]),
                "Fresh-Onset Recall": _mean_or_none([r["recall_fresh_onset"] for r in per_run]),
                "Already-Ongoing Recall": _mean_or_none([r["recall_already_ongoing"] for r in per_run]),
            }
        )
    return pd.DataFrame(records, columns=COLUMNS)


def to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(COLUMNS) + " |"
    separator = "|---|---:|" + "---:|" * (len(COLUMNS) - 2)
    lines = [header, separator]
    for _, row in df.iterrows():
        cells = [row["Model/Input"], str(row["n_runs"])] + [_fmt(row[c]) for c in COLUMNS[2:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--row", action="append", nargs="+", metavar="LABEL_OR_DIR",
        help="One row: a label followed by one or more run directories to average, e.g. --row \"Full GraphSAGE\" run1 run2",
    )
    parser.add_argument("--output", default=None, help="Write the markdown table to this path (also always printed).")
    args = parser.parse_args()

    if not args.row:
        parser.error("at least one --row is required")

    rows = [(entry[0], entry[1:]) for entry in args.row]
    for label, run_dirs in rows:
        if not run_dirs:
            parser.error(f"--row {label!r} needs at least one run directory")
        for d in run_dirs:
            if not os.path.isdir(d):
                parser.error(f"run directory does not exist: {d}")

    df = build_matrix(rows)
    table = to_markdown(df)
    print(table)

    if args.output:
        with open(args.output, "w") as f:
            f.write(f"# Ablation Matrix\n\n{table}\n")
        csv_path = os.path.splitext(args.output)[0] + ".csv"
        df.to_csv(csv_path, index=False)
        print(f"\nWritten -> {args.output}\nWritten -> {csv_path}")


if __name__ == "__main__":
    main()
