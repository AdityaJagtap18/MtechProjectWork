#!/usr/bin/env python3
"""Evaluate an already-trained GraphSAGE run directory (writes/refreshes
that same run's evaluation artifacts -- predictions, metrics, calibration,
risk ranking, plots -- never a new run directory, since it's re-evaluating
one existing trained model rather than producing a new experiment).
Re-loads the run's own saved `preprocessing/` artifacts rather than
refitting them (plan §15: evaluation must never refit on its own data mix).

Usage:
    python scripts/evaluate_graphsage.py --run-dir experiments/classical_gnn/<run_id>
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import torch

from scm_dataset.modeling.config import load_config
from scm_dataset.modeling.evaluate import evaluate_experiment, render_all_plots
from scm_dataset.modeling.experiment import write_json
from scm_dataset.modeling.graphsage import build_model
from scm_dataset.modeling.pipeline import prepare
from scm_dataset.modeling.preprocessing import FeaturePreprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    config = load_config(os.path.join(args.run_dir, "config.yaml"))
    preprocessor = FeaturePreprocessor.load(os.path.join(args.run_dir, "preprocessing"))
    prepared = prepare(config, preprocessor=preprocessor)

    in_dims = {nt.value: dim for nt, dim in prepared.snapshot_builder.feature_dims().items()}
    edge_types = list(prepared.snapshot_builder.topology.edge_index_dict.keys())
    model = build_model(
        in_dims=in_dims, edge_types=edge_types, hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers,
        dropout=config.model.dropout, aggregation=config.model.aggregation,
    )
    model.load_state_dict(torch.load(os.path.join(args.run_dir, "model.pt"), weights_only=True))

    eval_result = evaluate_experiment(model, prepared, config.threshold)

    eval_result.predictions.to_csv(os.path.join(args.run_dir, "predictions.csv"), index=False)
    eval_result.risk_ranking.to_csv(os.path.join(args.run_dir, "supplier_risk_ranking.csv"), index=False)
    eval_result.warning_times.to_csv(os.path.join(args.run_dir, "early_warning.csv"), index=False)
    write_json(os.path.join(args.run_dir, "onset_breakdown.json"), eval_result.onset_breakdown)
    write_json(os.path.join(args.run_dir, "metrics.json"), {
        "threshold_policy": eval_result.threshold_policy, "threshold": eval_result.threshold,
        "by_split": eval_result.metrics_by_split,
    })
    write_json(os.path.join(args.run_dir, "classification_report.json"), eval_result.metrics_by_split.get("test", {}))
    cm = eval_result.metrics_by_split.get("test", {}).get("confusion_matrix")
    if cm:
        pd.DataFrame([cm]).to_csv(os.path.join(args.run_dir, "confusion_matrix.csv"), index=False)
    write_json(os.path.join(args.run_dir, "calibration.json"), eval_result.calibration_by_split)

    history_path = os.path.join(args.run_dir, "training_history.csv")
    history = pd.read_csv(history_path) if os.path.exists(history_path) else pd.DataFrame()
    render_all_plots(eval_result, history, os.path.join(args.run_dir, "plots"))

    print(f"Re-evaluated -> {args.run_dir}")
    if "test" in eval_result.metrics_by_split:
        m = eval_result.metrics_by_split["test"]
        print(f"  test: pr_auc={m['pr_auc']} roc_auc={m['roc_auc']} f1={m['f1']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f}")


if __name__ == "__main__":
    main()
