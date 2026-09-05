#!/usr/bin/env python3
"""End-to-end classical GraphSAGE experiment (plan §35/§69/§75/§76):

    load benchmark -> leakage-safe features -> train-only preprocessing
    -> HeteroData -> train (early stopping on validation PR-AUC)
    -> evaluate (predictions, metrics, calibration, risk ranking, plots)
    -> save experiment artifacts under experiments/classical_gnn/<run_id>/

One run per seed; pass multiple `--seeds` to repeat the same prepared data
(loaded/preprocessed once) across seeds and additionally write a
mean +/- std summary (plan §59/§62/§70). Pass `--baselines` to also run the
Majority and Logistic Regression baselines (plan §53-57) against the same
prepared data/split/target.

Usage:
    python scripts/run_graphsage_experiment.py --config configs/graphsage.yaml --seeds 42
    python scripts/run_graphsage_experiment.py --config configs/graphsage.yaml --seeds 42,43,44,45,46 --baselines
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from scm_dataset.modeling.baselines import run_logistic_regression_baseline, run_majority_baseline
from scm_dataset.modeling.config import load_config
from scm_dataset.modeling.evaluate import evaluate_experiment, render_all_plots
from scm_dataset.modeling.experiment import build_run_metadata, new_run_dir, save_config, write_json
from scm_dataset.modeling.pipeline import prepare
from scm_dataset.modeling.train import class_balance_summary, train_graphsage


_PRIMARY_DATASET_ID = "scm_v1_black_swan_seed43"


def _strategy_tag(config) -> str:
    # Keeps existing "temporal"/"full"/seed43 run directory names
    # unchanged; distinguishes severity/scenario generalization runs,
    # feature-mode ablations (GRAPH_SAGE_IMPROVEMENT_PLAN.md §6), and a
    # non-primary dataset (D1's seed44 diagnostic,
    # GRAPHSAGE_FINAL_GENERALIZATION_AND_IMPROVEMENT_PLAN.md) so they
    # don't collide with or read as more primary runs once several kinds
    # accumulate side by side.
    split_tag = "" if config.split.strategy == "temporal" else f"_{config.split.strategy}"
    feature_tag = "" if config.features.feature_mode == "full" else f"_{config.features.feature_mode}"
    dataset_tag = "" if config.dataset.dataset_id == _PRIMARY_DATASET_ID else f"_{config.dataset.dataset_id.replace('scm_v1_black_swan_', '')}"
    return split_tag + feature_tag + dataset_tag


def _save_graphsage_run(prepared, train_result, eval_result, config, seed: int) -> str:
    run_dir = new_run_dir(config.experiment.output_dir, f"hetero_graphsage{_strategy_tag(config)}_seed{seed}")
    save_config(run_dir, config)

    import torch

    torch.save(train_result.model.state_dict(), os.path.join(run_dir, "model.pt"))
    prepared.preprocessor.save(os.path.join(run_dir, "preprocessing"))

    eval_result.predictions.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)
    eval_result.risk_ranking.to_csv(os.path.join(run_dir, "supplier_risk_ranking.csv"), index=False)
    eval_result.warning_times.to_csv(os.path.join(run_dir, "early_warning.csv"), index=False)
    write_json(os.path.join(run_dir, "onset_breakdown.json"), eval_result.onset_breakdown)
    write_json(os.path.join(run_dir, "temporal_variation.json"), eval_result.temporal_variation)
    train_result.history.to_csv(os.path.join(run_dir, "training_history.csv"), index=False)

    test_variation = eval_result.temporal_variation.get("test", {})
    if test_variation.get("fraction_time_varying") is not None and test_variation["fraction_time_varying"] < 0.5:
        print(
            f"  WARNING: only {test_variation['fraction_time_varying']:.0%} of test-split suppliers have a "
            f"risk_probability that varies across their own prediction times -- this run may have degenerated "
            f"into a fixed per-supplier score rather than a genuine temporal prediction (see "
            f"evaluate.py::temporal_variation_summary's docstring for why feature_mode=static_plus_graph does this)."
        )

    test_metrics = eval_result.metrics_by_split.get("test", {})
    write_json(os.path.join(run_dir, "metrics.json"), {
        "threshold_policy": eval_result.threshold_policy,
        "threshold": eval_result.threshold,
        "best_epoch": train_result.best_epoch,
        "best_val_pr_auc": train_result.best_val_pr_auc,
        "pos_weight": train_result.pos_weight,
        "by_split": eval_result.metrics_by_split,
    })
    write_json(os.path.join(run_dir, "classification_report.json"), test_metrics)
    cm = test_metrics.get("confusion_matrix")
    if cm:
        pd.DataFrame([cm]).to_csv(os.path.join(run_dir, "confusion_matrix.csv"), index=False)
    write_json(os.path.join(run_dir, "calibration.json"), eval_result.calibration_by_split)

    metadata = build_run_metadata(
        config, prepared.benchmark, prepared.frames, seed,
        split_summary=class_balance_summary(prepared.examples),
        training_duration_seconds=train_result.training_duration_seconds,
        extra={"best_epoch": train_result.best_epoch, "stopped_early": train_result.stopped_early, "pos_weight": train_result.pos_weight},
    )
    write_json(os.path.join(run_dir, "run_metadata.json"), metadata)

    render_all_plots(eval_result, train_result.history, os.path.join(run_dir, "plots"))
    return run_dir


def _save_baseline_run(name: str, result, prepared, config) -> str:
    run_dir = new_run_dir(config.experiment.output_dir, f"{name}{_strategy_tag(config)}")
    save_config(run_dir, config)
    result.predictions.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)
    write_json(os.path.join(run_dir, "metrics.json"), {"threshold": result.threshold, "by_split": result.metrics_by_split})
    write_json(os.path.join(run_dir, "calibration.json"), result.calibration_by_split)
    write_json(os.path.join(run_dir, "onset_breakdown.json"), result.onset_breakdown)
    metadata = build_run_metadata(
        config, prepared.benchmark, prepared.frames, seed=config.seed,
        split_summary=class_balance_summary(prepared.examples), training_duration_seconds=None,
        extra={"model_name": name},
    )
    write_json(os.path.join(run_dir, "run_metadata.json"), metadata)
    shutil.rmtree(os.path.join(run_dir, "plots"))  # baselines don't produce training curves / graph-specific plots
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/graphsage.yaml")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds; defaults to the config's `experiment.seeds` first entry only unless --all-seeds is given.")
    parser.add_argument("--all-seeds", action="store_true", help="Run every seed listed in the config's experiment.seeds.")
    parser.add_argument("--baselines", action="store_true", help="Also run Majority + Logistic Regression baselines.")
    parser.add_argument(
        "--feature-mode", default=None,
        choices=["full", "dynamic_only", "static_only", "region_risk_only", "static_plus_graph"],
        help="Override config's features.feature_mode (GRAPH_SAGE_IMPROVEMENT_PLAN.md Phase B ablations).",
    )
    parser.add_argument(
        "--dataset-id", default=None,
        help="Override config's dataset.dataset_id (e.g. scm_v1_black_swan_seed44 for the D1 within-seed44 diagnostic in GRAPHSAGE_FINAL_GENERALIZATION_AND_IMPROVEMENT_PLAN.md).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.feature_mode:
        config.features.feature_mode = args.feature_mode
    if args.dataset_id:
        config.dataset.dataset_id = args.dataset_id
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    elif args.all_seeds:
        seeds = list(config.experiment.seeds)
    else:
        seeds = [config.seed]

    print(f"Preparing benchmark data (dataset={config.dataset.dataset_id}, split={config.split.strategy}, feature_mode={config.features.feature_mode}) ...")
    prepared = prepare(config)
    print(f"Prediction examples: {len(prepared.examples)} (supplier x prediction_time)")

    test_metrics_by_seed = []
    onset_by_seed = []
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        train_result = train_graphsage(prepared, seed=seed)
        eval_result = evaluate_experiment(train_result.model, prepared, config.threshold)
        run_dir = _save_graphsage_run(prepared, train_result, eval_result, config, seed)
        print(f"Saved GraphSAGE run -> {run_dir}")
        if "test" in eval_result.metrics_by_split:
            test_metrics_by_seed.append(eval_result.metrics_by_split["test"])
            m = eval_result.metrics_by_split["test"]
            print(f"  test: pr_auc={m['pr_auc']} roc_auc={m['roc_auc']} f1={m['f1']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f}")
        onset = eval_result.onset_breakdown.get("test")
        if onset:
            onset_by_seed.append(onset)
            print(
                f"  test onset breakdown: fresh_onset(n={onset['n_fresh_onset']}, recall={onset['recall_fresh_onset']}, "
                f"pr_auc={onset['pr_auc_fresh_onset']}, roc_auc={onset['roc_auc_fresh_onset']}) | "
                f"already_ongoing(n={onset['n_already_ongoing']}, recall={onset['recall_already_ongoing']}, "
                f"pr_auc={onset['pr_auc_already_ongoing']}, roc_auc={onset['roc_auc_already_ongoing']})"
            )

    if len(seeds) > 1 and test_metrics_by_seed:
        summary = {}
        for key in ("pr_auc", "roc_auc", "f1", "precision", "recall", "balanced_accuracy", "brier_score"):
            values = [m[key] for m in test_metrics_by_seed if m.get(key) is not None]
            if values:
                summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "n_seeds": len(values)}
        onset_summary = {}
        for key in (
            "recall_fresh_onset", "recall_already_ongoing", "pr_auc_fresh_onset", "roc_auc_fresh_onset",
            "pr_auc_already_ongoing", "roc_auc_already_ongoing",
        ):
            values = [o[key] for o in onset_by_seed if o.get(key) is not None]
            if values:
                onset_summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "n_seeds": len(values)}
        summary_name = f"multiseed_summary{_strategy_tag(config)}_seeds_{'-'.join(str(s) for s in seeds)}.json"
        summary_path = os.path.join(config.experiment.output_dir, summary_name)
        write_json(summary_path, {"seeds": seeds, "test_metrics": summary, "test_onset_breakdown": onset_summary})
        print(f"\nMulti-seed test summary (mean +/- std over {len(seeds)} seeds) -> {summary_path}")
        for key, stats in summary.items():
            print(f"  {key}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
        if onset_summary:
            print("  onset breakdown:")
            for key, stats in onset_summary.items():
                print(f"    {key}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    if args.baselines:
        print("\n=== baselines ===")
        majority = run_majority_baseline(prepared, config.threshold)
        run_dir = _save_baseline_run("majority_baseline", majority, prepared, config)
        print(f"Saved majority baseline -> {run_dir}")
        if "test" in majority.metrics_by_split:
            m = majority.metrics_by_split["test"]
            print(f"  test: pr_auc={m['pr_auc']} f1={m['f1']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f}")

        logreg = run_logistic_regression_baseline(prepared, config.threshold, seed=seeds[0])
        run_dir = _save_baseline_run("logistic_regression_baseline", logreg, prepared, config)
        print(f"Saved logistic regression baseline -> {run_dir}")
        if "test" in logreg.metrics_by_split:
            m = logreg.metrics_by_split["test"]
            print(f"  test: pr_auc={m['pr_auc']} f1={m['f1']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f}")


if __name__ == "__main__":
    main()
