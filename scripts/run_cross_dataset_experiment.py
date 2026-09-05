#!/usr/bin/env python3
"""D2 -- TRUE cross-dataset generalization experiment
(GRAPHSAGE_FINAL_GENERALIZATION_AND_IMPROVEMENT_PLAN.md).

    SOURCE dataset train split
        |
        v
    GraphSAGE (+ Majority, + Logistic Regression, all fit on SOURCE only)
        |
        +---> evaluated WITHIN-SEED on SOURCE's own test split (reference)
        |
        +---> evaluated CROSS-DATASET on the ENTIRE TARGET dataset, using
              SOURCE's fitted preprocessor and a threshold already
              selected on SOURCE's validation split. No TARGET label ever
              participates in preprocessing, class weighting, threshold
              selection, or model selection -- it is only ever scored.

This is NOT `target train -> target test` (that is within-seed
replication of the target, not cross-dataset generalization) -- run this
script a second time with --source-dataset-id/--target-dataset-id swapped
to get the reverse direction, if D0 (scripts/compare_datasets.py) found
the two datasets compatible in both directions.

Usage:
    python scripts/run_cross_dataset_experiment.py \\
        --config configs/graphsage.yaml \\
        --source-dataset-id scm_v1_black_swan_seed43 \\
        --target-dataset-id scm_v1_black_swan_seed44 \\
        --seeds 42,43,44,45,46
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from scm_dataset.modeling.baselines import (
    run_logistic_regression_baseline,
    run_logistic_regression_baseline_cross_dataset,
    run_majority_baseline,
    run_majority_baseline_cross_dataset,
)
from scm_dataset.modeling.config import load_config
from scm_dataset.modeling.data import load_benchmark
from scm_dataset.modeling.evaluate import evaluate_experiment, evaluate_on_target_dataset, render_all_plots
from scm_dataset.modeling.experiment import build_run_metadata, new_run_dir, save_config, write_json
from scm_dataset.modeling.pipeline import prepare, prepare_for_cross_dataset_eval
from scm_dataset.modeling.train import class_balance_summary, train_graphsage


def _short(dataset_id: str) -> str:
    return dataset_id.replace("scm_v1_black_swan_", "")


def _save_metrics_pair(run_dir: str, prefix: str, eval_result, extra_metadata: dict) -> None:
    eval_result.predictions.to_csv(os.path.join(run_dir, f"predictions_{prefix}.csv"), index=False)
    write_json(os.path.join(run_dir, f"onset_breakdown_{prefix}.json"), eval_result.onset_breakdown)
    write_json(os.path.join(run_dir, f"temporal_variation_{prefix}.json"), getattr(eval_result, "temporal_variation", {}))
    write_json(os.path.join(run_dir, f"calibration_{prefix}.json"), eval_result.calibration_by_split)
    write_json(os.path.join(run_dir, f"metrics_{prefix}.json"), {
        "threshold": eval_result.threshold,
        "threshold_policy": getattr(eval_result, "threshold_policy", None),  # BaselineResult has no threshold_policy field
        "by_split": eval_result.metrics_by_split,
        **extra_metadata,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/graphsage.yaml")
    parser.add_argument("--source-dataset-id", default="scm_v1_black_swan_seed43")
    parser.add_argument("--target-dataset-id", default="scm_v1_black_swan_seed44")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--feature-mode", default=None, choices=["full", "dynamic_only", "static_only", "region_risk_only", "static_plus_graph"])
    parser.add_argument("--epochs", type=int, default=None, help="Override config's training.epochs (useful for a quick smoke test).")
    args = parser.parse_args()

    config = load_config(args.config)
    config.dataset.dataset_id = args.source_dataset_id
    if args.feature_mode:
        config.features.feature_mode = args.feature_mode
    if args.epochs:
        config.training.epochs = args.epochs
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"D2 cross-dataset experiment: {args.source_dataset_id} (train) -> {args.target_dataset_id} (test), feature_mode={config.features.feature_mode}")
    print("Preparing SOURCE (fits preprocessing on source's own train split only) ...")
    prepared_source = prepare(config)
    print(f"  source examples: {len(prepared_source.examples)}")

    target_benchmark = load_benchmark(config.dataset.benchmark_path, args.target_dataset_id)
    print("Preparing TARGET (using source's preprocessor, never refit; every example marked split=test) ...")
    prepared_target = prepare_for_cross_dataset_eval(config, target_benchmark, prepared_source.preprocessor)
    print(f"  target examples: {len(prepared_target.examples)}")

    tag = f"crossdataset_{_short(args.source_dataset_id)}_to_{_short(args.target_dataset_id)}"
    within_seed_pr_aucs, cross_dataset_pr_aucs = [], []

    for seed in seeds:
        print(f"\n=== model seed {seed} ===")
        train_result = train_graphsage(prepared_source, seed=seed, verbose=False)
        within_eval = evaluate_experiment(train_result.model, prepared_source, config.threshold)
        cross_eval = evaluate_on_target_dataset(train_result.model, prepared_target, threshold=within_eval.threshold)

        within_test = within_eval.metrics_by_split.get("test", {})
        cross_test = cross_eval.metrics_by_split.get("test", {})
        print(f"  WITHIN-SEED  ({args.source_dataset_id} -> {args.source_dataset_id}): pr_auc={within_test.get('pr_auc')} roc_auc={within_test.get('roc_auc')} n={within_test.get('n_examples')}")
        print(f"  CROSS-DATASET({args.source_dataset_id} -> {args.target_dataset_id}): pr_auc={cross_test.get('pr_auc')} roc_auc={cross_test.get('roc_auc')} n={cross_test.get('n_examples')}")
        cross_onset = cross_eval.onset_breakdown.get("test", {})
        n_fresh = cross_onset.get("n_fresh_onset", 0)
        fresh_recall = cross_onset.get("recall_fresh_onset")
        print(f"    cross-dataset onset breakdown: n_fresh_onset={n_fresh}, fresh_recall={'N/A (n=0)' if not n_fresh else fresh_recall}, "
              f"n_ongoing={cross_onset.get('n_already_ongoing')}, ongoing_recall={cross_onset.get('recall_already_ongoing')}")

        run_dir = new_run_dir(config.experiment.output_dir, f"{tag}_seed{seed}")
        save_config(run_dir, config)
        torch.save(train_result.model.state_dict(), os.path.join(run_dir, "model.pt"))
        prepared_source.preprocessor.save(os.path.join(run_dir, "preprocessing"))
        train_result.history.to_csv(os.path.join(run_dir, "training_history.csv"), index=False)

        _save_metrics_pair(run_dir, "within_seed", within_eval, {"dataset": args.source_dataset_id})
        _save_metrics_pair(run_dir, "cross_dataset", cross_eval, {"train_dataset": args.source_dataset_id, "test_dataset": args.target_dataset_id})

        metadata = build_run_metadata(
            config, prepared_source.benchmark, prepared_source.frames, seed,
            split_summary=class_balance_summary(prepared_source.examples),
            training_duration_seconds=train_result.training_duration_seconds,
            extra={
                "experiment_type": "D2_true_cross_dataset_generalization",
                "source_dataset_id": args.source_dataset_id,
                "target_dataset_id": args.target_dataset_id,
                "best_epoch": train_result.best_epoch,
                "stopped_early": train_result.stopped_early,
                "pos_weight": train_result.pos_weight,
                "within_seed_test_pr_auc": within_test.get("pr_auc"),
                "cross_dataset_test_pr_auc": cross_test.get("pr_auc"),
            },
        )
        write_json(os.path.join(run_dir, "run_metadata.json"), metadata)

        plots_dir = os.path.join(run_dir, "plots")
        render_all_plots(within_eval, train_result.history, os.path.join(plots_dir, "within_seed"))
        render_all_plots(cross_eval, train_result.history, os.path.join(plots_dir, "cross_dataset"))

        print(f"  Saved -> {run_dir}")
        if within_test.get("pr_auc") is not None:
            within_seed_pr_aucs.append(within_test["pr_auc"])
        if cross_test.get("pr_auc") is not None:
            cross_dataset_pr_aucs.append(cross_test["pr_auc"])

    if len(seeds) > 1:
        summary = {
            "source_dataset_id": args.source_dataset_id,
            "target_dataset_id": args.target_dataset_id,
            "seeds": seeds,
            "within_seed_pr_auc": {"mean": float(np.mean(within_seed_pr_aucs)), "std": float(np.std(within_seed_pr_aucs)), "min": float(np.min(within_seed_pr_aucs)), "max": float(np.max(within_seed_pr_aucs))} if within_seed_pr_aucs else None,
            "cross_dataset_pr_auc": {"mean": float(np.mean(cross_dataset_pr_aucs)), "std": float(np.std(cross_dataset_pr_aucs)), "min": float(np.min(cross_dataset_pr_aucs)), "max": float(np.max(cross_dataset_pr_aucs))} if cross_dataset_pr_aucs else None,
        }
        summary_path = os.path.join(config.experiment.output_dir, f"multiseed_summary_{tag}_seeds_{'-'.join(str(s) for s in seeds)}.json")
        write_json(summary_path, summary)
        print(f"\nMulti-seed summary -> {summary_path}")
        print(f"  within-seed  PR-AUC: {summary['within_seed_pr_auc']}")
        print(f"  cross-dataset PR-AUC: {summary['cross_dataset_pr_auc']}")

    print("\n=== baselines (fit on source only, evaluated within-seed and cross-dataset) ===")
    majority_within = run_majority_baseline(prepared_source, config.threshold)
    majority_cross = run_majority_baseline_cross_dataset(prepared_source, prepared_target, config.threshold)
    logreg_within = run_logistic_regression_baseline(prepared_source, config.threshold, seed=seeds[0])
    logreg_cross = run_logistic_regression_baseline_cross_dataset(prepared_source, prepared_target, config.threshold, seed=seeds[0])

    baseline_run_dir = new_run_dir(config.experiment.output_dir, f"{tag}_baselines")
    save_config(baseline_run_dir, config)
    _save_metrics_pair(baseline_run_dir, "majority_within_seed", majority_within, {"dataset": args.source_dataset_id})
    _save_metrics_pair(baseline_run_dir, "majority_cross_dataset", majority_cross, {"train_dataset": args.source_dataset_id, "test_dataset": args.target_dataset_id})
    _save_metrics_pair(baseline_run_dir, "logreg_within_seed", logreg_within, {"dataset": args.source_dataset_id})
    _save_metrics_pair(baseline_run_dir, "logreg_cross_dataset", logreg_cross, {"train_dataset": args.source_dataset_id, "test_dataset": args.target_dataset_id})
    print(f"Saved baselines -> {baseline_run_dir}")
    print(f"  majority   within={majority_within.metrics_by_split.get('test', {}).get('pr_auc')}  cross={majority_cross.metrics_by_split.get('test', {}).get('pr_auc')}")
    print(f"  logreg     within={logreg_within.metrics_by_split.get('test', {}).get('pr_auc')}  cross={logreg_cross.metrics_by_split.get('test', {}).get('pr_auc')}")


if __name__ == "__main__":
    main()
