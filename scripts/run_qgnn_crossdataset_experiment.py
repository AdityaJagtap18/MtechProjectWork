#!/usr/bin/env python3
"""GraphSAGE-Reduced vs QGNN-Reduced -- TRUE cross-dataset generalization
(QGNN_FAIR_BENCHMARK_AND_IMPLEMENTATION_PLAN.md section 17, reusing
GRAPHSAGE_FINAL_GENERALIZATION_AND_IMPROVEMENT_PLAN.md's D2 protocol):

    SOURCE dataset train split
        -> fit reduction (PCA / domain-selected) on source train only
        -> fit preprocessing on source train only
        -> train GraphSAGE-Reduced and QGNN-Reduced
        -> evaluated WITHIN-SEED on source's own test split (reference)
        -> evaluated CROSS-DATASET on the ENTIRE TARGET dataset, using
           source's fitted reducer + preprocessor and a threshold already
           selected on source's validation split. No TARGET label ever
           participates in reduction, preprocessing, threshold selection,
           or model selection -- it is only ever scored.

Trains fresh each invocation (rather than reloading a saved model +
reducer from disk) -- training is fast enough (~1-2 min total for both
models, 5 seeds, per scripts/run_qgnn_experiment.py's measured timings)
that this is simpler than adding reducer deserialization, and keeps the
reducer object exactly as fit in-memory for both directions this script
touches.

Usage:
    python scripts/run_qgnn_crossdataset_experiment.py \\
        --config configs/qgnn.yaml \\
        --source-dataset-id scm_v1_black_swan_seed43 \\
        --target-dataset-id scm_v1_black_swan_seed44 \\
        --n-components 8 --reduction-method domain_selected --seeds 42,43,44,45,46
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from scm_dataset.modeling.data import load_benchmark
from scm_dataset.modeling.evaluate import evaluate_experiment, evaluate_on_target_dataset
from scm_dataset.modeling.experiment import build_run_metadata, new_run_dir, save_config, write_json
from scm_dataset.modeling.pipeline import prepare_reduced, prepare_reduced_for_cross_dataset_eval
from scm_dataset.modeling.qgnn import evaluate_qgnn, evaluate_qgnn_on_target, load_qgnn_config, train_qgnn
from scm_dataset.modeling.train import class_balance_summary, train_graphsage


def _short(dataset_id: str) -> str:
    return dataset_id.replace("scm_v1_black_swan_", "")


def _save_pair(run_dir: str, prefix: str, eval_result, extra: dict) -> None:
    eval_result.predictions.to_csv(os.path.join(run_dir, f"predictions_{prefix}.csv"), index=False)
    write_json(os.path.join(run_dir, f"onset_breakdown_{prefix}.json"), eval_result.onset_breakdown)
    write_json(os.path.join(run_dir, f"calibration_{prefix}.json"), eval_result.calibration_by_split)
    write_json(os.path.join(run_dir, f"metrics_{prefix}.json"), {
        "threshold": eval_result.threshold, "threshold_policy": eval_result.threshold_policy,
        "by_split": eval_result.metrics_by_split, **extra,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/qgnn.yaml")
    parser.add_argument("--source-dataset-id", default="scm_v1_black_swan_seed43")
    parser.add_argument("--target-dataset-id", default="scm_v1_black_swan_seed44")
    parser.add_argument("--n-components", type=int, default=None)
    parser.add_argument("--reduction-method", default=None, choices=["pca", "domain_selected"])
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    args = parser.parse_args()

    cfg = load_qgnn_config(args.config)
    config = cfg.base
    qgnn_arch = cfg.qgnn
    config.dataset.dataset_id = args.source_dataset_id
    if args.n_components:
        qgnn_arch.n_components = args.n_components
    if args.reduction_method:
        qgnn_arch.reduction_method = args.reduction_method
    if args.n_layers:
        qgnn_arch.n_layers = args.n_layers
    if args.device:
        qgnn_arch.device = args.device
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"D2 (reduced) cross-dataset: {args.source_dataset_id} (train) -> {args.target_dataset_id} (test), "
          f"reduction={qgnn_arch.reduction_method} n_components={qgnn_arch.n_components}")
    prepared_source, reducer = prepare_reduced(config, qgnn_arch.reduction_method, qgnn_arch.n_components, columns=qgnn_arch.reduction_columns)
    print(f"  source examples: {len(prepared_source.examples)}, reduced columns: {reducer.columns}")

    target_benchmark = load_benchmark(config.dataset.benchmark_path, args.target_dataset_id)
    prepared_target = prepare_reduced_for_cross_dataset_eval(config, target_benchmark, prepared_source.preprocessor, reducer)
    print(f"  target examples: {len(prepared_target.examples)}")

    tag = f"crossdataset_reduced_{_short(args.source_dataset_id)}_to_{_short(args.target_dataset_id)}_dim{qgnn_arch.n_components}"
    gs_within, gs_cross, q_within, q_cross = [], [], [], []

    for seed in seeds:
        print(f"\n=== model seed {seed} ===")

        gs_train = train_graphsage(prepared_source, seed=seed, verbose=False)
        gs_within_eval = evaluate_experiment(gs_train.model, prepared_source, config.threshold)
        gs_cross_eval = evaluate_on_target_dataset(gs_train.model, prepared_target, threshold=gs_within_eval.threshold)
        gs_w, gs_c = gs_within_eval.metrics_by_split.get("test", {}), gs_cross_eval.metrics_by_split.get("test", {})
        print(f"  GraphSAGE-Reduced  within pr_auc={gs_w.get('pr_auc')}  cross pr_auc={gs_c.get('pr_auc')} roc_auc={gs_c.get('roc_auc')}")

        q_train = train_qgnn(prepared_source, seed=seed, n_layers=qgnn_arch.n_layers, mlp_hidden=qgnn_arch.mlp_hidden, device_name=qgnn_arch.device, verbose=False)
        q_within_eval = evaluate_qgnn(q_train.model, prepared_source, config.threshold)
        q_cross_eval = evaluate_qgnn_on_target(q_train.model, prepared_target, threshold=q_within_eval.threshold)
        q_w, q_c = q_within_eval.metrics_by_split.get("test", {}), q_cross_eval.metrics_by_split.get("test", {})
        print(f"  QGNN-Reduced       within pr_auc={q_w.get('pr_auc')}  cross pr_auc={q_c.get('pr_auc')} roc_auc={q_c.get('roc_auc')}")

        run_dir = new_run_dir(config.experiment.output_dir, f"{tag}_seed{seed}")
        save_config(run_dir, config)
        torch.save(gs_train.model.state_dict(), os.path.join(run_dir, "graphsage_reduced_model.pt"))
        torch.save(q_train.model.state_dict(), os.path.join(run_dir, "qgnn_reduced_model.pt"))
        reducer.save(os.path.join(run_dir, "reduction"))
        _save_pair(run_dir, "graphsage_within", gs_within_eval, {"dataset": args.source_dataset_id})
        _save_pair(run_dir, "graphsage_cross", gs_cross_eval, {"train_dataset": args.source_dataset_id, "test_dataset": args.target_dataset_id})
        _save_pair(run_dir, "qgnn_within", q_within_eval, {"dataset": args.source_dataset_id})
        _save_pair(run_dir, "qgnn_cross", q_cross_eval, {"train_dataset": args.source_dataset_id, "test_dataset": args.target_dataset_id, "quantum": q_train.model.quantum_resource_summary()})
        metadata = build_run_metadata(
            config, prepared_source.benchmark, prepared_source.frames, seed,
            split_summary=class_balance_summary(prepared_source.examples), training_duration_seconds=None,
            extra={
                "experiment_type": "QGNN_D2_reduced_cross_dataset_generalization",
                "source_dataset_id": args.source_dataset_id, "target_dataset_id": args.target_dataset_id,
                "reduction_method": qgnn_arch.reduction_method, "n_components": qgnn_arch.n_components,
            },
        )
        write_json(os.path.join(run_dir, "run_metadata.json"), metadata)
        print(f"  saved -> {run_dir}")

        if gs_w.get("pr_auc") is not None:
            gs_within.append(gs_w["pr_auc"])
        if gs_c.get("pr_auc") is not None:
            gs_cross.append(gs_c["pr_auc"])
        if q_w.get("pr_auc") is not None:
            q_within.append(q_w["pr_auc"])
        if q_c.get("pr_auc") is not None:
            q_cross.append(q_c["pr_auc"])

    def _stats(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))} if vals else None

    summary = {
        "source_dataset_id": args.source_dataset_id, "target_dataset_id": args.target_dataset_id,
        "reduction_method": qgnn_arch.reduction_method, "n_components": qgnn_arch.n_components, "seeds": seeds,
        "graphsage_reduced_within_pr_auc": _stats(gs_within), "graphsage_reduced_cross_pr_auc": _stats(gs_cross),
        "qgnn_reduced_within_pr_auc": _stats(q_within), "qgnn_reduced_cross_pr_auc": _stats(q_cross),
    }
    summary_path = os.path.join(config.experiment.output_dir, f"multiseed_summary_{tag}_seeds_{'-'.join(str(s) for s in seeds)}.json")
    write_json(summary_path, summary)
    print(f"\nMulti-seed summary -> {summary_path}")
    print(f"  GraphSAGE-Reduced  within={summary['graphsage_reduced_within_pr_auc']}  cross={summary['graphsage_reduced_cross_pr_auc']}")
    print(f"  QGNN-Reduced       within={summary['qgnn_reduced_within_pr_auc']}  cross={summary['qgnn_reduced_cross_pr_auc']}")


if __name__ == "__main__":
    main()
