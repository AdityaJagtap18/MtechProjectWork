#!/usr/bin/env python3
"""GraphSAGE-Reduced vs QGNN-Reduced experiment
(QGNN_FAIR_BENCHMARK_AND_IMPLEMENTATION_PLAN.md sections 3, 12-15).

Fits ONE reduction (PCA or domain-selected, on the dataset's own train
split) per invocation, then for each model seed trains and evaluates both:

    GraphSAGE-Reduced -- the EXISTING, unmodified HeteroGraphSAGE
    architecture (scripts/run_graphsage_experiment.py's own model code),
    fed the reduced supplier representation instead of the full one.

    QGNN-Reduced -- the hybrid quantum model in modeling/qgnn.py, fed the
    exact same reduced representation.

This script does not retrain or touch GraphSAGE-Full: that is the
existing frozen baseline in experiments/classical_gnn/, already
extensively benchmarked -- see GRAPHSAGE_WORK_SUMMARY.md and
GRAPHSAGE_PHASE_A_GENERALIZATION_FINDINGS.md. Compare this script's
output against those existing numbers for the "QGNN-Reduced vs
GraphSAGE-Full" secondary comparison (plan section 3) rather than
rerunning it here.

Usage:
    python scripts/run_qgnn_experiment.py --config configs/qgnn.yaml --tag smoke --seeds 42 --epochs 5
    python scripts/run_qgnn_experiment.py --config configs/qgnn.yaml --tag dimension_6 --n-components 6 --seeds 42,43,44,45,46
"""

from __future__ import annotations

import argparse
import os

import torch

from scm_dataset.modeling.evaluate import evaluate_experiment, render_all_plots
from scm_dataset.modeling.experiment import build_run_metadata, new_run_dir, save_config, write_json
from scm_dataset.modeling.pipeline import prepare_reduced
from scm_dataset.modeling.qgnn import evaluate_qgnn, load_qgnn_config, train_qgnn
from scm_dataset.modeling.train import class_balance_summary, train_graphsage


def _save_run(run_dir: str, config, model, model_kind: str, prepared, train_result, eval_result, seed: int, extra_metadata: dict) -> None:
    save_config(run_dir, config)
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    prepared.preprocessor.save(os.path.join(run_dir, "preprocessing"))
    train_result.history.to_csv(os.path.join(run_dir, "training_history.csv"), index=False)
    eval_result.predictions.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)
    eval_result.risk_ranking.to_csv(os.path.join(run_dir, "supplier_risk_ranking.csv"), index=False)
    write_json(os.path.join(run_dir, "metrics.json"), {"threshold": eval_result.threshold, "threshold_policy": eval_result.threshold_policy, "by_split": eval_result.metrics_by_split})
    write_json(os.path.join(run_dir, "calibration.json"), eval_result.calibration_by_split)
    write_json(os.path.join(run_dir, "onset_breakdown.json"), eval_result.onset_breakdown)

    metadata = build_run_metadata(
        config, prepared.benchmark, prepared.frames, seed=seed,
        split_summary=class_balance_summary(prepared.examples),
        training_duration_seconds=train_result.training_duration_seconds,
        extra={"model_kind": model_kind, "best_epoch": train_result.best_epoch, "stopped_early": train_result.stopped_early, "pos_weight": train_result.pos_weight, **extra_metadata},
    )
    write_json(os.path.join(run_dir, "run_metadata.json"), metadata)
    render_all_plots(eval_result, train_result.history, os.path.join(run_dir, "plots"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/qgnn.yaml")
    parser.add_argument("--tag", default="dimension_run", help="Subdirectory-ish tag for run names, e.g. 'smoke', 'dimension_6', 'final'.")
    parser.add_argument("--seeds", default=None, help="Comma-separated model seeds; defaults to the config's experiment.seeds.")
    parser.add_argument("--n-components", type=int, default=None, help="Override qgnn.n_components (and the reduced supplier dimension for GraphSAGE-Reduced too).")
    parser.add_argument("--reduction-method", default=None, choices=["pca", "domain_selected"])
    parser.add_argument("--n-layers", type=int, default=None, help="Override qgnn.n_layers (variational layers).")
    parser.add_argument("--device", default=None, help="Override qgnn.device (e.g. lightning.gpu, default.qubit).")
    parser.add_argument("--epochs", type=int, default=None, help="Override config's training.epochs (useful for a smoke test).")
    args = parser.parse_args()

    cfg = load_qgnn_config(args.config)
    config = cfg.base
    qgnn_arch = cfg.qgnn
    if args.n_components:
        qgnn_arch.n_components = args.n_components
    if args.reduction_method:
        qgnn_arch.reduction_method = args.reduction_method
    if args.n_layers:
        qgnn_arch.n_layers = args.n_layers
    if args.device:
        qgnn_arch.device = args.device
    if args.epochs:
        config.training.epochs = args.epochs
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(config.experiment.seeds)

    print(f"GraphSAGE-Reduced vs QGNN-Reduced: dataset={config.dataset.dataset_id} reduction={qgnn_arch.reduction_method} n_components={qgnn_arch.n_components} n_layers={qgnn_arch.n_layers} device={qgnn_arch.device}")
    prepared, reducer = prepare_reduced(config, qgnn_arch.reduction_method, qgnn_arch.n_components, columns=qgnn_arch.reduction_columns)
    print(f"  examples: {len(prepared.examples)}, reduced supplier columns: {reducer.columns}")

    graphsage_reduced_pr_aucs, qgnn_reduced_pr_aucs = [], []

    for seed in seeds:
        print(f"\n=== model seed {seed} ===")

        print("-- GraphSAGE-Reduced --")
        gs_train_result = train_graphsage(prepared, seed=seed, verbose=False)
        gs_eval_result = evaluate_experiment(gs_train_result.model, prepared, config.threshold)
        gs_test = gs_eval_result.metrics_by_split.get("test", {})
        print(f"  test: pr_auc={gs_test.get('pr_auc')} roc_auc={gs_test.get('roc_auc')}")
        gs_run_dir = new_run_dir(config.experiment.output_dir, f"{args.tag}_graphsage_reduced_seed{seed}")
        reducer.save(os.path.join(gs_run_dir, "reduction"))
        _save_run(
            gs_run_dir, config, gs_train_result.model, "graphsage_reduced", prepared, gs_train_result, gs_eval_result, seed,
            extra_metadata={"reduction_method": qgnn_arch.reduction_method, "n_components": qgnn_arch.n_components},
        )
        print(f"  saved -> {gs_run_dir}")
        if gs_test.get("pr_auc") is not None:
            graphsage_reduced_pr_aucs.append(gs_test["pr_auc"])

        print("-- QGNN-Reduced --")
        q_train_result = train_qgnn(prepared, seed=seed, n_layers=qgnn_arch.n_layers, mlp_hidden=qgnn_arch.mlp_hidden, device_name=qgnn_arch.device, verbose=False)
        q_eval_result = evaluate_qgnn(q_train_result.model, prepared, config.threshold)
        q_test = q_eval_result.metrics_by_split.get("test", {})
        print(f"  test: pr_auc={q_test.get('pr_auc')} roc_auc={q_test.get('roc_auc')}")
        resource_summary = q_train_result.model.quantum_resource_summary()
        print(f"  quantum resource summary: {resource_summary}")
        q_run_dir = new_run_dir(config.experiment.output_dir, f"{args.tag}_qgnn_reduced_seed{seed}")
        reducer.save(os.path.join(q_run_dir, "reduction"))
        _save_run(
            q_run_dir, config, q_train_result.model, "qgnn_reduced", prepared, q_train_result, q_eval_result, seed,
            extra_metadata={"reduction_method": qgnn_arch.reduction_method, "n_components": qgnn_arch.n_components, "quantum": resource_summary},
        )
        write_json(os.path.join(q_run_dir, "quantum_resource_summary.json"), resource_summary)
        print(f"  saved -> {q_run_dir}")
        if q_test.get("pr_auc") is not None:
            qgnn_reduced_pr_aucs.append(q_test["pr_auc"])

    if len(seeds) > 1:
        import numpy as np

        summary = {
            "reduction_method": qgnn_arch.reduction_method, "n_components": qgnn_arch.n_components, "n_layers": qgnn_arch.n_layers,
            "seeds": seeds,
            "graphsage_reduced_pr_auc": {"mean": float(np.mean(graphsage_reduced_pr_aucs)), "std": float(np.std(graphsage_reduced_pr_aucs)), "min": float(np.min(graphsage_reduced_pr_aucs)), "max": float(np.max(graphsage_reduced_pr_aucs))} if graphsage_reduced_pr_aucs else None,
            "qgnn_reduced_pr_auc": {"mean": float(np.mean(qgnn_reduced_pr_aucs)), "std": float(np.std(qgnn_reduced_pr_aucs)), "min": float(np.min(qgnn_reduced_pr_aucs)), "max": float(np.max(qgnn_reduced_pr_aucs))} if qgnn_reduced_pr_aucs else None,
        }
        summary_path = os.path.join(config.experiment.output_dir, f"multiseed_summary_{args.tag}_seeds_{'-'.join(str(s) for s in seeds)}.json")
        write_json(summary_path, summary)
        print(f"\nMulti-seed summary -> {summary_path}")
        print(f"  GraphSAGE-Reduced PR-AUC: {summary['graphsage_reduced_pr_auc']}")
        print(f"  QGNN-Reduced      PR-AUC: {summary['qgnn_reduced_pr_auc']}")


if __name__ == "__main__":
    main()
