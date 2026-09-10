#!/usr/bin/env python3
"""QGNN-v2 depth-3 baseline vs depth-3 + data re-uploading -- controlled
follow-up experiment. Purely additive: does not modify or call
`scripts/run_qgnn_v2_experiment.py`; reuses the same frozen-encoder +
PCA pipeline (`graph_embedding_reduction.py`, `qgnn_v2.py`, both
unchanged) for both models, so the only intentional difference between
runs A and B is the circuit itself
(`qgnn.build_qgnn_model` vs `qgnn_v2_reupload.build_qgnn_reupload_model`).

Usage:
    python scripts/run_qgnn_v2_reupload_experiment.py --config configs/qgnn_v2.yaml --tag reupload_smoke --seeds 42 --epochs 5
    python scripts/run_qgnn_v2_reupload_experiment.py --config configs/qgnn_v2.yaml --tag reupload --seeds 42,43,44,45,46
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from scm_dataset.modeling.data import load_benchmark
from scm_dataset.modeling.evaluate import render_all_plots
from scm_dataset.modeling.experiment import build_run_metadata, new_run_dir, save_config, write_json
from scm_dataset.modeling.features import NodeFeatureFrames
from scm_dataset.modeling.graph_embedding_reduction import (
    extract_supplier_embeddings,
    load_frozen_graphsage_encoder,
    prepare_frozen_encoder_input,
)
from scm_dataset.modeling.qgnn import build_qgnn_model
from scm_dataset.modeling.qgnn_v2 import build_v2_prepared, evaluate_v2, generate_predictions_v2, load_qgnn_v2_config, set_seed, train_v2_head
from scm_dataset.modeling.qgnn_v2_reupload import build_qgnn_reupload_model
from scm_dataset.modeling.train import class_balance_summary

from _analysis_common import find_latest_graphsage_full_checkpoint


def _save_run(run_dir: str, config, model, model_kind: str, v2prepared, train_result, eval_result, seed: int, inference_seconds: float, extra_metadata: dict) -> None:
    save_config(run_dir, config)
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    train_result.history.to_csv(os.path.join(run_dir, "training_history.csv"), index=False)
    eval_result.predictions.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)
    eval_result.risk_ranking.to_csv(os.path.join(run_dir, "supplier_risk_ranking.csv"), index=False)
    write_json(os.path.join(run_dir, "metrics.json"), {"threshold": eval_result.threshold, "threshold_policy": eval_result.threshold_policy, "by_split": eval_result.metrics_by_split})
    write_json(os.path.join(run_dir, "calibration.json"), eval_result.calibration_by_split)
    write_json(os.path.join(run_dir, "onset_breakdown.json"), eval_result.onset_breakdown)
    render_all_plots(eval_result, train_result.history, os.path.join(run_dir, "plots"))
    metadata = build_run_metadata(
        config, v2prepared.benchmark, NodeFeatureFrames(frames={}, numeric_columns={}, categorical_columns={}), seed=seed,
        split_summary=class_balance_summary(v2prepared.examples),
        training_duration_seconds=train_result.training_duration_seconds,
        extra={"model_kind": model_kind, "best_epoch": train_result.best_epoch, "stopped_early": train_result.stopped_early, "pos_weight": train_result.pos_weight, "inference_seconds_full_test_split": inference_seconds, **extra_metadata},
    )
    write_json(os.path.join(run_dir, "run_metadata.json"), metadata)


def _timed_inference(model, v2prepared) -> tuple[float, "object"]:
    start = time.time()
    predictions = generate_predictions_v2(model, v2prepared)
    elapsed = time.time() - start
    return elapsed, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/qgnn_v2.yaml")
    parser.add_argument("--tag", default="reupload")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--n-components", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override config's training.epochs (useful for a smoke test).")
    parser.add_argument("--encoder-experiments-dir", default="experiments/classical_gnn")
    args = parser.parse_args()

    cfg = load_qgnn_v2_config(args.config)
    config = cfg.base
    v2_arch = cfg.qgnn_v2
    v2_arch.n_layers = 3  # fixed: this experiment is a controlled follow-up to the already-selected depth-3 configuration, not a new depth sweep
    if args.n_components:
        v2_arch.n_components = args.n_components
    if args.device:
        v2_arch.device = args.device
    if args.epochs:
        config.training.epochs = args.epochs
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(config.experiment.seeds)

    print(f"QGNN-v2 depth-3 baseline vs +data-reuploading: dataset={config.dataset.dataset_id} n_components={v2_arch.n_components} device={v2_arch.device}")

    baseline_test_pr, baseline_val_pr = [], []
    reupload_test_pr, reupload_val_pr = [], []

    for seed in seeds:
        print(f"\n=== model seed {seed} (encoder + heads) ===")
        checkpoint_dir = find_latest_graphsage_full_checkpoint(args.encoder_experiments_dir, seed)
        benchmark = load_benchmark(config.dataset.benchmark_path, config.dataset.dataset_id)
        prepared_full = prepare_frozen_encoder_input(config, benchmark, os.path.join(checkpoint_dir, "preprocessing"))
        encoder = load_frozen_graphsage_encoder(os.path.join(checkpoint_dir, "model.pt"), prepared_full)
        assert all(not p.requires_grad for p in encoder.parameters())

        all_times = sorted(prepared_full.examples["time"].unique())
        embedding_frame = extract_supplier_embeddings(encoder, prepared_full, all_times)
        v2prepared, reducer = build_v2_prepared(config, benchmark, prepared_full.examples, embedding_frame, v2_arch.n_components)

        print("-- A: baseline QGNN-v2 depth-3 --")
        set_seed(seed)
        baseline_model = build_qgnn_model(n_qubits=v2_arch.n_components, n_layers=3, mlp_hidden=v2_arch.mlp_hidden, device_name=v2_arch.device)
        baseline_train = train_v2_head(v2prepared, baseline_model, seed=seed, verbose=False)
        baseline_eval = evaluate_v2(baseline_train.model, v2prepared, config.threshold)
        b_test, b_val = baseline_eval.metrics_by_split.get("test", {}), baseline_eval.metrics_by_split.get("validation", {})
        b_inference_s, _ = _timed_inference(baseline_train.model, v2prepared)
        print(f"  test: pr_auc={b_test.get('pr_auc')} roc_auc={b_test.get('roc_auc')}  val: pr_auc={b_val.get('pr_auc')}")
        b_run_dir = new_run_dir(config.experiment.output_dir, f"{args.tag}_baseline_seed{seed}")
        reducer.save(os.path.join(b_run_dir, "reduction"))
        write_json(os.path.join(b_run_dir, "encoder_checkpoint.json"), {"checkpoint_dir": checkpoint_dir})
        write_json(os.path.join(b_run_dir, "quantum_resource_summary.json"), baseline_train.model.quantum_resource_summary())
        _save_run(b_run_dir, config, baseline_train.model, "qgnn_v2_depth3_baseline", v2prepared, baseline_train, baseline_eval, seed, b_inference_s, {"n_components": v2_arch.n_components, "quantum": baseline_train.model.quantum_resource_summary()})
        print(f"  saved -> {b_run_dir}")
        if b_test.get("pr_auc") is not None:
            baseline_test_pr.append(b_test["pr_auc"])
        if b_val.get("pr_auc") is not None:
            baseline_val_pr.append(b_val["pr_auc"])

        print("-- B: QGNN-v2 depth-3 + data re-uploading --")
        set_seed(seed)
        reupload_model = build_qgnn_reupload_model(n_qubits=v2_arch.n_components, n_layers=3, mlp_hidden=v2_arch.mlp_hidden, device_name=v2_arch.device)
        reupload_train = train_v2_head(v2prepared, reupload_model, seed=seed, verbose=False)
        reupload_eval = evaluate_v2(reupload_train.model, v2prepared, config.threshold)
        r_test, r_val = reupload_eval.metrics_by_split.get("test", {}), reupload_eval.metrics_by_split.get("validation", {})
        r_inference_s, _ = _timed_inference(reupload_train.model, v2prepared)
        resource_summary = reupload_train.model.quantum_resource_summary()
        print(f"  test: pr_auc={r_test.get('pr_auc')} roc_auc={r_test.get('roc_auc')}  val: pr_auc={r_val.get('pr_auc')}")
        print(f"  quantum resource summary: {resource_summary}")
        r_run_dir = new_run_dir(config.experiment.output_dir, f"{args.tag}_dataupload_seed{seed}")
        reducer.save(os.path.join(r_run_dir, "reduction"))
        write_json(os.path.join(r_run_dir, "encoder_checkpoint.json"), {"checkpoint_dir": checkpoint_dir})
        write_json(os.path.join(r_run_dir, "quantum_resource_summary.json"), resource_summary)
        _save_run(r_run_dir, config, reupload_train.model, "qgnn_v2_depth3_data_reupload", v2prepared, reupload_train, reupload_eval, seed, r_inference_s, {"n_components": v2_arch.n_components, "quantum": resource_summary})
        print(f"  saved -> {r_run_dir}")
        if r_test.get("pr_auc") is not None:
            reupload_test_pr.append(r_test["pr_auc"])
        if r_val.get("pr_auc") is not None:
            reupload_val_pr.append(r_val["pr_auc"])

    if len(seeds) > 1:
        paired_diffs = [r - b for r, b in zip(reupload_test_pr, baseline_test_pr)]
        summary = {
            "seeds": seeds,
            "baseline_test_pr_auc": {"mean": float(np.mean(baseline_test_pr)), "std": float(np.std(baseline_test_pr)), "min": float(np.min(baseline_test_pr)), "max": float(np.max(baseline_test_pr))},
            "baseline_val_pr_auc": {"mean": float(np.mean(baseline_val_pr)), "std": float(np.std(baseline_val_pr))},
            "reupload_test_pr_auc": {"mean": float(np.mean(reupload_test_pr)), "std": float(np.std(reupload_test_pr)), "min": float(np.min(reupload_test_pr)), "max": float(np.max(reupload_test_pr))},
            "reupload_val_pr_auc": {"mean": float(np.mean(reupload_val_pr)), "std": float(np.std(reupload_val_pr))},
            "paired_test_pr_auc_delta": {"per_seed": dict(zip(seeds, paired_diffs)), "mean": float(np.mean(paired_diffs)), "std": float(np.std(paired_diffs)), "n_improving": sum(1 for d in paired_diffs if d > 0), "n_degrading": sum(1 for d in paired_diffs if d < 0)},
        }
        summary_path = os.path.join(config.experiment.output_dir, f"multiseed_summary_{args.tag}_seeds_{'-'.join(str(s) for s in seeds)}.json")
        write_json(summary_path, summary)
        print(f"\nMulti-seed summary -> {summary_path}")
        print(f"  A baseline  test PR-AUC: {summary['baseline_test_pr_auc']}")
        print(f"  B reupload  test PR-AUC: {summary['reupload_test_pr_auc']}")
        print(f"  paired delta (B-A): {summary['paired_test_pr_auc_delta']}")


if __name__ == "__main__":
    main()
