#!/usr/bin/env python3
"""QGNN-v2: GraphSAGE-Hybrid-Control vs Hybrid-QGNN
(QGNN_V2_HYBRID_IMPLEMENTATION_AND_DEPTH_BENCHMARK.md sections 7-8, 11,
15).

For each model seed, reuses the EXISTING, already-frozen GraphSAGE-Full
checkpoint trained with that same seed (experiments/classical_gnn/) as a
frozen encoder -- never retrains it. Extracts supplier embeddings,
fits a train-only PCA down to `--n-components` dims, then trains a
matched classical head and a quantum head (configurable variational-layer
depth) on the identical reduced representation.

Usage:
    python scripts/run_qgnn_v2_experiment.py --config configs/qgnn_v2.yaml --tag smoke --seeds 42 --n-layers 1 --epochs 5
    python scripts/run_qgnn_v2_experiment.py --config configs/qgnn_v2.yaml --tag depth3 --n-layers 3 --seeds 42,43,44,45,46
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from scm_dataset.modeling.evaluate import render_all_plots
from scm_dataset.modeling.experiment import build_run_metadata, new_run_dir, save_config, write_json
from scm_dataset.modeling.graph_embedding_reduction import (
    extract_supplier_embeddings,
    load_frozen_graphsage_encoder,
    prepare_frozen_encoder_input,
)
from scm_dataset.modeling.qgnn import build_qgnn_model
from scm_dataset.modeling.qgnn_v2 import build_classical_head, build_v2_prepared, evaluate_v2, load_qgnn_v2_config, set_seed, train_v2_head
from scm_dataset.modeling.train import class_balance_summary

from _analysis_common import find_latest_graphsage_full_checkpoint


def _save_run(run_dir: str, config, model, model_kind: str, v2prepared, train_result, eval_result, seed: int, extra_metadata: dict) -> None:
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
        config, v2prepared.benchmark, _fake_frames(), seed=seed,
        split_summary=class_balance_summary(v2prepared.examples),
        training_duration_seconds=train_result.training_duration_seconds,
        extra={"model_kind": model_kind, "best_epoch": train_result.best_epoch, "stopped_early": train_result.stopped_early, "pos_weight": train_result.pos_weight, **extra_metadata},
    )
    write_json(os.path.join(run_dir, "run_metadata.json"), metadata)


def _fake_frames():
    """`build_run_metadata` needs a `NodeFeatureFrames`-shaped object only
    to list feature column names per node type -- meaningless for a v2
    head that consumes PCA'd embedding dimensions, not named features.
    Rather than modify `build_run_metadata` (frozen shared infra), pass an
    empty-but-shaped stand-in; the real feature provenance (encoder
    checkpoint path, n_components, reduction method) is recorded in
    `extra_metadata` instead."""
    from scm_dataset.modeling.features import NodeFeatureFrames

    return NodeFeatureFrames(frames={}, numeric_columns={}, categorical_columns={})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/qgnn_v2.yaml")
    parser.add_argument("--tag", default="depth_run")
    parser.add_argument("--seeds", default=None, help="Comma-separated model seeds; defaults to the config's experiment.seeds.")
    parser.add_argument("--n-components", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None, help="Override qgnn_v2.n_layers (variational layers) -- the primary depth experiment's independent variable.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override config's training.epochs (useful for a smoke test).")
    parser.add_argument("--encoder-experiments-dir", default="experiments/classical_gnn", help="Where to find the existing GraphSAGE-Full checkpoints being reused as the frozen encoder.")
    args = parser.parse_args()

    cfg = load_qgnn_v2_config(args.config)
    config = cfg.base
    v2_arch = cfg.qgnn_v2
    if args.n_components:
        v2_arch.n_components = args.n_components
    if args.n_layers:
        v2_arch.n_layers = args.n_layers
    if args.device:
        v2_arch.device = args.device
    if args.epochs:
        config.training.epochs = args.epochs
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(config.experiment.seeds)

    print(f"QGNN-v2: dataset={config.dataset.dataset_id} n_components={v2_arch.n_components} n_layers={v2_arch.n_layers} device={v2_arch.device}")

    classical_pr_aucs, qgnn_pr_aucs = [], []

    for seed in seeds:
        print(f"\n=== model seed {seed} (encoder + heads) ===")
        checkpoint_dir = find_latest_graphsage_full_checkpoint(args.encoder_experiments_dir, seed)
        print(f"  frozen encoder checkpoint: {checkpoint_dir}")

        from scm_dataset.modeling.data import load_benchmark

        benchmark = load_benchmark(config.dataset.benchmark_path, config.dataset.dataset_id)
        prepared_full = prepare_frozen_encoder_input(config, benchmark, os.path.join(checkpoint_dir, "preprocessing"))
        encoder = load_frozen_graphsage_encoder(os.path.join(checkpoint_dir, "model.pt"), prepared_full)
        assert all(not p.requires_grad for p in encoder.parameters()), "frozen encoder must have requires_grad=False on every parameter"

        all_times = sorted(prepared_full.examples["time"].unique())
        embedding_frame = extract_supplier_embeddings(encoder, prepared_full, all_times)
        print(f"  extracted embeddings: {embedding_frame.shape}")

        v2prepared, reducer = build_v2_prepared(config, benchmark, prepared_full.examples, embedding_frame, v2_arch.n_components)

        print("-- GraphSAGE-Hybrid-Control (classical) --")
        set_seed(seed)  # must precede model construction so weight init is reproducible too, not just training-time shuffling
        classical_model = build_classical_head(v2_arch.n_components, v2_arch.mlp_hidden)
        classical_train = train_v2_head(v2prepared, classical_model, seed=seed, verbose=False)
        classical_eval = evaluate_v2(classical_train.model, v2prepared, config.threshold)
        c_test = classical_eval.metrics_by_split.get("test", {})
        print(f"  test: pr_auc={c_test.get('pr_auc')} roc_auc={c_test.get('roc_auc')}")
        c_run_dir = new_run_dir(config.experiment.output_dir, f"{args.tag}_classical_seed{seed}")
        reducer.save(os.path.join(c_run_dir, "reduction"))
        write_json(os.path.join(c_run_dir, "encoder_checkpoint.json"), {"checkpoint_dir": checkpoint_dir})
        _save_run(c_run_dir, config, classical_train.model, "graphsage_hybrid_control", v2prepared, classical_train, classical_eval, seed, {"n_components": v2_arch.n_components, "encoder_checkpoint": checkpoint_dir})
        print(f"  saved -> {c_run_dir}")
        if c_test.get("pr_auc") is not None:
            classical_pr_aucs.append(c_test["pr_auc"])

        print(f"-- Hybrid-QGNN (depth={v2_arch.n_layers}) --")
        set_seed(seed)
        qgnn_model = build_qgnn_model(n_qubits=v2_arch.n_components, n_layers=v2_arch.n_layers, mlp_hidden=v2_arch.mlp_hidden, device_name=v2_arch.device)
        qgnn_train = train_v2_head(v2prepared, qgnn_model, seed=seed, verbose=False)
        qgnn_eval = evaluate_v2(qgnn_train.model, v2prepared, config.threshold)
        q_test = qgnn_eval.metrics_by_split.get("test", {})
        resource_summary = qgnn_train.model.quantum_resource_summary()
        print(f"  test: pr_auc={q_test.get('pr_auc')} roc_auc={q_test.get('roc_auc')}")
        print(f"  quantum resource summary: {resource_summary}")
        q_run_dir = new_run_dir(config.experiment.output_dir, f"{args.tag}_qgnn_seed{seed}")
        reducer.save(os.path.join(q_run_dir, "reduction"))
        write_json(os.path.join(q_run_dir, "encoder_checkpoint.json"), {"checkpoint_dir": checkpoint_dir})
        write_json(os.path.join(q_run_dir, "quantum_resource_summary.json"), resource_summary)
        _save_run(q_run_dir, config, qgnn_train.model, "hybrid_qgnn", v2prepared, qgnn_train, qgnn_eval, seed, {"n_components": v2_arch.n_components, "n_layers": v2_arch.n_layers, "encoder_checkpoint": checkpoint_dir, "quantum": resource_summary})
        print(f"  saved -> {q_run_dir}")
        if q_test.get("pr_auc") is not None:
            qgnn_pr_aucs.append(q_test["pr_auc"])

    if len(seeds) > 1:
        summary = {
            "n_components": v2_arch.n_components, "n_layers": v2_arch.n_layers, "seeds": seeds,
            "graphsage_hybrid_control_pr_auc": {"mean": float(np.mean(classical_pr_aucs)), "std": float(np.std(classical_pr_aucs)), "min": float(np.min(classical_pr_aucs)), "max": float(np.max(classical_pr_aucs))} if classical_pr_aucs else None,
            "hybrid_qgnn_pr_auc": {"mean": float(np.mean(qgnn_pr_aucs)), "std": float(np.std(qgnn_pr_aucs)), "min": float(np.min(qgnn_pr_aucs)), "max": float(np.max(qgnn_pr_aucs))} if qgnn_pr_aucs else None,
        }
        summary_path = os.path.join(config.experiment.output_dir, f"multiseed_summary_{args.tag}_seeds_{'-'.join(str(s) for s in seeds)}.json")
        write_json(summary_path, summary)
        print(f"\nMulti-seed summary -> {summary_path}")
        print(f"  GraphSAGE-Hybrid-Control PR-AUC: {summary['graphsage_hybrid_control_pr_auc']}")
        print(f"  Hybrid-QGNN              PR-AUC: {summary['hybrid_qgnn_pr_auc']}")


if __name__ == "__main__":
    main()
