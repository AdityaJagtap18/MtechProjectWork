#!/usr/bin/env python3
"""QGNN-v4: StronglyEntanglingLayers hybrid head vs. its matched-capacity
classical control (RQ-Q3), both on a raw (unreduced) frozen GraphSAGE-Full
embedding (PENNYLANE_QGNN_IMPLEMENTATION_PLAN.md sections 4, 7-10).

For each model seed, reuses the EXISTING, already-frozen GraphSAGE-Full
checkpoint trained with that same seed (experiments/classical_gnn/) as a
frozen encoder -- never retrains it. Extracts the raw hidden_dim supplier
embedding (no PCA -- the trainable Linear(in_dim, n_qubits) inside each
head IS the reduction, see quantum/model.py::build_v4_prepared), then
trains a HybridQuantumHead and a MatchedCapacityClassicalHead on the
identical input.

Usage:
    python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4.yaml --tag smoke --seeds 42 --epochs 5
    python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4.yaml --tag primary --seeds 42,43,44,45,46
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
from scm_dataset.modeling.qgnn_v2 import evaluate_v2, set_seed, train_v2_head
from scm_dataset.modeling.quantum import (
    HybridQuantumHead,
    MatchedCapacityClassicalHead,
    build_v4_prepared,
    load_quantum_v4_config,
    train_v4_head_with_diagnostics,
)
from scm_dataset.modeling.train import class_balance_summary

from _analysis_common import find_latest_graphsage_full_checkpoint


def _save_run(run_dir: str, config, model, model_kind: str, v4prepared, train_result, eval_result, seed: int, extra_metadata: dict) -> None:
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
        config, v4prepared.benchmark, _fake_frames(), seed=seed,
        split_summary=class_balance_summary(v4prepared.examples),
        training_duration_seconds=train_result.training_duration_seconds,
        extra={"model_kind": model_kind, "best_epoch": train_result.best_epoch, "stopped_early": train_result.stopped_early, "pos_weight": train_result.pos_weight, **extra_metadata},
    )
    write_json(os.path.join(run_dir, "run_metadata.json"), metadata)


def _fake_frames():
    """`build_run_metadata` needs a `NodeFeatureFrames`-shaped object only
    to list feature column names per node type -- meaningless for a v4
    head that consumes a raw frozen-encoder embedding, not named features.
    Mirrors `run_qgnn_v2_experiment.py`'s own stand-in exactly; the real
    feature provenance (encoder checkpoint path, in_dim) is recorded in
    `extra_metadata` instead."""
    from scm_dataset.modeling.features import NodeFeatureFrames

    return NodeFeatureFrames(frames={}, numeric_columns={}, categorical_columns={})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/qgnn_v4.yaml")
    parser.add_argument("--tag", default="v4_run")
    parser.add_argument("--seeds", default=None, help="Comma-separated model seeds; defaults to the config's experiment.seeds.")
    parser.add_argument("--n-qubits", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None, help="Override quantum_v4.n_layers (variational layers).")
    parser.add_argument("--ansatz", default=None, choices=["strongly_entangling", "basic_entangler"])
    parser.add_argument("--diff-method", default=None, choices=["backprop", "parameter-shift"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override config's training.epochs (useful for a smoke test).")
    parser.add_argument("--encoder-experiments-dir", default="experiments/classical_gnn", help="Where to find the existing GraphSAGE-Full checkpoints being reused as the frozen encoder.")
    parser.add_argument("--diagnostics", action="store_true", help="Use quantum.train.train_v4_head_with_diagnostics instead of qgnn_v2.train_v2_head: logs per-epoch train PR-AUC and quantum/reduce-layer gradient norms into training_history.csv. Same optimizer/loss/early-stopping setup either way -- only the logging differs, so results are directly comparable to non-diagnostic runs.")
    args = parser.parse_args()

    train_fn = train_v4_head_with_diagnostics if args.diagnostics else train_v2_head

    cfg = load_quantum_v4_config(args.config)
    config = cfg.base
    v4_arch = cfg.quantum_v4
    if args.n_qubits:
        v4_arch.n_qubits = args.n_qubits
    if args.n_layers:
        v4_arch.n_layers = args.n_layers
    if args.ansatz:
        v4_arch.ansatz = args.ansatz
    if args.diff_method:
        v4_arch.diff_method = args.diff_method
    if args.device:
        v4_arch.device = args.device
    if args.epochs:
        config.training.epochs = args.epochs
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(config.experiment.seeds)

    print(f"QGNN-v4: dataset={config.dataset.dataset_id} n_qubits={v4_arch.n_qubits} n_layers={v4_arch.n_layers} ansatz={v4_arch.ansatz} diff_method={v4_arch.diff_method} device={v4_arch.device}")

    classical_pr_aucs, quantum_pr_aucs = [], []

    for seed in seeds:
        print(f"\n=== model seed {seed} (encoder + heads) ===")
        split_suffix = config.split.strategy if config.split.strategy != "temporal" else None
        checkpoint_dir = find_latest_graphsage_full_checkpoint(args.encoder_experiments_dir, seed, split_suffix=split_suffix)
        print(f"  frozen encoder checkpoint: {checkpoint_dir}")

        from scm_dataset.modeling.data import load_benchmark

        benchmark = load_benchmark(config.dataset.benchmark_path, config.dataset.dataset_id)
        prepared_full = prepare_frozen_encoder_input(config, benchmark, os.path.join(checkpoint_dir, "preprocessing"))
        encoder = load_frozen_graphsage_encoder(os.path.join(checkpoint_dir, "model.pt"), prepared_full)
        assert all(not p.requires_grad for p in encoder.parameters()), "frozen encoder must have requires_grad=False on every parameter"

        all_times = sorted(prepared_full.examples["time"].unique())
        embedding_frame = extract_supplier_embeddings(encoder, prepared_full, all_times)
        print(f"  extracted raw embeddings (no PCA): {embedding_frame.shape}")

        v4prepared = build_v4_prepared(config, benchmark, prepared_full.examples, embedding_frame)
        in_dim = embedding_frame.shape[1]

        print("-- Matched-Capacity-Classical-v4 (RQ-Q3 control) --")
        set_seed(seed)  # must precede model construction so weight init is reproducible too, not just training-time shuffling
        classical_model = MatchedCapacityClassicalHead(in_dim, v4_arch.n_qubits)
        classical_train = train_fn(v4prepared, classical_model, seed=seed, verbose=False)
        classical_eval = evaluate_v2(classical_train.model, v4prepared, config.threshold)
        c_test = classical_eval.metrics_by_split.get("test", {})
        c_params = sum(p.numel() for p in classical_model.parameters())
        print(f"  test: pr_auc={c_test.get('pr_auc')} roc_auc={c_test.get('roc_auc')}  params={c_params}")
        c_run_dir = new_run_dir(config.experiment.output_dir, f"{args.tag}_matched_classical_seed{seed}")
        write_json(os.path.join(c_run_dir, "encoder_checkpoint.json"), {"checkpoint_dir": checkpoint_dir})
        _save_run(c_run_dir, config, classical_train.model, "matched_capacity_classical_v4", v4prepared, classical_train, classical_eval, seed, {"n_qubits": v4_arch.n_qubits, "in_dim": in_dim, "encoder_checkpoint": checkpoint_dir, "total_trainable_parameters": c_params})
        print(f"  saved -> {c_run_dir}")
        if c_test.get("pr_auc") is not None:
            classical_pr_aucs.append(c_test["pr_auc"])

        print(f"-- Hybrid-Quantum-v4 (n_qubits={v4_arch.n_qubits}, n_layers={v4_arch.n_layers}, ansatz={v4_arch.ansatz}) --")
        set_seed(seed)
        quantum_model = HybridQuantumHead(
            in_dim, n_qubits=v4_arch.n_qubits, n_layers=v4_arch.n_layers,
            ansatz=v4_arch.ansatz, diff_method=v4_arch.diff_method, device_name=v4_arch.device,
        )
        quantum_train = train_fn(v4prepared, quantum_model, seed=seed, verbose=False)
        quantum_eval = evaluate_v2(quantum_train.model, v4prepared, config.threshold)
        q_test = quantum_eval.metrics_by_split.get("test", {})
        resource_summary = quantum_train.model.quantum_resource_summary()
        print(f"  test: pr_auc={q_test.get('pr_auc')} roc_auc={q_test.get('roc_auc')}")
        print(f"  quantum resource summary: {resource_summary}")
        q_run_dir = new_run_dir(config.experiment.output_dir, f"{args.tag}_quantum_seed{seed}")
        write_json(os.path.join(q_run_dir, "encoder_checkpoint.json"), {"checkpoint_dir": checkpoint_dir})
        write_json(os.path.join(q_run_dir, "quantum_resource_summary.json"), resource_summary)
        _save_run(q_run_dir, config, quantum_train.model, "hybrid_quantum_v4", v4prepared, quantum_train, quantum_eval, seed, {"in_dim": in_dim, "encoder_checkpoint": checkpoint_dir, "quantum": resource_summary})
        print(f"  saved -> {q_run_dir}")
        if q_test.get("pr_auc") is not None:
            quantum_pr_aucs.append(q_test["pr_auc"])

    if len(seeds) > 1:
        summary = {
            "n_qubits": v4_arch.n_qubits, "n_layers": v4_arch.n_layers, "ansatz": v4_arch.ansatz, "seeds": seeds,
            "matched_capacity_classical_pr_auc": {"mean": float(np.mean(classical_pr_aucs)), "std": float(np.std(classical_pr_aucs)), "min": float(np.min(classical_pr_aucs)), "max": float(np.max(classical_pr_aucs))} if classical_pr_aucs else None,
            "hybrid_quantum_v4_pr_auc": {"mean": float(np.mean(quantum_pr_aucs)), "std": float(np.std(quantum_pr_aucs)), "min": float(np.min(quantum_pr_aucs)), "max": float(np.max(quantum_pr_aucs))} if quantum_pr_aucs else None,
        }
        summary_path = os.path.join(config.experiment.output_dir, f"multiseed_summary_{args.tag}_seeds_{'-'.join(str(s) for s in seeds)}.json")
        write_json(summary_path, summary)
        print(f"\nMulti-seed summary -> {summary_path}")
        print(f"  Matched-Capacity-Classical-v4 PR-AUC: {summary['matched_capacity_classical_pr_auc']}")
        print(f"  Hybrid-Quantum-v4             PR-AUC: {summary['hybrid_quantum_v4_pr_auc']}")


if __name__ == "__main__":
    main()
