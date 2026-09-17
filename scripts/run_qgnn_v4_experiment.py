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
from scm_dataset.modeling.qgnn_v2 import build_v2_prepared, evaluate_v2, set_seed, train_v2_head
from scm_dataset.modeling.quantum import (
    HybridQuantumHead,
    HybridQuantumHeadLayerNorm,
    HybridQuantumHeadOutputScale,
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


def _build_quantum_model(variant: str, in_dim: int, v4_arch, alpha_init: float, projection: dict, quantum_init: str):
    """Phase 2b (output-scale/calibration investigation, QGNN_V4_PHASE2B_REPORT.md):
    `variant="baseline"` is the unmodified HybridQuantumHead (identical to
    every prior phase). The other three variants all use
    HybridQuantumHeadOutputScale with the SAME qubits/layers/ansatz --
    only the scale/bias mechanism on the pre-Linear(n_qubits,1)
    representation differs. No architecture, encoding, or circuit change
    in any variant.

    `projection` (Phase 4 Stage 2, Track B): {"projection_type",
    "projection_hidden_dim", "pre_projection_norm"} -- only meaningful for
    the two LayerNorm variants (Stage 2's fixed output-side control).
    Every other variant requires the all-defaults ("linear" projection, no
    pre-norm) projection dict -- passing a non-default projection with a
    non-LayerNorm variant is a configuration error, not silently ignored.

    `quantum_init` (Phase 4 Stage 4, Track F): "default"/"small_gaussian"/
    "identity_like" -- same restriction as `projection`, only meaningful
    for the two LayerNorm variants; a non-default value with any other
    variant is a configuration error."""
    common = dict(
        n_qubits=v4_arch.n_qubits, n_layers=v4_arch.n_layers,
        ansatz=v4_arch.ansatz, diff_method=v4_arch.diff_method, device_name=v4_arch.device,
    )
    is_default_projection = projection == {"projection_type": "linear", "projection_hidden_dim": 32, "pre_projection_norm": False}
    is_default_init = quantum_init == "default"
    incompatible_msg = "--projection-type/--pre-projection-norm/--quantum-init require --head-variant layernorm_noaffine or layernorm_affine (Stage 2/4's fixed output-side control)"
    if variant == "baseline":
        if not (is_default_projection and is_default_init):
            raise ValueError(incompatible_msg)
        return HybridQuantumHead(in_dim, **common)
    if variant == "scale":
        if not (is_default_projection and is_default_init):
            raise ValueError(incompatible_msg)
        return HybridQuantumHeadOutputScale(in_dim, **common, alpha_init=alpha_init, use_bias=False, trainable_scale=True)
    if variant == "scale_bias":
        if not (is_default_projection and is_default_init):
            raise ValueError(incompatible_msg)
        return HybridQuantumHeadOutputScale(in_dim, **common, alpha_init=alpha_init, use_bias=True, trainable_scale=True)
    if variant == "fixed_scale":
        if not (is_default_projection and is_default_init):
            raise ValueError(incompatible_msg)
        return HybridQuantumHeadOutputScale(in_dim, **common, alpha_init=alpha_init, use_bias=False, trainable_scale=False)
    if variant == "layernorm_noaffine":
        return HybridQuantumHeadLayerNorm(in_dim, **common, elementwise_affine=False, quantum_init=quantum_init, **projection)
    if variant == "layernorm_affine":
        return HybridQuantumHeadLayerNorm(in_dim, **common, elementwise_affine=True, quantum_init=quantum_init, **projection)
    raise ValueError(f"unknown head variant {variant!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/qgnn_v4.yaml")
    parser.add_argument("--tag", default="v4_run")
    parser.add_argument("--seeds", default=None, help="Comma-separated model seeds; defaults to the config's experiment.seeds.")
    parser.add_argument("--n-qubits", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None, help="Override quantum_v4.n_layers (variational layers).")
    parser.add_argument("--ansatz", default=None, choices=["strongly_entangling", "basic_entangler", "hardware_efficient_ring", "reduced_entanglement"])
    parser.add_argument("--diff-method", default=None, choices=["backprop", "parameter-shift"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override config's training.epochs (useful for a smoke test).")
    parser.add_argument("--patience", type=int, default=None, help="Override config's training.early_stopping_patience (Phase 2 stability investigation -- QGNN_V4_BENCHMARK.md/QGNN_V4_PHASE1_DIAGNOSTICS.md). max_epochs stays whatever --epochs/config already set; only the patience changes.")
    parser.add_argument("--encoder-experiments-dir", default="experiments/classical_gnn", help="Where to find the existing GraphSAGE-Full checkpoints being reused as the frozen encoder.")
    parser.add_argument("--diagnostics", action="store_true", help="Use quantum.train.train_v4_head_with_diagnostics instead of qgnn_v2.train_v2_head: logs per-epoch train PR-AUC and quantum/reduce-layer gradient norms into training_history.csv. Same optimizer/loss/early-stopping setup either way -- only the logging differs, so results are directly comparable to non-diagnostic runs.")
    parser.add_argument("--head-variant", default="baseline", choices=["baseline", "scale", "scale_bias", "fixed_scale", "layernorm_noaffine", "layernorm_affine"], help="Phase 2b/2c output-scale and normalization investigation (QGNN_V4_PHASE2B_REPORT.md, QGNN_V4_PHASE2C_REPORT.md). 'baseline'=unmodified HybridQuantumHead (default, identical to every prior phase). 'scale'=trainable alpha before the existing Linear(n_qubits,1). 'scale_bias'=trainable alpha+beta. 'fixed_scale'=non-trainable alpha at --alpha-init. 'layernorm_noaffine'/'layernorm_affine'=LayerNorm(n_qubits) on the PauliZ output before Linear(n_qubits,1), without/with a learnable per-qubit scale+bias. Same qubits/layers/ansatz/encoder in every case.")
    parser.add_argument("--alpha-init", type=float, default=1.0, help="Initial (or, for --head-variant fixed_scale, fixed) value of the output-scale alpha. Ignored for --head-variant baseline.")
    parser.add_argument("--quantum-only", action="store_true", help="Skip the matched-capacity classical arm -- use when the classical control is unchanged from an already-saved baseline run (Phase 2b: classical is never modified, so re-running it would just reproduce existing results).")
    parser.add_argument("--projection-type", default="linear", choices=["linear", "nonlinear"], help="Phase 4 Stage 2 Track B2 (QGNN_V4_PHASE4_PLAN.md): 'linear'=the existing single Linear(in_dim,n_qubits) bottleneck (default, identical to every prior phase). 'nonlinear'=Linear(in_dim,--projection-hidden-dim)->GELU->Linear(--projection-hidden-dim,n_qubits). Only valid with --head-variant layernorm_noaffine/layernorm_affine.")
    parser.add_argument("--projection-hidden-dim", type=int, default=32, help="Hidden width for --projection-type nonlinear. Ignored otherwise.")
    parser.add_argument("--pre-projection-norm", action="store_true", help="Track B3: applies LayerNorm(in_dim) to the frozen embedding BEFORE the projection into the quantum circuit. Only valid with --head-variant layernorm_noaffine/layernorm_affine. Independent of --projection-type (can combine with either).")
    parser.add_argument("--pca-components", type=int, default=None, help="Track B4: train-split-only PCA (reusing qgnn_v2.build_v2_prepared/PCASupplierReducer, the same infra qgnn_v2.py already uses) reduces the frozen embedding to this many dimensions BEFORE it reaches the head -- in_dim becomes --pca-components instead of the raw embedding width. n_qubits/n_layers/ansatz stay whatever --n-qubits/etc. already set (unchanged quantum circuit, only the classical input changes). Default None = no PCA, raw embedding used unchanged (Track B1/B2/B3's setting).")
    parser.add_argument("--quantum-init", default="default", choices=["default", "small_gaussian", "identity_like"], help="Phase 4 Stage 4 Track F (QGNN_V4_PHASE4_PLAN.md): initial values of the quantum circuit's own trainable parameters -- same shape/count either way. 'default' (F1, the control): PennyLane's own TorchLayer default, uniform[0,2*pi] -- identical to every prior phase. 'small_gaussian' (F2): mean=0,std=0.01. 'identity_like' (F3): every rotation parameter starts at exactly 0.0 (Rot(0,0,0)/RY(0)/RZ(0)/RX(0) are each exactly the single-qubit identity) -- the fixed entangling CNOT pattern is NOT parameterized and still fires regardless, so this is identity-like for the rotation gates only, not the whole circuit. Only valid with --head-variant layernorm_noaffine/layernorm_affine.")
    parser.add_argument("--output-subdir", default=None, help="Joined onto config.experiment.output_dir before every run/summary path (e.g. 'phase4_stage4/small_gaussian') -- keeps a stage's configurations in separate directories instead of all landing in the same experiments/qgnn_v4/ flat listing. Default None = today's behavior, unchanged.")
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
    if args.patience:
        config.training.early_stopping_patience = args.patience
    if args.output_subdir:
        config.experiment.output_dir = os.path.join(config.experiment.output_dir, args.output_subdir)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(config.experiment.seeds)

    print(f"QGNN-v4: dataset={config.dataset.dataset_id} n_qubits={v4_arch.n_qubits} n_layers={v4_arch.n_layers} ansatz={v4_arch.ansatz} diff_method={v4_arch.diff_method} device={v4_arch.device} epochs={config.training.epochs} patience={config.training.early_stopping_patience}")

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

        pca_reducer = None
        if args.pca_components:
            # Track B4: train-split-only PCA fit, exactly the mechanism
            # qgnn_v2.py's own QGNN-v2 experiments already use -- NOT a new
            # reduction implementation. `examples` (supplier/time/target/
            # split) is unchanged; only the input columns the head sees
            # shrink from raw hidden_dim to --pca-components.
            v4prepared, pca_reducer = build_v2_prepared(config, benchmark, prepared_full.examples, embedding_frame, args.pca_components)
            in_dim = args.pca_components
            print(f"  PCA-reduced embeddings (train-fit only): {in_dim} components")
        else:
            v4prepared = build_v4_prepared(config, benchmark, prepared_full.examples, embedding_frame)
            in_dim = embedding_frame.shape[1]

        if not args.quantum_only:
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

        projection = {
            "projection_type": args.projection_type,
            "projection_hidden_dim": args.projection_hidden_dim,
            "pre_projection_norm": args.pre_projection_norm,
        }
        print(f"-- Hybrid-Quantum-v4 (variant={args.head_variant}, n_qubits={v4_arch.n_qubits}, n_layers={v4_arch.n_layers}, ansatz={v4_arch.ansatz}, projection={projection}, pca_components={args.pca_components}, quantum_init={args.quantum_init}) --")
        set_seed(seed)
        quantum_model = _build_quantum_model(args.head_variant, in_dim, v4_arch, args.alpha_init, projection, args.quantum_init)
        # Phase 4 Stage 4 Track F's initialization audit (QGNN_V4_PHASE4_PLAN.md
        # section 16): captured BEFORE train_fn touches the model at all, so
        # this is genuinely the pre-training distribution, not a snapshot
        # after any optimizer step.
        init_stats = quantum_model.quantum_parameter_stats() if hasattr(quantum_model, "quantum_parameter_stats") else None
        quantum_train = train_fn(v4prepared, quantum_model, seed=seed, verbose=False)
        quantum_eval = evaluate_v2(quantum_train.model, v4prepared, config.threshold)
        q_test = quantum_eval.metrics_by_split.get("test", {})
        resource_summary = quantum_train.model.quantum_resource_summary()
        final_stats = quantum_train.model.quantum_parameter_stats() if hasattr(quantum_train.model, "quantum_parameter_stats") else None
        print(f"  test: pr_auc={q_test.get('pr_auc')} roc_auc={q_test.get('roc_auc')}")
        print(f"  quantum resource summary: {resource_summary}")
        if init_stats is not None:
            print(f"  quantum param stats: initial={init_stats}  final={final_stats}")
        q_run_dir = new_run_dir(config.experiment.output_dir, f"{args.tag}_quantum_seed{seed}")
        write_json(os.path.join(q_run_dir, "encoder_checkpoint.json"), {"checkpoint_dir": checkpoint_dir})
        write_json(os.path.join(q_run_dir, "quantum_resource_summary.json"), resource_summary)
        if init_stats is not None:
            write_json(os.path.join(q_run_dir, "quantum_init_stats.json"), {"initial": init_stats, "final": final_stats})
        _save_run(q_run_dir, config, quantum_train.model, "hybrid_quantum_v4", v4prepared, quantum_train, quantum_eval, seed, {"in_dim": in_dim, "encoder_checkpoint": checkpoint_dir, "quantum": resource_summary, "head_variant": args.head_variant, "alpha_init": args.alpha_init, "pca_components": args.pca_components, "quantum_init": args.quantum_init})
        print(f"  saved -> {q_run_dir}")
        if q_test.get("pr_auc") is not None:
            quantum_pr_aucs.append(q_test["pr_auc"])

    if len(seeds) > 1:
        summary = {
            "n_qubits": v4_arch.n_qubits, "n_layers": v4_arch.n_layers, "ansatz": v4_arch.ansatz, "seeds": seeds,
            "matched_capacity_classical_pr_auc": {"mean": float(np.mean(classical_pr_aucs)), "std": float(np.std(classical_pr_aucs)), "min": float(np.min(classical_pr_aucs)), "max": float(np.max(classical_pr_aucs))} if classical_pr_aucs else None,
            "hybrid_quantum_v4_pr_auc": {"mean": float(np.mean(quantum_pr_aucs)), "std": float(np.std(quantum_pr_aucs)), "min": float(np.min(quantum_pr_aucs)), "max": float(np.max(quantum_pr_aucs))} if quantum_pr_aucs else None,
        }
        # config.split.strategy is included here (not just args.tag) because
        # configs/qgnn_v4.yaml and configs/qgnn_v4_severity.yaml share the
        # same experiment.output_dir -- running primary then severity with
        # the same --tag would otherwise silently overwrite this convenience
        # summary file (discovered during Phase 3; each per-seed run
        # directory is unaffected, only this aggregate).
        summary_path = os.path.join(config.experiment.output_dir, f"multiseed_summary_{args.tag}_{config.split.strategy}_seeds_{'-'.join(str(s) for s in seeds)}.json")
        write_json(summary_path, summary)
        print(f"\nMulti-seed summary -> {summary_path}")
        print(f"  Matched-Capacity-Classical-v4 PR-AUC: {summary['matched_capacity_classical_pr_auc']}")
        print(f"  Hybrid-Quantum-v4             PR-AUC: {summary['hybrid_quantum_v4_pr_auc']}")


if __name__ == "__main__":
    main()
