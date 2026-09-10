#!/usr/bin/env python3
"""QGNN-v3 controlled architecture study: entanglement topology (ring,
all-to-all) and trainable input scaling, all at the selected depth-3, 8
qubits. Purely additive -- does not modify or call
`scripts/run_qgnn_v2_experiment.py` / `run_qgnn_v2_reupload_experiment.py`.
Reuses the same frozen-encoder + PCA pipeline for every variant, so the
only intentional difference between runs is the architecture change
itself:

    A -- existing baseline (qgnn.build_qgnn_model, unmodified)
    B -- ring entanglement (qgnn_v3.build_qgnn_ring_model)
    C -- all-to-all entanglement (qgnn_v3.build_qgnn_all_to_all_model)
    D -- trainable input scaling (qgnn_v3.build_qgnn_trainable_scaling_model)

Variant E (best justified combination) is deliberately not implemented
here -- the plan only calls for it if B/C/D show a clear validation
improvement over A, which is not yet established at the time this script
runs.

Usage:
    python scripts/run_qgnn_v3_experiment.py --config configs/qgnn_v2.yaml --tag v3_smoke --seeds 42 --epochs 5
    python scripts/run_qgnn_v3_experiment.py --config configs/qgnn_v2.yaml --tag v3 --seeds 42,43,44,45,46
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
from scm_dataset.modeling.qgnn_v3 import build_qgnn_all_to_all_model, build_qgnn_ring_model, build_qgnn_trainable_scaling_model
from scm_dataset.modeling.train import class_balance_summary

from _analysis_common import find_latest_graphsage_full_checkpoint

VARIANTS = {
    "A_baseline": lambda n_qubits, mlp_hidden, device: build_qgnn_model(n_qubits=n_qubits, n_layers=3, mlp_hidden=mlp_hidden, device_name=device),
    "B_ring": lambda n_qubits, mlp_hidden, device: build_qgnn_ring_model(n_qubits, 3, mlp_hidden, device),
    "C_all_to_all": lambda n_qubits, mlp_hidden, device: build_qgnn_all_to_all_model(n_qubits, 3, mlp_hidden, device),
    "D_trainable_scaling": lambda n_qubits, mlp_hidden, device: build_qgnn_trainable_scaling_model(n_qubits, 3, mlp_hidden, device),
}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/qgnn_v2.yaml")
    parser.add_argument("--tag", default="v3")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--n-components", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override config's training.epochs (useful for a smoke test).")
    parser.add_argument("--encoder-experiments-dir", default="experiments/classical_gnn")
    args = parser.parse_args()

    cfg = load_qgnn_v2_config(args.config)
    config = cfg.base
    v2_arch = cfg.qgnn_v2
    v2_arch.n_layers = 3  # fixed: this study is about topology/encoding at the already-selected depth, not a new depth sweep
    if args.n_components:
        v2_arch.n_components = args.n_components
    if args.device:
        v2_arch.device = args.device
    if args.epochs:
        config.training.epochs = args.epochs
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(config.experiment.seeds)

    print(f"QGNN-v3 architecture study: dataset={config.dataset.dataset_id} n_components={v2_arch.n_components} device={v2_arch.device} variants={list(VARIANTS)}")

    test_pr_by_variant = {name: [] for name in VARIANTS}
    val_pr_by_variant = {name: [] for name in VARIANTS}

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

        for variant_name, build_fn in VARIANTS.items():
            print(f"-- {variant_name} --")
            set_seed(seed)
            model = build_fn(v2_arch.n_components, v2_arch.mlp_hidden, v2_arch.device)
            train_result = train_v2_head(v2prepared, model, seed=seed, verbose=False)
            eval_result = evaluate_v2(train_result.model, v2prepared, config.threshold)
            test_m, val_m = eval_result.metrics_by_split.get("test", {}), eval_result.metrics_by_split.get("validation", {})
            infer_start = time.time()
            generate_predictions_v2(train_result.model, v2prepared)
            inference_s = time.time() - infer_start
            resource_summary = train_result.model.quantum_resource_summary()
            print(f"  test: pr_auc={test_m.get('pr_auc')} roc_auc={test_m.get('roc_auc')}  val: pr_auc={val_m.get('pr_auc')}")
            print(f"  resource summary: {resource_summary}")

            run_dir = new_run_dir(config.experiment.output_dir, f"{args.tag}_{variant_name}_seed{seed}")
            reducer.save(os.path.join(run_dir, "reduction"))
            write_json(os.path.join(run_dir, "encoder_checkpoint.json"), {"checkpoint_dir": checkpoint_dir})
            write_json(os.path.join(run_dir, "quantum_resource_summary.json"), resource_summary)
            _save_run(run_dir, config, train_result.model, variant_name, v2prepared, train_result, eval_result, seed, inference_s, {"n_components": v2_arch.n_components, "variant": variant_name, "quantum": resource_summary})
            print(f"  saved -> {run_dir}")

            if test_m.get("pr_auc") is not None:
                test_pr_by_variant[variant_name].append(test_m["pr_auc"])
            if val_m.get("pr_auc") is not None:
                val_pr_by_variant[variant_name].append(val_m["pr_auc"])

    if len(seeds) > 1:
        summary = {"seeds": seeds, "variants": {}}
        baseline_test = test_pr_by_variant["A_baseline"]
        baseline_val = val_pr_by_variant["A_baseline"]
        for variant_name in VARIANTS:
            test_vals, val_vals = test_pr_by_variant[variant_name], val_pr_by_variant[variant_name]
            entry = {
                "test_pr_auc": {"mean": float(np.mean(test_vals)), "std": float(np.std(test_vals)), "min": float(np.min(test_vals)), "max": float(np.max(test_vals))},
                "val_pr_auc": {"mean": float(np.mean(val_vals)), "std": float(np.std(val_vals))},
            }
            if variant_name != "A_baseline":
                test_diffs = [v - b for v, b in zip(test_vals, baseline_test)]
                val_diffs = [v - b for v, b in zip(val_vals, baseline_val)]
                entry["paired_test_pr_auc_delta_vs_A"] = {"per_seed": dict(zip(seeds, test_diffs)), "mean": float(np.mean(test_diffs)), "std": float(np.std(test_diffs)), "n_improving": sum(1 for d in test_diffs if d > 0), "n_degrading": sum(1 for d in test_diffs if d < 0)}
                entry["paired_val_pr_auc_delta_vs_A"] = {"per_seed": dict(zip(seeds, val_diffs)), "mean": float(np.mean(val_diffs)), "std": float(np.std(val_diffs)), "n_improving": sum(1 for d in val_diffs if d > 0), "n_degrading": sum(1 for d in val_diffs if d < 0)}
            summary["variants"][variant_name] = entry

        summary_path = os.path.join(config.experiment.output_dir, f"multiseed_summary_{args.tag}_seeds_{'-'.join(str(s) for s in seeds)}.json")
        write_json(summary_path, summary)
        print(f"\nMulti-seed summary -> {summary_path}")
        for variant_name, entry in summary["variants"].items():
            print(f"  {variant_name}: test={entry['test_pr_auc']} val={entry['val_pr_auc']}")
            if "paired_val_pr_auc_delta_vs_A" in entry:
                print(f"      paired val delta vs A: {entry['paired_val_pr_auc_delta_vs_A']}")


if __name__ == "__main__":
    main()
