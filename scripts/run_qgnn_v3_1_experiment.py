#!/usr/bin/env python3
"""QGNN-v3.1: trainable input scaling (D) confirmation benchmark against
the existing depth-3 baseline (A), expanded to 10 seeds. Purely additive
-- does not modify or call any of the run_qgnn_v2*/run_qgnn_v3*.py
scripts, and reuses `qgnn.build_qgnn_model` (A) and
`qgnn_v3.build_qgnn_trainable_scaling_model` (D) completely unmodified.

For each seed, A and D are trained on the SAME `V2PreparedData` object
(same frozen encoder, same extracted embeddings, same fitted PCA reducer,
same reduced 8D rows) -- built once per seed and passed to both, not
rebuilt per model, so "identical input" is true by construction, not by
argument (verified additionally by test).

Usage:
    # single-seed reproduction check (variant A only) before the full run
    python scripts/run_qgnn_v3_1_experiment.py --config configs/qgnn_v2.yaml --tag repro_check --seeds 42 --variants A

    # full 10-seed confirmation
    python scripts/run_qgnn_v3_1_experiment.py --config configs/qgnn_v2.yaml --tag confirm --seeds 42,43,44,45,46,47,48,49,50,51
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
from scm_dataset.modeling.qgnn_v3 import build_qgnn_trainable_scaling_model
from scm_dataset.modeling.risk_ranking_metrics import risk_ranking_summary
from scm_dataset.modeling.train import class_balance_summary

from _analysis_common import find_latest_graphsage_full_checkpoint

DEFAULT_OUTPUT_DIR = "experiments/qgnn_v3_1"

BUILDERS = {
    "A": lambda n_qubits, mlp_hidden, device: build_qgnn_model(n_qubits=n_qubits, n_layers=3, mlp_hidden=mlp_hidden, device_name=device),
    "D": lambda n_qubits, mlp_hidden, device: build_qgnn_trainable_scaling_model(n_qubits, 3, mlp_hidden, device),
}


def _save_run(run_dir: str, config, model, model_kind: str, v2prepared, train_result, eval_result, seed: int, inference_seconds: float, ranking: dict, extra_metadata: dict) -> None:
    save_config(run_dir, config)
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    train_result.history.to_csv(os.path.join(run_dir, "training_history.csv"), index=False)
    eval_result.predictions.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)
    eval_result.risk_ranking.to_csv(os.path.join(run_dir, "supplier_risk_ranking.csv"), index=False)
    write_json(os.path.join(run_dir, "metrics.json"), {"threshold": eval_result.threshold, "threshold_policy": eval_result.threshold_policy, "by_split": eval_result.metrics_by_split})
    write_json(os.path.join(run_dir, "calibration.json"), eval_result.calibration_by_split)
    write_json(os.path.join(run_dir, "onset_breakdown.json"), eval_result.onset_breakdown)
    write_json(os.path.join(run_dir, "risk_ranking_at_k.json"), ranking)
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
    parser.add_argument("--tag", default="confirm")
    parser.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--variants", default="A,D", help="Comma-separated subset of {A,D} to run -- use 'A' alone for the pre-flight reproduction check.")
    parser.add_argument("--n-components", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override config's training.epochs (useful for a smoke test).")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Kept separate from experiments/qgnn_v2/ and experiments/qgnn_v3/ -- never overwrites prior artifacts.")
    parser.add_argument("--encoder-experiments-dir", default="experiments/classical_gnn")
    args = parser.parse_args()

    cfg = load_qgnn_v2_config(args.config)
    config = cfg.base
    config.experiment.output_dir = args.output_dir
    v2_arch = cfg.qgnn_v2
    v2_arch.n_layers = 3
    if args.n_components:
        v2_arch.n_components = args.n_components
    if args.device:
        v2_arch.device = args.device
    if args.epochs:
        config.training.epochs = args.epochs
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = args.variants.split(",")

    print(f"QGNN-v3.1 confirmation: dataset={config.dataset.dataset_id} n_components={v2_arch.n_components} device={v2_arch.device} variants={variants} seeds={seeds}")

    val_pr = {v: [] for v in variants}
    test_pr = {v: [] for v in variants}

    for seed in seeds:
        print(f"\n=== model seed {seed} (encoder + heads) ===")
        checkpoint_dir = find_latest_graphsage_full_checkpoint(args.encoder_experiments_dir, seed)
        benchmark = load_benchmark(config.dataset.benchmark_path, config.dataset.dataset_id)
        prepared_full = prepare_frozen_encoder_input(config, benchmark, os.path.join(checkpoint_dir, "preprocessing"))
        encoder = load_frozen_graphsage_encoder(os.path.join(checkpoint_dir, "model.pt"), prepared_full)
        assert all(not p.requires_grad for p in encoder.parameters())

        all_times = sorted(prepared_full.examples["time"].unique())
        embedding_frame = extract_supplier_embeddings(encoder, prepared_full, all_times)
        # Built ONCE per seed, passed to both A and D below -- identical
        # frozen embeddings and identical fitted PCA reducer by
        # construction, not merely by intent.
        v2prepared, reducer = build_v2_prepared(config, benchmark, prepared_full.examples, embedding_frame, v2_arch.n_components)

        for variant in variants:
            print(f"-- variant {variant} --")
            set_seed(seed)
            model = BUILDERS[variant](v2_arch.n_components, v2_arch.mlp_hidden, v2_arch.device)
            train_result = train_v2_head(v2prepared, model, seed=seed, verbose=False)
            eval_result = evaluate_v2(train_result.model, v2prepared, config.threshold)
            test_m, val_m = eval_result.metrics_by_split.get("test", {}), eval_result.metrics_by_split.get("validation", {})
            infer_start = time.time()
            generate_predictions_v2(train_result.model, v2prepared)
            inference_s = time.time() - infer_start
            ranking = risk_ranking_summary(eval_result.predictions, split="test", k_percents=(5.0, 10.0))
            resource_summary = train_result.model.quantum_resource_summary()
            print(f"  test: pr_auc={test_m.get('pr_auc')} roc_auc={test_m.get('roc_auc')}  val: pr_auc={val_m.get('pr_auc')}")
            print(f"  risk ranking (test): {ranking}")
            print(f"  resource summary: {resource_summary}")

            run_dir = new_run_dir(config.experiment.output_dir, f"{args.tag}_{variant}_seed{seed}")
            reducer.save(os.path.join(run_dir, "reduction"))
            write_json(os.path.join(run_dir, "encoder_checkpoint.json"), {"checkpoint_dir": checkpoint_dir})
            write_json(os.path.join(run_dir, "quantum_resource_summary.json"), resource_summary)
            _save_run(run_dir, config, train_result.model, variant, v2prepared, train_result, eval_result, seed, inference_s, ranking, {"n_components": v2_arch.n_components, "variant": variant, "quantum": resource_summary})
            print(f"  saved -> {run_dir}")

            if test_m.get("pr_auc") is not None:
                test_pr[variant].append(test_m["pr_auc"])
            if val_m.get("pr_auc") is not None:
                val_pr[variant].append(val_m["pr_auc"])

    if len(seeds) > 1 and len(variants) > 1:
        summary = {"seeds": seeds, "variants": {}}
        for variant in variants:
            summary["variants"][variant] = {
                "val_pr_auc": {"mean": float(np.mean(val_pr[variant])), "std": float(np.std(val_pr[variant])), "median": float(np.median(val_pr[variant])), "min": float(np.min(val_pr[variant])), "max": float(np.max(val_pr[variant]))},
                "test_pr_auc": {"mean": float(np.mean(test_pr[variant])), "std": float(np.std(test_pr[variant])), "median": float(np.median(test_pr[variant])), "min": float(np.min(test_pr[variant])), "max": float(np.max(test_pr[variant]))},
            }
        if "A" in variants and "D" in variants:
            val_diffs = [d - a for d, a in zip(val_pr["D"], val_pr["A"])]
            test_diffs = [d - a for d, a in zip(test_pr["D"], test_pr["A"])]
            summary["paired_val_pr_auc_delta_D_minus_A"] = {
                "per_seed": dict(zip(seeds, val_diffs)), "mean": float(np.mean(val_diffs)), "median": float(np.median(val_diffs)),
                "std": float(np.std(val_diffs)), "min": float(np.min(val_diffs)), "max": float(np.max(val_diffs)),
                "n_improving": sum(1 for d in val_diffs if d > 0), "n_degrading": sum(1 for d in val_diffs if d < 0),
                "pct_improving": 100.0 * sum(1 for d in val_diffs if d > 0) / len(val_diffs),
            }
            summary["paired_test_pr_auc_delta_D_minus_A"] = {
                "per_seed": dict(zip(seeds, test_diffs)), "mean": float(np.mean(test_diffs)), "median": float(np.median(test_diffs)),
                "std": float(np.std(test_diffs)), "min": float(np.min(test_diffs)), "max": float(np.max(test_diffs)),
                "n_improving": sum(1 for d in test_diffs if d > 0), "n_degrading": sum(1 for d in test_diffs if d < 0),
                "pct_improving": 100.0 * sum(1 for d in test_diffs if d > 0) / len(test_diffs),
            }
        summary_path = os.path.join(config.experiment.output_dir, f"multiseed_summary_{args.tag}_seeds_{'-'.join(str(s) for s in seeds)}.json")
        write_json(summary_path, summary)
        print(f"\nMulti-seed summary -> {summary_path}")
        for variant, entry in summary["variants"].items():
            print(f"  {variant}: val={entry['val_pr_auc']} test={entry['test_pr_auc']}")
        if "paired_val_pr_auc_delta_D_minus_A" in summary:
            print(f"  paired VAL delta (D-A): {summary['paired_val_pr_auc_delta_D_minus_A']}")
            print(f"  paired TEST delta (D-A): {summary['paired_test_pr_auc_delta_D_minus_A']}")


if __name__ == "__main__":
    main()
