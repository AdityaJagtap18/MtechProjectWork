#!/usr/bin/env python3
"""Quantum Encoding Investigation (QGNN_V4_QUANTUM_ENCODING_RESULTS.md,
Sections 9-10) -- loads each already-saved model.pt (no retraining),
rebuilds the exact architecture (including its own encoding_type/
encoding_scale/data_reuploading) from its own run directory, and runs one
forward pass over that run's real TEST split with hooks capturing:

    encoded angles (HybridQuantumHeadLayerNorm.encode(h), BEFORE the
    quantum circuit) -- Section 10's "encoded angle distribution"
    six PauliZ expectation channels (BEFORE LayerNorm) -- Section 9's
    "quantum representation analysis"

Mirrors `inspect_qgnn_v4_stage5_quantum_features.py`'s pattern, extended
to also capture the encoded angles (Stage 5 didn't need this since it
never varied the encoding).
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pointbiserialr

from _analysis_common import find_latest_graphsage_full_checkpoint
from scm_dataset.modeling.config import load_config
from scm_dataset.modeling.data import load_benchmark
from scm_dataset.modeling.graph_embedding_reduction import (
    extract_supplier_embeddings,
    load_frozen_graphsage_encoder,
    prepare_frozen_encoder_input,
)
from scm_dataset.modeling.quantum import HybridQuantumHeadLayerNorm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments", "qgnn_v4")
ENC_DIR = os.path.join(EXPERIMENTS_DIR, "encoding_investigation")
OUT_DIR = ENC_DIR
SEEDS = [42, 43]
PI = 3.141592653589793

# (config_label, encoding_type, encoding_scale, data_reuploading, base_dir, subdir_or_None, tag)
CONFIGS = [
    ("E0_baseline", "tanh", PI, False, EXPERIMENTS_DIR, None, "phase2c_layernorm_noaffine"),
    ("E1_reduced_scale", "tanh", 0.5 * PI, False, ENC_DIR, "e1_reduced_scale", "encpilot_e1"),
    ("E2_increased_scale", "tanh", 2 * PI, False, ENC_DIR, "e2_increased_scale", "encpilot_e2"),
    ("E3_linear_clip", "clip", PI, False, ENC_DIR, "e3_linear_clip", "encpilot_e3"),
    ("E4_data_reuploading", "tanh", PI, True, ENC_DIR, "e4_data_reuploading", "encpilot_e4"),
]


def find_run_dir(base_dir: str, subdir: str | None, tag: str, seed: int, split_strategy: str) -> str:
    search_dir = os.path.join(base_dir, subdir) if subdir else base_dir
    pattern = os.path.join(search_dir, f"*_{tag}_quantum_seed{seed}")
    matches = sorted(glob.glob(pattern))
    kept = []
    for d in matches:
        cfg_path = os.path.join(d, "config.yaml")
        if not os.path.exists(cfg_path):
            continue
        cfg = yaml.safe_load(open(cfg_path))
        if cfg.get("split", {}).get("strategy") == split_strategy:
            kept.append(d)
    if not kept:
        raise FileNotFoundError(f"no run directory found under {pattern!r} with split.strategy={split_strategy!r}")
    return kept[-1]


def angle_distribution_stats(angles_np: np.ndarray, scale: float) -> dict:
    flat = angles_np.flatten()
    near_pi_thresh = 0.9 * scale
    near_zero_thresh = 0.1 * scale
    return {
        "mean": float(np.mean(flat)), "std": float(np.std(flat)),
        "min": float(np.min(flat)), "max": float(np.max(flat)),
        "p10": float(np.percentile(flat, 10)), "p50": float(np.percentile(flat, 50)), "p90": float(np.percentile(flat, 90)),
        "frac_near_plusminus_scale": float((np.abs(flat) >= near_pi_thresh).mean()),
        "frac_near_zero": float((np.abs(flat) <= near_zero_thresh).mean()),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    angle_rows, feature_rows = [], []

    for config_label, enc_type, enc_scale, reupload, base_dir, subdir, tag in CONFIGS:
        for split_label, config_path, strategy in [
            ("primary", "configs/qgnn_v4.yaml", "temporal"),
            ("severity", "configs/qgnn_v4_severity.yaml", "severity"),
        ]:
            config = load_config(config_path)
            benchmark = load_benchmark(config.dataset.benchmark_path, config.dataset.dataset_id)
            for seed in SEEDS:
                run_dir = find_run_dir(base_dir, subdir, tag, seed, strategy)

                split_suffix = None if strategy == "temporal" else strategy
                checkpoint_dir = find_latest_graphsage_full_checkpoint(
                    os.path.join(REPO_ROOT, "experiments", "classical_gnn"), seed, split_suffix=split_suffix
                )
                prepared_full = prepare_frozen_encoder_input(config, benchmark, os.path.join(checkpoint_dir, "preprocessing"))
                encoder = load_frozen_graphsage_encoder(os.path.join(checkpoint_dir, "model.pt"), prepared_full)
                all_times = sorted(prepared_full.examples["time"].unique())
                embedding_frame = extract_supplier_embeddings(encoder, prepared_full, all_times)

                test_examples = prepared_full.examples[prepared_full.examples["split"] == "test"]
                joined = test_examples.set_index(["supplier_id", "time"]).join(embedding_frame, how="left").reset_index()
                emb_cols = list(embedding_frame.columns)
                h = torch.tensor(joined[emb_cols].values.astype(np.float32))
                target = joined["target"].values.astype(int)

                state_dict = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu")
                model = HybridQuantumHeadLayerNorm(
                    in_dim=h.shape[1], n_qubits=6, n_layers=2, ansatz="strongly_entangling",
                    diff_method="backprop", device_name="default.qubit", elementwise_affine=False,
                    encoding_type=enc_type, encoding_scale=enc_scale, data_reuploading=reupload,
                )
                model.load_state_dict(state_dict)
                model.eval()
                with torch.no_grad():
                    angles = model.encode(h)
                    q_out = model.quantum(angles).to(torch.float32)

                angles_np = angles.numpy()
                pauliz_np = q_out.numpy()

                # --- Section 10: encoded angle distribution ---
                stats = angle_distribution_stats(angles_np, enc_scale)
                angle_rows.append({"config": config_label, "split": split_label, "seed": seed, **stats})
                print(f"[angles] {config_label} {split_label} seed{seed}: mean={stats['mean']:.4f} std={stats['std']:.4f} frac_near_scale={stats['frac_near_plusminus_scale']:.3f} frac_near_zero={stats['frac_near_zero']:.3f}")

                # --- Section 9: quantum feature (PauliZ) analysis ---
                if pauliz_np.shape[0] > 1:
                    channel_corr = np.corrcoef(pauliz_np.T)
                    off_diag = channel_corr[np.triu_indices(6, k=1)]
                    max_abs_cross = float(np.nanmax(np.abs(off_diag))) if len(off_diag) else None
                    mean_abs_cross = float(np.nanmean(np.abs(off_diag))) if len(off_diag) else None
                else:
                    max_abs_cross = mean_abs_cross = None

                target_corrs = []
                for ch in range(6):
                    vals = pauliz_np[:, ch]
                    if len(np.unique(target)) > 1 and np.std(vals) > 0:
                        r, p = pointbiserialr(vals, target)
                    else:
                        r, p = None, None
                    target_corrs.append(r)
                    feature_rows.append({
                        "config": config_label, "split": split_label, "seed": seed, "channel": ch,
                        "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                        "min": float(np.min(vals)), "max": float(np.max(vals)),
                        "target_correlation": float(r) if r is not None else None,
                        "max_abs_cross_channel_corr": max_abs_cross, "mean_abs_cross_channel_corr": mean_abs_cross,
                    })
                abs_corrs = [abs(r) for r in target_corrs if r is not None]
                max_abs_t = max(abs_corrs) if abs_corrs else None
                mean_abs_t = float(np.mean(abs_corrs)) if abs_corrs else None
                for row in feature_rows[-6:]:
                    row["max_abs_target_correlation_this_run"] = max_abs_t
                    row["mean_abs_target_correlation_this_run"] = mean_abs_t
                print(f"[features] {config_label} {split_label} seed{seed}: max|corr(ch,target)|={max_abs_t}, max|cross-ch corr|={max_abs_cross}")

    angle_df = pd.DataFrame(angle_rows)
    angle_df.to_csv(os.path.join(OUT_DIR, "encoding_encoded_angles.csv"), index=False)
    print(f"\nencoding_encoded_angles.csv: {len(angle_df)} rows")

    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(os.path.join(OUT_DIR, "encoding_quantum_features.csv"), index=False)
    print(f"encoding_quantum_features.csv: {len(feature_df)} rows")


if __name__ == "__main__":
    main()
