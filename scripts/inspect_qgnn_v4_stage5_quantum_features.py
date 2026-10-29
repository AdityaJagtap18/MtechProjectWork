#!/usr/bin/env python3
"""QGNN-v4 Phase 4 Stage 5 (QGNN_V4_PHASE4_PLAN.md, Section 18/19: quantum
feature / representation-collapse analysis) -- loads each already-saved
Stage 5 model.pt (no retraining), rebuilds the exact architecture from its
own run directory, and runs one forward pass over that run's real TEST
split with a hook capturing the six PauliZ expectation values BEFORE
LayerNorm. Mirrors `inspect_seed45_quantum_features.py`'s (Phase 2d)
pattern exactly, generalized across three ansätze and both splits instead
of one variant/one split.

Per-channel mean/std/min/max and point-biserial correlation with the
target are aggregated directly to `stage5_quantum_features.csv` (30
ansatz x split x seed combinations x 6 channels = 180 rows) rather than
dumping full per-example CSVs, since Section 18 asks for the channel
statistics, not the raw per-example values.
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
STAGE5_DIR = os.path.join(EXPERIMENTS_DIR, "phase4_stage5")
OUT_DIR = STAGE5_DIR
SEEDS = [42, 43, 44, 45, 46]

# (config_label, ansatz, base_dir, subdir_or_None, tag)
ANSATZ_SOURCES = [
    ("A0_strongly_entangling", "strongly_entangling", EXPERIMENTS_DIR, None, "phase2c_layernorm_noaffine"),
    ("A1_hardware_ring", "hardware_efficient_ring", STAGE5_DIR, "hardware_ring", "stage5_hardware_ring"),
    ("A2_reduced_entanglement", "reduced_entanglement", STAGE5_DIR, "reduced_entanglement", "stage5_reduced_entanglement"),
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


def forward_with_hooks(model: HybridQuantumHeadLayerNorm, h: torch.Tensor) -> dict:
    with torch.no_grad():
        angles = torch.pi * torch.tanh(model.reduce(h))
        q_out = model.quantum(angles).to(torch.float32)
        normed = model.norm(q_out)
        logits = model.out(normed).squeeze(-1)
        probs = torch.sigmoid(logits)
    return {"pauliz": q_out, "normed": normed, "logits": logits, "probs": probs}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    for config_label, ansatz, base_dir, subdir, tag in ANSATZ_SOURCES:
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
                    in_dim=h.shape[1], n_qubits=6, n_layers=2, ansatz=ansatz,
                    diff_method="backprop", device_name="default.qubit", elementwise_affine=False,
                )
                model.load_state_dict(state_dict)
                model.eval()
                out = forward_with_hooks(model, h)
                pauliz_np = out["pauliz"].numpy()

                # cross-channel correlation matrix (for the "highly correlated
                # channels" representation-collapse check, Section 19)
                if pauliz_np.shape[0] > 1:
                    channel_corr = np.corrcoef(pauliz_np.T)
                    off_diag = channel_corr[np.triu_indices(6, k=1)]
                    max_abs_channel_corr = float(np.nanmax(np.abs(off_diag))) if len(off_diag) else None
                else:
                    max_abs_channel_corr = None

                target_corrs = []
                for ch in range(6):
                    vals = pauliz_np[:, ch]
                    if len(np.unique(target)) > 1 and np.std(vals) > 0:
                        r, p = pointbiserialr(vals, target)
                    else:
                        r, p = None, None
                    target_corrs.append(r)
                    rows.append({
                        "config": config_label, "ansatz": ansatz, "split": split_label, "seed": seed, "channel": ch,
                        "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                        "min": float(np.min(vals)), "max": float(np.max(vals)),
                        "target_correlation": float(r) if r is not None else None,
                        "target_correlation_p": float(p) if p is not None else None,
                        "max_abs_cross_channel_corr": max_abs_channel_corr,
                    })
                abs_corrs = [abs(r) for r in target_corrs if r is not None]
                max_abs = max(abs_corrs) if abs_corrs else None
                for row in rows[-6:]:
                    row["max_abs_target_correlation_this_run"] = max_abs
                print(f"{config_label} {split_label} seed{seed}: max|corr(channel,target)|={max_abs}, max|cross-channel corr|={max_abs_channel_corr}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "stage5_quantum_features.csv"), index=False)
    print(f"\nstage5_quantum_features.csv: {len(df)} rows -> {os.path.join(OUT_DIR, 'stage5_quantum_features.csv')}")


if __name__ == "__main__":
    main()
