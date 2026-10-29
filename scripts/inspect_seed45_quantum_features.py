#!/usr/bin/env python3
"""Phase 2d seed-45 diagnostic: loads the already-saved model.pt for
baseline/LN-no-affine/LN-affine (severity split, seed 45, and comparison
seeds), rebuilds the exact architecture from each run's own config, and
runs one forward pass over the real severity test set with hooks
capturing every intermediate stage:

    frozen embedding -> Linear(128,6) -> angles (pi*tanh)
    -> quantum circuit -> pauliz (pre-norm)
    -> [LayerNorm] -> normed (post-norm, LN runs only)
    -> Linear(6,1) -> logit

No retraining -- inference only, using the exact saved weights. Writes
one CSV per run with per-example values for every stage, for direct
comparison across seeds and configurations (QGNN_V4_PHASE2D_REPORT.md).
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import yaml

from _analysis_common import find_latest_graphsage_full_checkpoint
from scm_dataset.modeling.config import load_config
from scm_dataset.modeling.data import load_benchmark
from scm_dataset.modeling.graph_embedding_reduction import (
    extract_supplier_embeddings,
    load_frozen_graphsage_encoder,
    prepare_frozen_encoder_input,
)
from scm_dataset.modeling.quantum import HybridQuantumHead, HybridQuantumHeadLayerNorm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "experiments", "qgnn_v4", "phase2d_seed45_inspection")

RUNS = {
    **{("baseline", s): "diag_severity" for s in (42, 43, 44, 45, 46)},
    **{("ln_noaffine", s): "phase2c_layernorm_noaffine" for s in (42, 43, 44, 45, 46)},
    **{("ln_affine", s): "phase2c_layernorm_affine" for s in (42, 43, 44, 45, 46)},
}


def find_run_dir(tag: str, seed: int) -> str:
    for d in sorted(glob.glob(os.path.join(REPO_ROOT, "experiments", "qgnn_v4", f"*_{tag}_quantum_seed{seed}"))):
        cfg = yaml.safe_load(open(os.path.join(d, "config.yaml")))
        if cfg["split"]["strategy"] == "severity":
            return d
    raise FileNotFoundError(f"no severity run for tag={tag!r} seed={seed}")


def build_model(variant: str, in_dim: int, state_dict: dict):
    common = dict(n_qubits=6, n_layers=2, ansatz="strongly_entangling", diff_method="backprop", device_name="default.qubit")
    if variant == "baseline":
        model = HybridQuantumHead(in_dim, **common)
    elif variant == "ln_noaffine":
        model = HybridQuantumHeadLayerNorm(in_dim, **common, elementwise_affine=False)
    elif variant == "ln_affine":
        model = HybridQuantumHeadLayerNorm(in_dim, **common, elementwise_affine=True)
    else:
        raise ValueError(variant)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def forward_with_hooks(model, h: torch.Tensor, is_layernorm: bool) -> dict:
    with torch.no_grad():
        angles = torch.pi * torch.tanh(model.reduce(h))
        q_out = model.quantum(angles).to(torch.float32)
        if is_layernorm:
            normed = model.norm(q_out)
            logits = model.out(normed).squeeze(-1)
        else:
            normed = None
            logits = model.out(q_out).squeeze(-1)
        probs = torch.sigmoid(logits)
    return {"angles": angles, "pauliz": q_out, "normed": normed, "logits": logits, "probs": probs}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    config = load_config("configs/qgnn_v4_severity.yaml")
    benchmark = load_benchmark(config.dataset.benchmark_path, config.dataset.dataset_id)

    for (variant, seed), tag in RUNS.items():
        run_dir = find_run_dir(tag, seed)
        checkpoint_dir = find_latest_graphsage_full_checkpoint(
            os.path.join(REPO_ROOT, "experiments", "classical_gnn"), seed, split_suffix="severity"
        )
        prepared_full = prepare_frozen_encoder_input(config, benchmark, os.path.join(checkpoint_dir, "preprocessing"))
        encoder = load_frozen_graphsage_encoder(os.path.join(checkpoint_dir, "model.pt"), prepared_full)
        all_times = sorted(prepared_full.examples["time"].unique())
        embedding_frame = extract_supplier_embeddings(encoder, prepared_full, all_times)

        test_examples = prepared_full.examples[prepared_full.examples["split"] == "test"]
        joined = test_examples.set_index(["supplier_id", "time"]).join(embedding_frame, how="left").reset_index()
        emb_cols = [c for c in embedding_frame.columns]
        h = torch.tensor(joined[emb_cols].values.astype(np.float32))

        state_dict = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu")
        model = build_model(variant, in_dim=h.shape[1], state_dict=state_dict)
        out = forward_with_hooks(model, h, is_layernorm=(variant != "baseline"))

        pauliz_np = out["pauliz"].numpy()
        rows = {
            "supplier_id": joined["supplier_id"], "time": joined["time"], "actual_disruption": joined["target"],
            "logit": out["logits"].numpy(), "probability": out["probs"].numpy(),
        }
        for i in range(pauliz_np.shape[1]):
            rows[f"pauliz_{i}"] = pauliz_np[:, i]
        if out["normed"] is not None:
            normed_np = out["normed"].numpy()
            for i in range(normed_np.shape[1]):
                rows[f"normed_{i}"] = normed_np[:, i]
        angles_np = out["angles"].numpy()
        for i in range(angles_np.shape[1]):
            rows[f"angle_{i}"] = angles_np[:, i]

        out_df = pd.DataFrame(rows)
        out_path = os.path.join(OUT_DIR, f"{variant}_seed{seed}_severity_test_features.csv")
        out_df.to_csv(out_path, index=False)
        print(f"{variant} seed{seed}: saved {len(out_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
