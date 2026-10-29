#!/usr/bin/env python3
"""Table 4: quantum resource cost vs. performance for the paper's five
main architecture variants (standing reference + the 4 Phase 3 qubit-
count/circuit-depth ablations), primary/temporal split.

Pure aggregation of already-produced files: quantum_resource_summary.json
for parameter counts, calibration.json (mean of 5 seeds) for ECE, and
metrics.json (mean of 5 seeds) for PR-AUC -- matches the exact
methodology already used for Table 1 / Table 3 of the paper. No new
training or inference.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QGNN_V4 = os.path.join(REPO_ROOT, "experiments", "qgnn_v4")

SEEDS = [42, 43, 44, 45, 46]

# Standing reference (6q/2L) primary-task dirs, verified earlier against
# Table 1 (test PR-AUC seed42=0.868).
STANDING_REF_DIRS = {
    42: "20260914T163451Z_phase2c_layernorm_noaffine_quantum_seed42",
    43: "20260914T163503Z_phase2c_layernorm_noaffine_quantum_seed43",
    44: "20260914T163519Z_phase2c_layernorm_noaffine_quantum_seed44",
    45: "20260914T163541Z_phase2c_layernorm_noaffine_quantum_seed45",
    46: "20260914T163601Z_phase2c_layernorm_noaffine_quantum_seed46",
}

CONFIGS = [
    ("Standing reference", 6, 2, STANDING_REF_DIRS),
    ("Qubit count: 4 qubits", 4, 2, None),  # resolved below via phase3_q4
    ("Qubit count: 8 qubits", 8, 2, None),
    ("Circuit depth: 1 layer", 6, 1, None),
    ("Circuit depth: 3 layers", 6, 3, None),
]
TAGS = {
    "Qubit count: 4 qubits": "phase3_q4",
    "Qubit count: 8 qubits": "phase3_q8",
    "Circuit depth: 1 layer": "phase3_d1",
    "Circuit depth: 3 layers": "phase3_d3",
}


def resolve_primary_dirs(tag: str) -> dict[int, str]:
    dirs = sorted(glob.glob(os.path.join(QGNN_V4, f"*_{tag}_quantum_seed*")))
    out = {}
    for d in dirs:
        cfg = yaml.safe_load(open(os.path.join(d, "config.yaml")))
        if cfg["split"]["strategy"] == "temporal":
            seed = int(os.path.basename(d).split("seed")[-1])
            out[seed] = os.path.basename(d)  # last match (sorted) wins
    return out


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(probs, bin_edges[1:-1], right=False)
    n = len(probs)
    total = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        total += (mask.sum() / n) * abs(acc - conf)
    return float(total)


def aggregate(dirs: dict[int, str]) -> dict:
    pr_aucs, eces, briers = [], [], []
    qubits = layers = trainable_q = total_params = None
    for seed, d in dirs.items():
        run_dir = os.path.join(QGNN_V4, d)
        m = json.load(open(os.path.join(run_dir, "metrics.json")))
        pr_aucs.append(m["by_split"]["test"]["pr_auc"])
        briers.append(m["by_split"]["test"]["brier_score"])

        preds = pd.read_csv(os.path.join(run_dir, "predictions.csv"))
        preds = preds[preds["split"] == "test"]
        eces.append(ece(preds["risk_probability"].values.astype(float),
                         preds["actual_disruption"].values.astype(int)))

        if qubits is None:
            res = json.load(open(os.path.join(run_dir, "quantum_resource_summary.json")))
            qubits = res["qubits"]
            layers = res["variational_layers"]
            trainable_q = res["trainable_quantum_parameters"]
            total_params = res["total_trainable_parameters"]

    return {
        "qubits": qubits, "layers": layers,
        "trainable_quantum_params": trainable_q, "total_params": total_params,
        "pr_auc_mean": float(np.mean(pr_aucs)), "pr_auc_std": float(np.std(pr_aucs)),
        "ece_mean": float(np.mean(eces)), "ece_std": float(np.std(eces)),
        "brier_mean": float(np.mean(briers)), "brier_std": float(np.std(briers)),
    }


def main() -> None:
    rows = []
    for label, qubits, layers, fixed_dirs in CONFIGS:
        dirs = fixed_dirs if fixed_dirs is not None else resolve_primary_dirs(TAGS[label])
        assert len(dirs) == 5, f"{label}: expected 5 seeds, got {len(dirs)}"
        agg = aggregate(dirs)
        rows.append({"configuration": label, **agg})
        print(f"{label}: qubits={agg['qubits']} layers={agg['layers']} "
              f"params={agg['total_params']} PR-AUC={agg['pr_auc_mean']:.3f}+/-{agg['pr_auc_std']:.3f} "
              f"ECE={agg['ece_mean']:.3f}+/-{agg['ece_std']:.3f}")

    df = pd.DataFrame(rows)
    out_dir = os.path.join(REPO_ROOT, "final_tables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "QGNN_V4_RESOURCE_VS_PERFORMANCE_TABLE.csv")
    df.to_csv(out_path, index=False)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
