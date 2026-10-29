#!/usr/bin/env python3
"""Pooled reliability diagram (classical vs. quantum, primary/test split)
for the paper's standing-reference comparison (Table 1: ECE 0.049 vs 0.182).

Pools per-example predictions.csv test rows across the 5 standing-reference
seeds for each head (no retraining, no new inference -- reads files already
produced by prior runs), then bins into deciles exactly as the project's
own ECE definition does (Eq. 10 in the paper).
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QGNN_V4 = os.path.join(REPO_ROOT, "experiments", "qgnn_v4")
CLASSICAL_GNN = os.path.join(REPO_ROOT, "experiments", "classical_gnn")
OUT_DIR = os.path.join(REPO_ROOT, "final_figures")

SEEDS = [42, 43, 44, 45, 46]
# "Classical GNN (full)" baseline exactly as used in Table 1 / Table 3 of
# the paper (QGNN_V4_CLASSICAL_BASELINE.md, QGNN_V4_PHASE2C_REPORT.md) --
# verified against classical_gnn_all_metrics.csv test_pr_auc per seed.
CLASSICAL_BASE = CLASSICAL_GNN
CLASSICAL_DIRS = {
    42: "20260905T041456Z_hetero_graphsage_seed42",
    43: "20260905T041808Z_hetero_graphsage_seed43",
    44: "20260905T041857Z_hetero_graphsage_seed44",
    45: "20260905T041943Z_hetero_graphsage_seed45",
    46: "20260905T042034Z_hetero_graphsage_seed46",
}
# QGNN-v4 standing reference (6q/2L StronglyEntangling, LayerNorm-no-affine),
# primary/temporal split -- verified test_pr_auc=0.868 for seed42 matches
# the 163451-series (not the 163614-series, which is the severity split).
QUANTUM_BASE = QGNN_V4
QUANTUM_DIRS = {
    42: "20260914T163451Z_phase2c_layernorm_noaffine_quantum_seed42",
    43: "20260914T163503Z_phase2c_layernorm_noaffine_quantum_seed43",
    44: "20260914T163519Z_phase2c_layernorm_noaffine_quantum_seed44",
    45: "20260914T163541Z_phase2c_layernorm_noaffine_quantum_seed45",
    46: "20260914T163601Z_phase2c_layernorm_noaffine_quantum_seed46",
}

N_BINS = 10


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(probs, bin_edges[1:-1], right=False)
    mean_pred, obs_freq, counts = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        n = mask.sum()
        counts.append(int(n))
        if n == 0:
            mean_pred.append(np.nan)
            obs_freq.append(np.nan)
        else:
            mean_pred.append(float(probs[mask].mean()))
            obs_freq.append(float(labels[mask].mean()))
    return np.array(mean_pred), np.array(obs_freq), np.array(counts)


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> float:
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


def per_seed_curves(base_dir: str, dirs: dict[int, str]):
    """Per-seed ECE + reliability curve, averaged bin-wise across seeds --
    matches the paper's own Table 3 methodology (mean of 5 per-seed scalar
    ECE values), rather than pooling raw predictions across seeds first."""
    eces, means, obss = [], [], []
    for seed, d in dirs.items():
        df = pd.read_csv(os.path.join(base_dir, d, "predictions.csv"))
        df = df[df["split"] == "test"]
        probs = df["risk_probability"].values.astype(float)
        labels = df["actual_disruption"].values.astype(int)
        eces.append(ece(probs, labels))
        m, o, _ = reliability_curve(probs, labels)
        means.append(m)
        obss.append(o)
    mean_ece = float(np.mean(eces))
    std_ece = float(np.std(eces))
    avg_mean = np.nanmean(np.vstack(means), axis=0)
    avg_obs = np.nanmean(np.vstack(obss), axis=0)
    return mean_ece, std_ece, avg_mean, avg_obs, eces


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    c_ece, c_ece_std, c_mean, c_obs, c_per_seed = per_seed_curves(CLASSICAL_BASE, CLASSICAL_DIRS)
    q_ece, q_ece_std, q_mean, q_obs, q_per_seed = per_seed_curves(QUANTUM_BASE, QUANTUM_DIRS)

    print(f"Mean-of-5-seed test ECE -- classical: {c_ece:.4f}+/-{c_ece_std:.4f} (per-seed: {c_per_seed})")
    print(f"Mean-of-5-seed test ECE -- quantum:   {q_ece:.4f}+/-{q_ece_std:.4f} (per-seed: {q_per_seed})")
    print("(Table 1 of the paper reports classical=0.0493, quantum=0.1823)")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    ax.plot(c_mean, c_obs, marker="o", color="#2b7a3d", label=f"Classical GNN (ECE={c_ece:.3f})")
    ax.plot(q_mean, q_obs, marker="o", color="#b23a3a", label=f"QGNN (ECE={q_ece:.3f})")
    ax.set_xlabel("Mean predicted probability", fontweight="bold")
    ax.set_ylabel("Observed frequency", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    legend = ax.legend(loc="upper left")
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "fig7_calibration_reliability_diagram.png")
    fig.savefig(out_path, dpi=200)
    print(f"saved {out_path}")

    summary = pd.DataFrame({
        "bin_mean_predicted_classical": c_mean,
        "bin_observed_freq_classical": c_obs,
        "bin_mean_predicted_quantum": q_mean,
        "bin_observed_freq_quantum": q_obs,
    })
    csv_path = os.path.join(OUT_DIR, "fig7_calibration_reliability_diagram_data.csv")
    summary.to_csv(csv_path, index=False)
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
