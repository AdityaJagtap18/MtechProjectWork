#!/usr/bin/env python3
"""PauliZ channel-correlation figure for the QGNN-v4 standing reference
(6q/2L StronglyEntangling, LayerNorm-no-affine, primary split), supporting
the paper's own claim ("cross-channel correlation 0.68-0.93") with an
actual figure instead of a number stated only in prose.

Built entirely from experiments/qgnn_v4/phase4_stage5/stage5_quantum_features.csv,
which already holds per-channel target-correlation and max-cross-channel-
correlation for this exact config across all 5 seeds (produced by
scripts/inspect_qgnn_v4_stage5_quantum_features.py's prior run; no new
inference -- a fresh forward pass was attempted but blocked by a dataset
schema drift unrelated to this figure, see run notes).
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE5_CSV = os.path.join(REPO_ROOT, "experiments", "qgnn_v4", "phase4_stage5", "stage5_quantum_features.csv")
OUT_DIR = os.path.join(REPO_ROOT, "final_figures")

SEEDS = [42, 43, 44, 45, 46]
N_QUBITS = 6


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(STAGE5_CSV)
    sub = df[(df["config"] == "A0_strongly_entangling") & (df["split"] == "primary")].copy()
    assert len(sub) == len(SEEDS) * N_QUBITS, f"expected {len(SEEDS) * N_QUBITS} rows, got {len(sub)}"

    target_corr = sub.pivot(index="channel", columns="seed", values="target_correlation")
    target_corr = target_corr[SEEDS]
    max_cross = sub.drop_duplicates("seed").set_index("seed")["max_abs_cross_channel_corr"].loc[SEEDS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.3, 1]})

    seed_labels = [f"seed {i + 1}" for i in range(len(SEEDS))]

    im = ax1.imshow(target_corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax1.set_xticks(range(len(SEEDS)))
    ax1.set_xticklabels(seed_labels)
    ax1.set_yticks(range(N_QUBITS))
    ax1.set_yticklabels([f"Q{i}" for i in range(N_QUBITS)])
    for i in range(N_QUBITS):
        for j in range(len(SEEDS)):
            v = target_corr.values[i, j]
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center",
                      color="white" if abs(v) > 0.6 else "black", fontsize=8)
    ax1.set_title("(a) Per-channel correlation with target")
    fig.colorbar(im, ax=ax1, label="point-biserial r", fraction=0.046, pad=0.04)

    bars = ax2.bar(seed_labels, max_cross.values, color="#6a3fb0")
    ax2.axhline(max_cross.mean(), linestyle="--", color="gray",
                label=f"mean = {max_cross.mean():.2f}")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("seed", fontweight="bold")
    ax2.set_ylabel("max |cross-channel correlation|", fontweight="bold")
    ax2.set_title("(b) Max pairwise PauliZ\nchannel correlation")
    ax2.legend()
    for b, v in zip(bars, max_cross.values):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "fig8_pauliz_channel_correlation.png")
    fig.savefig(out_path, dpi=200)
    print(f"saved {out_path}")
    print(f"max|cross-channel corr| per seed: {dict(max_cross)}")
    print(f"mean: {max_cross.mean():.3f}, range: {max_cross.min():.3f}-{max_cross.max():.3f}")

    out_csv = os.path.join(OUT_DIR, "fig8_pauliz_channel_correlation_data.csv")
    sub[["seed", "channel", "target_correlation", "max_abs_cross_channel_corr"]].to_csv(out_csv, index=False)
    print(f"saved {out_csv}")


if __name__ == "__main__":
    main()
