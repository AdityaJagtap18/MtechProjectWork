#!/usr/bin/env python3
"""Table 3: calibration (ECE / Brier) across the same six ablated factors
and two references as the paper's existing Table 2 (which only reports
PR-AUC). Pure aggregation of already-produced per-phase summary CSVs --
no new runs. Mirrors Table 2's structure: single configs get mean+/-std,
multi-variant rows (Projection, Initialization pilot, Encoding) get a
min-max range across their pilot variants, exactly as Table 2 does for
PR-AUC.
"""
import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QGNN_V4 = os.path.join(REPO_ROOT, "experiments", "qgnn_v4")
CLASSICAL_GNN = os.path.join(REPO_ROOT, "experiments", "classical_gnn")
OUT_DIR = os.path.join(REPO_ROOT, "final_tables")


def classical_reference_ece_brier() -> tuple[float, float, float, float]:
    """Classical GNN (full), primary/test, mean of 5 seeds -- verified
    against Table 1 (test_pr_auc matches classical_gnn_all_metrics.csv,
    test_ece mean = 0.0493 matches Table 1's reported 0.0493)."""
    seeds = [42, 43, 44, 45, 46]
    df = pd.read_csv(os.path.join(CLASSICAL_GNN, "baseline_extended_metrics", "classical_gnn_all_metrics.csv"))
    df = df[(df["config"] == "classical_gnn_full") & (df["split"] == "primary") & (df["seed"].isin(seeds))]
    return df["test_ece"].mean(), df["test_ece"].std(), df["test_brier_score"].mean(), df["test_brier_score"].std()


def row_from_summary(csv_path: str, config: str, split: str = "primary") -> dict:
    df = pd.read_csv(csv_path)
    r = df[(df["config"] == config) & (df["split"] == split)].iloc[0]
    return {
        "ece_mean": r["ece_mean"], "ece_std": r["ece_std"],
        "brier_mean": r["brier_score_mean"], "brier_std": r["brier_score_std"],
        "pr_auc_mean": r["pr_auc_mean"], "n_seeds": r["n_seeds"],
    }


def range_from_summaries(rows: list[dict]) -> dict:
    eces = [r["ece_mean"] for r in rows]
    briers = [r["brier_mean"] for r in rows]
    return {"ece_range": (min(eces), max(eces)), "brier_range": (min(briers), max(briers))}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    # --- References ---
    c_ece_mean, c_ece_std, c_brier_mean, c_brier_std = classical_reference_ece_brier()
    rows.append({"factor": "Reference", "configuration": "Classical GraphSAGE-Full",
                 "ece": f"{c_ece_mean:.3f}±{c_ece_std:.3f}",
                 "brier": f"{c_brier_mean:.3f}±{c_brier_std:.3f}"})

    standing = row_from_summary(os.path.join(QGNN_V4, "phase4_stage5", "stage5_summary.csv"),
                                 "A0_strongly_entangling", "primary")
    rows.append({"factor": "Reference", "configuration": "QGNN standing reference",
                 "ece": f"{standing['ece_mean']:.3f}±{standing['ece_std']:.3f}",
                 "brier": f"{standing['brier_mean']:.3f}±{standing['brier_std']:.3f}"})

    # --- Qubit count (Phase 3; reuse computed values, verified vs Table 2 PR-AUC) ---
    rc_path = os.path.join(REPO_ROOT, "final_tables", "QGNN_V4_RESOURCE_VS_PERFORMANCE_TABLE.csv")
    rc = pd.read_csv(rc_path)
    for label, factor in [("Qubit count: 4 qubits", "Qubit count"), ("Qubit count: 8 qubits", "Qubit count"),
                           ("Circuit depth: 1 layer", "Circuit depth"), ("Circuit depth: 3 layers", "Circuit depth")]:
        r = rc[rc["configuration"] == label].iloc[0]
        rows.append({"factor": factor, "configuration": label,
                     "ece": f"{r['ece_mean']:.3f}±{r['ece_std']:.3f}",
                     "brier": f"{r['brier_mean']:.3f}±{r['brier_std']:.3f}"})

    # --- Ansatz (Stage 5) ---
    for config, label in [("A1_hardware_ring", "Hardware-efficient ring"), ("A2_reduced_entanglement", "Reduced entanglement")]:
        r = row_from_summary(os.path.join(QGNN_V4, "phase4_stage5", "stage5_summary.csv"), config, "primary")
        rows.append({"factor": "Ansatz", "configuration": label,
                     "ece": f"{r['ece_mean']:.3f}±{r['ece_std']:.3f}",
                     "brier": f"{r['brier_mean']:.3f}±{r['brier_std']:.3f}"})

    # --- Projection (Stage 2 pilot, 2 seeds): nonlinear / pre-norm / PCA(4,6,8) ---
    proj_rows = [row_from_summary(os.path.join(QGNN_V4, "phase4_stage2", "stage2_summary.csv"), c, "primary")
                 for c in ["B2_nonlinear_projection", "B3_pre_projection_norm", "B4_pca4", "B4_pca6", "B4_pca8"]]
    pr = range_from_summaries(proj_rows)
    rows.append({"factor": "Projection", "configuration": "Nonlinear / pre-norm / PCA (2-seed pilot)",
                 "ece": f"{pr['ece_range'][0]:.3f}–{pr['ece_range'][1]:.3f}",
                 "brier": f"{pr['brier_range'][0]:.3f}–{pr['brier_range'][1]:.3f}"})

    # --- Initialization: small-Gaussian / identity-like (2-pilot) ---
    init_pilot_rows = [row_from_summary(os.path.join(QGNN_V4, "phase4_stage4", "stage4_summary.csv"), c, "primary")
                        for c in ["F2_small_gaussian", "F3_identity_like"]]
    ir = range_from_summaries(init_pilot_rows)
    rows.append({"factor": "Initialization", "configuration": "Small-Gaussian / identity-like (2-seed pilot)",
                 "ece": f"{ir['ece_range'][0]:.3f}–{ir['ece_range'][1]:.3f}",
                 "brier": f"{ir['brier_range'][0]:.3f}–{ir['brier_range'][1]:.3f}"})

    # --- Initialization: identity-like, 5-seed expansion ---
    r5 = row_from_summary(os.path.join(QGNN_V4, "phase4_stage4b", "stage4b_summary.csv"), "F3_identity_like_5seed", "primary")
    rows.append({"factor": "Initialization", "configuration": "Identity-like, 5-seed expansion",
                 "ece": f"{r5['ece_mean']:.3f}±{r5['ece_std']:.3f}",
                 "brier": f"{r5['brier_mean']:.3f}±{r5['brier_std']:.3f}"})

    # --- Encoding: scale / clip / re-uploading (2-pilot) ---
    enc = pd.read_csv(os.path.join(QGNN_V4, "encoding_investigation", "encoding_summary.csv"))
    enc_variant_rows = []
    for c in ["E1_reduced_scale", "E2_increased_scale", "E3_linear_clip", "E4_data_reuploading"]:
        r = enc[(enc["config"] == c) & (enc["split"] == "primary")].iloc[0]
        enc_variant_rows.append({"ece_mean": r["ece_mean"], "brier_mean": r["brier_score_mean"]})
    er = range_from_summaries(enc_variant_rows)
    rows.append({"factor": "Encoding", "configuration": "Scale / clip / re-uploading (2-seed pilot)",
                 "ece": f"{er['ece_range'][0]:.3f}–{er['ece_range'][1]:.3f}",
                 "brier": f"{er['brier_range'][0]:.3f}–{er['brier_range'][1]:.3f}"})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    out_path = os.path.join(OUT_DIR, "QGNN_V4_CALIBRATION_ABLATION_TABLE.csv")
    df.to_csv(out_path, index=False)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
