#!/usr/bin/env python3
"""QGNN-v4 Final Research Consolidation: builds the three master CSVs from
values verified directly against each phase's own already-published report
(cited per row via the `source` column) -- this is a consolidation of
already-reported, already-verified results, not a new analysis or rerun.
Every number here was cross-checked against the cited .md report (and,
where that report itself cites a CSV/JSON artifact, against that artifact)
before being hard-coded -- not retyped from memory alone.

Where a metric was not recorded for a configuration (e.g. F1/MCC for a
2-seed pilot that only reported PR-AUC/ROC-AUC/MCC headline numbers),
the cell is left as None -> written as empty in the CSV, never invented.
"""

from __future__ import annotations

import csv
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = REPO_ROOT

MASTER_COLUMNS = [
    "phase", "experiment", "configuration", "split", "n_seeds", "seed_type",
    "primary_pr_auc", "primary_pr_auc_std", "severity_pr_auc", "severity_pr_auc_std",
    "pr_auc", "pr_auc_std", "roc_auc", "f1", "precision", "recall", "specificity",
    "balanced_accuracy", "mcc", "brier", "ece", "decision", "source",
]

# Each row: one (configuration, split) pair. `pr_auc`/`roc_auc`/etc. are for
# THIS row's split; `primary_pr_auc`/`severity_pr_auc` columns are filled
# only on a convenience "both splits" summary row where useful, else left
# blank to avoid duplicating the per-split rows below them.
ROWS = [
    # --- Standing references ---
    dict(phase="Reference", experiment="Classical GraphSAGE-Full", configuration="full graph, hidden_dim=128",
         split="primary", n_seeds=5, seed_type="full",
         pr_auc=0.8070, pr_auc_std=0.0655, roc_auc=0.9871, f1=0.6771, precision=0.5220, recall=0.9678,
         specificity=0.9353, balanced_accuracy=0.9515, mcc=0.6845, brier=0.0418, ece=0.0493,
         decision="Reference", source="QGNN_V4_CLASSICAL_BASELINE.md"),
    dict(phase="Reference", experiment="Classical GraphSAGE-Full", configuration="full graph, hidden_dim=128",
         split="severity", n_seeds=5, seed_type="full",
         pr_auc=0.4487, pr_auc_std=0.0131, roc_auc=0.7275, f1=0.4113, precision=0.3871, recall=0.4406,
         specificity=0.9438, balanced_accuracy=0.6922, mcc=0.3624, brier=0.0857, ece=0.0904,
         decision="Reference", source="QGNN_V4_CLASSICAL_BASELINE.md"),
    dict(phase="Reference", experiment="QGNN-v4 standing reference", configuration="6q/2L StronglyEntangling, LayerNorm-no-affine, default init, pi*tanh",
         split="primary", n_seeds=5, seed_type="full",
         pr_auc=0.8112, pr_auc_std=0.0737, roc_auc=0.9775, f1=0.6565, precision=0.5191, recall=0.8975,
         specificity=0.9400, balanced_accuracy=0.9188, mcc=0.6539, brier=0.0655, ece=0.1823,
         decision="Reference", source="QGNN_V4_PHASE2C_REPORT.md"),
    dict(phase="Reference", experiment="QGNN-v4 standing reference", configuration="6q/2L StronglyEntangling, LayerNorm-no-affine, default init, pi*tanh",
         split="severity", n_seeds=5, seed_type="full",
         pr_auc=0.3976, pr_auc_std=0.0571, roc_auc=0.6566, f1=0.4187, precision=0.4608, recall=0.4166,
         specificity=0.9513, balanced_accuracy=0.6840, mcc=0.3824, brier=0.1225, ece=0.2405,
         decision="Reference", source="QGNN_V4_PHASE2C_REPORT.md"),

    # --- Phase 1/Phase 2b family: QGNN-v4 raw baseline (no LayerNorm) ---
    dict(phase="Phase 1-2b", experiment="QGNN-v4 raw baseline", configuration="HybridQuantumHead, alpha=1, no output normalization",
         split="primary", n_seeds=5, seed_type="full",
         pr_auc=0.8328, pr_auc_std=0.1397, roc_auc=0.9829, f1=0.5693, precision=0.4373, recall=0.9603,
         specificity=0.7512, balanced_accuracy=0.8557, mcc=0.6839, brier=0.1283, ece=0.2845,
         decision="Superseded by LayerNorm", source="QGNN_V4_PHASE2C_REPORT.md Table 1/3"),
    dict(phase="Phase 1-2b", experiment="QGNN-v4 raw baseline", configuration="HybridQuantumHead, alpha=1, no output normalization",
         split="severity", n_seeds=5, seed_type="full",
         pr_auc=0.3915, pr_auc_std=0.0237, roc_auc=0.6257, f1=0.2436, precision=0.4354, recall=0.5667,
         specificity=0.5991, balanced_accuracy=0.5829, mcc=0.2305, brier=0.2105, ece=0.3448,
         decision="Superseded by LayerNorm", source="QGNN_V4_PHASE2C_REPORT.md Table 2/3"),
    dict(phase="Phase 2b", experiment="Output scale: trainable alpha", configuration="scale (alpha init=1, trainable, no bias)",
         split="primary", n_seeds=5, seed_type="full",
         pr_auc=0.8417, pr_auc_std=0.1160, roc_auc=None, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=None, brier=0.118, ece=0.253,
         decision="Not supported as major contributor", source="QGNN_V4_PHASE2B_REPORT.md Table (4)"),
    dict(phase="Phase 2b", experiment="Output scale: trainable alpha", configuration="scale (alpha init=1, trainable, no bias)",
         split="severity", n_seeds=5, seed_type="full",
         pr_auc=0.4040, pr_auc_std=0.0447, roc_auc=None, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=None, brier=0.212, ece=0.351,
         decision="Not supported as major contributor", source="QGNN_V4_PHASE2B_REPORT.md Table (4)"),
    dict(phase="Phase 2b", experiment="Output scale: trainable alpha+beta", configuration="scale_bias",
         split="primary", n_seeds=5, seed_type="full",
         pr_auc=0.8412, pr_auc_std=0.1229, roc_auc=None, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=None, brier=0.107, ece=0.243,
         decision="Not supported as major contributor", source="QGNN_V4_PHASE2B_REPORT.md Table (4)"),
    dict(phase="Phase 2b", experiment="Output scale: trainable alpha+beta", configuration="scale_bias",
         split="severity", n_seeds=5, seed_type="full",
         pr_auc=0.3936, pr_auc_std=0.0255, roc_auc=None, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=None, brier=0.211, ece=0.348,
         decision="Not supported as major contributor", source="QGNN_V4_PHASE2B_REPORT.md Table (4)"),
    dict(phase="Phase 2b", experiment="Output scale: fixed 0.5", configuration="fixed_scale, alpha=0.5",
         split="primary", n_seeds=5, seed_type="full",
         pr_auc=0.8197, pr_auc_std=0.1425, roc_auc=None, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=None, brier=0.166, ece=0.352,
         decision="Not supported as major contributor", source="QGNN_V4_PHASE2B_REPORT.md Table (4)"),
    dict(phase="Phase 2b", experiment="Output scale: fixed 0.5", configuration="fixed_scale, alpha=0.5",
         split="severity", n_seeds=5, seed_type="full",
         pr_auc=0.3882, pr_auc_std=0.0516, roc_auc=None, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=None, brier=0.218, ece=0.367,
         decision="Not supported as major contributor", source="QGNN_V4_PHASE2B_REPORT.md Table (4)"),
    dict(phase="Phase 2b", experiment="Output scale: fixed 2.0", configuration="fixed_scale, alpha=2.0",
         split="primary", n_seeds=5, seed_type="full",
         pr_auc=0.8420, pr_auc_std=0.1224, roc_auc=None, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=None, brier=0.099, ece=0.224,
         decision="Not supported as major contributor", source="QGNN_V4_PHASE2B_REPORT.md Table (4)"),
    dict(phase="Phase 2b", experiment="Output scale: fixed 2.0", configuration="fixed_scale, alpha=2.0",
         split="severity", n_seeds=5, seed_type="full",
         pr_auc=0.4026, pr_auc_std=0.0375, roc_auc=None, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=None, brier=0.197, ece=0.332,
         decision="Not supported as major contributor", source="QGNN_V4_PHASE2B_REPORT.md Table (4)"),

    dict(phase="Phase 2c", experiment="LayerNorm affine", configuration="6q/2L, LayerNorm(elementwise_affine=True)",
         split="primary", n_seeds=5, seed_type="full",
         pr_auc=0.8002, pr_auc_std=0.0730, roc_auc=0.9765, f1=0.6476, precision=0.5191, recall=0.8612,
         specificity=0.9426, balanced_accuracy=0.9019, mcc=0.6393, brier=0.0568, ece=0.1456,
         decision="Mixed evidence", source="QGNN_V4_PHASE2C_REPORT.md Table 1/3"),
    dict(phase="Phase 2c", experiment="LayerNorm affine", configuration="6q/2L, LayerNorm(elementwise_affine=True)",
         split="severity", n_seeds=5, seed_type="full",
         pr_auc=0.4077, pr_auc_std=0.0710, roc_auc=0.6505, f1=0.4151, precision=0.4401, recall=0.4164,
         specificity=0.9505, balanced_accuracy=0.6835, mcc=0.3747, brier=0.1184, ece=0.2304,
         decision="Mixed evidence", source="QGNN_V4_PHASE2C_REPORT.md Table 2/3"),

    # --- Phase 3: qubit count / depth ablation (LayerNorm-no-affine output stage) ---
    dict(phase="Phase 3", experiment="Qubit count: 4 qubits", configuration="4q/2L StronglyEntangling, LayerNorm-no-affine",
         split="primary", n_seeds=5, seed_type="full",
         pr_auc=0.847, pr_auc_std=0.102, roc_auc=None, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=None, brier=None, ece=None,
         decision="Mixed evidence", source="QGNN_V4_PHASE4_PLAN.md section 0 (Phase 3 carryover)"),
    dict(phase="Phase 3", experiment="Qubit count: 4 qubits", configuration="4q/2L StronglyEntangling, LayerNorm-no-affine",
         split="severity", n_seeds=5, seed_type="full",
         pr_auc=0.396, pr_auc_std=0.030, roc_auc=None, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=None, brier=None, ece=None,
         decision="Mixed evidence", source="QGNN_V4_PHASE4_PLAN.md section 0 (Phase 3 carryover)"),
    dict(phase="Phase 3", experiment="Qubit count: 8 qubits", configuration="8q/2L StronglyEntangling, LayerNorm-no-affine",
         split="primary", n_seeds=5, seed_type="full",
         pr_auc=0.810, pr_auc_std=0.091, roc_auc=None, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=None, brier=None, ece=None,
         decision="Not supported as major contributor", source="QGNN_V4_PHASE4_PLAN.md section 0 (Phase 3 carryover)"),
    dict(phase="Phase 3", experiment="Qubit count: 8 qubits", configuration="8q/2L StronglyEntangling, LayerNorm-no-affine",
         split="severity", n_seeds=5, seed_type="full",
         pr_auc=0.364, pr_auc_std=0.054, roc_auc=None, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=None, brier=None, ece=None,
         decision="Not supported as major contributor", source="QGNN_V4_PHASE4_PLAN.md section 0 (Phase 3 carryover)"),
    dict(phase="Phase 3", experiment="Circuit depth: 1 layer", configuration="6q/1L StronglyEntangling, LayerNorm-no-affine",
         split="primary", n_seeds=5, seed_type="full",
         pr_auc=0.693, pr_auc_std=0.198, roc_auc=None, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=None, brier=None, ece=None,
         decision="Not supported as major contributor", source="QGNN_V4_PHASE4_PLAN.md section 0 (Phase 3 carryover)"),
    dict(phase="Phase 3", experiment="Circuit depth: 1 layer", configuration="6q/1L StronglyEntangling, LayerNorm-no-affine",
         split="severity", n_seeds=5, seed_type="full",
         pr_auc=0.386, pr_auc_std=0.017, roc_auc=None, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=None, brier=None, ece=None,
         decision="Mixed evidence (tightest severity std observed, but weak primary)", source="QGNN_V4_PHASE4_PLAN.md section 0 (Phase 3 carryover)"),
    dict(phase="Phase 3", experiment="Circuit depth: 3 layers", configuration="6q/3L StronglyEntangling, LayerNorm-no-affine",
         split="primary", n_seeds=5, seed_type="full",
         pr_auc=0.804, pr_auc_std=0.186, roc_auc=None, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=None, brier=None, ece=None,
         decision="Not supported as major contributor", source="QGNN_V4_PHASE4_PLAN.md section 0 (Phase 3 carryover)"),
    dict(phase="Phase 3", experiment="Circuit depth: 3 layers", configuration="6q/3L StronglyEntangling, LayerNorm-no-affine",
         split="severity", n_seeds=0, seed_type="killed, no data",
         pr_auc=None, pr_auc_std=None, roc_auc=None, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=None, brier=None, ece=None,
         decision="Inconclusive (run killed, zero usable data)", source="Phase 3 batch, killed by user request"),

    # --- Phase 4 Stage 1: representation diagnostics (frozen embedding, not a QGNN head) ---
    dict(phase="Phase 4 Stage 1", experiment="Representation diagnostic: Logistic Regression", configuration="raw 128D frozen embedding, sklearn LogisticRegression -- DIAGNOSTIC, not an end-to-end model",
         split="primary", n_seeds=5, seed_type="full",
         pr_auc=0.6706, pr_auc_std=0.0426, roc_auc=0.9662, f1=0.5739, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=0.5532, brier=None, ece=None,
         decision="Diagnostic (representation contains signal; not a competing end-to-end model)", source="QGNN_V4_PHASE4_RESULTS.md (Stage 1) A1, re-verified against experiments/qgnn_v4/phase4_stage1/representation_diagnostics.csv"),
    dict(phase="Phase 4 Stage 1", experiment="Representation diagnostic: Logistic Regression", configuration="raw 128D frozen embedding, sklearn LogisticRegression -- DIAGNOSTIC, not an end-to-end model",
         split="severity", n_seeds=5, seed_type="full",
         pr_auc=0.4646, pr_auc_std=0.0051, roc_auc=0.7311, f1=0.4710, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=0.4336, brier=None, ece=None,
         decision="Diagnostic (numerically exceeds both end-to-end references on severity PR-AUC; not a competing end-to-end model, see QGNN_V4_FINAL_RESEARCH_SUMMARY.md section 9's correction)", source="QGNN_V4_PHASE4_RESULTS.md (Stage 1) A1, re-verified against experiments/qgnn_v4/phase4_stage1/representation_diagnostics.csv"),
    dict(phase="Phase 4 Stage 1", experiment="Representation diagnostic: MLP", configuration="raw 128D frozen embedding, sklearn MLPClassifier(64)",
         split="primary", n_seeds=5, seed_type="full",
         pr_auc=0.733, pr_auc_std=0.063, roc_auc=0.977, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=None, brier=None, ece=None,
         decision="Supported contributor to understanding", source="QGNN_V4_PHASE4_RESULTS.md (Stage 1) A1"),
    dict(phase="Phase 4 Stage 1", experiment="Representation diagnostic: MLP", configuration="raw 128D frozen embedding, sklearn MLPClassifier(64)",
         split="severity", n_seeds=5, seed_type="full",
         pr_auc=0.463, pr_auc_std=0.007, roc_auc=0.730, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=None, brier=None, ece=None,
         decision="Supported contributor to understanding", source="QGNN_V4_PHASE4_RESULTS.md (Stage 1) A1"),
    dict(phase="Phase 4 Stage 1", experiment="PCA-4 + Logistic Regression", configuration="PCA(4, train-fit) of frozen embedding + LogisticRegression",
         split="primary", n_seeds=5, seed_type="full",
         pr_auc=0.781, pr_auc_std=0.085, roc_auc=None, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=None, brier=None, ece=None,
         decision="Supported contributor (primary signal is low-rank)", source="QGNN_V4_PHASE4_RESULTS.md (Stage 1) A2"),
    dict(phase="Phase 4 Stage 1", experiment="PCA-128 + Logistic Regression", configuration="PCA(128, train-fit) of frozen embedding + LogisticRegression",
         split="severity", n_seeds=5, seed_type="full",
         pr_auc=0.480, pr_auc_std=0.011, roc_auc=None, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=None, brier=None, ece=None,
         decision="Supported contributor (severity signal is distributed, not low-rank)", source="QGNN_V4_PHASE4_RESULTS.md (Stage 1) A2"),

    # --- Phase 4 Stage 2: projection/bottleneck (2-seed pilot) ---
    dict(phase="Phase 4 Stage 2", experiment="Projection: nonlinear (B2)", configuration="Linear(128,32)->GELU->Linear(32,6)",
         split="primary", n_seeds=2, seed_type="pilot",
         pr_auc=0.650, pr_auc_std=0.017, roc_auc=0.963, f1=0.583, precision=0.454, recall=0.816,
         specificity=0.930, balanced_accuracy=0.873, mcc=0.573, brier=0.092, ece=0.225,
         decision="Not supported as major contributor", source="QGNN_V4_PHASE4_STAGE2_RESULTS.md"),
    dict(phase="Phase 4 Stage 2", experiment="Projection: nonlinear (B2)", configuration="Linear(128,32)->GELU->Linear(32,6)",
         split="severity", n_seeds=2, seed_type="pilot",
         pr_auc=0.389, pr_auc_std=0.004, roc_auc=0.633, f1=0.405, precision=0.419, recall=0.397,
         specificity=0.954, balanced_accuracy=0.675, mcc=0.360, brier=0.103, ece=0.162,
         decision="Not supported as major contributor", source="QGNN_V4_PHASE4_STAGE2_RESULTS.md"),
    dict(phase="Phase 4 Stage 2", experiment="Projection: pre-projection LayerNorm (B3)", configuration="LayerNorm(128)->Linear(128,6)",
         split="primary", n_seeds=2, seed_type="pilot",
         pr_auc=0.729, pr_auc_std=0.032, roc_auc=0.980, f1=0.641, precision=0.488, recall=0.940,
         specificity=0.929, balanced_accuracy=0.934, mcc=0.647, brier=0.083, ece=0.205,
         decision="Not supported as major contributor", source="QGNN_V4_PHASE4_STAGE2_RESULTS.md"),
    dict(phase="Phase 4 Stage 2", experiment="Projection: pre-projection LayerNorm (B3)", configuration="LayerNorm(128)->Linear(128,6)",
         split="severity", n_seeds=2, seed_type="pilot",
         pr_auc=0.263, pr_auc_std=0.045, roc_auc=0.702, f1=0.425, precision=0.413, recall=0.441,
         specificity=0.949, balanced_accuracy=0.695, mcc=0.378, brier=0.099, ece=0.183,
         decision="Not supported as major contributor (severity collapse)", source="QGNN_V4_PHASE4_STAGE2_RESULTS.md"),
    dict(phase="Phase 4 Stage 2", experiment="Projection: PCA-4 (B4)", configuration="train-fit PCA(4)->Linear(4,6)",
         split="primary", n_seeds=2, seed_type="pilot",
         pr_auc=0.391, pr_auc_std=0.094, roc_auc=0.905, f1=0.350, precision=0.237, recall=0.802,
         specificity=0.776, balanced_accuracy=0.789, mcc=0.352, brier=0.161, ece=0.290,
         decision="Investigate (contradicts Stage 1's diagnostic finding, cause not isolated)", source="QGNN_V4_PHASE4_STAGE2_RESULTS.md"),
    dict(phase="Phase 4 Stage 2", experiment="Projection: PCA-4 (B4)", configuration="train-fit PCA(4)->Linear(4,6)",
         split="severity", n_seeds=2, seed_type="pilot",
         pr_auc=0.289, pr_auc_std=0.015, roc_auc=0.698, f1=0.298, precision=0.218, recall=0.470,
         specificity=0.865, balanced_accuracy=0.667, mcc=0.240, brier=0.143, ece=0.248,
         decision="Investigate", source="QGNN_V4_PHASE4_STAGE2_RESULTS.md"),

    # --- Phase 4 Stage 4: initialization (2-seed pilot) ---
    dict(phase="Phase 4 Stage 4", experiment="Init: small Gaussian (F2, std=0.01)", configuration="quantum_init=small_gaussian",
         split="primary", n_seeds=2, seed_type="pilot",
         pr_auc=0.828, pr_auc_std=0.083, roc_auc=0.980, f1=0.772, precision=0.701, recall=0.899,
         specificity=0.964, balanced_accuracy=0.931, mcc=0.767, brier=0.049, ece=0.145,
         decision="Investigate (primary gain, severity cost)", source="QGNN_V4_PHASE4_STAGE4_RESULTS.md"),
    dict(phase="Phase 4 Stage 4", experiment="Init: small Gaussian (F2, std=0.01)", configuration="quantum_init=small_gaussian",
         split="severity", n_seeds=2, seed_type="pilot",
         pr_auc=0.265, pr_auc_std=0.057, roc_auc=0.653, f1=0.400, precision=0.367, recall=0.440,
         specificity=0.939, balanced_accuracy=0.690, mcc=0.349, brier=0.084, ece=0.144,
         decision="Not supported as major contributor (severity collapse)", source="QGNN_V4_PHASE4_STAGE4_RESULTS.md"),
    dict(phase="Phase 4 Stage 4", experiment="Init: identity-like (F3), 2-seed pilot", configuration="quantum_init=identity_like",
         split="primary", n_seeds=2, seed_type="pilot",
         pr_auc=0.726, pr_auc_std=0.017, roc_auc=0.975, f1=0.688, precision=0.540, recall=0.948,
         specificity=0.942, balanced_accuracy=0.945, mcc=0.691, brier=0.080, ece=0.179,
         decision="Superseded by 5-seed result below", source="QGNN_V4_PHASE4_STAGE4_RESULTS.md"),
    dict(phase="Phase 4 Stage 4", experiment="Init: identity-like (F3), 2-seed pilot", configuration="quantum_init=identity_like",
         split="severity", n_seeds=2, seed_type="pilot",
         pr_auc=0.380, pr_auc_std=0.102, roc_auc=0.647, f1=0.403, precision=0.374, recall=0.441,
         specificity=0.940, balanced_accuracy=0.690, mcc=0.353, brier=0.126, ece=0.206,
         decision="Superseded by 5-seed result below", source="QGNN_V4_PHASE4_STAGE4_RESULTS.md"),

    # --- Phase 4 Stage 4b: initialization scale sweep (2-seed pilot) + F3 5-seed expansion ---
    dict(phase="Phase 4 Stage 4b", experiment="Init scale: Gaussian std=0.001", configuration="quantum_init=gaussian, gaussian_std=0.001",
         split="primary", n_seeds=2, seed_type="pilot",
         pr_auc=0.827, pr_auc_std=0.085, roc_auc=0.981, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=0.767, brier=None, ece=None,
         decision="Not supported as major contributor (redundant with std=0.01)", source="QGNN_V4_PHASE4_STAGE4B_RESULTS.md"),
    dict(phase="Phase 4 Stage 4b", experiment="Init scale: Gaussian std=0.001", configuration="quantum_init=gaussian, gaussian_std=0.001",
         split="severity", n_seeds=2, seed_type="pilot",
         pr_auc=0.263, pr_auc_std=0.063, roc_auc=0.645, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=0.347, brier=None, ece=None,
         decision="Not supported as major contributor", source="QGNN_V4_PHASE4_STAGE4B_RESULTS.md"),
    dict(phase="Phase 4 Stage 4b", experiment="Init scale: Gaussian std=0.025", configuration="quantum_init=gaussian, gaussian_std=0.025",
         split="primary", n_seeds=2, seed_type="pilot",
         pr_auc=0.812, pr_auc_std=0.099, roc_auc=0.980, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=0.764, brier=None, ece=None,
         decision="Not supported as major contributor", source="QGNN_V4_PHASE4_STAGE4B_RESULTS.md"),
    dict(phase="Phase 4 Stage 4b", experiment="Init scale: Gaussian std=0.025", configuration="quantum_init=gaussian, gaussian_std=0.025",
         split="severity", n_seeds=2, seed_type="pilot",
         pr_auc=0.277, pr_auc_std=0.037, roc_auc=0.677, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=0.353, brier=None, ece=None,
         decision="Not supported as major contributor", source="QGNN_V4_PHASE4_STAGE4B_RESULTS.md"),
    dict(phase="Phase 4 Stage 4b", experiment="Init scale: Gaussian std=0.050", configuration="quantum_init=gaussian, gaussian_std=0.050",
         split="primary", n_seeds=2, seed_type="pilot",
         pr_auc=0.813, pr_auc_std=0.097, roc_auc=0.979, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=0.764, brier=None, ece=None,
         decision="Investigate (possible severity recovery, single-seed-driven)", source="QGNN_V4_PHASE4_STAGE4B_RESULTS.md"),
    dict(phase="Phase 4 Stage 4b", experiment="Init scale: Gaussian std=0.050", configuration="quantum_init=gaussian, gaussian_std=0.050",
         split="severity", n_seeds=2, seed_type="pilot",
         pr_auc=0.386, pr_auc_std=0.074, roc_auc=0.612, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=0.365, brier=None, ece=None,
         decision="Investigate (single-seed-driven, not confirmed)", source="QGNN_V4_PHASE4_STAGE4B_RESULTS.md"),
    dict(phase="Phase 4 Stage 4b", experiment="Init: identity-like (F3), 5-seed expansion", configuration="quantum_init=identity_like",
         split="primary", n_seeds=5, seed_type="full",
         pr_auc=0.777, pr_auc_std=0.059, roc_auc=0.982, f1=0.687, precision=0.535, recall=0.960,
         specificity=0.939, balanced_accuracy=0.950, mcc=0.692, brier=0.070, ece=0.169,
         decision="Not supported as major contributor (variance reduction did not survive 5 seeds)", source="QGNN_V4_PHASE4_STAGE4B_RESULTS.md"),
    dict(phase="Phase 4 Stage 4b", experiment="Init: identity-like (F3), 5-seed expansion", configuration="quantum_init=identity_like",
         split="severity", n_seeds=5, seed_type="full",
         pr_auc=0.386, pr_auc_std=0.112, roc_auc=0.654, f1=0.367, precision=0.328, recall=0.454,
         specificity=0.902, balanced_accuracy=0.678, mcc=0.314, brier=0.129, ece=0.234,
         decision="Not supported as major contributor (severity variance WORSE than reference)", source="QGNN_V4_PHASE4_STAGE4B_RESULTS.md"),

    # --- Phase 4 Stage 5: ansatz comparison (5-seed) ---
    dict(phase="Phase 4 Stage 5", experiment="Ansatz: hardware-efficient ring", configuration="6q/2L, RY+RZ + CNOT ring, 24 quantum params",
         split="primary", n_seeds=5, seed_type="full",
         pr_auc=0.8065, pr_auc_std=0.1474, roc_auc=0.9826, f1=0.6743, precision=0.5177, recall=0.9744,
         specificity=0.9331, balanced_accuracy=0.9537, mcc=0.6833, brier=0.0650, ece=0.1799,
         decision="Investigate (comparable mean, ~2x primary variance of reference)", source="QGNN_V4_PHASE4_STAGE5_RESULTS.md"),
    dict(phase="Phase 4 Stage 5", experiment="Ansatz: hardware-efficient ring", configuration="6q/2L, RY+RZ + CNOT ring, 24 quantum params",
         split="severity", n_seeds=5, seed_type="full",
         pr_auc=0.3540, pr_auc_std=0.0550, roc_auc=0.6731, f1=0.4236, precision=0.4096, recall=0.4406,
         specificity=0.9488, balanced_accuracy=0.6947, mcc=0.3766, brier=0.0918, ece=0.1585,
         decision="Investigate (better severity calibration than reference)", source="QGNN_V4_PHASE4_STAGE5_RESULTS.md"),
    dict(phase="Phase 4 Stage 5", experiment="Ansatz: reduced entanglement", configuration="6q/2L, RY only + CNOT chain, 12 quantum params",
         split="primary", n_seeds=5, seed_type="full",
         pr_auc=0.8138, pr_auc_std=0.1047, roc_auc=0.9840, f1=0.6345, precision=0.4923, recall=0.9025,
         specificity=0.9339, balanced_accuracy=0.9182, mcc=0.6359, brier=0.0642, ece=0.1802,
         decision="Investigate (comparable mean, most parameter-efficient)", source="QGNN_V4_PHASE4_STAGE5_RESULTS.md"),
    dict(phase="Phase 4 Stage 5", experiment="Ansatz: reduced entanglement", configuration="6q/2L, RY only + CNOT chain, 12 quantum params",
         split="severity", n_seeds=5, seed_type="full",
         pr_auc=0.3596, pr_auc_std=0.0574, roc_auc=0.6713, f1=0.3136, precision=0.2625, recall=0.4837,
         specificity=0.8279, balanced_accuracy=0.6558, mcc=0.2541, brier=0.1457, ece=0.2450,
         decision="Investigate (reproducible specificity collapse on 2/5 seeds)", source="QGNN_V4_PHASE4_STAGE5_RESULTS.md"),

    # --- Quantum Encoding Investigation (2-seed pilot) ---
    dict(phase="Encoding Investigation", experiment="Encoding: reduced scale (E1, 0.5*pi)", configuration="encoding_scale=0.5*pi, tanh",
         split="primary", n_seeds=2, seed_type="pilot",
         pr_auc=0.8034, pr_auc_std=0.1433, roc_auc=0.980, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=0.6456, brier=None, ece=0.185,
         decision="Investigate (nominal gain, high variance, seed-inconsistent)", source="QGNN_V4_QUANTUM_ENCODING_RESULTS.md"),
    dict(phase="Encoding Investigation", experiment="Encoding: reduced scale (E1, 0.5*pi)", configuration="encoding_scale=0.5*pi, tanh",
         split="severity", n_seeds=2, seed_type="pilot",
         pr_auc=0.4315, pr_auc_std=0.0309, roc_auc=0.620, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=0.3644, brier=None, ece=None,
         decision="Investigate", source="QGNN_V4_QUANTUM_ENCODING_RESULTS.md"),
    dict(phase="Encoding Investigation", experiment="Encoding: increased scale (E2, 2*pi)", configuration="encoding_scale=2*pi, tanh",
         split="primary", n_seeds=2, seed_type="pilot",
         pr_auc=0.5812, pr_auc_std=0.2649, roc_auc=0.937, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=0.4795, brier=None, ece=0.229,
         decision="Not supported as major contributor (clear regression)", source="QGNN_V4_QUANTUM_ENCODING_RESULTS.md"),
    dict(phase="Encoding Investigation", experiment="Encoding: increased scale (E2, 2*pi)", configuration="encoding_scale=2*pi, tanh",
         split="severity", n_seeds=2, seed_type="pilot",
         pr_auc=0.3703, pr_auc_std=0.0257, roc_auc=0.678, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=0.3320, brier=None, ece=None,
         decision="Not supported as major contributor", source="QGNN_V4_QUANTUM_ENCODING_RESULTS.md"),
    dict(phase="Encoding Investigation", experiment="Encoding: linear clip (E3)", configuration="encoding_type=clip, scale=pi",
         split="primary", n_seeds=2, seed_type="pilot",
         pr_auc=0.6660, pr_auc_std=0.0785, roc_auc=0.965, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=0.5540, brier=0.090, ece=0.198,
         decision="Not supported as major contributor (measured information loss at saturation)", source="QGNN_V4_QUANTUM_ENCODING_RESULTS.md"),
    dict(phase="Encoding Investigation", experiment="Encoding: linear clip (E3)", configuration="encoding_type=clip, scale=pi",
         split="severity", n_seeds=2, seed_type="pilot",
         pr_auc=0.3154, pr_auc_std=0.0264, roc_auc=0.638, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=0.2924, brier=None, ece=None,
         decision="Not supported as major contributor", source="QGNN_V4_QUANTUM_ENCODING_RESULTS.md"),
    dict(phase="Encoding Investigation", experiment="Encoding: data re-uploading (E4)", configuration="data_reuploading=True, encode before each of 2 layers",
         split="primary", n_seeds=2, seed_type="pilot",
         pr_auc=0.5994, pr_auc_std=0.2024, roc_auc=0.956, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=0.5355, brier=None, ece=0.232,
         decision="Not supported as major contributor", source="QGNN_V4_QUANTUM_ENCODING_RESULTS.md"),
    dict(phase="Encoding Investigation", experiment="Encoding: data re-uploading (E4)", configuration="data_reuploading=True, encode before each of 2 layers",
         split="severity", n_seeds=2, seed_type="pilot",
         pr_auc=0.3787, pr_auc_std=0.0303, roc_auc=0.686, f1=None, precision=None, recall=None,
         specificity=None, balanced_accuracy=None, mcc=0.3935, brier=None, ece=None,
         decision="Not supported as major contributor", source="QGNN_V4_QUANTUM_ENCODING_RESULTS.md"),
]


def write_master_csv():
    path = os.path.join(OUT_DIR, "QGNN_V4_MASTER_RESULTS.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in ROWS:
            out = {c: row.get(c, "") for c in MASTER_COLUMNS}
            out = {k: ("" if v is None else v) for k, v in out.items()}
            writer.writerow(out)
    print(f"{path}: {len(ROWS)} rows")


ABLATION_ROWS = [
    dict(factor="Training patience (10 vs. 25)", phase="Phase 2",
         evidence="18/20 runs select the exact same checkpoint at patience=25; positive control confirms validation PR-AUC genuinely never improves beyond the patience=10 optimum for every unchanged seed.",
         interpretation="Not supported as major contributor"),
    dict(factor="Output scale (trainable/fixed alpha, +bias)", phase="Phase 2b",
         evidence="All four variants land within [0.82, 0.84] primary / [0.388, 0.404] severity -- statistically indistinguishable from the raw baseline (0.833/0.392) and each other; mathematically redundant with the existing Linear(6,1) by construction.",
         interpretation="Not supported as major contributor"),
    dict(factor="LayerNorm (no-affine / affine) on PauliZ output", phase="Phase 2c",
         evidence="Genuine, non-redundant change (per-example normalization). Cuts severity recall/F1 std by ~10x, eliminates threshold-0.5 collapse (2->0 seeds), improves calibration on both splits substantially. Costs primary PR-AUC mean (0.833->0.800-0.811) and reverses ranking for one severity seed (45).",
         interpretation="Mixed evidence -- adopted as the standing output stage for its stability/calibration benefit, not because it closes the Classical gap"),
    dict(factor="Seed-specific behavior (seed 45 reversal)", phase="Phase 2d",
         evidence="Seed 45's baseline circuit concentrated signal into one dominant PauliZ channel (corr=0.524, highest of any seed); LayerNorm removes the optimization incentive for this, and that seed's best channel correlation nearly halves (0.524->0.282) while every other seed's improves or holds flat. Gradients, dataset composition, and general training dynamics were all ruled out as explanations.",
         interpretation="Supported contributor (to seed-dependent representation distortion specifically, medium confidence, not proven causal)"),
    dict(factor="Qubit count (4, 6, 8)", phase="Phase 3",
         evidence="4q gives the highest primary mean (0.847) with the tightest severity std among the three; 8q is comparable to 6q on both metrics. Confounded with classical Linear(128,n_qubits) parameter count, which scales with qubit count too.",
         interpretation="Mixed evidence -- primary is genuinely low-rank (corroborated independently by Stage 1's PCA finding), but the qubit-count effect itself is capacity-confounded"),
    dict(factor="Circuit depth (1, 2, 3 layers)", phase="Phase 3",
         evidence="1 layer: worst primary (0.693) but tightest severity std of any configuration in the whole project (0.017). 3 layers: comparable primary to 2-layer reference, severity run killed (no data).",
         interpretation="Inconclusive (3-layer severity incomplete; 1-layer's severity stability not otherwise explained)"),
    dict(factor="Frozen-embedding representation quality (diagnostic classifiers, PCA)", phase="Phase 4 Stage 1",
         evidence="A plain logistic regression on the same frozen embedding matches/exceeds both the classical head and every QGNN configuration on severity PR-AUC (0.465-0.469 vs. 0.449/0.398), with far tighter cross-seed std. PCA shows primary signal is low-rank (4 components best) while severity signal is distributed (128 components best).",
         interpretation="Supported contributor -- the representation itself is not the ceiling either downstream head is hitting; the severity problem is not solved by more capacity or better projection alone"),
    dict(factor="Projection/bottleneck architecture (nonlinear, pre-norm, PCA-informed)", phase="Phase 4 Stage 2",
         evidence="Every alternative underperforms the existing single Linear(128,6) bottleneck on both splits in a 2-seed pilot; PCA-informed projection underperforms despite far fewer parameters and contradicts Stage 1's own diagnostic finding on the same PCA representation.",
         interpretation="Not supported as major contributor (2-seed pilot; PCA-projection anomaly flagged as unresolved)"),
    dict(factor="Quantum parameter initialization (default / small Gaussian / identity-like)", phase="Phase 4 Stage 4 + 4b",
         evidence="Small-Gaussian and identity-like both improve primary mean and calibration but degrade severity, in a 2-seed pilot. Identity-like's apparent variance reduction (std 0.094->0.017) did NOT survive 5-seed expansion (std rose to 0.059; severity std became WORSE than the reference, 0.112 vs 0.057). Root-cause analysis attributes the primary/severity split to a ranking-specific overfitting-to-validation failure, not a general calibration or optimization defect.",
         interpretation="Not supported as major contributor -- a real trade-off exists but not a net improvement, and the stability claim did not replicate at full seed count"),
    dict(factor="Ansatz / entanglement topology (StronglyEntangling / hardware-ring / reduced-chain)", phase="Phase 4 Stage 5",
         evidence="All three land within ~1% of each other on primary mean and ~4% on severity mean (5 seeds each). StronglyEntangling (reference) has the tightest primary variance. Reduced-entanglement shows a reproducible severity specificity collapse on 2/5 seeds and the most redundant (cross-correlated) quantum feature channels of the three. Not parameter-matched (36/24/12 quantum params).",
         interpretation="Not supported as major contributor to the Classical gap; mixed evidence on secondary properties (stability, calibration, representation redundancy)"),
    dict(factor="Quantum encoding (scale, tanh vs. clip, data re-uploading)", phase="Encoding Investigation",
         evidence="2x scale (E2) is a clear, multiply-corroborated regression (widest angle spread, worst channel-target correlation, distinct severity convergence delay). Linear clip (E3) regresses with a directly measured mechanism (46% of angles saturate exactly at the boundary). Re-uploading (E4) changes representation redundancy without a metrics benefit. 0.5pi (E1) shows a nominally higher primary mean but 1.5x the reference's variance -- not seed-consistent.",
         interpretation="Not supported as major contributor (2-seed pilot; no configuration expanded to 5 seeds per the investigation's own decision rule)"),
    dict(factor="Fresh-onset detection", phase="Multiple (Phase 1 through Encoding Investigation)",
         evidence="Fresh-onset PR-AUC sits at 0.004-0.012 across every QGNN-v4 configuration AND the full classical GraphSAGE-Full baseline tested to date -- the full classical model, with complete graph/message-passing access, shows the identical near-zero floor.",
         interpretation="Supported contributor (to understanding the limitation) -- ruled out as an architecture-specific problem; a property of the frozen input features at the prediction horizon used, not of any model tested"),
]

ABLATION_COLUMNS = ["factor", "phase", "evidence", "interpretation"]


def write_ablation_csv():
    path = os.path.join(OUT_DIR, "QGNN_V4_ABLATION_TABLE.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ABLATION_COLUMNS)
        writer.writeheader()
        for row in ABLATION_ROWS:
            writer.writerow(row)
    print(f"{path}: {len(ABLATION_ROWS)} rows")


REFERENCE_COMPARISON_ROWS = [
    dict(split="primary", metric="pr_auc", classical=0.8070, qgnn=0.8112,
         absolute_gap=0.8070 - 0.8112, relative_gap=(0.8070 - 0.8112) / 0.8070,
         note="QGNN mean nominally exceeds classical here; both stds (0.066 vs 0.074) are wide enough that this is not a robust directional claim"),
    dict(split="primary", metric="roc_auc", classical=0.9871, qgnn=0.9775,
         absolute_gap=0.9871 - 0.9775, relative_gap=(0.9871 - 0.9775) / 0.9871, note=""),
    dict(split="primary", metric="f1", classical=0.6771, qgnn=0.6565,
         absolute_gap=0.6771 - 0.6565, relative_gap=(0.6771 - 0.6565) / 0.6771, note=""),
    dict(split="primary", metric="recall", classical=0.9678, qgnn=0.8975,
         absolute_gap=0.9678 - 0.8975, relative_gap=(0.9678 - 0.8975) / 0.9678, note=""),
    dict(split="primary", metric="mcc", classical=0.6845, qgnn=0.6539,
         absolute_gap=0.6845 - 0.6539, relative_gap=(0.6845 - 0.6539) / 0.6845, note=""),
    dict(split="primary", metric="brier", classical=0.0418, qgnn=0.0655,
         absolute_gap=0.0418 - 0.0655, relative_gap=(0.0418 - 0.0655) / 0.0418, note="Lower is better for Brier/ECE -- negative gap here means classical is better calibrated"),
    dict(split="primary", metric="ece", classical=0.0493, qgnn=0.1823,
         absolute_gap=0.0493 - 0.1823, relative_gap=(0.0493 - 0.1823) / 0.0493, note="Classical substantially better calibrated"),
    dict(split="severity", metric="pr_auc", classical=0.4487, qgnn=0.3976,
         absolute_gap=0.4487 - 0.3976, relative_gap=(0.4487 - 0.3976) / 0.4487,
         note="Classical leads with tight, non-overlapping-in-practice variance on both sides"),
    dict(split="severity", metric="roc_auc", classical=0.7275, qgnn=0.6566,
         absolute_gap=0.7275 - 0.6566, relative_gap=(0.7275 - 0.6566) / 0.7275, note=""),
    dict(split="severity", metric="f1", classical=0.4113, qgnn=0.4187,
         absolute_gap=0.4113 - 0.4187, relative_gap=(0.4113 - 0.4187) / 0.4113, note="QGNN LayerNorm-no-affine essentially at parity with classical on F1 specifically"),
    dict(split="severity", metric="recall", classical=0.4406, qgnn=0.4166,
         absolute_gap=0.4406 - 0.4166, relative_gap=(0.4406 - 0.4166) / 0.4406, note=""),
    dict(split="severity", metric="mcc", classical=0.3624, qgnn=0.3824,
         absolute_gap=0.3624 - 0.3824, relative_gap=(0.3624 - 0.3824) / 0.3624, note="QGNN slightly higher MCC on severity"),
    dict(split="severity", metric="brier", classical=0.0857, qgnn=0.1225,
         absolute_gap=0.0857 - 0.1225, relative_gap=(0.0857 - 0.1225) / 0.0857, note="Classical better calibrated"),
    dict(split="severity", metric="ece", classical=0.0904, qgnn=0.2405,
         absolute_gap=0.0904 - 0.2405, relative_gap=(0.0904 - 0.2405) / 0.0904, note="Classical substantially better calibrated"),
]
REFERENCE_COMPARISON_COLUMNS = ["split", "metric", "classical", "qgnn", "absolute_gap", "relative_gap", "note"]


def write_reference_comparison_csv():
    path = os.path.join(OUT_DIR, "QGNN_V4_REFERENCE_COMPARISON.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REFERENCE_COMPARISON_COLUMNS)
        writer.writeheader()
        for row in REFERENCE_COMPARISON_ROWS:
            row = dict(row)
            row["absolute_gap"] = round(row["absolute_gap"], 4)
            row["relative_gap"] = round(row["relative_gap"], 4)
            writer.writerow(row)
    print(f"{path}: {len(REFERENCE_COMPARISON_ROWS)} rows")


if __name__ == "__main__":
    write_master_csv()
    write_ablation_csv()
    write_reference_comparison_csv()
