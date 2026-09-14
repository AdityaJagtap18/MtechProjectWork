# Classical GNN (GraphSAGE-Full) Baseline — Extended Metrics

Part 1 of this phase's request: extracts the full ranking/classification/
calibration/generalization metric set for the **full classical GraphSAGE
GNN** (`HeteroGraphSAGE`, message-passing, `hidden_dim=128`) from its
already-existing, already-trained run artifacts in
`experiments/classical_gnn/` — **no retraining**. This is a different
baseline than the "matched-capacity classical control"
(`Linear(128,6)→ReLU→Linear(6,1)`) used throughout Phases 1–2b's RQ-Q3
ablation — that control isolates "does bottlenecking hurt," this one is
the real classical architecture QGNN-v4 is ultimately meant to be
compared against.

**These are the exact same frozen-encoder checkpoints QGNN-v4 has used
as its encoder in every phase so far** (`scripts/extract_classical_gnn_baseline.py`
uses `_analysis_common.find_latest_graphsage_full_checkpoint`, the same
resolver `run_qgnn_v4_experiment.py` calls) — confirmed by identical
train/validation/test row counts between classical's and QGNN-v4's own
saved `predictions.csv` (18600/4500/3600 primary, matching exactly).
Threshold policy: `fixed, value: 0.5` for both classical
(`configs/graphsage.yaml`/`graphsage_severity.yaml`) and every QGNN-v4
config — identical operating point, not recomputed or re-tuned.

**One field required recomputation, not extraction:** classical's saved
`onset_breakdown.json` predates the `pr_auc_fresh_onset`/
`roc_auc_fresh_onset` fields (an older schema — those runs were produced
before that addition to `evaluate.disruption_onset_breakdown`). Recomputed
here by calling that exact, current, shared function on the already-saved
`predictions.csv` — not a different or invented definition, the same
function QGNN-v4's own onset breakdown already uses.

**Genuinely unavailable, stated rather than guessed at:** per-epoch train
PR-AUC (`train.py`'s history logs `train_loss` per epoch only, never
`train_pr_auc` — only the final, best-checkpoint train PR-AUC is
recoverable, from `metrics.json`) and any per-layer gradient norm
(classical's training loop never logged one, unlike QGNN-v4's
`--diagnostics` mode). Both are `None` in the saved CSV, not derived from
an unrelated quantity.

Structured data: `experiments/classical_gnn/baseline_extended_metrics/classical_gnn_all_metrics.csv` (10 rows), `classical_gnn_severity_level_breakdown.csv` (20 rows) — gitignored with every other run artifact; this report and `scripts/extract_classical_gnn_baseline.py` are the committed record.

---

## Classical GNN Baseline Table (per-seed, test split)

### Primary (temporal) split

| Seed | PR-AUC | ROC-AUC | F1 | Precision | Recall | Specificity | Bal. Acc. | MCC | Brier | ECE | MCE | Best Epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.7874 | 0.9872 | 0.7236 | 0.5796 | 0.9628 | 0.9497 | 0.9563 | 0.7255 | 0.0333 | 0.0441 | 0.5131 | 11 |
| 43 | 0.6888 | 0.9713 | 0.6382 | 0.4784 | 0.9587 | 0.9247 | 0.9417 | 0.6478 | 0.0639 | 0.0668 | 0.6812 | 12 |
| 44 | 0.8745 | 0.9943 | 0.7032 | 0.5566 | 0.9545 | 0.9452 | 0.9499 | 0.7055 | 0.0299 | 0.0357 | 0.7324 | 12 |
| 45 | 0.8491 | 0.9926 | 0.6865 | 0.5227 | 1.0000 | 0.9342 | 0.9671 | 0.6988 | 0.0397 | 0.0476 | 0.6843 | 9 |
| 46 | 0.8354 | 0.9901 | 0.6340 | 0.4726 | 0.9628 | 0.9226 | 0.9427 | 0.6449 | 0.0422 | 0.0523 | 0.6502 | 10 |

### Severity/OOD split

| Seed | PR-AUC | ROC-AUC | F1 | Precision | Recall | Specificity | Bal. Acc. | MCC | Brier | ECE | MCE | Best Epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.4430 | 0.7298 | 0.3838 | 0.3399 | 0.4406 | 0.9320 | 0.6863 | 0.3312 | 0.0977 | 0.1012 | 0.8553 | 20 |
| 43 | 0.4560 | 0.7271 | 0.4431 | 0.4455 | 0.4406 | 0.9564 | 0.6985 | 0.3991 | 0.0748 | 0.0792 | 0.8534 | 26 |
| 44 | 0.4378 | 0.7159 | 0.4228 | 0.4063 | 0.4406 | 0.9488 | 0.6947 | 0.3753 | 0.0798 | 0.0859 | 0.8606 | 22 |
| 45 | 0.4358 | 0.7097 | 0.4022 | 0.3700 | 0.4406 | 0.9404 | 0.6905 | 0.3518 | 0.0874 | 0.0926 | 0.8626 | 24 |
| 46 | 0.4708 | 0.7550 | 0.4045 | 0.3738 | 0.4406 | 0.9414 | 0.6910 | 0.3544 | 0.0888 | 0.0930 | 0.8657 | 18 |

**Severity recall is exactly 0.4406 in every one of the 5 seeds** (std =
0.0000) — this is a known, previously-documented finding
(`evaluate.disruption_onset_breakdown`'s own docstring), not new here:
the classical model catches the identical set of already-ongoing
positives every single independently-seeded run, and zero fresh onsets
in every run. Confirmed again by the recomputed onset breakdown below.

## Cross-Seed Stability (mean ± std, median, min, max, IQR)

| Split | Metric | Mean | Std | Median | Min | Max | IQR |
|---|---|---:|---:|---:|---:|---:|---:|
| primary | PR-AUC | 0.8070 | 0.0655 | 0.8354 | 0.6888 | 0.8745 | 0.0617 |
| primary | ROC-AUC | 0.9871 | 0.0083 | 0.9901 | 0.9713 | 0.9943 | 0.0055 |
| primary | F1 | 0.6771 | 0.0355 | 0.6865 | 0.6340 | 0.7236 | 0.0650 |
| primary | Precision | 0.5220 | 0.0421 | 0.5227 | 0.4726 | 0.5796 | 0.0783 |
| primary | Recall | 0.9678 | 0.0164 | 0.9628 | 0.9545 | 1.0000 | 0.0041 |
| primary | MCC | 0.6845 | 0.0324 | 0.6988 | 0.6449 | 0.7255 | 0.0576 |
| primary | Brier | 0.0418 | 0.0119 | 0.0397 | 0.0299 | 0.0639 | 0.0089 |
| primary | ECE | 0.0493 | 0.0103 | 0.0476 | 0.0357 | 0.0668 | 0.0083 |
| severity | PR-AUC | 0.4487 | 0.0131 | 0.4430 | 0.4358 | 0.4708 | 0.0181 |
| severity | ROC-AUC | 0.7275 | 0.0156 | 0.7271 | 0.7097 | 0.7550 | 0.0139 |
| severity | F1 | 0.4113 | 0.0201 | 0.4045 | 0.3838 | 0.4431 | 0.0205 |
| severity | Precision | 0.3871 | 0.0360 | 0.3738 | 0.3399 | 0.4455 | 0.0363 |
| severity | Recall | 0.4406 | 0.0000 | 0.4406 | 0.4406 | 0.4406 | 0.0000 |
| severity | MCC | 0.3624 | 0.0231 | 0.3544 | 0.3312 | 0.3991 | 0.0234 |
| severity | Brier | 0.0857 | 0.0079 | 0.0874 | 0.0748 | 0.0977 | 0.0090 |
| severity | ECE | 0.0904 | 0.0074 | 0.0926 | 0.0792 | 0.1012 | 0.0071 |

**Classical PR-AUC 0.807±0.066 (primary) matches the headline figure
already documented elsewhere in this project's prior classical-GNN
work** (`QGNN_V4_BENCHMARK.md`'s own reference to "GraphSAGE, PR-AUC
0.807±0.066") — a direct consistency check that this extraction is
correct, not a new number.

## Fresh-Onset Analysis (severity split, recomputed)

| Seed | Fresh-Onset PR-AUC | Fresh-Onset Recall | N Fresh-Onset |
|---:|---:|---:|---:|
| 42 | 0.0064 | 0.0000 | 84 |
| 43 | 0.0056 | 0.0000 | 84 |
| 44 | 0.0065 | 0.0000 | 84 |
| 45 | 0.0056 | 0.0000 | 84 |
| 46 | 0.0062 | 0.0000 | 84 |

**Fresh-onset PR-AUC (0.0056–0.0065) sits at the same near-zero noise
floor QGNN-v4's own fresh-onset PR-AUC does (0.0045–0.0102 across every
Phase 2b configuration)** — this is an important cross-check: the
absence of a fresh-onset signal is not specific to any QGNN-v4 head or
to the frozen-embedding bottleneck. The **full** classical GNN, with
complete access to the graph and all message-passing information, shows
the identical failure. This strengthens Phase 1's original conclusion
that the missing signal is a property of the benchmark's input features
themselves (no genuine leading indicator of a fresh onset exists in the
dynamic features at the prediction horizon used), not a limitation of
any model architecture tested in this project to date.

## Severity-Level Breakdown (classical GNN, same reconstruction method as QGNN-v4's)

| Severity Level | n (≈) | n Positive | PR-AUC (mean of 5 seeds) |
|---:|---:|---:|---:|
| 0 (baseline) | 8400 | 43 | 0.964 |
| 1 (unreliable, 1 positive total) | 600 | 1 | 0.550 |
| 2 | 7800 | 311 | 0.892 |
| 5 (= severity split's test set) | 9900 | 733 | 0.449 |

Same qualitative pattern as QGNN-v4's own severity-level breakdown
(`QGNN_V4_PHASE2B_EXTENDED_METRICS.md`): high at severity 0/2, a sharp
drop at severity 5 — not a gradual decline. The classical GNN's absolute
PR-AUC is meaningfully higher than any QGNN-v4 configuration's at every
level, consistent with it having access to the full graph and full
feature set rather than a 6-dimensional post-hoc bottleneck.

## What's unavailable and why

| Field | Status | Reason |
|---|---|---|
| Per-epoch train PR-AUC / best_train_pr_auc | Not available | `train.py`'s `history_rows` logs `train_loss` per epoch, never `train_pr_auc` — only the final (best-checkpoint) train PR-AUC exists, via `metrics.json`'s `by_split.train.pr_auc` (reported in the CSV as `train_pr_auc`, a single final value, not a curve) |
| Quantum gradient norm (any) | N/A | Classical has no quantum component |
| Reduction-layer gradient norm | Not available | Classical's training loop never logged any per-layer gradient norm, unlike QGNN-v4's `--diagnostics` instrumentation |
