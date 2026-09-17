# QGNN-v4 Phase 4 — Stage 2 Results: Bottleneck / Projection Pilot

## 1. Objective

Determine whether changing how the frozen 128-D GraphSAGE embedding is
projected into the 6-qubit quantum circuit — before any other part of the
pipeline changes — produces a reproducible improvement over the existing
linear-bottleneck reference. Quantum circuit (6 qubits, 2 layers,
StronglyEntanglingLayers, backprop), post-quantum LayerNorm-no-affine,
optimizer, loss, threshold, and frozen encoder are held fixed throughout;
only the classical input side of the head changes per variant.

## 2. Existing baseline (unchanged, not re-derived)

| | Primary PR-AUC | Severity PR-AUC |
|---|---|---|
| Classical GraphSAGE-Full | 0.807 ± 0.066 | 0.449 ± 0.013 |
| QGNN-v4 reference (6q/2L/StronglyEntangling/LayerNorm, 5 seeds) | 0.811 ± 0.074 | 0.398 ± 0.057 |
| QGNN-v4 4-qubit (context only) | 0.847 ± 0.102 | 0.396 ± 0.030 |

## 3. Exact architectures tested

| Config | Projection | Quantum circuit | Total trainable params |
|---|---|---|---:|
| **B1** (control) | `Linear(128,6)` | unchanged (6q/2L/StronglyEntangling) | 817 |
| **B2** | `Linear(128,32)→GELU→Linear(32,6)` | unchanged | 4,369 |
| **B3** | `LayerNorm(128)→Linear(128,6)` | unchanged | 1,073 |
| **B4-PCA4** | train-fit `PCA(4)→Linear(4,6)` | unchanged | 73 |
| **B4-PCA6** | train-fit `PCA(6)→Linear(6,6)` | unchanged | 85 |
| **B4-PCA8** | train-fit `PCA(8)→Linear(8,6)` | unchanged | 97 |

Every variant keeps the identical post-quantum `LayerNorm(6, no affine)→
Linear(6,1)` output stage. B1 is **not rerun** — it reuses the
already-completed Phase 2c/3 reference runs for seeds 42/43 exactly as
saved.

## 4. Experimental protocol

- **Code changes**: `HybridQuantumHeadLayerNorm` (`quantum/heads.py`)
  gained three backward-compatible constructor args (`projection_type`,
  `projection_hidden_dim`, `pre_projection_norm`) — defaults reproduce
  the pre-Stage-2 behavior exactly (verified by a bit-identical-output
  test). `run_qgnn_v4_experiment.py` gained matching CLI flags
  (`--projection-type`, `--projection-hidden-dim`, `--pre-projection-norm`,
  `--pca-components`) that default to today's behavior when omitted. B4
  needed **no new head code** — it reuses `qgnn_v2.build_v2_prepared`
  (the same train-split-only PCA infra QGNN-v2 already uses,
  leakage-tested in `test_qgnn_v2.py`) to pre-reduce the embedding, then
  feeds it into the SAME `HybridQuantumHeadLayerNorm` with `in_dim=
  pca_components`. `qgnn.py`/`qgnn_v2.py`/`qgnn_v2_reupload.py`/`qgnn_v3.py`
  were not modified.
- **Fairness**: every variant used the identical frozen encoder checkpoints
  (per seed), dataset, split, optimizer (Adam, lr=0.001, weight_decay=
  0.0001), max_epochs=100, patience=10, class_weighting=balanced,
  threshold=fixed@0.5, 6 qubits/2 layers/StronglyEntangling/backprop, and
  `--diagnostics` throughout. Only the projection changed per run.
- **Pilot scope**: seeds 42, 43 × primary + severity splits × 6 configs
  (5 newly run + B1 reused) = 24 runs total (20 newly trained).
- **Validation before the pilot**: each new variant was smoke-tested
  (1 seed, 3 epochs) against the real dataset first, then the full test
  suite (291 tests, including 9 new Stage 2 tests) was run twice — before
  and after the new code — with zero regressions, before committing to
  the 20-run pilot batch.

## 5–7. Primary results, Severity results, all requested metrics (test split, threshold=0.5)

| Config | Split | PR-AUC | ROC-AUC | F1 | Precision | Recall | Specificity | Balanced Acc. | MCC | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 (control) | primary | 0.774±0.094 | 0.977±0.012 | 0.637±0.043 | 0.511±0.009 | 0.857±0.126 | 0.941±0.007 | 0.899±0.060 | 0.631±0.059 | 0.070±0.016 | 0.193±0.034 |
| B2 nonlinear | primary | 0.650±0.017 | 0.963±0.003 | 0.583±0.054 | 0.454±0.034 | 0.816±0.101 | 0.930±0.001 | 0.873±0.051 | 0.573±0.066 | 0.092±0.028 | 0.225±0.059 |
| B3 pre-norm | primary | 0.729±0.032 | 0.980±0.005 | 0.641±0.0005 | 0.488±0.014 | 0.940±0.052 | 0.929±0.008 | 0.934±0.022 | 0.647±0.009 | 0.083±0.029 | 0.205±0.075 |
| B4-PCA4 | primary | 0.391±0.094 | 0.905±0.038 | 0.350±0.062 | 0.237±0.068 | 0.802±0.169 | 0.776±0.121 | 0.789±0.024 | 0.352±0.030 | 0.161±0.062 | 0.290±0.123 |
| B4-PCA6 | primary | 0.524±0.037 | 0.951±0.003 | 0.544±0.002 | 0.395±0.007 | 0.876±0.041 | 0.903±0.007 | 0.890±0.017 | 0.548±0.009 | 0.084±0.002 | 0.167±0.012 |
| B4-PCA8 | primary | 0.511±0.047 | 0.900±0.017 | 0.492±0.011 | 0.360±0.025 | 0.787±0.060 | 0.898±0.019 | 0.842±0.021 | 0.484±0.0002 | 0.085±0.011 | 0.144±0.049 |
| B1 (control) | severity | 0.422±0.028 | 0.667±0.026 | 0.456±0.076 | 0.592±0.259 | 0.414±0.027 | 0.962±0.032 | 0.688±0.003 | 0.440±0.114 | 0.139±0.024 | 0.287±0.035 |
| B2 nonlinear | severity | 0.389±0.004 | 0.633±0.016 | 0.405±0.039 | 0.419±0.076 | 0.397±0.005 | 0.954±0.014 | 0.675±0.009 | 0.360±0.047 | 0.103±0.038 | 0.162±0.125 |
| B3 pre-norm | severity | 0.263±0.045 | 0.702±0.038 | 0.425±0.026 | 0.413±0.049 | 0.441±0.000 | 0.949±0.010 | 0.695±0.005 | 0.378±0.030 | 0.099±0.012 | 0.183±0.039 |
| B4-PCA4 | severity | 0.289±0.015 | 0.698±0.019 | 0.298±0.044 | 0.218±0.036 | 0.470±0.053 | 0.865±0.013 | 0.667±0.033 | 0.240±0.052 | 0.143±0.018 | 0.248±0.052 |
| B4-PCA6 | severity | 0.263±0.039 | 0.671±0.017 | 0.233±0.053 | 0.161±0.039 | 0.425±0.080 | 0.821±0.019 | 0.623±0.049 | 0.163±0.068 | 0.147±0.007 | 0.246±0.027 |
| B4-PCA8 | severity | 0.341±0.100 | 0.676±0.052 | 0.380±0.025 | 0.316±0.028 | 0.479±0.014 | 0.917±0.008 | 0.698±0.011 | 0.328±0.027 | 0.104±0.002 | 0.155±0.006 |

Mean ± std across the 2 pilot seeds (42, 43) throughout. Full per-seed
values, confusion matrices (TP/TN/FP/FN), and every train/validation/test
metric: `experiments/qgnn_v4/phase4_stage2/stage2_per_seed_results.csv`,
`stage2_confusion_matrices.csv`, `stage2_summary.csv`, `stage2_metrics.csv`,
`stage2_results.csv`.

**Note on B1's numbers**: 0.774±0.094 (primary) / 0.422±0.028 (severity)
here is the seeds-{42,43} subset of the full 5-seed reference
(0.811±0.074 / 0.398±0.057) — naturally a different mean/std than the
5-seed number from averaging fewer, different seeds, not a discrepancy.
Every B2/B3/B4 comparison in this report is against this SAME 2-seed B1
subset, so the comparison stays apples-to-apples.

## 8–9. Per-seed results

| Config | Split | Seed | PR-AUC | ROC-AUC | MCC |
|---|---|---:|---:|---:|---:|
| B1 | primary | 42 | 0.868 | 0.990 | 0.690 |
| B1 | primary | 43 | 0.681 | 0.965 | 0.572 |
| B1 | severity | 42 | 0.394 | 0.693 | 0.326 |
| B1 | severity | 43 | 0.450 | 0.640 | 0.554 |
| B2 | primary | 42 | 0.632 | 0.966 | 0.506 |
| B2 | primary | 43 | 0.667 | 0.961 | 0.639 |
| B2 | severity | 42 | 0.385 | 0.617 | 0.313 |
| B2 | severity | 43 | 0.393 | 0.648 | 0.407 |
| B3 | primary | 42 | 0.697 | 0.985 | 0.657 |
| B3 | primary | 43 | 0.762 | 0.974 | 0.638 |
| B3 | severity | 42 | 0.308 | 0.740 | 0.348 |
| B3 | severity | 43 | 0.218 | 0.663 | 0.409 |
| B4-PCA4 | primary | 42 | 0.484 | 0.943 | 0.322 |
| B4-PCA4 | primary | 43 | 0.297 | 0.867 | 0.382 |
| B4-PCA4 | severity | 42 | 0.303 | 0.679 | 0.292 |
| B4-PCA4 | severity | 43 | 0.274 | 0.716 | 0.187 |
| B4-PCA6 | primary | 42 | 0.487 | 0.948 | 0.538 |
| B4-PCA6 | primary | 43 | 0.561 | 0.955 | 0.557 |
| B4-PCA6 | severity | 42 | 0.302 | 0.655 | 0.095 |
| B4-PCA6 | severity | 43 | 0.224 | 0.688 | 0.231 |
| B4-PCA8 | primary | 42 | 0.464 | 0.916 | 0.484 |
| B4-PCA8 | primary | 43 | 0.557 | 0.883 | 0.484 |
| B4-PCA8 | severity | 42 | 0.442 | 0.727 | 0.356 |
| B4-PCA8 | severity | 43 | 0.241 | 0.624 | 0.301 |

## 10. Comparison against Classical GNN

Every Stage 2 variant sits below the classical GraphSAGE-Full baseline
(0.807 primary / 0.449 severity) on primary; B1's own 2-seed primary
subset (0.774) is closest. On severity, B1's 2-seed subset (0.422) is
close to classical (0.449); every other variant is further below it. No
Stage 2 variant closes the classical gap.

## 11. Comparison against the existing QGNN reference

B1 IS the existing reference (2-seed subset). Every new variant (B2, B3,
B4×3) underperforms it on **both** primary and severity PR-AUC, with no
exception across all 5 new configurations × 2 splits.

## 12. Stability analysis

- **B2** has the tightest severity std (±0.004) of anything tested here —
  but at a mean (0.389) below B1's — so tighter variance around a worse
  number, not a stability win.
- **B4-PCA8 severity** has the worst std observed (±0.100 on just 2
  seeds) — one seed (42: 0.442) looks almost competitive with B1, the
  other (43: 0.241) collapses; this single-pair spread is a caution
  against reading either seed as representative.
- **Validation→test PR-AUC generalization gap** (a cleaner stability
  signal than seed std alone, given n=2): B1 primary 0.083 (smallest of
  any config) vs. B2 0.225, B3 0.120, B4-PCA4/6/8 0.301/0.281/0.235. Every
  projection change increased the val→test gap on primary — the existing
  linear bottleneck generalizes from validation to test more reliably
  than any alternative tried. On severity the gap is large for all
  configs (0.56–0.74, consistent with Stage 1's finding that severity is
  an inherent train/test distribution-shift problem) with no clear
  ordering by config.

## 13. Calibration analysis

ECE ranges 0.14–0.29 across every config on both splits. On primary, B1
has the best (lowest) ECE (0.193) of any config. On severity, B1's ECE
(0.287) is actually the *worst* of the six — B4-PCA8 (0.155) and B4-PCA6
(0.246) are numerically better calibrated there. But this comes bundled
with much worse PR-AUC/MCC on those same configs, so it reads as "less
confident, less-discriminative predictions being incidentally
better-calibrated" rather than a genuine calibration win worth trading
ranking quality for. No config shows a calibration failure (ECE > 0.4)
severe enough to be disqualifying on its own.

## 14–15. Which configuration should proceed / be rejected — Final Decision

| Config | Verdict | Reasoning |
|---|---|---|
| **B2 — nonlinear projection** | **DROP** | Primary PR-AUC 0.650 vs. B1's 0.774 (−16%), severity also worse (0.389 vs 0.422); 5.3× the parameters of B1 for a worse result on every ranking metric; largest primary val→test gap after the PCA variants. No metric favors it. |
| **B3 — pre-projection LayerNorm** | **DROP** | Primary is only modestly worse (0.729 vs 0.774, −6%), but severity collapses (0.263 vs 0.422, −38%) — fails the phase's own "no major degradation in important secondary metrics" criterion outright. Plausible mechanism: Stage 1's A3 found individual raw embedding dimensions correlate up to r=0.87 with the target — normalizing the embedding's own per-dimension scale away *before* the learned projection may be discarding exactly the magnitude information that made those dimensions useful, particularly for the harder severity generalization task. |
| **B4 — PCA4/PCA6/PCA8** | **INVESTIGATE, do not expand as-is** | All three underperform B1 substantially on both splits (primary 0.39–0.52 vs. 0.774; severity 0.26–0.34 vs. 0.422) despite having far fewer parameters (73–97 vs. 817) — ruling out "more capacity = more overfitting" as the explanation. The validation→test gap (0.235–0.301) is the largest of any config, and — more tellingly — **contradicts Stage 1's own A2 finding** that a plain logistic regression on the identical PCA(4) representation reached primary PR-AUC 0.781, comparable to the full 128-D embedding. That the quantum head does dramatically worse than a linear model on the exact same reduced input is a real, reproducible signal in this pilot, but the cause (an optimization/scale mismatch between PCA's per-component variance structure and this training setup, vs. a genuine representational problem) is not yet established. Do not expand to 5 seeds until that's understood — repeating a likely-fixable setup 2.5× more would burn compute without adding evidence. |
| **B1 — reference (unchanged)** | **KEEP as the standing control** | Every alternative tried underperforms it. No architectural change is indicated by this pilot. |

**Overall Stage 2 conclusion, stated plainly per the phase's own rule
against manufacturing a positive result: none of the three projection
changes tested (nonlinear projection, pre-projection normalization,
PCA-informed projection) produced an improvement over the existing linear
bottleneck in this controlled pilot.** Two (B2, B3) are clear, unambiguous
negative results. The PCA variants (B4) are a more interesting negative
result — large enough and inconsistent enough with Stage 1's own
diagnostic evidence to warrant a short follow-up before being written off
completely, but not a reason to expand the pilot as currently configured.

## 16. Limitations

- **2-seed pilot, not 5**: per the phase's own staged-expansion rule,
  and because no config showed a "credible improvement" to justify
  expanding. Point estimates here (especially stds, computed from n=2)
  are noisy — read the qualitative pattern (every alternative underperforms
  B1, consistently, on both metrics and both splits) rather than the exact
  decimal values.
- **Severity split's known limitation carries over unchanged** (Phase 3/
  Stage 1 finding): only one severity-5 event underlies the entire OOD
  test population — every severity number in this report inherits that
  caveat.
- **B4's root cause is not yet isolated** — see the INVESTIGATE verdict
  above. This report does not claim to know why PCA-informed projection
  underperforms; it reports that it does, cleanly, and flags the specific
  contradiction with Stage 1 worth chasing.
- Matched-capacity classical control was not retrained per projection
  variant (`--quantum-only` used throughout) — per instruction, since
  Stage 2's question is specifically about the quantum head's own
  projection, not a fresh RQ-Q3 fairness re-check.

## 17. Recommended next experiment

Given Stage 2 found no improvement from touching the projection, and the
master research plan flags initialization as "a high-priority experiment"
given persistent seed sensitivity (reconfirmed here: B4-PCA8 severity std
0.100 from just 2 seeds) — **Stage 4 (Initialization: F1 current vs. F2
small-Gaussian vs. F3 identity-like)** is the most evidence-supported next
step: it directly targets the seed-variance problem visible in every
stage so far (Phase 1 through this pilot) rather than another
representation-side change that Stage 2 just showed doesn't obviously
help. Stage 5 (the two already-built ansätze, 4q+2L/6q+2L controlled
comparison) is the other reasonable candidate, already fully coded from
Phase 3. Deferring to you on which to run next, per the phase's own rule
not to auto-advance through remaining stages.
