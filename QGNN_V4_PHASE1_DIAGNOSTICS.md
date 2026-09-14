# QGNN-v4 Phase 1: Instability Diagnostics

Phase 1 of the QGNN-v4 stability investigation (research question: *"Under
what conditions does the quantum head improve or fail to improve upon an
equally constrained classical bottleneck, and what factors explain its
stability or instability?"*). Per the investigation's own instruction, no
architecture change was made in this phase — only instrumentation. This
document reports what's derivable from the 20 runs already on disk
(`QGNN_V4_BENCHMARK.md`'s primary + severity 5-seed runs), then describes
the new instrumentation built to capture what those runs didn't log
(gradient norms, per-epoch train PR-AUC), with the exact command to
produce the remaining diagnostics via a real rerun.

**default.qubit + `diff_method="backprop"` throughout — every result and
diagnostic below is an exact-gradient, noise-free classical simulation.
None of the instability found here is attributable to hardware noise or
decoherence; there is no hardware in the loop.**

---

## 1. What's already answerable from existing artifacts (no new training)

Extracted from `training_history.csv`, `metrics.json`, `calibration.json`,
`predictions.csv`, and `run_metadata.json` in each of the 20 existing run
directories (`experiments/qgnn_v4/`).

### Primary (temporal) split

| Seed | Arm | Train PR-AUC | Val PR-AUC | Test PR-AUC | Best epoch | Final epoch | Prob std (test) | Frac ≥0.5 | Final train loss |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | classical | 0.878 | 0.871 | 0.775 | 2 | 12 | 0.091 | 0.143 | 0.284 |
| 42 | quantum | **0.547** | 0.889 | **0.962** | 35 | 45 | 0.198 | 0.120 | 0.329 |
| 43 | classical | 0.898 | 0.866 | 0.682 | 8 | 18 | 0.284 | 0.141 | 0.196 |
| 43 | quantum | 0.863 | 0.860 | **0.570** | 34 | 44 | 0.195 | 0.127 | 0.269 |
| 44 | classical | 0.892 | 0.881 | 0.872 | 8 | 18 | 0.211 | 0.098 | 0.215 |
| 44 | quantum | 0.688 | 0.870 | 0.927 | 22 | 32 | 0.078 | 0.114 | 0.733 |
| 45 | classical | 0.827 | 0.885 | 0.822 | 7 | 17 | 0.233 | 0.135 | 0.255 |
| 45 | quantum | 0.825 | 0.870 | 0.820 | 9 | 19 | **0.023** | **1.000** | 1.062 |
| 46 | classical | 0.865 | 0.886 | 0.830 | 22 | 32 | 0.281 | 0.135 | 0.179 |
| 46 | quantum | 0.818 | 0.868 | 0.886 | 3 | 13 | 0.085 | 0.122 | 0.712 |

### Severity split

| Seed | Arm | Train PR-AUC | Val PR-AUC | Test PR-AUC | Best epoch | Final epoch | Prob std (test) | Frac ≥0.5 | Final train loss |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | classical | 0.943 | 1.000 | 0.431 | 6 | 16 | 0.155 | 0.086 | 0.350 |
| 42 | quantum | 0.967 | 0.971 | 0.361 | **1** | 11 | **0.022** | **0.935** | 1.038 |
| 43 | classical | 0.980 | 1.000 | 0.462 | 1 | 11 | 0.132 | 0.057 | 0.111 |
| 43 | quantum | 0.994 | 0.998 | 0.374 | **3** | 13 | 0.022 | 0.0001 | 1.026 |
| 44 | classical | 0.923 | 1.000 | 0.424 | 2 | 12 | 0.154 | 0.049 | 0.309 |
| 44 | quantum | 0.904 | 1.000 | 0.425 | **5** | 15 | 0.017 | 0.053 | 1.140 |
| 45 | classical | 0.937 | 1.000 | 0.417 | 3 | 13 | 0.176 | 0.098 | 0.224 |
| 45 | quantum | 0.769 | 1.000 | 0.385 | **3** | 13 | **0.004** | **1.000** | 1.271 |
| 46 | classical | 0.961 | 1.000 | 0.467 | 4 | 14 | 0.154 | 0.079 | 0.370 |
| 46 | quantum | 0.962 | 1.000 | 0.412 | **85** | 95 | 0.237 | 0.079 | **0.106** |

### Finding 1: Training loss convergence is consistently worse for quantum

Final train loss is 2–5x higher for quantum than classical in 7 of 10
seed/split combinations (e.g. primary seed 45: 1.06 vs. 0.26; severity
seed 43: 1.03 vs. 0.11). Classical's final train loss stays in a tight
0.11–0.37 range throughout; quantum's ranges 0.11–1.27, with most severity
runs sitting above 1.0. This is a real optimization-difficulty signal —
the quantum head is not fitting its own training objective as well as the
classical control, independent of test-set generalization.

### Finding 2: Early stopping truncates most severity-split quantum runs before the head develops any output spread

Best epoch is 1, 3, 5, 3 for four of five severity-split quantum seeds
(only seed 46 trains to epoch 85). Mechanism, read directly from the
data: severity validation PR-AUC is near-saturated for both arms (classical
exactly 1.000 every seed; quantum 0.97–1.00) — the validation carve-out is
drawn from the same severity-1-3 training distribution, so it's an easy
target. The model hits a "good enough" validation checkpoint almost
immediately, gets no further validation improvement for 10 epochs
(patience), and stops — `final_epoch = best_epoch + 10` holds exactly in
every one of these 20 runs, confirming patience-triggered stopping, not
the 100-epoch budget being exhausted anywhere.

This directly explains the near-collapsed test-probability distributions
in those same seeds: prob std as low as 0.004–0.022 (seed 45: literally
±0.004 around a mean of 0.573 — barely distinguishable from a constant
predictor), against classical's 0.13–0.18 in the same runs. The one
severity seed that trained substantially longer (46, stopped at epoch 95)
has both the lowest final train loss (0.106, actually *better* than its
own classical counterpart's 0.370) and by far the widest, most
classical-like probability spread (std 0.237) among the five quantum
severity seeds — consistent with "more epochs let the quantum head's
output scale develop," though seed 46's test PR-AUC (0.412) isn't
dramatically higher than the truncated seeds (0.36–0.43), so more
training epochs alone did not translate into a large PR-AUC win on the
severity-4-5 OOD test set — it mainly fixed calibration/spread, not
ranking quality.

### Finding 3: Two distinct failure modes are hiding under "high seed variance," not one

- **Seed 43 (primary):** train PR-AUC 0.863, val 0.860, test **0.570**. A
  genuine train/val-to-test generalization failure — the head fits both
  its own training objective and the validation split well, but that
  doesn't transfer to the test window.
- **Seed 42 (primary):** train PR-AUC **0.547** (the *worst* train score
  of any primary-split run, classical or quantum), val 0.889, test
  **0.962** (the *best* test score of any run in this benchmark). The
  reverse pattern — poor training-set ranking, excellent val/test ranking.
  This is unusual enough that a mechanism should not be guessed at without
  the per-epoch curve (§2); it's flagged here as a distinct, unexplained
  anomaly, not folded into "seed 43-style instability."

These are opposite-shaped anomalies. A single "quantum is unstable"
label would obscure that seed 43's problem (overfits train/val, fails
test) and seed 42's problem (underfits train, generalizes anyway) likely
have different causes and may need different fixes.

### Finding 4: The classical control shows none of this

Across all 20 runs, classical's final train loss stays in 0.11–0.37, its
probability distributions never collapse to the extreme narrow ranges
quantum shows, and its train/val/test PR-AUC ordering never inverts the
way seed 42/43 do. Whatever is driving Findings 1–3, it is specific to
the quantum arm, not an artifact of the shared frozen-embedding input or
evaluation code both arms use identically.

---

## 2. What existing artifacts cannot answer — instrumentation built

Not logged by the original run: per-epoch **train PR-AUC** (only final
train-loss was tracked per epoch; train PR-AUC was computed once, on the
final best-checkpoint) and **quantum-circuit gradient norms** (never
logged at all). Both are needed to distinguish, e.g., "seed 42's odd
train/test pattern is a genuine mid-training inflection" from "it's noise
in a single final snapshot," and to test directly whether vanishing/exploding
quantum gradients are the mechanism behind Findings 1–2, versus
early-stopping simply cutting a slower-converging optimization short.

**Built** (additive only — `qgnn.py`/`qgnn_v2.py`/`qgnn_v2_reupload.py`/`qgnn_v3.py`
untouched, no forward-pass or architecture change):

- [`src/scm_dataset/modeling/quantum/train.py`](src/scm_dataset/modeling/quantum/train.py) —
  `train_v4_head_with_diagnostics`, a copy of `qgnn_v2.train_v2_head`'s
  loop with two additions: per-epoch train PR-AUC (an extra `no_grad`
  forward pass over the full train set each epoch), and per-epoch L2 norm
  of the quantum circuit's own gradients (`quantum_grad_norm_mean/max/min`)
  plus the shared `reduce` bottleneck's gradient norm (present for both
  arms, so classical's own gradient health is visible too, not just
  quantum's).
- `scripts/run_qgnn_v4_experiment.py --diagnostics` — swaps in the
  instrumented loop; every other part of the run (checkpoint resolution,
  embedding extraction, artifact saving, run metadata) is unchanged.
- Verified by test (`tests/test_quantum_v4.py`, 2 new tests, 259 total
  passing, 0 regressions): the instrumented loop reaches **bit-identical**
  `best_val_pr_auc`/`best_epoch` to the existing `train_v2_head` given the
  same seed — confirming the new logging changes nothing about
  optimization, only what's recorded.

## 3. Reproducibility — Phase 1 completion

```bash
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4.yaml --tag diag_primary --seeds 42,43,44,45,46 --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4_severity.yaml --tag diag_severity --seeds 42,43,44,45,46 --diagnostics
```

Writes to new, non-overwriting run directories under
`experiments/qgnn_v4/` (same non-overwrite guarantee as every other run in
this project) with two extra `training_history.csv` columns
(`train_pr_auc`, `quantum_grad_norm_mean/max/min`, `reduce_grad_norm_mean`)
per epoch, on top of everything already saved. Same optimizer, loss,
early stopping, seeds, and splits as the original runs — directly
comparable, not a re-tuned rerun.

## 4. Not yet done

Gradient-norm magnitudes and the full per-epoch train-PR-AUC curves for
seeds 42–46 (both splits) — pending the rerun above. Phase 1's report
will be completed with those numbers once that run is executed; Phases
2–6 (calibration/init experiments, ablation grid, 5-seed validation,
scientific comparison, quantum-specific analysis) follow only after
Phase 1's actual mechanism is identified, per the investigation's own
sequencing.
