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

## 3. Gradient and per-epoch curve analysis (real data, `--diagnostics` rerun)

All 20 diagnostic runs reproduced **bit-identical PR-AUC** to the original
(non-instrumented) runs, confirming the added logging changes nothing
about optimization.

### 4.1 Gradients are not vanishing

| | primary quantum | severity quantum |
|---|---|---|
| Quantum-circuit grad norm, mean over training | 0.18–0.25 | 0.14–0.23 |
| Quantum-circuit grad norm, first-epoch → last-epoch | **grows** in 4/5 seeds (e.g. seed 42: 0.088→0.155, seed 45: 0.069→0.309) | grows in 4/5 seeds; **shrinks** smoothly in seed 46 (0.198→0.062, the one run that trained to near-convergence) |
| `reduce` (shared bottleneck) grad norm, first epoch | 0.29–2.14 | 0.23–3.00 |

Quantum-circuit gradients stay substantial (order 0.1–0.3) throughout
every run and generally **grow**, not shrink, over the course of the
(early-stopped) training window — the opposite of a vanishing-gradient/
barren-plateau signature, which would show gradients decaying toward zero
early and staying there. **Barren plateaus are ruled out as the mechanism
at this scale (6 qubits, 2 layers).** The one case where gradients do
shrink smoothly (severity seed 46) is the one run that trained long enough
to approach an actual optimum — normal end-of-training decay, not
pathology.

The shared `reduce` layer's gradient norm starts 3–15x larger than the
quantum circuit's own, in every run. The quantum circuit consistently
receives a smaller share of the gradient signal than the classical
bottleneck it sits behind in the same head — a real, quantifiable
asymmetry, though not one that prevents learning (§3.1's magnitudes are
still well above numerical noise throughout).

### 4.2 A train-PR-AUC/loss decoupling exists, is quantum-specific, and is concentrated on the primary split

| | Classical, both splits (10 runs) | Quantum, severity (5 runs) | Quantum, primary (5 runs) |
|---|---:|---:|---:|
| Train PR-AUC decline from peak to final epoch | 0.0000–0.0074 | 0.0000–0.0299 | **0.0165–0.2080** |

Classical's train PR-AUC rises quickly then holds flat for the rest of
training, in all 10 runs, on both splits — train loss keeps falling every
epoch, but train-set ranking quality never degrades once reached. Quantum
on severity is nearly as stable (small declines, ≤0.03). **Quantum on
primary is qualitatively different**: in 4 of 5 seeds (42, 44, 45, 46),
train PR-AUC peaks within the first 3–11 epochs, then *declines*
substantially (0.11–0.21 absolute) for the remainder of training, even as
train loss keeps monotonically falling every single epoch and validation
PR-AUC stays flat-to-rising throughout. Example, seed 42: train PR-AUC
peaks at 0.874 (epoch 11), falls to 0.714 by epoch 45, while train loss
falls from 1.29 to 0.33 and validation PR-AUC holds at 0.883–0.887 the
whole time. Full per-epoch curves for seeds 42/43 (quantum) and their
classical counterparts are in the run's `training_history.csv` files
(`*_diag_primary_*_seed{42,43}`); classical's curves for the same two
seeds show no such decline (train PR-AUC plateaus within 2–7 epochs and
stays flat for the rest of the run).

**This is not a "declining = failing" story — the correlation runs the
other way.** Across the 5 primary quantum seeds, the size of the
train-PR-AUC decline correlates strongly (r = 0.93) with a *higher* test
PR-AUC, not a lower one:

| Seed | Train PR-AUC decline | Test PR-AUC |
|---:|---:|---:|
| 44 | 0.208 | 0.927 |
| 42 | 0.160 | 0.962 |
| 46 | 0.129 | 0.886 |
| 45 | 0.106 | 0.820 |
| **43** | **0.017** | **0.570** |

Seed 43 — the one primary-split quantum seed that behaves like the
classical control (train PR-AUC rises and holds, no decline) — is the
one with by far the worst test result. Its actual problem (established
in §1, Finding 3, and unchanged by this data) is a genuine train/val-to-test
generalization gap, not the decline phenomenon; the other four seeds'
decline coincides with, and does not prevent, their strong test
generalization. **The decline should not be read as evidence of
instability causing poor results — if anything it tracks the opposite of
overfitting in this small sample.**

Read together with §1's Finding 2 (severity's early stopping cuts
training off after 1–5 epochs in 4/5 seeds, while the training that *does*
run 95 epochs — severity seed 46 — shows smooth, monotonic, decline-free
convergence in both loss and train PR-AUC): the most defensible
mechanism, given what's actually in the data, is that early stopping
(keyed on validation PR-AUC, patience 10) is catching the quantum head at
different, seed-dependent points along a genuinely longer and less
monotonic optimization trajectory than the classical head's — not that
the quantum head is failing to learn. **This is a caveated, 5-seed-sample
finding (r=0.93 on n=5 is suggestive, not a claim of statistical
significance), and the split/training-length confound (only one run in
the whole diagnostic set trained long enough to observe a smooth
late-training regime, and it happens to be on the severity split) means
"is it the split or the training length that matters" cannot be cleanly
separated from this data alone** — flagged as an open question for
Phase 2, not resolved here.

### 4.3 Calibration (Brier/ECE) confirms the same split asymmetry

Quantum's Brier/ECE is worse than classical's in every one of the 20
diagnostic runs (unchanged from `QGNN_V4_BENCHMARK.md`), consistent
throughout §1–4: whatever is driving the train-ranking non-monotonicity
on primary and the truncated training on severity, both leave a
measurable trace in output calibration, not just in the headline PR-AUC.

---

## 5. Phase 1 conclusion — answering the original checklist

1. **Quantum parameter initialization:** `qml.qnn.TorchLayer`'s default
   init (uniform, not seed-pathological — verified: `quantum_grad_norm` is
   already 0.07–0.22 at epoch 1 in every run, i.e. gradients flow from the
   very first step in every seed, on both splits).
2. **Gradient magnitudes during training:** stay in the 0.1–0.3 range for
   the quantum circuit throughout, generally *growing* over an
   early-stopped run and shrinking only when a run trains long enough to
   approach convergence (§3.1).
3. **Do gradients become extremely small/unstable?** No — see above.
   Barren plateaus / vanishing gradients are ruled out at this scale as
   the explanation for the instability found in `QGNN_V4_BENCHMARK.md`.
4. **Output probability distributions across seeds:** vary enormously
   (test-split std from 0.004 to 0.28) — driven primarily by how many
   epochs each run actually completed before patience-triggered stopping
   (§1 Finding 2), not by gradient pathology.
5. **Is the quantum head poorly calibrated?** Yes, consistently — Brier/
   ECE roughly 2–3x worse than classical's in nearly every run (§3.3,
   `QGNN_V4_BENCHMARK.md` §9).
6. **Is the final `Linear(6,1)` output scale causing the threshold
   instability?** Consistent with this: `reduce_grad_norm` (which
   determines how much signal reaches the bottleneck feeding both the
   circuit and, indirectly, the output layer) varies by 10x+ across seeds
   at epoch 1 (0.23–3.00), and the seeds with the narrowest final
   probability spread are exactly the ones that stopped earliest, before
   the output layer had many updates to find an appropriate scale. Not
   independently isolated from "just needs more epochs" in this phase.
7. **Train/validation PR-AUC curves per seed:** produced for all 20 runs
   (`training_history.csv`, `train_pr_auc` and `validation_pr_auc`
   columns); primary seeds 42/43 detailed in §3.2.
8. **Why does seed 43 fail, specifically?** Not initialization or
   gradient pathology (its gradient norms and epoch-1 behavior are
   unremarkable next to the other seeds) — its train and validation PR-AUC
   both rise together and hold (0.86–0.89), the classical-like pattern
   from §3.2, yet test PR-AUC is 0.57, far below train/val. This is a
   genuine train/val-to-test distributional generalization failure specific
   to this seed's frozen encoder/temporal test window, not the same
   phenomenon as the other seeds' train-PR-AUC decline (§3.2) and not
   resolved by this phase's instrumentation — a candidate for its own
   follow-up (e.g., inspecting which suppliers/timepoints seed 43's test
   window actually contains) rather than an architecture change.

**Overall**: the instability documented in `QGNN_V4_BENCHMARK.md` is not
explained by vanishing gradients, barren plateaus, or bad initialization.
It's better explained by (a) an early-stopping policy keyed on validation
PR-AUC that, on the severity split, triggers almost immediately because
validation is near-saturated, truncating training before the quantum
head's output scale develops (§1 Finding 2, §3.1), and (b) a
non-monotonic train-ranking trajectory on the primary split that appears
specific to the quantum head and does not, in this sample, predict worse
test generalization (§3.2) — with seed 43 standing as a separate,
unexplained generalization failure. None of this is attributable to
hardware noise or decoherence — every result here is `default.qubit`,
exact backprop, zero shots.

**Recommended Phase 2 candidates, informed by these findings** (final
choice is the user's, per the original request's Phase 2 candidate list):
- **Training budget / patience** as an added candidate alongside
  init/output-scale/calibration — untested by the original Phase 2 list,
  but directly motivated by §3.1–3.2 (only one run in this entire
  diagnostic set trained long enough to reach smooth, decline-free
  convergence). Worth testing before output-scale changes, since it's the
  simplest change and most directly targets what the data shows.
- Output-scale stabilization (candidate C) remains well-motivated by the
  `reduce_grad_norm` asymmetry (§3.1) and the severity-split narrow-spread
  seeds (§1 Finding 2), independent of the patience question.
- Alternative initialization (candidate B) is the least supported by this
  phase's data — gradients flow from epoch 1 in every seed, so init
  itself doesn't look broken; deprioritize unless patience/output-scale
  changes don't resolve the instability.

## 6. Reproducibility

```bash
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4.yaml --tag diag_primary --seeds 42,43,44,45,46 --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4_severity.yaml --tag diag_severity --seeds 42,43,44,45,46 --diagnostics
```

Writes to new, non-overwriting run directories under
`experiments/qgnn_v4/` with two extra `training_history.csv` columns
(`train_pr_auc`, `quantum_grad_norm_mean/max/min`, `reduce_grad_norm_mean`)
per epoch, on top of everything already saved. Same optimizer, loss,
early stopping, seeds, and splits as the original (non-diagnostic) runs —
verified bit-identical PR-AUC, directly comparable, not a re-tuned rerun.
