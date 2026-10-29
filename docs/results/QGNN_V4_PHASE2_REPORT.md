# QGNN-v4 Phase 2: Training-Stability Investigation (Patience 10 vs. 25)

Phase 2 of the QGNN-v4 stability investigation. Tests whether the
instability found in `QGNN_V4_BENCHMARK.md` and diagnosed in
`QGNN_V4_PHASE1_DIAGNOSTICS.md` is an artifact of early stopping cutting
training short, by changing **only** early-stopping patience (10 → 25)
and re-running the exact same architecture, seeds, and splits.

**default.qubit + `diff_method="backprop"` throughout — an ideal,
noise-free classical simulation. Nothing below is attributable to
hardware noise or decoherence.**

---

## 1. What changed

- `training.early_stopping_patience`: 10 → 25, via a new `--patience`
  CLI override in `scripts/run_qgnn_v4_experiment.py` (mirrors the
  existing `--epochs` override exactly).

## 2. What did NOT change

Architecture (6 qubits, 2 `StronglyEntanglingLayers` layers, `RY`
`AngleEmbedding`, `Linear(128,6)` trainable reduction, `Linear(6,1)`
output), the frozen GraphSAGE-Full encoder, the classical control's
architecture, `max_epochs` (100, unchanged), optimizer, learning rate,
weight decay, class weighting, seeds (42–46), splits (primary + severity),
or threshold policy. No checkpoint was ever selected using the test set —
early stopping and best-checkpoint selection remained validation-PR-AUC-only,
exactly as before.

## 3. Experimental configuration

```bash
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4.yaml --tag phase2_primary --seeds 42,43,44,45,46 --patience 25 --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4_severity.yaml --tag phase2_severity --seeds 42,43,44,45,46 --patience 25 --diagnostics
```

`--diagnostics` used throughout (same instrumented loop as Phase 1's
baseline, verified bit-identical-optimization by test) so Phase 1's
`diag_primary`/`diag_severity` runs (patience=10) serve directly as the
baseline arm — no baseline rerun needed, and the existing baseline
artifacts were never touched. 20 new run directories under
`experiments/qgnn_v4/*_phase2_*`, non-overwriting. Git commit recorded
per-run in each `run_metadata.json` (`model_git_commit`), as for every run
in this project.

## 4. Per-seed results

`best_epoch` under patience=10 vs. patience=25 — the single number that
determines every downstream metric (identical best_epoch ⇒ byte-identical
test PR-AUC, val PR-AUC, probability distribution, Brier, ECE):

| Split | Arm | Seed | best_epoch (p=10) | best_epoch (p=25) | Changed? |
|---|---|---:|---:|---:|:---:|
| primary | classical | 42 | 2 | 2 | no |
| primary | classical | 43 | 8 | 8 | no |
| primary | classical | 44 | 8 | 8 | no |
| primary | classical | 45 | 7 | 7 | no |
| primary | classical | 46 | 22 | 22 | no |
| primary | quantum | 42 | 35 | 35 | no |
| primary | quantum | 43 | 34 | 34 | no |
| primary | quantum | **44** | **22** | **82** | **yes** |
| primary | quantum | 45 | 9 | 9 | no |
| primary | quantum | 46 | 3 | 3 | no |
| severity | classical | 42 | 6 | 6 | no |
| severity | classical | 43 | 1 | 1 | no |
| severity | classical | 44 | 2 | 2 | no |
| severity | classical | 45 | 3 | 3 | no |
| severity | classical | 46 | 4 | 4 | no |
| severity | quantum | 42 | 1 | 1 | no |
| severity | quantum | 43 | 3 | 3 | no |
| severity | quantum | 44 | 5 | 5 | no |
| severity | quantum | 45 | 3 | 3 | no |
| severity | quantum | **46** | **85** | **96** | **yes** |

**18 of 20 runs (all 10 classical, 8 of 10 quantum) select the exact same
checkpoint under patience=25 as under patience=10.** Positive control,
confirmed directly from the saved per-epoch curves: for every one of the
4 "unchanged" primary-quantum seeds, validation PR-AUC **never exceeded**
its patience=10 peak even across the full extra 15–25 epochs patience=25
allowed (e.g. seed 42: val PR-AUC 0.8886 at epoch 35, max 0.8881 across
epochs 36–60). Patience=10 was not cutting these runs off prematurely —
there was genuinely nothing better to find later.

Test PR-AUC, both arms, both patience settings:

| Split | Arm | Seed | Test PR-AUC (p=10) | Test PR-AUC (p=25) | Δ |
|---|---|---:|---:|---:|---:|
| primary | classical | 42–46 | 0.775 / 0.682 / 0.872 / 0.822 / 0.830 | *identical* | 0 |
| primary | quantum | 42 | 0.9615 | 0.9615 | 0 |
| primary | quantum | 43 | 0.5696 | 0.5696 | 0 |
| primary | quantum | **44** | 0.9266 | **0.9136** | **−0.0130** |
| primary | quantum | 45 | 0.8203 | 0.8203 | 0 |
| primary | quantum | 46 | 0.8861 | 0.8861 | 0 |
| severity | classical | 42–46 | 0.431 / 0.462 / 0.424 / 0.417 / 0.467 | *identical* | 0 |
| severity | quantum | 42 | 0.3610 | 0.3610 | 0 |
| severity | quantum | 43 | 0.3744 | 0.3744 | 0 |
| severity | quantum | 44 | 0.4248 | 0.4248 | 0 |
| severity | quantum | 45 | 0.3851 | 0.3851 | 0 |
| severity | quantum | **46** | 0.4123 | **0.4289** | **+0.0166** |

## 5. Aggregate results

| Split | Arm | Patience | Mean | Std | Min | Max |
|---|---|---|---:|---:|---:|---:|
| primary | classical | 10 | 0.7959 | 0.0724 | 0.6819 | 0.8716 |
| primary | classical | 25 | 0.7959 | 0.0724 | 0.6819 | 0.8716 |
| primary | quantum | 10 | 0.8328 | 0.1562 | 0.5696 | 0.9615 |
| primary | quantum | 25 | 0.8302 | 0.1544 | 0.5696 | 0.9615 |
| severity | classical | 10 | 0.4402 | 0.0229 | 0.4167 | 0.4672 |
| severity | classical | 25 | 0.4402 | 0.0229 | 0.4167 | 0.4672 |
| severity | quantum | 10 | 0.3915 | 0.0265 | 0.3610 | 0.4248 |
| severity | quantum | 25 | 0.3948 | 0.0305 | 0.3610 | 0.4289 |

Classical: **exactly unchanged**, both splits (expected — none of its 10
best_epochs moved). Quantum, primary: mean −0.0026 (noise-level),
std −0.0018 (no meaningful stability change). Quantum, severity: mean
+0.0033, but **std increased** (0.0265 → 0.0305) — patience=25 did not
reduce seed-to-seed variance on either split; if anything severity's
variance went up slightly, driven entirely by seed 46 moving while the
other four stayed fixed.

## 6. Baseline vs. Phase 2 comparison — seeds where QGNN "improves"

Using the strict per-seed-delta convention from `QGNN_V4_BENCHMARK.md`:
**1 of 10 quantum seeds improves (severity seed 46, +0.017), 1 of 10
gets worse (primary seed 44, −0.013), 8 of 10 are unchanged.** No seed
that failed under patience=10 (severity's four fast-stopping seeds; primary
seed 43) is fixed by patience=25.

## 7. Calibration comparison

| Split | Metric | Quantum, p=10 | Quantum, p=25 |
|---|---|---:|---:|
| primary | Brier (mean of 5) | 0.1283 | 0.1079 |
| primary | ECE (mean of 5) | 0.2845 | 0.2339 |
| severity | Brier (mean of 5) | 0.2105 | 0.2105 |
| severity | ECE (mean of 5) | 0.3448 | 0.3463 |

Primary's apparent calibration improvement is driven **entirely by the
one seed that changed** (44: Brier 0.134→0.032, ECE 0.327→0.075) — the
other four seeds are byte-identical, so this is not a general patience
effect, it's one seed's result pulling a 5-seed mean. Severity shows no
calibration change at all (four of five seeds identical; the fifth,
seed 46, moved by <0.001 Brier).

## 8. Probability-spread comparison

| Split | prob_std, p=10 (mean of 5) | prob_std, p=25 (mean of 5) |
|---|---:|---:|
| primary | 0.1158 | 0.1497 |
| severity | 0.0604 | 0.0611 |

Same story as §7: primary's apparent widening is seed 44 alone (0.078 →
0.247); the other four primary seeds and all five severity seeds are
unchanged. **The collapsed probability distributions documented in Phase 1
(severity seeds with std as low as 0.004–0.022) are still collapsed under
patience=25** — four of five severity quantum seeds still stop by epoch
1–5, patience=25 included, because validation PR-AUC genuinely plateaus
that early for those seeds (§4's positive control).

## 9. Gradient comparison

For the two seeds that did change, the gradient norm stayed in the same
0.1–0.25 range already reported in Phase 1 throughout the *entire*
extended run — including the additional epochs patience=25 unlocked.
Primary quantum seed 44 (best_epoch 22→82): `quantum_grad_norm_mean`
ranges 0.081–0.237 across epochs 1–91, never collapsing toward zero even
as training continues for 70 more epochs than before — consistent with
Phase 1's finding that gradients are not vanishing, now confirmed over a
longer window for the one seed that actually used it. No new gradient
pathology appears with more patience.

## 10. Training-duration comparison

| Split | Metric | Patience 10 | Patience 25 | Ratio |
|---|---|---:|---:|---:|
| primary | avg epochs (quantum) | 30.6 | 56.2 | 1.84x |
| primary | avg duration, s (quantum) | 18.1 | 33.3 | 1.84x |
| severity | avg epochs (quantum) | 29.4 | 42.4 | 1.44x |
| severity | avg duration, s (quantum) | 12.6 | 18.6 | 1.48x |

Patience=25 costs 1.4–1.8x more wall-clock time on average (all 10
quantum runs combined: 18.1s → 33.3s and 12.6s → 18.6s per run) for a
net effect that is neutral-to-negative on PR-AUC and calibration, once
the one-seed-driven aggregate changes in §5/§7/§8 are read at the
per-seed level rather than the mean.

## 11. Interpretation

**Patience is not the bottleneck.** The central, decisive finding is the
positive control in §4: every "unchanged" seed's validation PR-AUC
genuinely never improved beyond what patience=10 already found, even
given 15–25 additional epochs of search room. This directly rules out
"early stopping is cutting training short before a better optimum" as the
explanation for the instability documented in `QGNN_V4_BENCHMARK.md` and
`QGNN_V4_PHASE1_DIAGNOSTICS.md`. Two seeds (8%) did find a later, marginally
different checkpoint under more patience — one got a modest calibration
improvement at a small PR-AUC cost (primary seed 44), the other got a
small PR-AUC improvement with no calibration change (severity seed 46) —
too inconsistent and too rare to read as a general benefit of more
patience.

Seed 43's specific failure (§1 Finding 3 in Phase 1: fits train/val well,
fails test) is **unaffected by patience** — its best_epoch, and therefore
every downstream metric, is identical at patience=10 and patience=25.
Whatever is driving that seed's generalization gap is not a
training-length issue.

Severity's near-immediate early stopping (4/5 seeds, best_epoch 1–5) also
survives patience=25 unchanged. This is not evidence of insufficient
patience — the positive control shows validation PR-AUC is genuinely
already at its ceiling by epoch 1–5 for these seeds (severity validation
is near-saturated, PR-AUC 0.97–1.0, as already established in
`QGNN_V4_BENCHMARK.md` §6). More patience cannot fix a validation signal
that has nothing further to reward.

## 12. Should longer patience become the new default?

**No.** 18 of 20 runs are byte-identical; the 2 that changed show mixed,
small, seed-specific effects (one better calibration/worse PR-AUC, one
better PR-AUC/same calibration); aggregate PR-AUC and variance are
unchanged within noise on both splits; training cost increases 1.4–1.8x
for this negligible-to-mixed benefit. Recommend reverting to patience=10
as the default for any further QGNN-v4 work — patience=25 buys
measurably more compute for no measurable, general improvement.

## 13. Recommended next experiment

Per the investigation's own pre-specified decision rule ("if it doesn't
help → move to output-scale/calibration investigation"): **proceed to the
output-scale/calibration candidates from Phase 2's original list**
(candidate C — output-scale stabilization — and, if warranted after that,
BatchNorm/normalization around the head), not the qubit/layer/ansatz
ablation grid (Phase 3), which stays correctly deferred per the
investigation's own sequencing (§9 of the original Phase 2 request: "only
after this move to qubit/layer/ansatz ablation"). Phase 1 (§6, gradient
findings) already deprioritized alternative initialization as a candidate
— gradients flow normally from epoch 1 in every seed, in this phase's
data too — so output-scale stabilization is the best-motivated next
change to test, still without altering qubits, layers, ansatz, encoding,
or the frozen encoder.

---

## Answers to the 10 scientific questions

1. **Does increasing patience allow the QNN to train longer?** Yes,
   mechanically (1.4–1.8x more epochs/wall-clock on average) — but only 2
   of 10 quantum runs actually found a different, later checkpoint as a
   result.
2. **Does it reduce the collapsed probability distributions?** No, in
   4 of 5 severity seeds — the ones with the most severely collapsed
   distributions in Phase 1 are still collapsed under patience=25 (§8).
3. **Does ECE improve?** Only in the one primary seed that changed (44);
   unchanged everywhere else, including all of severity.
4. **Does Brier score improve?** Same pattern as ECE — one-seed effect
   on primary, no effect on severity.
5. **Does QNN test PR-AUC improve?** No, in aggregate (primary: −0.0026;
   severity: +0.0033, both within noise); per-seed, 1 improves, 1
   worsens, 8 unchanged.
6. **Does seed-to-seed variance decrease?** No — primary std essentially
   flat (0.1562→0.1544), severity std slightly **increases**
   (0.0265→0.0305).
7. **Does the severity split improve?** No, materially — aggregate mean
   moves +0.0033 (noise), driven by one seed; the split's core problem
   (early, saturated-validation stopping) is unaffected.
8. **Does seed 43 improve?** No — byte-identical best_epoch, test PR-AUC
   unchanged at 0.5696.
9. **Does longer training simply improve calibration/output spread
   without improving ranking?** Sometimes, for the one seed where it
   applies (primary 44: calibration much better, PR-AUC slightly worse)
   — but this is a single-seed effect, not a general pattern; severity
   shows no calibration/spread change at all despite one seed's best_epoch
   moving.
10. **Does the classical control change in the same way?** No change
    whatsoever — all 10 classical best_epochs are identical between
    patience=10 and patience=25, confirming classical was already finding
    its true optimum well within patience=10's window on every seed,
    both splits.
