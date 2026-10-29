# QGNN-v4 Phase 2d: Seed-45 LayerNorm Reversal Investigation

Diagnostic-only phase — no architecture, hyperparameter, threshold, or
dataset change. Investigates why LayerNorm (Phase 2c) improves or holds
flat severity PR-AUC for seeds 42/43/44/46 but reverses it substantially
for seed 45 (0.385 → 0.290/0.276).

**default.qubit + `diff_method="backprop"` throughout. Nothing here is
attributable to hardware noise.**

---

## 1. Executive Summary

Seed 45's baseline (pre-LayerNorm) quantum circuit happened, during its
own training, to concentrate an unusually strong discriminative signal
into a single PauliZ output channel (point-biserial correlation with the
label = 0.524 — well above any channel in any other seed's baseline
circuit). LayerNorm removes the optimization incentive for a circuit to
produce one large-magnitude dominant channel, since every channel gets
rescaled to comparable magnitude per example regardless of its original
scale. Retrained under LayerNorm, seed 45's circuit loses this advantage
(its best channel drops to correlation 0.279 — roughly halved) while
every other seed's best channel *improves or holds flat* under the same
change (+0.024 to +0.045, or −0.006 for seed 46). This is the only
seed-specific, quantitatively distinctive effect found across eleven
diagnostic angles — gradients, dataset composition, and general training
dynamics are all normal for seed 45 and offer no seed-specific
explanation on their own.

## 2. Reproducibility Check

Verified directly from saved `config.yaml`/`run_metadata.json`/
`encoder_checkpoint.json` for all three seed-45 runs (baseline,
LN-no-affine, LN-affine): identical frozen encoder checkpoint
(`experiments/classical_gnn/20260905T063450Z_hetero_graphsage_severity_seed45`),
identical hyperparameters (epochs=100, lr=0.001, weight_decay=0.0001,
patience=10, class_weighting=balanced), identical seed=45. Test-set
`supplier_id`/`time`/`actual_disruption` rows are byte-identical across
all three runs (`pandas.DataFrame.equals` confirmed `True`), n=11100,
n_positive=817. **The three runs are genuinely comparable — only the
head architecture (LayerNorm present/absent/affine) differs.** A
from-scratch forward pass rebuilding each model from its saved `model.pt`
reproduced the saved `predictions.csv` probabilities with **zero**
numerical difference, confirming the inspection pipeline used for
sections 5–7 below is exact, not an approximation.

## 3. Training Dynamics

| | Baseline | LN-no-affine | LN-affine |
|---|---:|---:|---:|
| Total epochs | 13 | 26 | 26 |
| Best epoch | 3 | 16 | 16 |
| val PR-AUC at epoch 1 | 0.977 | 0.309 | 0.309 |
| val PR-AUC saturates (≥0.999) at epoch | 3 | 12 | 12 |
| Final validation F1 | **0.0645 (constant every epoch)** | 0.821 | 0.889 |
| Final train loss | 1.271 | 0.417 | 0.323 |
| val→test PR-AUC gap | 0.615 | **0.710** | **0.724** |

Baseline's validation PR-AUC saturates almost immediately (epoch 3) while
its validation F1 stays at the exact same value (0.0645) for all 13
epochs — a sign the decision boundary at threshold 0.5 is essentially
static and uninformative even as the ranking metric moves, consistent
with severity's known near-saturated-validation problem (Phase 1).
LayerNorm's validation trajectory is far more gradual and its F1
genuinely climbs (0.13→0.82–0.89) — a real, non-degenerate decision
boundary forms. **Despite this "healthier-looking" validation
trajectory, LayerNorm's validation→test PR-AUC gap for seed 45 (0.71–0.72)
is the largest of any seed under any configuration in this entire
investigation** (§7 confirms this is seed-45-specific, not a general
LayerNorm effect). This is closest to **Category C** (§11), with
elements of **Category D** (§7) — not A, B, or E: gradients are healthy
throughout (§9), ruling out A as the primary driver; F1 improves rather
than degrades, ruling out a pure classification-only story (not E
alone); and the *ranking* itself changes substantially (§5), ruling out
"calibration only."

## 4. Prediction Distribution

| | Baseline | LN-no-affine | LN-affine |
|---|---:|---:|---:|
| Prob mean | 0.5730 | 0.3492 | 0.3028 |
| Prob std | 0.0044 | 0.1137 | 0.1246 |
| Prob min | 0.5679 | 0.2574 | 0.2070 |
| Prob max | 0.5893 | 0.8223 | 0.8273 |

Baseline's predictions for seed 45 are nearly constant (std 0.004, a
0.021-wide range) — a collapsed distribution, consistent with Phase
1/2b's original finding. LayerNorm widens this by **~26–28x** (std
0.114–0.125, range now 0.56–0.57 wide) — by far the largest widening
effect of any severity seed under LayerNorm. The widening itself is not
the problem (§6 shows it is, on its own, the *intended* effect); the
problem is *how* the distribution widens (§5).

## 5. Ranking Analysis

| | Baseline | LN-no-affine | LN-affine |
|---|---:|---:|---:|
| PR-AUC | 0.385 | 0.290 | 0.276 |
| Positive-class mean prob | 0.5765 | 0.4869 | 0.4515 |
| Positive-class std | 0.0079 | 0.2133 | 0.2394 |
| Negative-class mean prob | 0.5728 | 0.3382 | 0.2910 |
| Negative-class std | 0.0038 | 0.0934 | 0.1016 |
| Mean separation (pos − neg) | +0.0038 | **+0.1486** | **+0.1605** |

**The apparent paradox — LayerNorm gives ~40x more raw mean separation
between classes, yet ranking (PR-AUC) gets worse — resolves once
within-class spread is included.** Under LayerNorm, the positive class's
own standard deviation (0.213–0.239) is *larger than the mean separation
itself* (0.149–0.161). A large fraction of positive examples now score
below many negative examples even though the class means moved further
apart — both distributions widened, and the positive class widened more
than the gap between the means grew, so more individual pairs are
mis-ordered than under baseline's tight-but-barely-separated
distributions. **Spearman rank correlation between baseline and
LayerNorm probabilities for seed 45 is 0.33 (no-affine) / 0.36
(affine)** — a real, substantial re-ranking, not a monotonic rescaling
(which would show ρ ≈ 1). This directly answers the "ranking vs.
classification" distinction the investigation asked to keep separate:
**both change, and they change in opposite directions for seed 45** —
F1 improves because the wider spread happens to place more mass on the
correct side of the *fixed 0.5 threshold*, while PR-AUC (threshold-free)
worsens because the wider within-class spread increases mis-ordered
pairs overall.

Seed 45 is not uniquely re-ranked, though — Spearman ρ vs. baseline for
seed 44 under `ln_affine` is 0.337, comparable to seed 45's 0.360, yet
seed 44's PR-AUC *improves* under that same configuration. **Rank change
alone does not distinguish seed 45 from the others** — the next section
identifies what does.

## 6. Quantum Feature Analysis

Raw PauliZ output (6 channels), pre-LayerNorm, baseline-trained circuits:

| Seed | Per-channel std (6 values) | Max &#124;correlation with label&#124; |
|---:|---|---:|
| 42 | 0.103, 0.292, 0.082, 0.069, 0.221, 0.043 | 0.270 |
| 45 | 0.174, 0.026, 0.025, 0.015, 0.099, 0.056 | **0.524** |
| 46 | 0.414, 0.508, 0.074, 0.461, 0.327, 0.397 | 0.415 |

Seed 45's raw circuit output is markedly more uneven than seed 42's or
46's — 3 of its 6 channels have std ≤0.026 (near-constant), while one
channel (the one driving the 0.524 correlation) carries essentially all
the discriminative signal. Seeds 42 and 46 spread their signal more
evenly across channels from the start.

**The decisive number — max &#124;correlation&#124; across all 6 channels, before
vs. after retraining with LayerNorm, all 5 seeds:**

| Seed | Baseline | LN-no-affine | LN-affine | Δ (no-affine) | Δ (affine) |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.270 | 0.314 | 0.315 | +0.044 | +0.045 |
| 43 | 0.360 | 0.401 | 0.402 | +0.041 | +0.042 |
| 44 | 0.387 | 0.411 | 0.424 | +0.024 | +0.037 |
| **45** | **0.524** | **0.282** | **0.279** | **−0.242** | **−0.245** |
| 46 | 0.415 | 0.409 | 0.410 | −0.006 | −0.006 |

**Seed 45 is the only seed where the best-discriminating channel's
signal drops under LayerNorm — and it drops by 5–8x more than any other
seed's (positive or negative) change.** Mechanistic reading: LayerNorm
normalizes every example's 6-channel output to comparable per-example
scale, removing any reward the optimizer had for letting one channel's
raw magnitude dominate the others. Seed 45's baseline circuit had, by a
wide margin, the most to lose from this — it had concentrated
essentially all its signal into one channel; seeds 42/43/44/46 had
already spread signal more evenly (§ table above) and so had comparably
little to lose, or something to gain, from equalization.

## 7. Classical Projection (pre-quantum)

`Linear(128,6) → π·tanh` output, baseline-trained heads:

| Seed | Mean per channel (6 values) | Std per channel (6 values) |
|---:|---|---|
| 42 | 1.66, 0.10, −1.06, 0.09, 2.09, 1.45 | 1.01, 0.68, 0.59, 1.01, 0.88, 0.75 |
| 45 | 1.67, −2.71, −2.52, 1.11, −1.03, −2.58 | 1.16, 1.17, 1.49, 0.67, 1.07, 0.88 |
| 46 | 2.99, 2.67, −0.79, −2.94, 0.01, −2.87 | 0.20, 1.52, 0.12, 0.66, 0.08, 0.84 |

No sign of tanh saturation or collapse specific to seed 45 — its
per-channel means/stds are in the same general range as seed 42's, and
if anything less extreme than seed 46's (which has several channels
pinned near ±3, close to the π boundary, yet seed 46 does *not* show the
LayerNorm reversal). **The pre-quantum representation does not
distinguish seed 45 from the other seeds** — whatever makes seed 45
different emerges specifically from how its quantum circuit's *training*
under each head architecture resolves, not from the classical
projection feeding into it.

## 8. Seed Comparison Summary

| Quantity | 42 | 43 | 44 | 45 | 46 |
|---|---:|---:|---:|---:|---:|
| Baseline max channel &#124;corr&#124; | 0.270 | 0.360 | 0.387 | **0.524** | 0.415 |
| Δ max channel corr under LN | +0.04 | +0.04 | +0.03 | **−0.24** | −0.01 |
| val→test PR-AUC gap, LN-affine | 0.605 | 0.541 | 0.524 | **0.724** | 0.559 |
| Severity PR-AUC, baseline → LN-affine | 0.361→0.396 | 0.374→0.450 | 0.425→0.476 | **0.385→0.276** | 0.412→0.441 |
| Quantum grad norm (mean, LN-affine) | 0.142 | 0.357 | 0.237 | 0.152 | 0.087 |
| Best epoch, baseline → LN-affine | 1→5 | 3→3 | 5→4 | 3→16 | 85→10 |

Seed 45 is the unique outlier on exactly two rows — baseline's
unusually concentrated signal, and the magnitude of its loss under
LayerNorm — and unremarkable on every other row (gradient norms are
mid-range, not extreme; its baseline best-epoch behavior, epoch 3, is
similar to seed 43's).

## 9. Gradient Analysis

`quantum_grad_norm_mean` (mean over training) and range, all 5 seeds,
both configurations:

| Seed | Baseline (mean, range) | LN-affine (mean, range) |
|---:|---|---|
| 42 | 0.173 (0.076–0.224) | 0.142 (0.082–0.354) |
| 43 | 0.179 (0.110–0.224) | 0.357 (0.178–0.641) |
| 44 | 0.231 (0.134–0.287) | 0.237 (0.085–1.059) |
| 45 | 0.142 (0.036–0.212) | 0.152 (0.070–0.306) |
| 46 | 0.179 (0.062–0.277) | 0.087 (0.046–0.273) |

**Seed 45's gradients are unremarkable under both configurations** —
mid-range magnitude, no vanishing (never near zero across any epoch), no
explosion. `reduce_grad_norm_mean` rises substantially under LayerNorm
for *every* seed (e.g. seed 43: 0.57→3.76, the largest rise of any
seed) — this is a general property of adding LayerNorm to the gradient
path, not something specific to seed 45. **No evidence supports a
barren-plateau or vanishing/exploding-gradient explanation for seed 45**
— explicitly ruling out Category A as the primary mechanism, consistent
with Phase 1's own established finding that this circuit shows no
barren-plateau behavior at this scale.

## 10. Dataset/Split Analysis

Confirmed directly: `severity_split.csv` is **seed-independent** — it is
a fixed mapping from time period to train/test, unrelated to model seed.
The severity test set's composition (n=11100, n_positive=817,
n_fresh_onset=84) is **byte-identical across all 5 seeds**, verified from
`phase2b_all_metrics.csv`. **Seed 45 does not see different data — it
sees the identical test examples every other seed sees.** Whatever
differs originates entirely from the seed-45-trained frozen GraphSAGE
encoder's weights and/or the head's own seed-45 random initialization —
not from a different data sample. This directly rules out any reading of
Category D that would require a different underlying distribution; the
correct reading of "data/interaction" here is narrower: the *same* data
interacting differently with a *different, independently-trained*
encoder+circuit combination.

## 11. Failure Category

**Primary: Category C — Representation distortion**, with a specific,
quantitatively demonstrated mechanism (§6): LayerNorm removes the reward
for a circuit to concentrate signal into one dominant channel, and seed
45's baseline circuit — for reasons not further traceable without
retraining under controlled seeds, out of scope for this diagnostic-only
phase — happened to rely on exactly that strategy far more than any
other seed's circuit did.

**Secondary: elements of Category D** (§10, narrow reading) — not
"different data," but a different frozen encoder (itself seeded by 45)
producing a starting representation from which the quantum circuit,
under LayerNorm's altered incentives, converges to a comparatively worse
severity-5 generalizing solution.

**Ruled out:** Category A (optimization instability/gradient pathology —
§9 shows healthy gradients throughout), Category B in the pure overfitting
sense (train loss decreases similarly to other seeds, not anomalously),
Category E alone (ranking genuinely changes, not just calibration — §5),
Category F (a specific, quantifiable, reproducible mechanism was found,
not unexplained stochasticity), Category G (reproducibility fully
confirmed in §2 — no implementation, checkpoint, or evaluation
inconsistency found).

**Confidence: Medium.** The mechanism in §6 (loss of a dominant channel)
is well-evidenced, quantitatively distinctive to seed 45, and consistent
with a documented, sensible interpretation of how LayerNorm changes
optimization incentives. It is not proven causal in the strict sense —
this diagnostic phase did not (and, per its own constraints, should not)
retrain seed 45 under controlled interventions (e.g. forcing the circuit
to preserve a dominant channel) to confirm the mechanism experimentally.
Stated as the strongest evidence-supported explanation available from
existing artifacts, not a certainty.

## 12. Decision

```
INVESTIGATE FURTHER — but this is not a blocker for Phase 3.
```

The mechanism found (§6, §8) is specific, quantifiable, and not
"ordinary stochastic seed sensitivity" — it has an identifiable,
reproducible signature (the channel-correlation collapse) absent in
every other seed, so Category F does not apply and a pure "proceed to
Phase 3" recommendation would understate what was found. At the same
time, nothing here points to an implementation defect (Category G ruled
out) or a problem that would invalidate Phase 3's own planned qubit/
layer/ansatz ablation — if anything, this finding is a *reason* to run
that ablation, since a different qubit count or ansatz changes exactly
the kind of channel-concentration dynamics implicated here. Recommended
path: proceed to Phase 3 as planned, but carry this finding forward as
a specific thing to watch for — check, for each new architecture
variant, whether any individual seed's best-single-channel correlation
collapses disproportionately under whatever output-stage the ablation
uses, rather than assuming Phase 2c's LayerNorm-specific mechanism is a
one-off.

---

## Reproducibility

```bash
python scripts/inspect_seed45_quantum_features.py
```

Loads only already-saved `model.pt` checkpoints (baseline, LN-no-affine,
LN-affine, seeds 42–46, severity split) and re-derives every intermediate
activation via a single forward pass per run — no training. Verified to
reproduce each run's saved `predictions.csv` probabilities with zero
numerical difference. Output: `experiments/qgnn_v4/phase2d_seed45_inspection/*.csv`
(11 files, per-example angle/PauliZ/normed/logit/probability values),
gitignored with every other run artifact; this report and the inspection
script are the committed record.
