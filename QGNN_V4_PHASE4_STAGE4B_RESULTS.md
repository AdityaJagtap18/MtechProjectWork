# QGNN-v4 Phase 4 — Stage 4b Results: Initialization Scale + Severity Investigation

## 1. Objective

Stage 4's 2-seed pilot left four open questions this stage answers:
**(A)** Does reducing the initial quantum parameter scale (not "Gaussian
vs. default" as a family) explain F2's primary improvement? **(B)** Why
does F2 improve primary while degrading severity? **(C)** Is there an
intermediate Gaussian std that keeps the primary gain without the
severity collapse? **(D)** Does F3's variance reduction survive all 5
fixed seeds?

## 2. Stage 4 findings motivating this stage

- F2 (small_gaussian, std=0.01): primary 0.774→0.828 (better), severity
  0.422→0.265 (much worse).
- F3 (identity_like): primary std 0.094→0.017 (an apparent ~82% cut, 2
  seeds only), primary mean 0.774→0.726 (modestly worse).
- F1 starts at a large quantum-parameter L2 norm (~20.8–22.2); F2/F3
  start near zero and stay far smaller after training.

## 3. Exact repository changes

`circuit.py`: added `_gaussian_init(std)` (a configurable-std Gaussian,
generalizing Stage 4's fixed-std `_small_gaussian_init`, kept unchanged
for byte-for-byte backward compatibility) and extended
`resolve_quantum_init(quantum_init, gaussian_std=None)` to accept
`quantum_init="gaussian"` + a required `gaussian_std`. `HybridQuantumHeadLayerNorm`
gained a matching `gaussian_std` constructor arg, threaded to
`build_quantum_layer`, and reported in `quantum_resource_summary()`.
`run_qgnn_v4_experiment.py` gained `--gaussian-std` and extended
`--quantum-init`'s choices with `"gaussian"`. **No circuit topology,
qubit count, layer count, ansatz, projection, loss, optimizer, or
threshold changes** — verified automatically: every Stage 4b run reports
`total_trainable_parameters=817`, identical to Stage 4. `qgnn.py`/
`qgnn_v2.py`/`qgnn_v2_reupload.py`/`qgnn_v3.py` untouched.

11 new unit tests (gaussian_std required/rejected-when-inapplicable,
scale sweep produces the requested magnitude, `gaussian std=0.01`
matches `small_gaussian`'s own distribution exactly by seeded output,
parameter-count invariance, forward/backward, resource-summary
reporting). **317/317 tests passed, run both before and after the code
change** — zero regressions (306 pre-existing + 11 new).

## 4. Experimental configurations

| Config | Description | Source |
|---|---|---|
| G0_default | uniform[0,2π] (Stage 4 F1) | **Reused** — verified valid (817 params, 6q/2L, correct dataset/seed/split) before reuse |
| G1 | gaussian std=0.001 | New |
| G2 | gaussian std=0.005 | New |
| G3 | gaussian std=0.010 (Stage 4 F2) | **Reused** — verified valid |
| G4 | gaussian std=0.025 | New |
| G5 | gaussian std=0.050 | New |
| F3 (5-seed) | identity_like | Seeds 42/43 **reused** (verified valid); seeds 44/45/46 **new** |

**Part A (F3 5-seed expansion)**: 6 new runs (seeds 44/45/46 × 2 splits).
**Part B/C (scale-sweep pilot, seeds 42/43 only)**: 16 new runs (G1/G2/G4/G5
× 2 seeds × 2 splits) + G0/G3 reused (8 runs, 0 new). **Total: 22 new
training runs, 12 reused, 34 (config, split, seed) rows analyzed.**

## 5. Experimental protocol

Identical frozen encoder checkpoints per seed, dataset, splits, projection
(`Linear(128,6)`, unchanged), 6 qubits, 2 layers, StronglyEntanglingLayers,
`default.qubit`, `backprop`, Adam (lr=0.001, weight_decay=0.0001),
max_epochs=100, patience=10, balanced class weighting, threshold=fixed@0.5,
`--diagnostics` throughout, `--quantum-only`. Only `--quantum-init`/
`--gaussian-std` varied. No implementation limitation was hit that forced
any other variable to change.

## 6. Initialization statistics

Confirmed (`stage4b_initialization_stats.csv`) every scale produces its
requested magnitude at construction: G1 initial std≈0.00097 (seed 42),
G3 (=F2) initial std≈0.0097, matching their nominal 0.001/0.010 targets
closely. **Key finding (answers Question A directly)**: across G1–G5
(std 0.001→0.050), seed 42's primary PR-AUC is nearly IDENTICAL
(0.9117, 0.9117, 0.9117, 0.9114, 0.9097) and seed 43's is close (0.742,
0.741, 0.745, 0.714, 0.716) — **the exact small std barely matters for
primary; what matters is being near zero at all, vs. G0's large/spread
uniform start.** This is a clean, seed-paired result across all 5 scales
tested, not a 2-point comparison.

## 7. Primary results (test, threshold=0.5, mean±std)

| Config | PR-AUC | ROC-AUC | F1 | Precision | Recall | Specificity | Bal.Acc | MCC | Brier | ECE | n_seeds |
|---|---|---|---|---|---|---|---|---|---|---|---:|
| G0 default | 0.774±.094 | 0.977 | 0.637 | 0.511 | 0.857 | 0.941 | 0.899 | 0.631 | 0.070 | 0.193 | 2 |
| G1 std=0.001 | 0.827±.085 | 0.981 | 0.772 | 0.701 | 0.899 | 0.964 | 0.931 | 0.767 | 0.049 | 0.145 | 2 |
| G2 std=0.005 | 0.827±.085 | 0.980 | 0.772 | 0.701 | 0.899 | 0.964 | 0.931 | 0.767 | 0.049 | 0.145 | 2 |
| G3 std=0.010 | 0.828±.083 | 0.980 | 0.772 | 0.701 | 0.899 | 0.964 | 0.931 | 0.767 | 0.049 | 0.145 | 2 |
| G4 std=0.025 | 0.812±.099 | 0.980 | 0.770 | 0.700 | 0.895 | 0.964 | 0.929 | 0.764 | 0.049 | 0.141 | 2 |
| G5 std=0.050 | 0.813±.097 | 0.979 | 0.769 | 0.697 | 0.899 | 0.963 | 0.931 | 0.764 | 0.050 | 0.142 | 2 |
| F3 (5-seed) | 0.777±.059 | 0.982 | 0.687 | 0.535 | 0.960 | 0.939 | 0.950 | 0.692 | 0.070 | 0.169 | **5** |

Every near-zero-start configuration (G1–G5, F3) beats G0's Brier and ECE
on primary — a consistent calibration improvement, not scale-dependent.

## 8. Severity results (test, threshold=0.5, mean±std)

| Config | PR-AUC | ROC-AUC | F1 | Precision | Recall | Specificity | Bal.Acc | MCC | Brier | ECE | n_seeds |
|---|---|---|---|---|---|---|---|---|---|---|---:|
| G0 default | 0.422±.028 | 0.667 | 0.456 | 0.592 | 0.414 | 0.962 | 0.688 | 0.440 | 0.139 | 0.287 | 2 |
| G1 std=0.001 | 0.263±.063 | 0.645 | 0.398 | 0.364 | 0.439 | 0.939 | 0.689 | 0.347 | 0.085 | 0.144 | 2 |
| G2 std=0.005 | 0.265±.059 | 0.652 | 0.399 | 0.366 | 0.439 | 0.939 | 0.689 | 0.348 | 0.085 | 0.144 | 2 |
| G3 std=0.010 | 0.265±.057 | 0.653 | 0.400 | 0.367 | 0.440 | 0.939 | 0.690 | 0.349 | 0.084 | 0.144 | 2 |
| G4 std=0.025 | 0.277±.037 | 0.677 | 0.403 | 0.372 | 0.440 | 0.941 | 0.690 | 0.353 | 0.084 | 0.144 | 2 |
| G5 std=0.050 | 0.386±.074 | 0.612 | 0.414 | 0.392 | 0.440 | 0.945 | 0.692 | 0.365 | 0.081 | 0.142 | 2 |
| F3 (5-seed) | 0.386±.112 | 0.654 | 0.367 | 0.328 | 0.454 | 0.902 | 0.678 | 0.314 | 0.129 | 0.234 | **5** |

Every near-zero-start configuration again beats G0's Brier/ECE — even
where PR-AUC is much worse. This decoupling (better calibration, worse
ranking) recurs on both splits and is a real pattern, not noise.

## 9. F3 5-seed stability analysis (answers Question D)

| Metric | F3 2-seed pilot (Stage 4) | F3 5-seed (this stage) | QGNN-v4 5-seed reference (F1-equivalent) |
|---|---|---|---|
| Primary PR-AUC mean±std | 0.726±0.017 | **0.777±0.059** | 0.811±0.074 |
| Primary PR-AUC range | 0.035 | 0.171 | — |
| Severity PR-AUC mean±std | 0.380±0.102 | **0.386±0.112** | 0.398±0.057 |
| Severity PR-AUC range | 0.204 | 0.262 | — |

Per-seed primary: 0.744(42), 0.709(43), 0.799(44), **0.880(45)**, 0.751(46).
Per-seed severity: 0.278(42), 0.481(43), 0.225(44), 0.487(45), 0.458(46).

**Question D's answer: NO, the 2-seed variance reduction did not
survive.** Primary std more than tripled (0.017→0.059) once seeds
44/45/46 were added — still somewhat tighter than the 5-seed reference's
0.074, but nowhere near as dramatic as the pilot suggested. Severity std
got **worse** than the reference (0.112 vs. 0.057) — F3 is now the LEAST
stable configuration on severity of anything in this report. This is
itself the most important, if unglamorous, finding of Stage 4b: **a
2-seed variance estimate is not reliable enough to certify a
reproducibility claim**, exactly the caution this stage's own protocol
was designed to test for.

## 10. Gaussian initialization-scale analysis (answers Questions A & C)

**Question A — confirmed.** §6's per-seed comparison shows primary
PR-AUC is essentially flat across std 0.001→0.050 (seed 42: 0.910–0.912;
seed 43: 0.714–0.745) — the transition from G0 (large/uniform) to
*any* small-Gaussian scale is what drives the primary improvement, not
the specific std value.

**Question C — suggestive, not confirmed.** Severity PR-AUC shows a
graded pattern: 0.263(G1)→0.265(G2)→0.265(G3)→0.277(G4)→**0.386(G5)** —
std=0.050 recovers substantially toward G0's 0.422. However, per-seed
inspection shows this is driven almost entirely by **one seed**: seed 42's
severity PR-AUC is nearly flat across the whole sweep (0.325→0.324→
0.322→0.314→0.313 — if anything, slightly *decreasing*), while seed 43's
jumps sharply between G4 (0.240) and G5 (0.460). A single seed's sharp
transition is not enough to confirm a real "sweet spot" — it is a
plausible lead (worth testing std=0.05–0.10 with more seeds), not an
established result. **Do not read G5 as a solved trade-off on this
evidence alone.**

## 11. F2 severity investigation (Part E, answers Question B)

Compared G0 vs. G3 (=F2) on seeds 42/43, both splits, across the 8
candidate explanations:

1. **Poor validation performance — not supported.** Both configs reach
   very high severity validation PR-AUC (G0: 1.000/0.991; G3: 0.974/0.870)
   — validation itself looks excellent in both cases.
2. **Validation→test generalization gap — supported, moderately.** Both
   configs show large severity gaps (G0: 0.606/0.542; G3: 0.652/0.662) —
   G3's are somewhat larger, consistent with more severe overfitting to
   the in-distribution validation set, but the gap is already large for
   the default too — this is a contributing factor, not the sole cause.
3. **Early stopping — not supported.** Both stop normally (best_epoch
   3–9, no premature or unusually late stops specific to G3).
4. **Different score distributions — supported.** G3's severity
   predictions are shifted toward lower probabilities overall
   (prob_mean 0.21–0.22 vs. G0's 0.30–0.40) with a much higher positive-class
   std (0.25–0.29 vs. G0's 0.13–0.18) — G3 is far less consistent at
   scoring true positives uniformly high.
5. **Poor ranking despite reasonable classification metrics — supported,
   and the clearest single finding.** G3's severity Brier (0.084) and
   ECE (0.144) are BOTH numerically better than G0's (0.139/0.287) — by
   standard calibration metrics G3 looks *better* — yet its PR-AUC
   (0.265) is far worse than G0's (0.422). This decoupling directly shows
   the failure is ranking-specific (can't consistently separate
   positives from negatives across the full score range), not a general
   prediction-quality problem.
6. **Unusual quantum parameter movement — supported, and mechanistically
   suggestive.** G3's quantum parameters move ~5× further from their
   start on PRIMARY (initial L2 norm ≈0.06 → final ≈0.30) than on
   SEVERITY (≈0.06 → ≈0.08). G0's parameters barely move on either split
   (≈1–2% change, starting from a much larger norm ≈21). This suggests
   the near-zero start specifically enables a large, primary-beneficial
   adaptation that the severity objective's loss landscape doesn't offer
   or reward in the same way.
7. **Different quantum feature (PauliZ) distributions — not
   investigated.** Would require a dedicated forward-hook probe (as
   Phase 2d's `inspect_seed45_quantum_features.py` did) that this stage
   did not build — flagged as a gap, not silently assumed either way.
8. **Calibration changes — supported** (see point 5) — this is real, not
   an artifact, and coexists with the ranking failure rather than
   explaining it away.

**Best-supported synthesis**: F2's severity collapse looks like a
**ranking-specific overfitting-to-validation failure** (points 2, 4, 5),
plausibly enabled by the near-zero start allowing a larger,
primary-tuned adaptation that doesn't transfer (point 6) — not a
collapse in general prediction quality (calibration is fine or better)
and not an optimization-stability artifact (no early-stopping or
gradient anomalies found).

## 12. Score-distribution analysis (Part 9)

Full table: `stage4b_score_distributions.csv`. Summary: G0's positive/
negative separation is more consistent across seeds (primary: 0.61/0.29;
severity: 0.13/0.09) than G3's (primary: 0.43/0.63; severity: 0.22/0.19)
or F3's (primary: 0.37–0.78, severity: 0.09–0.33) — both alternatives
show much larger seed-to-seed swings in how well-separated positive and
negative scores are, especially on severity, consistent with §9's
finding that neither alternative is more reproducible than default on
that split.

## 13. Training dynamics

All 34 runs stopped via early stopping in a normal range (best_epoch
1–89 across the full report, most under 30); no run failed to converge
or hit the 100-epoch ceiling in a way suggestive of pathology. No
gradient explosion or vanishing in any configuration.

## 14. Gradient analysis

Consistent with Stage 4: G0's quantum gradient norms run higher
(0.135–0.372) than the near-zero-start configs (0.058–0.137 across
G1–G5 and F3) in every case checked. **Per this stage's own explicit
instruction, this is reported as an optimization characteristic (larger
gradients when starting from a region of larger trigonometric-derivative
magnitude), not evidence of a barren plateau** — every configuration's
gradients stayed well above zero and finite throughout.

## 15. Comparison with Classical GraphSAGE

Classical: primary 0.807±0.066, severity 0.449±0.013. G1–G5's primary
means (0.812–0.828) all edge past classical's primary mean in this
pilot; every configuration's severity mean stays below classical's,
G5 (0.386) and F3 (0.386) coming closest.

## 16. Comparison with QGNN-v4 reference

Reference (5-seed): primary 0.811±0.074, severity 0.398±0.057. F3's
5-seed primary mean (0.777) sits modestly below the reference; its std
(0.059) is somewhat tighter but not dramatically so. No Stage 4b
configuration matches the reference on both mean and stability
simultaneously.

## 17. Limitations

- **Scale-sweep pilot is 2 seeds.** The std=0.05 severity "recovery"
  (§10) is visibly driven by one of those two seeds — treat as a lead,
  not a finding, until re-tested with more seeds.
- **F3's 5-seed result is itself only 5 seeds** — the standard caveat
  applies to every number in this report, same as every prior phase.
- **Severity's known limitation is unchanged**: one underlying
  severity-5 event, same as every phase since Stage 1.
- **Quantum feature (PauliZ) distributions were not directly probed**
  (§11, point 7) — a real gap in the F2 severity investigation, not
  claimed to be ruled in or out.
- **"Identity-like" remains exact for rotation gates only**, not the
  full circuit (Stage 4's own documented limitation, unchanged here).

## 18. KEEP / DROP / INVESTIGATE decisions

| Config | Verdict | Reasoning |
|---|---|---|
| G0 default | **KEEP as control** | Unchanged reference point. |
| G1–G4 (std 0.001–0.025) | **DROP for general use** | Same primary gain as G3 with no severity advantage over it — G3 already represents this whole cluster; no reason to prefer a different std in 0.001–0.025 specifically. |
| G5 (std 0.050) | **INVESTIGATE** | The one config showing a partial severity recovery alongside most of the primary gain — but driven by a single pilot seed (§10); worth a focused, more-seeded follow-up before any stronger claim, not worth a full 5-seed expansion yet on this evidence alone. |
| F3 identity-like | **DROP (revised from Stage 4's INVESTIGATE)** | Its sole distinguishing rationale — reduced primary variance — did not survive 5-seed testing (std rose from 0.017 to 0.059, severity std got worse than the reference). Mean primary performance (0.777) is also below the reference (0.811). No remaining reason to prefer it over the default. |
| Gaussian family as a whole (near-zero start) | **INVESTIGATE** | A real, reproducible primary improvement + calibration improvement exists across the whole near-zero regime — genuinely interesting — but every scale tested trades away meaningful severity performance, and the one hint of a better trade-off (G5) isn't confirmed. |

## 19. Recommended next experiment

Initialization has now been investigated in real depth across two
stages: a genuine primary/calibration benefit exists at near-zero start,
a real severity cost exists at every scale from 0.001–0.025, a possible
(not confirmed) partial recovery at 0.050, and F3's stability promise
did not hold at 5 seeds. Per the master decision tree's own guidance
("if initialization changes do not produce meaningful improvements,
stop spending experimental budget on initialization and move to the
next controlled factor") — this stage does not find a clean win, so the
recommendation is to **move to Stage 5 (controlled ansatz comparison,
already implemented from Phase 3: 4q+2L and 6q+2L for both
`hardware_efficient_ring` and `reduced_entanglement`)** rather than
continue narrowing the initialization search. A narrow, low-cost
alternative — a 3-seed check of G5 specifically on severity, to see if
§10's single-seed pattern replicates — remains available if you'd rather
close that thread first. Not run automatically either way.
