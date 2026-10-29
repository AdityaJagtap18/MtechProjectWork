# QGNN-v4 Phase 4 — Stage 5 Results: Controlled Quantum Ansatz Comparison

## 1. Objective

Determine whether the quantum circuit's **ansatz/entanglement structure**
— holding qubit count, projection, initialization, optimizer, loss, and
evaluation protocol fixed — is a meaningful source of the QGNN's seed
sensitivity, primary/severity behavior, or representation characteristics.
This stage asks *whether circuit structure matters*, not *which ansatz is
best*.

## 2. Research motivation

Phase 3 built two alternative ansätze (`hardware_efficient_ring`,
`reduced_entanglement`) but never ran them to completion (the batch was
killed mid-run at the user's request). Stage 4/4b investigated
initialization and found it to be a real but trade-off-laden factor.
Stage 5 is the natural next controlled-factor experiment per Stage 4b's
own recommendation, isolating architecture from everything already
ruled in/out.

## 3. Existing QGNN and Classical references (unchanged, not re-derived)

| | Primary PR-AUC | Severity PR-AUC |
|---|---|---|
| Classical GraphSAGE-Full | 0.807 ± 0.066 | 0.449 ± 0.013 |
| QGNN-v4 reference (= A0 below) | 0.811 ± 0.074 | 0.398 ± 0.057 |

## 4. Ansatz implementations (verified against the actual code, not assumed)

Read `circuit.py` directly before running anything:

- **A0 — StronglyEntanglingLayers**: PennyLane's own template. 3 rotation
  params/qubit/layer (`Rot(φ,θ,ω)`), PennyLane's own built-in
  range-parameterized entangling CNOT pattern.
- **A1 — hardware_efficient_ring**: `RY` then `RZ` per qubit per layer (2
  params/qubit/layer), followed by a CNOT **ring** (6 edges, wraparound
  included: `_ring_pairs`).
- **A2 — reduced_entanglement**: `RY` only per qubit per layer (1
  param/qubit/layer), followed by a CNOT **chain** (5 edges, no
  wraparound: `_chain_pairs`).

No code changes were needed — all three were already correctly
implemented from Phase 3; this stage only verifies (via smoke tests
against the real dataset) and runs them to completion.

## 5. Circuit/resource comparison — verified, not assumed

| Ansatz | Qubits | Layers | Quantum Params | Total Head Params | Single-qubit gates | Entanglement |
|---|---:|---:|---:|---:|---|---|
| A0 StronglyEntangling | 6 | 2 | **36** | **817** | `Rot` (3 params) | PennyLane's built-in range pattern |
| A1 Hardware Ring | 6 | 2 | **24** | **805** | `RY`+`RZ` (2 params) | CNOT ring (6 edges) |
| A2 Reduced Entanglement | 6 | 2 | **12** | **793** | `RY` (1 param) | CNOT chain (5 edges) |

**Explicit parameter-count caveat (Section 5's own requirement)**: this
is NOT a parameter-matched comparison. A0 has 3× A2's quantum parameters
and 1.5× A1's. Any difference found below could reflect capacity as well
as topology — addressed directly in §19.

## 6. Exact experimental protocol

Frozen GraphSAGE encoder (per seed), dataset, splits, `Linear(128,6)`
projection (unchanged, no PCA/nonlinear), `LayerNorm(6, no affine)`,
Adam (lr=0.001, weight_decay=0.0001), max_epochs=100, patience=10,
balanced class weighting, threshold=fixed@0.5, `default.qubit`,
`diff_method="backprop"`, **default initialization only** (uniform[0,2π]
— Stage 4/4b's alternative initializations deliberately excluded per this
stage's own isolation rule), 6 qubits, 2 layers throughout. Seeds
42,43,44,45,46, both splits.

**A0 was not rerun.** Its 10 runs are Phase 2c/3's own reference
(`phase2c_layernorm_noaffine`) — verified valid before reuse (817 params,
6q/2L, correct dataset/seed/split for all 10; aggregated numbers
reproduce the established 0.811±0.074 / 0.398±0.057 reference exactly).
**A1 and A2 are newly trained**: 10 runs each (5 seeds × 2 splits), smoke-tested
first (1 seed, 3 epochs, confirmed correct param counts 805/793 against
the real dataset) before the full batch. **20 new training runs, 10
reused, 30 total (ansatz, split, seed) rows analyzed.**

## 7. Primary results (test, threshold=0.5, mean±std, 5 seeds)

| Ansatz | PR-AUC | ROC-AUC | F1 | Precision | Recall | Specificity | Bal.Acc | MCC | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 StronglyEntangling | 0.811±.074 | 0.977±.008 | 0.657 | 0.519 | 0.898 | 0.940 | 0.919 | 0.654±.052 | 0.065 | 0.182 |
| A1 Hardware Ring | 0.807±.147 | 0.983±.014 | 0.674 | 0.518 | **0.974** | 0.933 | 0.954 | 0.683±.038 | 0.065 | 0.180 |
| A2 Reduced Entanglement | 0.814±.105 | 0.984±.012 | 0.635 | 0.492 | 0.902 | 0.934 | 0.918 | 0.636±.098 | 0.064 | 0.180 |

## 8. Severity results (test, threshold=0.5, mean±std, 5 seeds)

| Ansatz | PR-AUC | ROC-AUC | F1 | Precision | Recall | Specificity | Bal.Acc | MCC | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 StronglyEntangling | 0.398±.057 | 0.657±.050 | 0.419 | 0.461 | 0.417 | 0.951 | 0.684 | 0.382±.089 | 0.122 | 0.241 |
| A1 Hardware Ring | 0.354±.055 | 0.673±.047 | 0.424 | 0.410 | 0.441 | 0.949 | 0.695 | 0.377±.024 | 0.092 | 0.158 |
| A2 Reduced Entanglement | 0.360±.057 | 0.671±.060 | 0.314 | 0.262 | 0.484 | **0.828** | 0.656 | **0.254±.110** | 0.146 | 0.245 |

## 9. Per-seed results

| Ansatz | Split | Seed | PR-AUC | ROC-AUC | MCC | Specificity |
|---|---|---:|---:|---:|---:|---:|
| A0 | primary | 42 | 0.868 | 0.990 | 0.690 | 0.934 |
| A0 | primary | 43 | 0.681 | 0.965 | 0.572 | 0.948 |
| A0 | primary | 44 | 0.820 | 0.981 | 0.700 | 0.945 |
| A0 | primary | 45 | 0.795 | 0.976 | 0.613 | 0.929 |
| A0 | primary | 46 | 0.892 | 0.975 | 0.695 | 0.943 |
| A1 | primary | 42 | **0.963** | 0.997 | 0.725 | 0.943 |
| A1 | primary | 43 | 0.642 | 0.957 | 0.634 | 0.918 |
| A1 | primary | 44 | 0.894 | 0.992 | 0.692 | 0.939 |
| A1 | primary | 45 | 0.614 | 0.977 | 0.644 | 0.913 |
| A1 | primary | 46 | 0.919 | 0.991 | 0.722 | 0.951 |
| A2 | primary | 42 | 0.890 | 0.992 | 0.675 | 0.926 |
| A2 | primary | 43 | 0.653 | 0.960 | 0.442 | 0.942 |
| A2 | primary | 44 | 0.886 | 0.988 | 0.703 | 0.944 |
| A2 | primary | 45 | 0.725 | 0.987 | 0.685 | 0.929 |
| A2 | primary | 46 | 0.915 | 0.993 | 0.675 | 0.929 |
| A0 | severity | 42 | 0.394 | 0.693 | 0.326 | 0.930 |
| A0 | severity | 43 | 0.450 | 0.640 | 0.554 | 0.995 |
| A0 | severity | 44 | 0.416 | 0.687 | 0.378 | 0.950 |
| A0 | severity | 45 | 0.290 | 0.696 | 0.311 | 0.945 |
| A0 | severity | 46 | 0.439 | 0.566 | 0.343 | 0.937 |
| A1 | severity | 42 | 0.395 | 0.662 | 0.348 | 0.939 |
| A1 | severity | 43 | 0.405 | 0.671 | 0.386 | 0.952 |
| A1 | severity | 44 | 0.321 | 0.614 | 0.417 | 0.961 |
| A1 | severity | 45 | 0.261 | 0.660 | 0.358 | 0.943 |
| A1 | severity | 46 | 0.387 | 0.759 | 0.374 | 0.949 |
| A2 | severity | 42 | 0.355 | 0.591 | 0.144 | **0.788** |
| A2 | severity | 43 | 0.318 | 0.691 | 0.106 | **0.548** |
| A2 | severity | 44 | 0.427 | 0.766 | 0.356 | 0.942 |
| A2 | severity | 45 | 0.278 | 0.627 | 0.374 | 0.949 |
| A2 | severity | 46 | 0.419 | 0.681 | 0.290 | 0.912 |

**A2's severity specificity collapses on 2 of 5 seeds** (42: 0.788, 43:
0.548, vs. 0.912–0.949 on the other three) — a real, seed-specific
instability, not a uniform shift, and the direct cause of A2's much lower
mean severity MCC (0.254 vs. A0's 0.382).

## 10. Seed-stability analysis

| Ansatz | Primary PR-AUC std | Primary range | Severity PR-AUC std | Severity range | Primary MCC std | Severity MCC std |
|---|---:|---:|---:|---:|---:|---:|
| A0 | **0.074** | 0.212 | 0.057 | 0.160 | **0.052** | 0.089 |
| A1 | 0.147 | 0.349 | 0.055 | 0.144 | 0.038 | **0.024** |
| A2 | 0.105 | 0.261 | 0.057 | 0.149 | 0.098 | 0.110 |

**A0 (the existing reference) is the most seed-stable on primary PR-AUC
by a clear margin** — A1's std is roughly double A0's, driven by a
bimodal per-seed split (§9: seeds 42/44/46 ≈0.89–0.96, seeds 43/45
≈0.61–0.64). On severity, PR-AUC std is nearly identical across all
three (0.055–0.057) — ansatz choice does not appear to affect severity
PR-AUC *variance*, only (for A2) a specific failure mode in specificity/MCC.
A1 has the tightest severity MCC std (0.024); A2 the widest on both
primary and severity MCC.

## 11. Quantum resource analysis

See §5's table. A0's larger parameter count (36 vs. 24/12) does not
translate into a mean-PR-AUC advantage on either split — all three
ansätze land within ~1% of each other on primary mean and ~4% on
severity mean — but A0's smaller *variance* on primary coincides with its
larger parameter count, a pattern worth flagging rather than asserting a
causal "more parameters → more stable" claim from n=3 architectures.

## 12. Quantum feature analysis

Six PauliZ channels (pre-LayerNorm) inspected via a forward-pass hook on
every saved model (`stage5_quantum_features.csv`, 180 rows: 3 ansätze ×
2 splits × 5 seeds × 6 channels).

| Ansatz | Split | Mean max\|channel↔target corr\| | Mean max\|cross-channel corr\| | Mean per-channel std |
|---|---|---:|---:|---:|
| A0 | primary | 0.654 | 0.785 | 0.086 |
| A1 | primary | 0.632 | 0.773 | 0.120 |
| A2 | primary | 0.720 | **0.931** | **0.282** |
| A0 | severity | 0.363 | 0.848 | 0.107 |
| A1 | severity | 0.370 | 0.882 | 0.114 |
| A2 | severity | 0.370 | 0.928 | 0.255 |

**A2's channels are simultaneously more variable (highest per-channel
std) and more mutually redundant (highest cross-channel correlation) than
A0/A1's.** This is not contradictory: RY-only rotations (1 free parameter
vs. A0's 3, A1's 2) combined with chain entanglement give each qubit less
independent freedom, so its 6 channels move together more even as their
individual range grows. The practical consequence is that A2's nominal
6-channel output carries closer to the information content of a smaller
effective number of channels — consistent with (though not proof of) a
capacity/topology interaction, not just fewer raw parameters.

## 13. Representation/collapse analysis

No ansatz produced a literally near-constant channel (minimum per-channel
std across all 180 rows was 0.007, not vanishing) — so no outright dead
channel was found for any ansatz. What *was* found is A2's high
cross-channel correlation (§12), a milder, graded version of the "one
dominant channel" pattern Phase 2d documented for seed 45 under the
LayerNorm architecture — here it's a property of the ansatz itself
(present across all 5 seeds for A2, not seed-specific), rather than a
seed-specific failure. **This is a different mechanism than Phase 2d's
finding, not a replication of it** — stated explicitly since Section 19
warned not to assume the same mechanism recurs.

## 14. Training dynamics

| Ansatz | Split | Mean quantum grad norm | Mean best_epoch | Mean total epochs |
|---|---|---:|---:|---:|
| A0 | primary | 0.139±0.060 | 15.2 | 25.2 |
| A1 | primary | 0.175±0.064 | 10.8 | 20.8 |
| A2 | primary | 0.119±0.057 | 14.0 | 24.0 |
| A0 | severity | 0.197±0.110 | 7.6 | 17.6 |
| A1 | severity | 0.209±0.066 | 15.0 | 25.0 |
| A2 | severity | 0.226±0.115 | 9.0 | 19.0 |

All 30 runs stopped via ordinary early stopping (best_epoch 1–32, none
hit the 100-epoch ceiling). Gradient norms stayed well above zero and
finite throughout for all three ansätze — **no evidence of a barren
plateau or gradient pathology for any configuration**, consistent with
this stage's own instruction not to over-read ordinary gradient-magnitude
differences.

## 15. Validation→test gap

| Ansatz | Split | Mean gap | Std |
|---|---|---:|---:|
| A0 | primary | 0.052 | 0.076 |
| A1 | primary | 0.077 | 0.151 |
| A2 | primary | 0.061 | 0.103 |
| A0 | severity | 0.601 | 0.066 |
| A1 | severity | 0.645 | 0.062 |
| A2 | severity | 0.638 | 0.064 |

Severity's large val→test gap (0.60–0.65) is consistent across all three
ansätze — this is the dataset's own OOD generalization limitation
(established in Stage 1/H), not something ansatz choice changes. On
primary, A1 has the largest gap (0.077, and the largest std, 0.151),
consistent with its bimodal per-seed pattern (§9).

## 16. Calibration analysis

Primary: all three ansätze are essentially tied (Brier 0.064–0.065, ECE
0.180–0.182) — calibration does not differentiate them on primary.
Severity: A1 has the best calibration (Brier 0.092, ECE 0.158) despite a
lower mean PR-AUC than A0 — another instance (as in Stage 4b) of ranking
and calibration moving independently. A2 has the worst severity
calibration (Brier 0.146, ECE 0.245), coinciding with its specificity
collapse on 2 seeds.

## 17. Comparison with Classical GraphSAGE (0.807/0.449)

| Ansatz | Primary vs. classical | Severity vs. classical |
|---|---|---|
| A0 | above (+0.004) | below (−0.051) |
| A1 | below (−0.001, within noise) | below (−0.095) |
| A2 | above (+0.007) | below (−0.089) |

All three sit close to classical on primary (within noise) and all three
remain below it on severity — ansatz choice does not close the
classical-severity gap.

## 18. Comparison with QGNN reference (A0 itself: 0.811/0.398)

A1 and A2 do not exceed A0 on either split's mean PR-AUC (A1: −0.005
primary, −0.044 severity; A2: +0.003 primary, within noise, −0.038
severity), and both show worse or equal stability on primary. Neither
alternative ansatz improves on the standing reference while preserving
severity behavior.

## 19. Parameter-count limitations

This comparison is NOT parameter-matched (§5): A0 has 3× A2's and 1.5×
A1's quantum parameters. The results do not support attributing A0's
tighter primary variance to its topology alone — a smaller parameter
budget (A1, A2) coinciding with looser primary variance is consistent
with, but does not prove, a capacity effect. A properly parameter-matched
ansatz comparison (e.g. adding rotation-gate variety to A2 without adding
qubits) would be needed to separate these cleanly; not attempted here per
this stage's own scope limits.

## 20. Research-question answers

- **Q1 (Primary)**: No ansatz changes mean primary performance by more
  than ~1%; A1 substantially increases primary *variance*.
- **Q2 (Severity/OOD)**: No ansatz improves severity mean PR-AUC over A0;
  A2 shows a distinct specificity-collapse failure mode on 2/5 seeds not
  seen in A0/A1.
- **Q3 (Seed stability)**: Yes, for primary specifically — A0 is
  meaningfully more stable than A1 or A2. Severity PR-AUC variance is
  unaffected by ansatz choice.
- **Q4 (Quantum feature representation)**: Yes — A2 produces
  systematically more cross-channel-correlated (less differentiated)
  features than A0/A1, a real, seed-consistent architectural property.
- **Q5 (val→test gap)**: No ansatz meaningfully reduces the gap on
  either split; severity's large gap is dataset-driven, not ansatz-driven.
- **Q6 (Calibration)**: A1 shows a real severity-calibration improvement
  over A0; A2 shows worse severity calibration, coincident with its
  specificity collapse.
- **Q7 (Robustness across seeds)**: The one directionally interesting
  result (A1's higher primary recall/MCC) is not consistent across all 5
  seeds (§9's bimodal pattern) — not robust enough to call a improvement.
- **Q8 (Parameter-count confound)**: Plausible but unproven (§19) —
  flagged, not resolved, by this stage's design.
- **Q9 (Meaningful factor vs. Classical gap)**: No — none of the three
  ansätze closes or meaningfully narrows the severity gap to Classical
  GraphSAGE; ansatz choice does not appear to be the dominant factor
  behind that gap.

## 21. KEEP / DROP / INVESTIGATE decisions

| Ansatz | Verdict | Reasoning |
|---|---|---|
| A0 StronglyEntangling | **KEEP as standing reference** | Best primary stability of the three; no alternative improves on it without a new cost. |
| A1 Hardware Ring | **INVESTIGATE** | Comparable mean performance with better severity calibration and higher primary recall/MCC, but nearly 2× A0's primary variance driven by a clear bimodal seed split — worth understanding *why* seeds 43/45 diverge before any further use. |
| A2 Reduced Entanglement | **INVESTIGATE** | Comparable mean PR-AUC with the fewest parameters, but a real, reproducible specificity-collapse failure mode on 2/5 severity seeds and the most redundant (collapsed) quantum feature channels of the three — a genuine architectural weakness worth a closer look, not a reason to abandon the ansatz outright given its otherwise-competitive mean performance. |

No ansatz is dropped outright — none consistently underperforms without
any redeeming property, and none is a clean win either, per this stage's
explicit "no overall ranking" instruction.

## 22. Limitations

- **Not parameter-matched** (§19) — the dominant unresolved confound.
- **Quantum feature analysis used fresh inference forward passes**, not
  the exact per-epoch training trajectory — channel statistics describe
  the final trained model, not how representations evolved during
  training.
- **A2's specificity collapse is seen in 2 of 5 seeds** — real and
  reproducible within this experiment, but a 5-seed sample is still a
  small population for characterizing a bimodal failure mode precisely.
- Severity's single-underlying-event limitation (Stage 1) applies to
  every severity number in this report unchanged.

## 23. Recommended next experiment

Ansatz topology, at matched qubit count, does not produce a clean win
over the existing reference — consistent with Stage 4b's initialization
finding that no single architectural lever tried so far decisively closes
the severity/Classical gap. Two reasonable, evidence-motivated next
steps, not run automatically: **(1)** a focused follow-up on A2's
specificity collapse (does it correlate with a specific validation
selection artifact, similar to Stage 4b's F2 investigation?), or **(2)**
per the master research plan's own remaining tracks, Stage 3 (quantum
encoding scale / data re-uploading) or Stage 7 (encoder architecture,
only justified if representation-side evidence continues to point away
from the quantum head itself, per Stage 1's own finding that a simple
diagnostic classifier already matches/exceeds the quantum head on
severity). Deferring to you on which to pursue.
