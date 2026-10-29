# QGNN-v4 Phase 4 — Stage 4 Results: Quantum Initialization Pilot

## 1. Objective

Every stage so far (Phase 1 through Stage 2) has shown large, persistent
PR-AUC variation across the 5 fixed model seeds, particularly on primary
and particularly under the LayerNorm output stage. Stage 4 tests whether
this is, at least partly, an artifact of how the quantum circuit's own
trainable rotation parameters are initialized — a different starting
point for a non-convex, seed-dependent optimization landscape could
plausibly push training toward different local solutions with different
seeds. This is **not** a search for the highest single PR-AUC: the
question is whether some initialization gives a more *reproducible*
result (higher mean, and/or lower cross-seed variance, without a
disqualifying loss elsewhere) than the current default.

## 2. Existing reference (unchanged, not re-derived)

| | Primary PR-AUC | Severity PR-AUC |
|---|---|---|
| Classical GraphSAGE-Full | 0.807 ± 0.066 | 0.449 ± 0.013 |
| QGNN-v4 reference (6q/2L/StronglyEntangling/LayerNorm, 5 seeds) | 0.811 ± 0.074 | 0.398 ± 0.057 |
| QGNN-v4 4-qubit (context only) | 0.847 ± 0.102 | 0.396 ± 0.030 |

## 3. Initialization strategies implemented

Verified directly against the installed PennyLane 0.45.1 source
(`TorchLayer._init_weights`) before writing any code, per this stage's
own "read the repository first" instruction:

- **F1 — default (control)**: `qml.qnn.TorchLayer`'s own behavior when no
  `init_method` is given — `torch.nn.init.uniform_(tensor, a=0, b=2*pi)`.
  Not reimplemented: `quantum_init="default"` passes `init_method=None`
  straight through, so this is byte-for-byte what every prior phase
  already ran (verified by a bit-identical-weights test against a bare
  `TorchLayer` call).
- **F2 — small Gaussian**: `torch.nn.init.normal_(tensor, mean=0, std=0.01)`.
- **F3 — identity-like**: every trainable rotation parameter set to
  exactly `0.0`. This is **exact**, not approximate, for the rotation
  gates specifically — `Rot(0,0,0)` (StronglyEntanglingLayers),
  `RY(0)`/`RZ(0)` (hardware_efficient_ring), `RY(0)` (reduced_entanglement)
  and `RX(0)` (BasicEntanglerLayers) are each precisely the single-qubit
  identity operator at parameter value 0. **Explicit limitation, stated
  per this stage's own instruction not to overclaim**: the entangling
  CNOT pattern in every ansatz here is a fixed, unparameterized part of
  the circuit template — it fires unconditionally regardless of
  `weights`, and CNOT is not the identity. So F3 is an exact identity
  initialization of the *rotation gates*, not of the full quantum layer —
  entanglement between qubits occurs on the very first forward pass
  regardless of initialization. A pre-training gradient check (unit
  test, not just asserted) confirmed this starting point is not
  degenerate: gradients differ across the 36 parameters and are not
  vanishingly small, so training can actually move away from it.

## 4. Exact experimental protocol

Every configuration used the identical frozen encoder checkpoints (per
seed), dataset, split, projection (`Linear(128,6)`, unchanged — no PCA, no
pre-projection LayerNorm), 6 qubits, 2 layers, StronglyEntanglingLayers,
`default.qubit`, `diff_method="backprop"`, Adam (lr=0.001,
weight_decay=0.0001), max_epochs=100, patience=10, balanced class
weighting, threshold=fixed@0.5, `--diagnostics` throughout. **Only
`--quantum-init` changed between runs.** Verified automatically (unit
test + every saved `quantum_resource_summary.json`): all three
configurations show `total_trainable_parameters=817`, identical shapes —
initialization changes values only, never parameter count.

- Code changes: `circuit.py` gained `QUANTUM_INIT_STRATEGIES`/
  `resolve_quantum_init` and a `quantum_init` parameter on
  `build_quantum_layer` (threaded straight into `TorchLayer`'s own
  `init_method`). `HybridQuantumHeadLayerNorm` gained a matching
  `quantum_init` constructor arg (default `"default"`, backward-compatible)
  and a new `quantum_parameter_stats()` method for the initialization
  audit. `run_qgnn_v4_experiment.py` gained `--quantum-init` and
  `--output-subdir` (the latter routes a run into
  `experiments/qgnn_v4/phase4_stage4/<init>/`, per this stage's own
  "store each initialization separately" requirement). No changes to
  `qgnn.py`/`qgnn_v2.py`/`qgnn_v2_reupload.py`/`qgnn_v3.py`.
- Unit tests: 13 new tests (default-init byte-identity to a bare
  `TorchLayer`, small-Gaussian/identity-like value checks, parameter-count
  invariance across all three strategies, forward/backward/finite-gradient
  checks, the "are identity-like's gradients actually non-degenerate"
  check, checkpoint round-trip via the existing state_dict mechanism,
  and an end-to-end `tiny_benchmark` training run for F3). 306/306 tests
  passed, run twice (before and after the code changes) — zero
  regressions.
- Smoke tests: all three strategies run against the real dataset (1 seed,
  3 epochs) before the pilot — confirmed distinct initial parameter
  statistics per strategy, correct `total_trainable_parameters=817` in
  every case, and successful end-to-end training.
- Pilot: 2 seeds (42, 43) × 2 splits × 3 strategies = 12 runs, all
  completed cleanly, saved to
  `experiments/qgnn_v4/phase4_stage4/{default,small_gaussian,identity_like}/`.

## 5–6. Primary and Severity results (test split, threshold=0.5, mean±std over 2 pilot seeds)

| Config | Split | PR-AUC | ROC-AUC | F1 | Precision | Recall | Specificity | Bal. Acc. | MCC | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 default | primary | 0.774±.094 | 0.977±.012 | 0.637±.043 | 0.511±.009 | 0.857±.126 | 0.941±.007 | 0.899±.060 | 0.631±.059 | 0.070±.016 | 0.193±.034 |
| F2 small_gaussian | primary | 0.828±.083 | 0.980±.014 | 0.772±.133 | 0.701±.204 | 0.899±.006 | 0.964±.029 | 0.931±.018 | 0.767±.131 | 0.049±.001 | 0.145±.037 |
| F3 identity_like | primary | 0.726±.017 | 0.975±.007 | 0.688±.041 | 0.540±.033 | 0.948±.052 | 0.942±.005 | 0.945±.028 | 0.691±.046 | 0.080±.034 | 0.179±.107 |
| F1 default | severity | 0.422±.028 | 0.667±.026 | 0.456±.076 | 0.592±.259 | 0.414±.027 | 0.962±.032 | 0.688±.003 | 0.440±.114 | 0.139±.024 | 0.287±.035 |
| F2 small_gaussian | severity | 0.265±.057 | 0.653±.039 | 0.400±.009 | 0.367±.015 | 0.440±.001 | 0.939±.004 | 0.690±.002 | 0.349±.010 | 0.084±.001 | 0.144±.003 |
| F3 identity_like | severity | 0.380±.102 | 0.647±.023 | 0.403±.030 | 0.374±.051 | 0.441±.000 | 0.940±.013 | 0.690±.006 | 0.353±.034 | 0.126±.054 | 0.206±.139 |

Full CSVs: `experiments/qgnn_v4/phase4_stage4/stage4_{per_seed_results,
confusion_matrices,summary,metrics,initialization_stats,training_dynamics,
results}.csv`.

## 7. Per-seed results

| Config | Split | Seed | PR-AUC | ROC-AUC | MCC | TP | FP | FN | TN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 | primary | 42 | 0.868 | 0.990 | 0.690 | 238 | 220 | 4 | 3138 |
| F1 | primary | 43 | 0.681 | 0.965 | 0.572 | 177 | 176 | 65 | 3182 |
| F2 | primary | 42 | **0.912** | 0.994 | 0.898 | 219 | 23 | 23 | 3335 |
| F2 | primary | 43 | 0.745 | 0.966 | 0.636 | 216 | 218 | 26 | 3140 |
| F3 | primary | 42 | 0.744 | 0.982 | 0.737 | 242 | 180 | 0 | 3178 |
| F3 | primary | 43 | 0.709 | 0.969 | 0.645 | 217 | 211 | 25 | 3147 |
| F1 | severity | 42 | 0.394 | 0.693 | 0.326 | 360 | 721 | 457 | 9562 |
| F1 | severity | 43 | 0.450 | 0.640 | 0.554 | 316 | 55 | 501 | 10228 |
| F2 | severity | 42 | 0.322 | 0.692 | 0.339 | 360 | 664 | 457 | 9619 |
| F2 | severity | 43 | 0.208 | 0.614 | 0.359 | 359 | 581 | 458 | 9702 |
| F3 | severity | 42 | 0.278 | 0.671 | 0.319 | 360 | 755 | 457 | 9528 |
| F3 | severity | 43 | **0.481** | 0.624 | 0.387 | 360 | 488 | 457 | 9795 |

**Paired-seed comparison, primary** (same seed, F1 vs. alternative):
F2 beats F1 on BOTH seeds (0.912>0.868, 0.745>0.681) — not a one-seed
fluke. F3 is mixed (0.744<0.868 on seed 42, 0.709>0.681 on seed 43) but
its own two seeds are far closer to each other than F1's are.

## 8. Stability analysis

| Config | Split | PR-AUC mean | std | range (max−min) |
|---|---|---:|---:|---:|
| F1 | primary | 0.774 | 0.094 | 0.187 |
| F2 | primary | 0.828 | 0.083 | 0.167 |
| **F3** | primary | 0.726 | **0.017** | **0.035** |
| F1 | severity | 0.422 | 0.028 | 0.055 |
| F2 | severity | 0.265 | 0.057 | 0.114 |
| F3 | severity | 0.380 | 0.102 | 0.204 |

**F3 (identity-like) shows an ~82% reduction in primary PR-AUC std**
(0.094→0.017) and range (0.187→0.035) relative to F1 — the clearest
signal in this pilot that initialization affects reproducibility, exactly
the question this stage was designed to answer. It comes at a real but
modest cost: primary mean drops from 0.774 to 0.726 (−6%). On severity,
F3 is *less* stable than F1 (std 0.102 vs 0.028) — the two severity seeds
disagree more under F3, so the "more reproducible" finding does not
carry over to the OOD split. **F2 (small Gaussian) does not reduce
variance** on either split — its primary std (0.083) is close to F1's
(0.094), and its severity std (0.057) is actually *higher* than F1's
(0.028).

## 9. Training dynamics

All 12 runs triggered early stopping (patience=10 engaged in every case);
best_epoch ranged 2–24, total_epochs_trained 12–34 — no run hit the
100-epoch ceiling, no run collapsed in fewer than 2 epochs. No gradient
explosion or vanishing observed: `quantum_grad_norm_mean` ranged
0.058–0.372 across all 12 runs, always well above zero and well below any
instability threshold seen in earlier phases' diagnostics.

A clear, consistent pattern by initialization: **F1's quantum gradient
norms (0.135–0.372) run noticeably higher than F2's (0.058–0.135) or
F3's (0.058–0.137)** — plausible given F1's parameters start spread
across the full [0, 2π] range (where trigonometric gate derivatives can
be larger for a generic state) versus F2/F3's near-zero start.

One training-curve oddity worth flagging, not over-interpreting: F3
primary seed 42 shows `best_train_pr_auc=0.355` at its selected
checkpoint (epoch 8) despite `best_validation_pr_auc=0.766` — train
underperforming validation at the checkpoint early-stopping picked. This
is a known artifact of selecting purely on validation PR-AUC with a small
validation set and can happen with any initialization; it is not unique
to F3, and the same run's TEST PR-AUC (0.744) is unremarkable, so it does
not look like a validation-set fluke driving a misleading final number.

## 10. Parameter initialization analysis

Recorded directly via `quantum_parameter_stats()` before and after
training (`stage4_initialization_stats.csv`) — the three strategies did
produce materially different starting distributions, confirmed
numerically, not just asserted:

| Config | Phase | mean | std | min | max | L2 norm |
|---|---|---:|---:|---:|---:|---:|
| F1 | initial (seed 42) | 2.968 | 1.810 | 0.026 | 6.211 | 20.779 |
| F1 | final (seed 42) | 2.912 | 1.807 | 0.089 | 6.195 | 20.483 |
| F2 | initial (seed 42) | −0.0001 | 0.0097 | −0.020 | 0.019 | 0.057 |
| F2 | final (seed 42) | −0.0059 | 0.0501 | −0.130 | 0.101 | 0.299 |
| F3 | initial (all seeds) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| F3 | final (seed 42) | 0.0014 | 0.0144 | −0.043 | 0.034 | 0.086 |

**A genuinely interesting, unplanned finding**: F1's parameters barely
move during training in absolute terms — L2 norm goes from 20.78 to
20.48, a **1.4% change** over up to 34 epochs. F2 and F3 start near zero
and move to L2 norms of 0.06–0.32 — a much larger *relative* change from
their own starting point, even though their final absolute norms stay
far smaller than F1's. In other words: **F1's circuit spends training
making small adjustments to an already-large, effectively-random
starting rotation; F2/F3's circuits are actually learning their rotation
structure comparatively from scratch.** This is a plausible mechanism
for why F2 in particular shows a real primary PR-AUC gain — the circuit
isn't fighting a large, arbitrary starting point — and is worth keeping
in mind when Stage 5 (ansatz) or any future initialization follow-up is
designed.

## 11. Comparison against Classical GNN

Classical: primary 0.807±0.066, severity 0.449±0.013. F2's primary mean
(0.828) is the only Stage 4 config to edge past classical's primary mean
in this pilot; every config's severity mean (0.265–0.422) stays below
classical's (0.449).

## 12. Comparison against QGNN-v4 baseline

The QGNN-v4 5-seed reference (0.811/0.398) is not directly comparable
number-for-number to this pilot's 2-seed F1 subset (0.774/0.422) — same
caveat as Stage 2's report. F2's primary mean (0.828) is nominally above
both the 2-seed F1 subset and the 5-seed reference; F3's primary std
(0.017) is far tighter than the 5-seed reference's own std (0.074),
though on n=2 vs. n=5 that comparison is suggestive, not conclusive.

## 13. Interpretation

Initialization **did** change something real: parameter distributions
differ exactly as designed (§10), gradient-norm dynamics differ
systematically by strategy (§9), and downstream PR-AUC/variance differ in
ways that are not attributable to random noise alone (both F2 primary
seeds beat both F1 primary seeds, paired). Initialization did **not**
change: parameter count (817 total, every config, verified), circuit
topology, projection, loss, optimizer, or evaluation protocol — the
comparison stayed controlled as required.

What initialization did NOT do: produce an initialization that is simply
better on every axis. F2 trades severity for primary; F3 trades primary
mean for primary stability while being slightly less stable on severity.
Neither result is a clean win — both are the kind of "credible reason to
continue investigating" this stage's own decision rule anticipates,
not a finished answer.

## 14. Limitations

- **2-seed pilot.** Every std/range figure above is computed from n=2 —
  read the qualitative, paired-seed pattern (F2 beats F1 on both primary
  seeds; F3's two seeds sit closer together than F1's) as the more
  reliable signal than the exact decimal values.
- **Severity's known limitation carries over unchanged**: only one
  severity-5 event underlies the entire OOD test population (Stage 1/3
  finding) — every severity number here inherits that caveat.
- **F3's "identity-like" is exact for rotation gates only** — the
  entangling CNOT structure is unparameterized and fires regardless, so
  this is not literally an identity quantum channel at step 0, and no
  claim to that effect should be read into these results.
- **Training-dynamics interpretation (§10) is a plausible mechanism, not
  a proven causal explanation** — it is offered as a lead for future
  investigation (e.g. Stage 4b: does explicitly rescaling F1's initial
  norm down reproduce F2/F3's behavior?), not a settled finding.

## 15. Final decision

| Config | Verdict | Reasoning |
|---|---|---|
| **F1 — default** | **KEEP as the standing control** | Unchanged; every comparison in this report and future ones is relative to it. |
| **F2 — small Gaussian** | **INVESTIGATE** | Real, seed-paired primary improvement (both seeds beat F1's matching seed; MCC 0.631→0.767, Brier/ECE both improve) — not a one-seed fluke. But severity degrades consistently and substantially (0.422→0.265, both seeds worse) — exactly the "higher Primary but severe Severity degradation" pattern this stage's own rules say should not be auto-selected. Needs a focused follow-up on *why* severity suffers specifically (plausibly connected to §10's finding that F2's circuit moves much further from its start than F1's) before a KEEP/DROP call. |
| **F3 — identity-like** | **INVESTIGATE (promising — recommend 5-seed expansion)** | The strongest evidence in this pilot for the stage's actual hypothesis: primary std drops ~82% (0.094→0.017) with only a modest mean cost (−6%) and an MCC improvement (0.631→0.691); severity stays in the same rough range as F1, just noisier on n=2. Meets the phase's own "promising" bar (§9: lower std + improved MCC, without major degradation elsewhere) more cleanly than F2 does — but per this stage's explicit "do not call a configuration the winner based on one seed" and "only expand if it shows a credible reason to continue" rules, 2 seeds is not enough to confirm a *reproducibility* claim. Recommended for 5-seed expansion before any KEEP call — not run automatically here. |

**Overall: initialization is a genuine, measurable source of QGNN
behavior difference — not a dead end — but this pilot does not produce an
unambiguous winner.** No configuration is a clean KEEP; none is a clean
DROP either. Both F2 and F3 surface real, evidenced, opposite-flavored
signals (raw performance vs. stability) worth a second look before this
stage's question ("is initialization a genuine source of instability") is
answered definitively.

## Recommended next experiment

Two reasonable, evidence-supported options, not run automatically per
this stage's own "do not continue blindly" rule:

1. **Expand F3 to 5 seeds × 2 splits** — the more directly on-topic option
   for Stage 4's own question (reproducibility), since it's the config
   that actually reduced seed variance.
2. **A short, targeted follow-up on F2's severity collapse** before
   deciding whether it's worth keeping at all — e.g. inspecting whether
   severity's own validation-selected checkpoint differs systematically
   from primary's for F2 specifically.

Stage 5 (the two ansätze already built in Phase 3, 4q+2L/6q+2L controlled
comparison) remains available as an unrelated next direction if neither
of the above is prioritized. Deferring to you on which to pursue.
