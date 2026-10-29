# QGNN-v4 Benchmark: StronglyEntanglingLayers Hybrid Head vs. Matched-Capacity Classical Control

Implements `PENNYLANE_QGNN_IMPLEMENTATION_PLAN.md` as a new, additive
architecture variant (`src/scm_dataset/modeling/quantum/`) alongside the
existing QGNN v1-v3 (`qgnn.py`, `qgnn_v2.py`, `qgnn_v3.py`), which follow a
different, earlier planning document and are unmodified by this work. All
training below was executed by the user (`scripts/run_qgnn_v4_experiment.py`);
this document consolidates the results.

---

## 1. Executive Summary

**No evidence of a quantum advantage on this benchmark, and one split
shows a consistent classical advantage.** On the primary (temporal) split,
Hybrid-Quantum-v4 has a higher mean test PR-AUC than its matched-capacity
classical control (0.833 vs. 0.796) but with **more than double the
seed-to-seed variance** (std 0.140 vs. 0.065) and one outright reversal
(seed 43) — the mean gap is smaller than either arm's own noise, so this
is not a result either direction can lean on. On the severity split (the
harder generalization task, train severity 1-3 / test severity 4-5), the
matched classical control **beats quantum consistently, 5/5 seeds**, with
tight variance on both sides (std ≈0.02-0.024): classical 0.440±0.023 vs.
quantum 0.392±0.027.

**Neither arm shows real fresh-onset detection ability on the severity
split** — `pr_auc_fresh_onset` sits at 0.004-0.010 for every seed, every
arm, indistinguishable from noise. This confirms the plan's own named,
inherited constraint (§0/§15): the frozen embedding carries no
fresh-onset precursor signal, so no head downstream of it — quantum or
classical — can recover one.

**A genuine, useful finding: the quantum head's output calibration is
unstable across seeds**, and this — not fresh-onset detection — is what
produced a red herring during evaluation (§7). Two of five severity-split
seeds showed the quantum head's entire test-split probability
distribution collapse to near-uniformly-above-0.5 (93.5% and 100% of test
rows predicted positive), which trivially inflated recall at the fixed
0.5 threshold for every category including fresh-onset, while ranking
quality (`pr_auc_fresh_onset`) stayed flat at the same near-zero level as
every other seed. Caught before being reported as a finding — see §7 for
the full diagnosis. The quantum head's consistently higher Brier/ECE
(§9) is the same instability showing up in calibration metrics directly.

## 2. Architecture

```
existing, unmodified GraphSAGE-Full checkpoint (per seed, per split)
    -> frozen (requires_grad_(False)), encode()["supplier"]
    -> [n_suppliers, 128] RAW embedding, no PCA
         /                                              \
  MatchedCapacityClassicalHead                  HybridQuantumHead
  Linear(128,6) -> ReLU -> Linear(6,1)          Linear(128,6) -> pi*tanh -> AngleEmbedding(RY)
                                                  -> StronglyEntanglingLayers(2 layers, 6 qubits)
                                                  -> PauliZ expval per qubit -> Linear(6,1)
```

Distinct from v2 (`qgnn_v2.py`), which PCA-reduces the frozen embedding to
a fixed 8D before either head sees it: v4's dimensionality reduction is a
**trainable** `Linear(128, n_qubits)` inside each head, learned jointly
with the rest of the head via the training-split loss — no separate
leakage-sensitive reducer artifact to fit/save/load. Distinct from v1-v3's
circuit (`qgnn.py`, hand-rolled `RY` + CNOT chain, linear/ring/all-to-all
entanglement ablated directly): v4 uses PennyLane's own
`StronglyEntanglingLayers` ansatz via `qml.qnn.TorchLayer`, a standard,
citable "hardware-efficient ansatz" building block, per the plan's
explicit choice (§6).

## 3. Fairness Design

Both heads receive byte-for-byte the same raw 128D frozen embedding at
every example. The frozen encoder never receives a gradient — verified by
test (`test_backbone_frozen_after_v4_training`). The `Linear(128, 6)`
reduction and `Linear(6, 1)` output layers are **identical in shape**
between the two heads (verified by test,
`test_matched_capacity_control_shares_bottleneck_width_with_quantum_head`):
`MatchedCapacityClassicalHead`'s entire parameter budget (781 params) is
exactly that shared bottleneck, with nothing extra — the quantum head adds
36 circuit parameters (817 total) on top of the identical bottleneck. This
isolates "does a quantum circuit inside the bottleneck help/hurt" from
"does the bottleneck width itself help/hurt" (RQ-Q3). Dataset, target,
horizon, class weighting, optimizer, learning rate, weight decay, early
stopping, and evaluation code are all identical to every other model in
this repo.

## 4. Training Protocol

Per model seed, per split: load that seed's existing frozen GraphSAGE-Full
checkpoint (for the severity split, the **separately-trained** severity
checkpoint — `experiments/classical_gnn/*_hetero_graphsage_severity_seed<N>`,
resolved automatically via `_analysis_common.find_latest_graphsage_full_checkpoint`'s
`split_suffix` argument, added in this phase — see §13) → freeze → extract
the raw 128D embedding for every (supplier, time) row across train/
validation/test (no PCA) → train `MatchedCapacityClassicalHead` and,
separately, `HybridQuantumHead` (same seed for weight init and
data-shuffling, both) → early-stop on validation PR-AUC (patience 10,
100-epoch budget) → evaluate, frozen, on test. Threshold policy: `fixed`,
value 0.5 (relevant to §7's finding).

## 5. Primary (Temporal) Split — Results (5 seeds, 42-46)

### Test split, per seed

| Seed | Classical PR-AUC | Quantum PR-AUC | Δ (quantum − classical) |
|---:|---:|---:|---:|
| 42 | 0.7749 | 0.9615 | +0.1867 |
| 43 | 0.6819 | 0.5696 | **−0.1123** |
| 44 | 0.8716 | 0.9266 | +0.0550 |
| 45 | 0.8218 | 0.8203 | −0.0015 |
| 46 | 0.8295 | 0.8861 | +0.0566 |

3/5 seeds clearly favor quantum, 1/5 essentially tied, 1/5 (seed 43)
clearly favors classical.

### Aggregate (mean ± std)

| Model | Split | PR-AUC | ROC-AUC | Brier | ECE |
|---|---|---:|---:|---:|---:|
| MatchedCapacityClassical | validation | 0.8777 ± 0.0090 | 0.9369 ± 0.0057 | 0.0627 ± 0.0504 | 0.1050 ± 0.1143 |
| MatchedCapacityClassical | test | 0.7959 ± 0.0724 | 0.9861 ± 0.0091 | 0.0631 ± 0.0499 | 0.1135 ± 0.1151 |
| HybridQuantum-v4 | validation | 0.8712 ± 0.0105 | 0.9240 ± 0.0100 | 0.1261 ± 0.0932 | 0.2836 ± 0.1447 |
| HybridQuantum-v4 | test | 0.8328 ± 0.1562 | 0.9829 ± 0.0175 | 0.1283 ± 0.0923 | 0.2845 ± 0.1363 |

Test-split std is reported two ways above and in §1 — 0.1562 here is the
plain `np.std` matching the run's own multi-seed summary JSON; §1's 0.140
figure matches the script's ddof convention. Both tell the same story:
quantum's test variance is roughly double classical's.

**Zero fresh-onset test examples on this split**, every seed, both arms —
this split cannot speak to RQ-Q1's fresh-onset comparison at all (§0/§15's
known, inherited constraint; confirmed again here). Already-ongoing test
recall: classical 0.946-1.000, quantum 0.901-1.000 — comparable, no
degeneracy (`fraction_time_varying = 1.0` for every seed/arm, i.e. no
static-score collapse).

## 6. Severity Split — Results (5 seeds, 42-46)

Train severity 1-3, test severity 4-5 (composition fixed by the
benchmark's own `splits/severity_split.csv`). Frozen encoder: the
separately-trained severity checkpoint per seed (§4, §13) — not the
temporal one.

### Test split, per seed

| Seed | Classical PR-AUC | Quantum PR-AUC | Δ (quantum − classical) |
|---:|---:|---:|---:|
| 42 | 0.4307 | 0.3610 | −0.0696 |
| 43 | 0.4620 | 0.3744 | −0.0877 |
| 44 | 0.4242 | 0.4248 | +0.0006 |
| 45 | 0.4167 | 0.3851 | −0.0317 |
| 46 | 0.4672 | 0.4123 | −0.0550 |

**4/5 seeds clearly favor classical, 1/5 essentially tied. Zero seeds
favor quantum.** This is a materially cleaner signal than the primary
split: both arms' variance is much tighter here (std ≈0.02-0.024 vs.
0.065-0.16 on primary), so the consistent classical-favoring direction is
not an artifact of noisy individual runs.

### Aggregate (mean ± std)

| Model | Split | PR-AUC | ROC-AUC | Brier | ECE |
|---|---|---:|---:|---:|---:|
| MatchedCapacityClassical | validation | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0454 ± 0.0182 | 0.1465 ± 0.0630 |
| MatchedCapacityClassical | test | 0.4402 ± 0.0229 | 0.7295 ± 0.0164 | 0.0791 ± 0.0136 | 0.1142 ± 0.0532 |
| HybridQuantum-v4 | validation | 0.9938 ± 0.0126 | 0.9998 ± 0.0004 | 0.1954 ± 0.1189 | 0.3837 ± 0.1919 |
| HybridQuantum-v4 | test | 0.3915 ± 0.0265 | 0.6257 ± 0.0431 | 0.2105 ± 0.0925 | 0.3448 ± 0.1806 |

Both absolute PR-AUC levels are much lower than the primary split for
**both** arms (≈0.39-0.44 vs. ≈0.80-0.83) — severity generalization
(train low-severity, test high-severity) is a harder task overall, not a
quantum-specific weakness; the comparison between arms is what's
informative here, not the absolute level.

**Near-perfect validation PR-AUC (classical 1.000±0.000, quantum
0.994±0.013) alongside a much harder test PR-AUC (~0.39-0.44) is a real
easy-validation/hard-test gap**, not a bug: `split.strategy: severity`'s
validation carve-out (`generalization_val_frac`) is drawn from the
same severity-1-3 training distribution, while test is the held-out
severity-4-5 band — an in-distribution validation set and an
out-of-distribution test set by design. Flagged here rather than silently
left unremarked; not independently re-verified against the classical
GraphSAGE-Full severity baseline's own validation number in this phase.

## 7. Onset Breakdown — A Caught Artifact, Not a Finding

The severity test split has real fresh-onset examples (84 fresh-onset,
733 already-ongoing, every seed) — the first split in this repo's QGNN
work where a genuine fresh-onset comparison is possible. Before reporting
`recall_fresh_onset` as a result, per-seed values were inspected directly
because two of five quantum-arm seeds showed a striking pattern:

| Seed | Quantum recall (fresh-onset) | Quantum recall (already-ongoing) | Quantum `pr_auc_fresh_onset` |
|---:|---:|---:|---:|
| 42 | 0.619 | 0.990 | 0.0045 |
| 43 | 0.000 | 0.001 | 0.0100 |
| 44 | 0.000 | 0.491 | 0.0050 |
| 45 | **1.000** | **1.000** | 0.0049 |
| 46 | 0.000 | 0.490 | 0.0059 |

Classical's fresh-onset recall was exactly 0.000 on **every single
seed**. Per the plan's own explicit warning (§15): *"if only the quantum
arm shows [non-zero fresh-onset performance], treat it as a candidate
artifact to investigate, not a headline result to lead with."* Diagnosis:

| Seed | Frac. test predictions ≥ 0.5 | Mean prob., positives | Mean prob., negatives |
|---:|---:|---:|---:|
| 42 | 93.5% | 0.550 | 0.527 |
| 43 | 0.0% | 0.431 | 0.400 |
| 44 | 5.3% | 0.478 | 0.455 |
| 45 | 100% | 0.577 | 0.573 |
| 46 | 7.8% | 0.445 | 0.091 |

Seeds 42 and 45 — the two with "high" fresh-onset recall — collapsed to
predicting positive for nearly the entire test split (93.5% and 100%
respectively), with essentially no separation between the mean predicted
probability for positive vs. negative examples (0.550 vs. 0.527; 0.577
vs. 0.573). At a 93-100% positive rate under a fixed 0.5 threshold,
*every* category's recall is trivially inflated, fresh-onset included —
and `pr_auc_fresh_onset` (threshold-independent) confirms this directly:
it stays at the same ~0.005 noise floor in these seeds as in the
zero-recall seeds. **The ranking never improved in any seed; only where
the fixed 0.5 cutoff happened to fall did.** This is a threshold-
calibration instability in the quantum head's output scale across seeds
(consistent with §9's calibration numbers), not fresh-onset detection —
and it would have looked like a positive quantum finding if only the
recall-at-0.5 numbers had been reported without checking `pr_auc` and the
underlying probability distributions. No code or config was changed in
response to this finding in this phase; `threshold.policy: fixed, value:
0.5` remains as configured, but any future report should surface
`pr_auc_fresh_onset` (threshold-free) rather than recall-at-a-fixed-cutoff
when discussing fresh-onset — recall at a threshold that isn't adapted
per-run is not comparable across seeds for a head this unstably
calibrated. Switching `threshold.policy` to `f1_optimal` (already
supported by `config.py`/`metrics.select_threshold`, just unused in this
config) is a plausible next step, not yet taken.

## 8. Matched-Capacity Control (RQ-Q3)

| | Params | Shape |
|---|---:|---|
| `MatchedCapacityClassicalHead` | 781 | `Linear(128,6) -> ReLU -> Linear(6,1)`, exactly the shared bottleneck, nothing extra |
| `HybridQuantumHead` | 817 | identical `Linear(128,6)` + `Linear(6,1)` bottleneck, +36 circuit parameters |

The two heads differ by exactly 36 parameters (the `StronglyEntanglingLayers`
circuit weights: `n_layers × n_qubits × 3 = 2×6×3 = 36`) — verified by
test. Every PR-AUC comparison in §5-6 is therefore a comparison of "36
extra quantum parameters, same bottleneck width" vs. "no extra
parameters, same bottleneck width" — a controlled fairness ablation, not
a confound between input representation and head type.

## 9. Calibration

Quantum's Brier score and ECE are **worse than classical's on every split
and every subset measured** (primary test: ECE 0.285 vs. 0.114; severity
test: ECE 0.345 vs. 0.114) — roughly 2-3x worse throughout. This is
consistent with, and likely explains, §7's threshold-artifact: a head
whose predicted-probability distribution is this poorly anchored across
seeds will also produce unstable recall-at-a-fixed-threshold numbers.
Unlike v2's finding (graph-derived input improved QGNN calibration
substantially over v1, `QGNN_V2_DEPTH_BENCHMARK.md` §10), v4's quantum
head does not calibrate as well as its matched classical control at this
architecture.

## 10. Resource Analysis

| | Value |
|---|---:|
| Qubits | 6 |
| Variational layers | 2 |
| Ansatz | `StronglyEntanglingLayers` |
| Quantum trainable parameters | 36 |
| Shared bottleneck parameters (both heads) | 781 |
| Total trainable (HybridQuantumHead) | 817 |
| Total trainable (MatchedCapacityClassicalHead) | 781 |
| Frozen encoder (shared, not trainable) | ≈1.2M |
| Backend | `default.qubit` (CPU), `diff_method="backprop"` |

All 5 seeds × 2 splits × 2 heads (20 training runs total) completed in
**≈2.4 minutes total wall-clock** (`run_metadata.json`'s
`training_duration_seconds`, summed). No quantum computational advantage
is claimed — simulator-based throughout, consistent with every prior
phase in this project.

## 11. Limitations

- Single architecture point (6 qubits, 2 layers, `StronglyEntanglingLayers`,
  linear-Linear reduction) — no depth/qubit-count/ansatz ablation was run
  in this phase, unlike v2's depth sweep or v3's entanglement-topology
  study. The plan's own §14 Phase Q3 (ablation grid before spending full
  compute) was skipped; this run went straight to the primary-benchmark
  configuration.
- Onset breakdown is only meaningful on the severity split (§7); the
  primary split has zero fresh-onset test examples in every seed.
- `threshold.policy: fixed, value: 0.5` is not well-suited to comparing a
  head with this much cross-seed calibration variance (§7, §9) — no
  alternative threshold policy was tried in this phase.
- Five seeds is this project's standing convention, not a large-N
  statistical study; no formal significance test was run or is implied by
  any comparison above — read direction and relative magnitude only, as
  §5-6's per-seed tables do explicitly.
- Severity split's near-saturated validation PR-AUC (§6) was flagged but
  not independently cross-checked against the classical GraphSAGE-Full
  severity baseline's own validation numbers in this phase.
- Simulator-based only; no real quantum hardware; no claim of quantum
  advantage, per the plan's own explicit non-goal (§0).

## 12. Scientific Interpretation

Per the plan's own instruction (§30) not to assume QGNN superiority
going in, and that a result favoring the classical arm is scientifically
valid: **that is the result found here.** On the split with tighter,
more trustworthy variance (severity), the matched-capacity classical
control outperforms the quantum head consistently. On the split with a
higher quantum mean (primary/temporal), the variance is too large
relative to the mean gap to support a directional claim either way. No
evidence across either split supports "the quantum circuit adds value
over an equally-sized classical bottleneck on this frozen representation"
(RQ-Q1/RQ-Q3, taken together). The one clear, reportable, non-null
finding of this phase is methodological rather than a performance
comparison: the quantum head's output calibration is measurably less
stable across seeds than its classical control (§7, §9), and that
instability — not a fresh-onset breakthrough — is what produced this
phase's one dramatic-looking number.

## 13. Implementation Notes (this phase)

- New, additive package: `src/scm_dataset/modeling/quantum/`
  (`circuit.py`, `heads.py`, `model.py`) — does not modify `qgnn.py`,
  `qgnn_v2.py`, `qgnn_v2_reupload.py`, or `qgnn_v3.py`.
- `scripts/_analysis_common.py::find_latest_graphsage_full_checkpoint`
  gained an optional `split_suffix` parameter (default `None`,
  byte-for-byte the old behavior — v2/v3/v3.1 scripts unaffected) so a
  severity-split QGNN run automatically resolves to the
  separately-trained severity encoder checkpoint rather than silently
  reusing the temporal one, matching the classical GraphSAGE severity
  baseline's own protocol.
- `pyproject.toml` gained a `[quantum]` optional-dependency extra
  (`pennylane>=0.38`, `pennylane-lightning>=0.38`) — documented but never
  previously applied, despite PennyLane already being imported by v1-v3.
- 10 new tests (`tests/test_quantum_v4.py`): circuit output shape (both
  ansatz choices), unknown-ansatz rejection, encoding boundedness under
  extreme input magnitudes, gradient flow to quantum parameters, backbone
  frozen after v4 training, matched-bottleneck parameter-count equality,
  determinism given a fixed seed, and an end-to-end tiny-benchmark
  training run for both heads. Full suite: 257 passed, 0 regressions.

## 14. Reproducibility

```bash
# Primary (temporal) split, both heads, 5 seeds
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4.yaml --tag primary --seeds 42,43,44,45,46

# Severity split, both heads, 5 seeds (auto-resolves the severity-trained encoder)
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4_severity.yaml --tag severity --seeds 42,43,44,45,46
```

Each run reuses the existing GraphSAGE-Full checkpoints in
`experiments/classical_gnn/` as its frozen encoder (never retrained) and
writes artifacts (config, model checkpoint, predictions, metrics,
calibration, onset breakdown, training history, plots, quantum resource
summary, encoder-checkpoint provenance, run metadata including git commit
and timestamp) under `experiments/qgnn_v4/` — no run directory is ever
overwritten. `experiments/` is gitignored; only this report is committed.

## 15. Suggested Next Steps

1. Switch `threshold.policy` to `f1_optimal` (or otherwise report only
   `pr_auc_fresh_onset`, threshold-free) before drawing any further
   fresh-onset conclusions — §7's artifact is a direct consequence of a
   fixed 0.5 cutoff applied to an unstably-calibrated head.
2. A qubit-count/layer-count/ansatz ablation (the plan's own §14 Phase
   Q3), on the tiny fixture first, before spending more full-benchmark
   compute on a single untuned architecture point.
3. Investigate the quantum head's calibration instability directly — a
   `BatchNorm` or fixed output-scale on `self.out`, or a different weight
   initialization for the circuit, before concluding the instability is
   inherent to the ansatz rather than an easily-fixed init/scale issue.
4. Cross-check the severity split's near-saturated validation PR-AUC
   (§6, §11) against the classical GraphSAGE-Full severity baseline's own
   validation numbers, to confirm it's a property of the split design and
   not specific to these heads.
