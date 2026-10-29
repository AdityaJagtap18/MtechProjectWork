# QGNN Implementation and Benchmark

Implements `QGNN_FAIR_BENCHMARK_AND_IMPLEMENTATION_PLAN.md` through the
smoke test (plan section 12) and the infrastructure for the full
dimension sweep (sections 13-14). Source-of-truth documents read before
any code was written: `GRAPHSAGE_FINAL_GENERALIZATION_AND_IMPROVEMENT_PLAN.md`,
`GRAPHSAGE_PHASE_A_GENERALIZATION_FINDINGS.md`,
`GRAPHSAGE_PHASE_A5_ROOT_CAUSE_FINDINGS.md`, and the existing GraphSAGE
implementation (`src/scm_dataset/modeling/{graphsage,train,evaluate,
pipeline,features,preprocessing,hetero_graph,config,experiment}.py`) and
its tests.

**This document does not yet contain final results.** It covers
environment verification, the implementation itself, and one smoke test.
The dimension sweep (4/6/8), primary-configuration selection, the 5-seed
final run, and cross-dataset evaluation are the next phase (see
"Remaining Work" at the end) and will be reported in
`QGNN_FINAL_BENCHMARK.md`.

---

## 1. Research Question

> Does the quantum component add measurable value when the classical
> control (GraphSAGE-Reduced) receives exactly the same information and
> dimensionality as QGNN-Reduced?

Secondary: does QGNN-Reduced recover any of the performance lost to
dimensionality reduction, relative to GraphSAGE-Full?

## 2. Quantum Simulation Environment (plan section 2)

**`qqcuda` does not exist.** Verified directly: `pip index versions
qqcuda` against PyPI returns "No matching distribution found" — it is not
an installable package under any index this machine can reach. No
quantum library of any kind was installed in this environment before this
phase, and the existing PyTorch install was CPU-only
(`torch==2.14.0+cpu`) despite a physical GPU being present.

**Environment before this phase:**
- `torch 2.14.0+cpu`, `torch.cuda.is_available() == False`
- GPU present: NVIDIA GeForce RTX 5060 Ti, 16GB, driver 580.167.08 (supports up to CUDA 13.0), compute capability 12.0 (Blackwell)
- No quantum library installed

**Chosen stack, and why:**
- **PyTorch reinstalled as `torch==2.14.0+cu130`** — the identical version already in use, just CUDA-enabled, matching the driver's max supported CUDA (13.0) exactly. Chosen specifically to minimize risk to the frozen GraphSAGE baseline: same torch API version, only the backend changes. Verified: full existing test suite (195/195) still passes after this change.
- **PennyLane** (`pennylane==0.45.1`) for the quantum circuit, because:
  - it has a first-class PyTorch interface (`qml.qnode(dev, interface="torch")`), so the QGNN model is an ordinary `nn.Module` using the exact same autograd/optimizer machinery `train.py` already uses;
  - it runs correctly on CPU (`default.qubit`) with zero code changes needed to also run on GPU (`lightning.gpu`) — same high-level API either way.
- **`pennylane-lightning-gpu==0.45.0`** (built on NVIDIA cuStateVec) as the GPU backend. **Verified working** on the RTX 5060 Ti (a very new Blackwell-generation GPU where prebuilt-wheel compatibility was a real, explicitly-flagged risk before installation): a 6-qubit circuit executed with a real GPU memory footprint (confirmed via `nvidia-smi --query-compute-apps` showing the Python process's allocation during execution), batch/broadcast execution over `AngleEmbedding` worked natively, and gradients computed via `diff_method="adjoint"` were non-zero and correct.
- Automatic fallback to `default.qubit` (CPU) is built into `QuantumCircuitLayer` if the named device cannot be constructed, so this code does not hard-fail on a machine without a supported GPU.

No API was invented; every class/function used above is real, verified-installed library code.

## 3. Feature Pipeline Audit (plan section 4)

Confirmed via the live pipeline (`prepared.snapshot_builder.feature_dims()`), not estimated:

| Node type | Dimensionality (after `FeaturePreprocessor`) |
|---|---:|
| supplier | **60** |
| material | 48 |
| plant | 38 |
| product | 45 |
| region | 12 |
| procurement | 16 |

Supplier's 60 dimensions break down as: 26 raw numeric columns (12
dynamic — `order_volume`/`orders_count`/`delivery_volume`/`fulfillment_ratio`
× rolling windows [4, 8, 12] — plus 14 static — `tier`, `capacity`,
`capacity_utilization`, `reliability`, `financial_health`,
`lead_time_mean`, `lead_time_variability`, `quality_score`,
`inventory_buffer`, `substitution_availability`, `geopolitical_exposure`,
`disaster_exposure`, `cyber_exposure`, `criticality`), each expanded to
{value, missing-indicator} = 52, plus an 8-way one-hot `industry`
encoding (7 real categories + `__UNKNOWN__` bucket) = 60.

This is the representation compressed to 4/6/8 dimensions for QGNN — a
7.5–15× reduction. Every column above is real (verified via
`STATIC_NUMERIC_FIELDS`/`STATIC_CATEGORICAL_FIELDS` in `features.py` and
the `Supplier` dataclass in `schema/nodes.py`); nothing was invented, and
no target-derived column (`supplier_risk_score`, a label, is explicitly
excluded by `feature_audit.csv`) is used.

## 4. Reduction Methods (plan sections 5-6, 10-11)

Implemented in `src/scm_dataset/modeling/reduction.py`, both fitting only
on train-split (supplier, time) rows via the exact same `supplier_fit_mask`
leakage boundary `FeaturePreprocessor` itself uses (verified by a
dedicated leakage test — see §8):

- **`PCASupplierReducer`**: standardizes all 26 raw numeric supplier
  columns using train-only median (imputation) and mean/std, then fits
  `sklearn.decomposition.PCA(n_components=d, random_state=0)` on the
  standardized train rows only. Frozen for validation/test/target.
- **`DomainSelectedReducer`**: selects a small, fixed, human-interpretable
  subset of the supplier's own existing static fields (no PCA), verified
  to exist in the real schema:
  - 4 dims: `reliability`, `financial_health`, `capacity_utilization`, `criticality`
  - 6 dims: + `geopolitical_exposure`, `cyber_exposure`
  - 8 dims: + `lead_time_mean`, `quality_score`

  Standardized with train-only mean/std, same as PCA.

`ReducedGraphSnapshotBuilder` (also in `reduction.py`) composes a fitted
reducer with an ordinary, **unmodified** `HeteroGraphSnapshotBuilder`:
every node type other than SUPPLIER, every edge_index, and the graph
topology are exactly what the base builder already produces — only the
SUPPLIER tensor is replaced. Verified by test
(`test_prepare_reduced_only_changes_supplier_dimension`): reduced and
full-dimensional snapshots produce bit-identical tensors for every node
type except supplier, and identical edge_index for every relation.

`pipeline.prepare_reduced` / `prepare_reduced_from_benchmark` /
`prepare_reduced_for_cross_dataset_eval` wire this into the existing
pipeline as new, additive functions — `pipeline.py`'s existing `prepare`/
`prepare_from_benchmark`/`prepare_for_cross_dataset_eval` are untouched.

## 5. Model A — GraphSAGE-Full

Unmodified. This document does not retrain it — its numbers already exist
in `experiments/classical_gnn/` (see `GRAPHSAGE_WORK_SUMMARY.md` and the
Phase A/A.5 findings documents) and remain the reference for the
secondary "QGNN-Reduced vs GraphSAGE-Full" comparison.

## 6. Model B — GraphSAGE-Reduced

**No new model code.** It is `graphsage.HeteroGraphSAGE`/`train.train_graphsage`/
`evaluate.evaluate_experiment`, completely unchanged, given a
`PreparedData` whose `snapshot_builder` is a `ReducedGraphSnapshotBuilder`
(so `in_dims["supplier"]` is 4/6/8 instead of 60; every other node type
keeps its full width). This is the isolation the plan requires (section
3B): the *only* thing that differs from GraphSAGE-Full is the width of
one node type's input.

## 7. Model C — QGNN-Reduced

Implemented in `src/scm_dataset/modeling/qgnn.py`:

```
reduced supplier tensor [batch, d]  (from ReducedGraphSnapshotBuilder)
    -> angle_i = tanh(x_i) * pi                      (deterministic, label-free encoding)
    -> qml.AngleEmbedding(angles, rotation="Y")       (d qubits)
    -> for each of n_layers:
           RY(trainable weight) per qubit
           linear CNOT chain (wire i -> i+1)
    -> [qml.expval(PauliZ(i)) for i in range(d)]      (d measured values)
    -> small MLP: Linear(d, mlp_hidden) -> ReLU -> Linear(mlp_hidden, 1)
    -> risk logit (no sigmoid, BCEWithLogitsLoss -- same convention as HeteroGraphSAGE)
```

Deliberately does **not** do heterogeneous message passing — it reads
only the reduced supplier tensor, nothing from any other node type or
edge_index. This isolates the question the plan asks: does *quantum
processing* of the reduced representation add value over *graph+MLP
processing* (GraphSAGE-Reduced) of the identical input.

`train_qgnn` mirrors `train.train_graphsage`'s loop structure exactly
(same Adam optimizer construction, same class-weighted
`BCEWithLogitsLoss` via the existing `losses.py`, same validation-PR-AUC
early stopping) — the only structural difference is the forward pass
signature. `evaluate_qgnn` reuses `evaluate._finalize_evaluation` directly
(it was already factored out as the model-agnostic "shared tail" of
`evaluate_experiment`/`evaluate_on_target_dataset` from an earlier phase)
— metrics, calibration, risk ranking, onset breakdown, temporal
variation, and plotting are 100% the existing code, unmodified.

## 8. Leakage Controls (plan section 19)

- Reducer fitting uses `supplier_fit_mask(frame, train_examples)` —
  the identical function `FeaturePreprocessor` itself calls, not a
  reimplementation.
- **Verified by test** (`test_reducer_fit_never_uses_validation_or_test_rows`):
  corrupting every non-train row's raw feature values to `1e9` and
  refitting produces bit-identical fitted mean/std/PCA components to the
  uncorrupted fit — proof the fit path never reads those rows.
- No target label is used anywhere in `reduction.py` or `qgnn.py`'s
  forward/encoding path — angle scaling (`tanh(x) * pi`) is a fixed
  deterministic function of the (already leakage-safe) standardized
  input only.
- Threshold selection in `evaluate_qgnn` uses only the validation split,
  via the same `select_threshold` function GraphSAGE-Full/-Reduced use.
- Cross-dataset reduction (`prepare_reduced_for_cross_dataset_eval`)
  takes an already-fit reducer as a parameter and never calls `.fit()` on
  it — mirrors the existing `prepare_for_cross_dataset_eval`'s contract
  for `FeaturePreprocessor`. (Not yet exercised end-to-end — see
  "Remaining Work.")

## 9. Testing

- **Baseline recorded before any change**: 195/195 passing.
- **After installing CUDA PyTorch** (before any new code): 195/195 passing — proves the shared-dependency change didn't affect the existing GraphSAGE baseline.
- **New tests**: `tests/test_reduction.py` (5 tests: PCA output shape, domain-selection uses real columns, rejects unknown columns, leakage test, `prepare_reduced` isolation test) and `tests/test_qgnn.py` (5 tests: forward shape, non-zero gradients on every parameter including the quantum weights, resource-summary reporting, deterministic bounded angle encoding, and a short training run with decreasing loss and valid predictions) — all run on CPU (`default.qubit`) so they pass in any environment.
- **Full suite after implementation**: 205/205 passing (195 original + 10 new).

## 10. Smoke Test (plan section 12), and a Critical Backend Finding

Configuration: `configs/qgnn.yaml` defaults (6 dimensions,
`domain_selected` reduction, 1 variational layer), model seed 42, dataset
`scm_v1_black_swan_seed43`, existing temporal split — run via the real
CLI (`scripts/run_qgnn_experiment.py --tag smoke --seeds 42`), not a
standalone script. All required checks passed: forward pass, backward
pass, non-zero gradients on every parameter (quantum weights included),
correct prediction shape, quantum circuit execution, metrics computation,
and artifact saving (model checkpoint, config, predictions, risk ranking,
metrics/calibration/onset-breakdown JSON, training history, plots, and a
`quantum_resource_summary.json`) all verified under
`experiments/qgnn/*smoke*/`.

**GPU backend measurement, done first with `lightning.gpu` (5 epochs)**:
took **~16.5 minutes**. Profiling traced this to `lightning.gpu`'s
`diff_method="adjoint"` processing a "batched" call as a serialized loop
internally rather than a truly vectorized batch execution — confirmed
directly: a single forward+backward call on 2048 rows took **14.4s** on
`lightning.gpu`+adjoint versus **0.039s** on `default.qubit`+backprop
(CPU) — **~368x faster**, for bit-identical gradients. Restructuring the
training loop to batch across all (supplier, time) rows at once (instead
of one circuit call per timestep snapshot, which GraphSAGE genuinely
needs for message passing but QGNN-Reduced does not) only reduced the
`lightning.gpu` run to ~13.5 minutes — confirming the bottleneck is
per-row compute inside the adjoint execution path, not Python-level call
dispatch overhead.

**Switched the practical default to `default.qubit` (CPU,
`diff_method="backprop"`)** as a result. Verified numerically identical
training dynamics to `lightning.gpu` (same seed, same loss/PR-AUC
trajectory to 4 decimal places) at **~140x** faster wall-clock (5.8s vs
13.5 min for 5 epochs). `lightning.gpu` remains fully implemented,
verified working (confirmed real GPU memory allocation via `nvidia-smi
--query-compute-apps` during execution, correct gradients), and available
via `device_name="lightning.gpu"` — worth revisiting at a much larger
qubit count/circuit depth than this phase uses, where a truly vectorized
GPU statevector simulator would matter; not the right tool at 4-8 qubits.
This is recorded as a negative-adjacent finding per the plan's own
guidance (section 21: "simulation is much slower") rather than optimized
away.

**Full smoke-test run** (`configs/qgnn.yaml` defaults — up to 100 epochs
with early stopping, `default.qubit`, model seed 42, 1 seed only — not
yet a statistically meaningful multi-seed result):

| Model | Test PR-AUC | Test ROC-AUC |
|---|---:|---:|
| GraphSAGE-Reduced (6 dims, domain-selected) | 0.744 | 0.980 |
| QGNN-Reduced (6 qubits, 1 layer) | 0.190 | 0.617 |

Total wall-clock for both models together: **under 2 minutes.** This
single seed is not a conclusion — see "Remaining Work" for the actual
5-seed comparison this needs before it means anything statistically.

## 11. Dimension Sweep and Primary Configuration Selection (plan sections 6, 13-14)

Ran `scripts/run_qgnn_experiment.py` at 4/6/8 dimensions (`domain_selected`
reduction, 1 variational layer), **5 model seeds (42-46) at every
dimension** — the sweep already used the full final seed set, so it
doubles as the final run; no separate 5-seed rerun was needed.

**Selected on validation PR-AUC (plan section 6: never on final test
performance)**:

| Dimension | GraphSAGE-Reduced val PR-AUC | QGNN-Reduced val PR-AUC |
|---:|---:|---:|
| 4 | 0.833 ± 0.015 | 0.178 ± 0.045 |
| 6 | 0.834 ± 0.021 | 0.247 ± 0.039 |
| **8** | 0.825 ± 0.022 | **0.301 ± 0.046** |

QGNN-Reduced's validation PR-AUC climbs monotonically with dimension (a
real, non-overlapping-within-noise difference between 4 and 8 dims);
GraphSAGE-Reduced's is flat across all three (differences well within one
standard deviation of each other) — consistent with GraphSAGE-Reduced's
performance being dominated by its still-intact message passing from
every other node type, not by exactly which 4-8 supplier features it
starts with, while QGNN-Reduced has no such fallback since it reads only
the supplier tensor. Test-split PR-AUC (0.167→0.232→0.284) shows the
identical ordering, confirming the validation-based choice isn't an
artifact of validation-set noise.

**Selected configuration: 8 dimensions, `domain_selected` reduction, 1
variational layer** — highest QGNN-Reduced validation performance, same
circuit depth/simulator cost as 4 and 6 dims (all trivially fast on
`default.qubit`), comparable stability (std 0.046, in line with the other
dimensions).

### Primary and secondary comparisons (plan section 3), 8-dim, seeds 42-46

| Model | Test PR-AUC | Test ROC-AUC | Val PR-AUC | Val ROC-AUC |
|---|---:|---:|---:|---:|
| GraphSAGE-Full (60 dims, existing frozen baseline) | 0.807 ± 0.066 | 0.987 ± 0.008 | 0.880 ± 0.012 | 0.941 ± 0.008 |
| GraphSAGE-Reduced (8 dims) | 0.763 ± 0.032 | 0.983 ± 0.002 | 0.825 ± 0.022 | 0.922 ± 0.004 |
| **QGNN-Reduced (8 qubits, 1 layer)** | **0.284 ± 0.045** | 0.759 ± 0.030 | 0.301 ± 0.046 | 0.737 ± 0.028 |

**Primary comparison (GraphSAGE-Reduced vs QGNN-Reduced, identical 8-dim
input): QGNN-Reduced does not outperform its classical control.** The gap
is large and consistent across all 5 seeds, not a close call — this is
the plan's own anticipated "QGNN < GraphSAGE-Reduced" outcome (section
20: "a valid result").

**Secondary comparison (QGNN-Reduced vs GraphSAGE-Full): QGNN-Reduced
does not recover the performance lost to dimensionality reduction — it
loses far more than the reduction itself costs.** Reducing GraphSAGE from
60 to 8 dimensions costs only ~5% relative test PR-AUC (0.807→0.763);
replacing the classical processing of that same reduced input with the
quantum circuit costs ~65% relative (0.807→0.284). The dimensionality
reduction itself is nearly harmless to a graph-based model that still has
message passing to fall back on; the quantum circuit is not recovering
what a classical MLP-scale head could.

Negative result, preserved as-is (plan section 21): this specific QGNN
configuration (angle encoding, 1 shallow variational layer, linear
entanglement, no graph access) does not add measurable value over its
matched classical control at 4, 6, or 8 qubits.

## 12. Cross-Dataset Evaluation (plan section 17), Full 5 Seeds, Both Directions

Selected configuration (8 dims, `domain_selected`, 1 layer), via
`scripts/run_qgnn_crossdataset_experiment.py`. Source-only preprocessing
and reduction throughout — no target label ever used for fitting,
reduction, or threshold selection (verified by the same leakage pattern
as `evaluate_on_target_dataset`, plus a `test_reducer_fit_never_uses_validation_or_test_rows`-style
guarantee already covered by testing).

| Direction | Model | Condition | PR-AUC | ROC-AUC | Brier | ECE |
|---|---|---|---:|---:|---:|---:|
| seed43→seed44 | GraphSAGE-Reduced | within | 0.763 ± 0.032 | 0.983 ± 0.002 | 0.047 ± 0.007 | 0.053 ± 0.007 |
| seed43→seed44 | GraphSAGE-Reduced | **cross** | **0.053 ± 0.001** | **0.326 ± 0.015** | 0.122 ± 0.004 | 0.133 ± 0.004 |
| seed43→seed44 | QGNN-Reduced | within | 0.284 ± 0.045 | 0.759 ± 0.031 | 0.190 ± 0.022 | 0.354 ± 0.032 |
| seed43→seed44 | QGNN-Reduced | **cross** | **0.072 ± 0.003** | **0.468 ± 0.017** | 0.211 ± 0.015 | 0.345 ± 0.031 |
| seed44→seed43 | GraphSAGE-Reduced | within | 0.949 ± 0.043 | 0.997 ± 0.002 | 0.139 ± 0.017 | 0.178 ± 0.011 |
| seed44→seed43 | GraphSAGE-Reduced | **cross** | **0.299 ± 0.008** | **0.652 ± 0.009** | 0.157 ± 0.017 | 0.191 ± 0.016 |
| seed44→seed43 | QGNN-Reduced | within | **0.059 ± 0.018** | 0.514 ± 0.051 | 0.253 ± 0.013 | 0.457 ± 0.016 |
| seed44→seed43 | QGNN-Reduced | **cross** | **0.043 ± 0.011** | **0.428 ± 0.037** | 0.254 ± 0.012 | 0.460 ± 0.016 |

### Finding 1 — GraphSAGE-Reduced (8 dims) replicates the full-model asymmetry almost exactly

Compare to `GRAPHSAGE_PHASE_A_GENERALIZATION_FINDINGS.md`'s full 60-dim
numbers: seed43→seed44 cross PR-AUC 0.052 (full) vs **0.053** (8-dim
reduced); seed44→seed43 cross PR-AUC 0.294 (full) vs **0.299** (8-dim
reduced). Same below-chance collapse one direction, same real-but-partial
transfer the other. This is the single strongest piece of evidence yet
for `GRAPHSAGE_PHASE_A5_ROOT_CAUSE_FINDINGS.md`'s conclusion that the
asymmetry is a labeling/regional-geography artifact, not a feature-content
one — it survives discarding 52 of 60 input dimensions essentially
unchanged, using a completely different (quantum-benchmark-motivated,
hand-picked) 8-feature subset than any feature-mode ablation tested
before.

### Finding 2 — QGNN-Reduced does not replicate the asymmetry, but not in an encouraging way

GraphSAGE-Reduced swings from catastrophically below chance (ROC-AUC
0.326) to meaningfully above chance (0.652) depending on direction.
QGNN-Reduced stays close to chance in both directions (0.468, 0.428) —
never catastrophic, but never useful either. Its cross-dataset PR-AUC
(0.072, 0.043) is in the same range as the majority-class baseline from
the original full-model D2 experiment (0.079, 0.041) in both directions —
QGNN-Reduced adds no measurable value over guessing the training base
rate, in either direction.

### Finding 3 — unexplained: QGNN-Reduced trained on seed44 performs near-randomly even within-seed

QGNN-Reduced's own within-seed test PR-AUC is 0.284 when trained on
seed43, but only **0.059** (ROC-AUC 0.514, indistinguishable from chance)
when trained on seed44 — worse than its own cross-dataset transfer *into*
seed43 would suggest, and far below GraphSAGE-Reduced's 0.949 within-seed44
result on the identical reduced input. This is reproducible across all 5
seeds (std 0.018, not noise) but its cause is not established — plausibly
an optimization-difficulty interaction between seed44's different class
balance (6.08% vs 3.33% prevalence, different `pos_weight`) and this
small circuit's shallow, low-parameter-count optimization landscape, but
that is a hypothesis, not a diagnosed mechanism. Recorded as an open,
unresolved negative finding rather than explained away.

### Finding 4 — QGNN-Reduced's cross-dataset onset recall is higher but not trustworthy

Cross-dataset fresh-onset recall: GraphSAGE-Reduced 0.000 / 0.313
(seed43→44 / seed44→43) vs QGNN-Reduced 0.227 ± 0.121 / 0.475 ± 0.276.
QGNN does catch more fresh onsets — but its expected calibration error is
2.5-6x worse than GraphSAGE-Reduced's in every condition (0.34-0.46 vs
0.05-0.19), and the seed44→43 fresh-recall figure has a standard
deviation (0.276) nearly as large as its own mean — highly unstable
across seeds, not a reliable property of the model. Read together with
QGNN-Reduced's much lower PR-AUC, this looks like a poorly-calibrated
model crossing its decision threshold more liberally (catching more
positives at the cost of many more false ones) rather than genuine
discriminative skill at anticipating fresh onsets. Reported plainly,
without recommending any action from it.

## 13. Remaining Work

Not yet done (optional per the plan; to be reported in
`QGNN_FINAL_BENCHMARK.md` if pursued):

- Ablations: reduction method (PCA vs domain-selected at 8 dims), circuit depth (1 vs 2 layers) (plan section 18).
- `QGNN_FINAL_BENCHMARK.md` itself.

The plan's core research question (does QGNN add measurable value over
its matched classical control) already has a clear, consistent, 5-seed,
both-direction answer: no, in every condition tested.
