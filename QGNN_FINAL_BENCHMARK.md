# QGNN Final Benchmark

Executes `QGNN_FAIR_BENCHMARK_AND_IMPLEMENTATION_PLAN.md` in full. Source
documents: `QGNN_IMPLEMENTATION_AND_BENCHMARK.md` (implementation detail,
environment verification, smoke test, dimension sweep, cross-dataset
evaluation), `GRAPHSAGE_FINAL_GENERALIZATION_AND_IMPROVEMENT_PLAN.md`,
`GRAPHSAGE_PHASE_A_GENERALIZATION_FINDINGS.md`,
`GRAPHSAGE_PHASE_A5_ROOT_CAUSE_FINDINGS.md`, `GRAPHSAGE_WORK_SUMMARY.md`.
All training was executed by the user; this document consolidates results
already produced and verified, and adds the two remaining ablations
(reduction method, circuit depth).

---

## 1. Executive Summary

**Research question**: does a hybrid quantum circuit add measurable
predictive value over classical processing, when both receive the exact
same reduced supply-chain risk representation?

**Design**: three controlled models — GraphSAGE-Full (existing 60-dim
frozen baseline), GraphSAGE-Reduced (the identical GraphSAGE architecture
fed a reduced input), and QGNN-Reduced (a shallow variational quantum
circuit + small MLP fed the identical reduced input) — evaluated on 4/6/8
dimensions, 5 model seeds, within-dataset and true cross-dataset
generalization, and two ablations (reduction method, circuit depth).

**Principal result**: across every configuration tested — 3
dimensionalities, 2 reduction methods, 2 circuit depths, 2 cross-dataset
directions, 5 seeds each — **QGNN-Reduced never outperformed its matched
GraphSAGE-Reduced control.** The gap narrowed meaningfully with a deeper
circuit (1→2 variational layers) but was not closed by it, and was not
explained by the choice of reduction method.

**Important limitation**: this evaluates one specific shallow hybrid
architecture (angle encoding, linear entanglement, 1-2 layers, 4-8
qubits) in simulation. It does not test deeper circuits, alternative
ansätze, real quantum hardware, or claim anything about quantum computing
in general.

## 2. Research Questions

1. **Primary**: does QGNN-Reduced outperform GraphSAGE-Reduced when both receive an identical reduced representation?
2. **Secondary**: does QGNN-Reduced recover any of the performance GraphSAGE loses to dimensionality reduction (vs GraphSAGE-Full)?
3. **Robustness**: is the primary result sensitive to reduction method (PCA vs domain-selected) or circuit depth (1 vs 2 layers)?
4. **Generalization**: does either reduced model transfer across independently-generated dataset realizations (seed43 ↔ seed44)?

## 3. Experimental Design

- **Dataset**: `scm_v1_black_swan_seed43` (primary), cross-dataset target `scm_v1_black_swan_seed44`. Unchanged from the existing GraphSAGE benchmark.
- **Task**: `supplier_disrupted` binary classification, 4-period prediction horizon. Unchanged.
- **Graph**: existing heterogeneous SCM graph (6 node types, 9 base edge types + PyG reverse relations). Unchanged; GraphSAGE-Reduced/-Full both do full heterogeneous message passing over it. QGNN-Reduced reads only the reduced supplier tensor (see §4).
- **Split**: existing temporal split (0.7/0.15/0.15 train/val/test). Unchanged.
- **Seeds**: model seeds 42-46 for every reported multi-seed result. Dataset seeds (43, 44) are a separate axis — never conflated with model seeds.
- **Preprocessing**: existing `FeaturePreprocessor` (train-only fit) for every node type except the reduced supplier representation, which uses a separate reducer (§ below), also train-only fit.
- **Reduction**: `PCASupplierReducer` (26 raw numeric supplier columns → standardize with train-only mean/std → `sklearn.PCA`, train-only fit) and `DomainSelectedReducer` (a fixed, existing-column subset — `reliability`, `financial_health`, `capacity_utilization`, `criticality`, `geopolitical_exposure`, `cyber_exposure`, `lead_time_mean`, `quality_score` at 8 dims — standardized with train-only mean/std). Both in `src/scm_dataset/modeling/reduction.py`.
- **GraphSAGE**: existing, unmodified `HeteroGraphSAGE` (2-layer heterogeneous SAGEConv, hidden_dim 128, dropout 0.20, mean aggregation). GraphSAGE-Full: supplier input dim 60. GraphSAGE-Reduced: supplier input dim 4/6/8 (every other node type unchanged).
- **QGNN**: reduced supplier vector → `angle = tanh(x) * π` (deterministic, label-free) → `AngleEmbedding` (RY, d qubits) → 1 or 2 variational layers (trainable RY per qubit + linear CNOT chain) → PauliZ expectation per qubit → small MLP (`Linear(d, 8) → ReLU → Linear(8, 1)`) → logit. `src/scm_dataset/modeling/qgnn.py`.
- **Training**: Adam, lr 0.001, weight_decay 0.0001, class-weighted `BCEWithLogitsLoss`, early stopping on validation PR-AUC (patience 10), up to 100 epochs — identical hyperparameters for GraphSAGE-Reduced and QGNN-Reduced (`train_qgnn` mirrors `train_graphsage`'s setup exactly).
- **Backend**: PennyLane, `default.qubit` device, `diff_method="backprop"` (CPU) — see §12 for why.

## 4. Fairness Controls

GraphSAGE-Reduced and QGNN-Reduced are the **matched primary comparison**
because they are given byte-for-byte the same reduced supplier input
tensor at every timestep (same reducer object, same fitted parameters,
same train/val/test split, same labels, same class-weighting policy, same
early-stopping rule, same evaluation code). The only thing that differs
between them is what happens *after* that input: GraphSAGE-Reduced still
does full heterogeneous message passing from every other node type;
QGNN-Reduced processes the reduced supplier vector alone through a
quantum circuit. This isolates the question "does quantum processing of
this representation add value" from "how much does reducing to d
dimensions cost," which GraphSAGE-Full vs GraphSAGE-Reduced answers
separately (§5, §6, Question 1).

Verified (not assumed): `ReducedGraphSnapshotBuilder` produces
bit-identical tensors to the full-dimensional builder for every node type
except supplier, and identical `edge_index` for every relation
(`test_prepare_reduced_only_changes_supplier_dimension`). Reducer fitting
never reads validation/test rows (`test_reducer_fit_never_uses_validation_or_test_rows`,
tampering test).

## 5. Baseline Results (primary dataset, temporal split, test)

| Model | Dim | PR-AUC | ROC-AUC | Precision | Recall | F1 | Bal. Acc. | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GraphSAGE-Full (existing frozen baseline) | 60 | 0.807 ± 0.066 | 0.987 ± 0.008 | 0.522 ± 0.042 | 0.968 ± 0.016 | 0.677 ± 0.035 | 0.952 ± 0.009 | 0.042 ± 0.012 |
| GraphSAGE-Reduced (domain-selected) | 8 | 0.763 ± 0.032 | 0.983 ± 0.002 | 0.498 ± 0.030 | 0.941 ± 0.044 | 0.650 ± 0.023 | 0.936 ± 0.019 | 0.047 ± 0.007 |
| **QGNN-Reduced (domain-selected, 1 layer)** | 8 | **0.284 ± 0.045** | 0.759 ± 0.031 | 0.155 ± 0.031 | 0.570 ± 0.114 | 0.237 ± 0.031 | 0.662 ± 0.014 | 0.190 ± 0.022 |

## 6. Dimension Selection

Ran 4, 6, 8 dimensions (`domain_selected`, 1 layer), 5 seeds each.
**Selected on validation PR-AUC** — never final test performance:

| Dimension | GraphSAGE-Reduced val PR-AUC | QGNN-Reduced val PR-AUC |
|---:|---:|---:|
| 4 | 0.833 ± 0.015 | 0.178 ± 0.045 |
| 6 | 0.834 ± 0.021 | 0.247 ± 0.039 |
| **8** | 0.825 ± 0.022 | **0.301 ± 0.046** |

QGNN-Reduced's validation performance climbed monotonically with
dimension (a real difference between 4 and 8 dims, not overlapping within
one standard deviation); GraphSAGE-Reduced's stayed flat (differences
well within one standard deviation of each other), consistent with
GraphSAGE-Reduced's performance being carried mostly by intact message
passing rather than exactly which 4-8 supplier features it starts with.
**8 dimensions selected**, matching the plan's own feasible-qubit range
and giving QGNN-Reduced its best validated configuration.

## 7. Reduction Ablation (8 dims, 1 layer, test/validation, 5 seeds)

| Reduction | Model | Test PR-AUC | Test ROC-AUC | Val PR-AUC | Val ROC-AUC | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| Domain-selected | GraphSAGE-Reduced | 0.763 ± 0.032 | 0.983 ± 0.002 | 0.825 ± 0.022 | 0.922 ± 0.004 | 0.047 ± 0.007 | 0.053 ± 0.007 |
| Domain-selected | QGNN-Reduced | 0.284 ± 0.045 | 0.759 ± 0.031 | 0.301 ± 0.046 | 0.737 ± 0.028 | 0.190 ± 0.022 | 0.354 ± 0.032 |
| PCA | GraphSAGE-Reduced | 0.739 ± 0.107 | 0.969 ± 0.026 | 0.860 ± 0.024 | 0.920 ± 0.016 | 0.052 ± 0.009 | 0.057 ± 0.012 |
| PCA | QGNN-Reduced | 0.314 ± 0.115 | 0.771 ± 0.037 | 0.317 ± 0.113 | 0.737 ± 0.040 | 0.175 ± 0.028 | 0.331 ± 0.030 |

PCA's mean QGNN test PR-AUC (0.314) is nominally higher than
domain-selected's (0.284), but with more than double the standard
deviation (0.115 vs 0.045) — the two reduction methods' per-seed ranges
overlap substantially (domain-selected: 0.203-0.328; PCA: 0.169-0.437),
so this is not a clear, reliable improvement. **The primary gap persists
under both reduction methods**: GraphSAGE-Reduced beats QGNN-Reduced by
roughly the same large margin (≈0.48 domain-selected, ≈0.43 PCA)
regardless of which reduction is used. Reduction method choice does not
explain the negative result.

## 8. Circuit Depth Ablation (8 dims, domain-selected, test/validation, 5 seeds)

| Layers | Quantum params | Test PR-AUC | Test ROC-AUC | Val PR-AUC | Val ROC-AUC | Brier | ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.284 ± 0.045 | 0.759 ± 0.031 | 0.301 ± 0.046 | 0.737 ± 0.028 | 0.190 ± 0.022 | 0.354 ± 0.032 |
| **2** | 16 | **0.388 ± 0.064** | **0.833 ± 0.067** | 0.398 ± 0.066 | 0.790 ± 0.059 | 0.156 ± 0.031 | 0.302 ± 0.048 |

**Paired per-seed difference (2 layers − 1 layer), test PR-AUC**:

| Seed | 1 layer | 2 layers | Δ |
|---:|---:|---:|---:|
| 42 | 0.322 | 0.416 | +0.094 |
| 43 | 0.327 | 0.402 | +0.075 |
| 44 | 0.300 | 0.442 | +0.142 |
| 45 | 0.270 | 0.263 | −0.007 |
| 46 | 0.203 | 0.419 | +0.216 |

Mean paired Δ = **+0.104 ± 0.074**. 4 of 5 seeds show a clear, sizeable
improvement; one seed (45) shows essentially no change. No formal
significance test was run (none was warranted by the plan or performed
here — this is a direction-and-magnitude read, not a p-value claim), but
a consistent same-signed effect in 4 of 5 paired seeds, of a magnitude
(~0.10 mean, on a base of 0.28) larger than either configuration's own
seed-to-seed standard deviation, is a genuine effect, not sampling noise.

**Circuit depth matters.** Going from 1 to 2 variational layers
meaningfully improved QGNN-Reduced on every metric — PR-AUC, ROC-AUC, and
calibration (Brier 0.190→0.156, ECE 0.354→0.302) all moved in the same
favorable direction. This means the original 1-layer result understated
what this circuit family can do. **It did not, however, close the gap**:
even 2-layer QGNN-Reduced (0.388) remains far below GraphSAGE-Reduced
(0.763, bit-identical across both ablation runs, confirming
reproducibility) on the identical input.

## 9. Cross-Dataset Generalization

Preserved from `QGNN_IMPLEMENTATION_AND_BENCHMARK.md` §12, not rerun (no
implementation problem to repair). 8 dims, domain-selected, 1 layer, 5
seeds, both directions, source-only preprocessing/reduction throughout:

| Direction | Model | Condition | PR-AUC | ROC-AUC | Brier | ECE |
|---|---|---|---:|---:|---:|---:|
| seed43→seed44 | GraphSAGE-Reduced | within | 0.763 ± 0.032 | 0.983 ± 0.002 | 0.047 ± 0.007 | 0.053 ± 0.007 |
| seed43→seed44 | GraphSAGE-Reduced | **cross** | 0.053 ± 0.001 | 0.326 ± 0.015 | 0.122 ± 0.004 | 0.133 ± 0.004 |
| seed43→seed44 | QGNN-Reduced | within | 0.284 ± 0.045 | 0.759 ± 0.031 | 0.190 ± 0.022 | 0.354 ± 0.032 |
| seed43→seed44 | QGNN-Reduced | **cross** | 0.072 ± 0.003 | 0.468 ± 0.017 | 0.211 ± 0.015 | 0.345 ± 0.031 |
| seed44→seed43 | GraphSAGE-Reduced | within | 0.949 ± 0.043 | 0.997 ± 0.002 | 0.139 ± 0.017 | 0.178 ± 0.011 |
| seed44→seed43 | GraphSAGE-Reduced | **cross** | 0.299 ± 0.008 | 0.652 ± 0.009 | 0.157 ± 0.017 | 0.191 ± 0.016 |
| seed44→seed43 | QGNN-Reduced | within | 0.059 ± 0.018 | 0.514 ± 0.051 | 0.253 ± 0.013 | 0.457 ± 0.016 |
| seed44→seed43 | QGNN-Reduced | **cross** | 0.043 ± 0.011 | 0.428 ± 0.037 | 0.254 ± 0.012 | 0.460 ± 0.016 |

GraphSAGE-Reduced (8 dims) reproduces the full 60-dim model's
direction-asymmetric cross-dataset behavior almost exactly (cross PR-AUC
0.053 vs the full model's 0.052 in the failing direction; 0.299 vs 0.294
in the transferring direction — see
`GRAPHSAGE_PHASE_A_GENERALIZATION_FINDINGS.md`). **This result provides
supporting evidence** for `GRAPHSAGE_PHASE_A5_ROOT_CAUSE_FINDINGS.md`'s
conclusion that the asymmetry is a property of the two datasets' disjoint
disruption-region labeling rather than of any specific feature set — it
survived discarding 52 of 60 input dimensions under a completely
different (quantum-benchmark-motivated) feature selection. This is not
offered as definitive causal proof, only as a second, independent piece
of evidence consistent with the earlier finding.

QGNN-Reduced does not reproduce this asymmetry, but not favorably: its
cross-dataset ROC-AUC stays close to chance in both directions (0.468,
0.428) rather than swinging from catastrophic to genuinely useful, and
its cross-dataset PR-AUC (0.072, 0.043) sits at roughly the majority-class
baseline level in both directions (from the original full-model D2
experiment: 0.079, 0.041) — no measurable transfer value in either
direction. This negative cross-dataset result is not reinterpreted as
success on the basis of avoiding GraphSAGE's catastrophic failure mode;
avoiding catastrophe by never working in the first place is not the same
as generalizing.

**Unresolved finding (not rerun here, not explained)**: QGNN-Reduced
trained and evaluated entirely within seed44 scores near chance
(PR-AUC 0.059, ROC-AUC 0.514, reproducible across all 5 seeds — not
noise) despite GraphSAGE-Reduced scoring 0.949 on the identical input.
Neither the reduction-method nor the circuit-depth ablation in this
document varied the training dataset, so neither speaks to this anomaly
directly. A plausible but undiagnosed hypothesis is an optimization
interaction between seed44's different class balance (6.08% vs 3.33%
prevalence, hence different `pos_weight`) and this small circuit's
shallow, low-parameter-count landscape — this is a hypothesis, not an
established mechanism, and is preserved as an open finding rather than
resolved here.

## 10. Calibration and Risk Ranking

QGNN-Reduced is substantially worse-calibrated than either GraphSAGE
variant in every configuration tested: ECE 0.30-0.46 for QGNN vs 0.05-0.19
for GraphSAGE-Reduced/-Full, a 2.5-9x gap depending on condition. The
2-layer ablation improved QGNN's calibration somewhat (ECE 0.354→0.302)
alongside its PR-AUC improvement, but it remains far behind GraphSAGE's
calibration quality. Risk-ranking artifacts (`supplier_risk_ranking.csv`)
were produced for every run via the existing, unmodified
`build_risk_ranking` — no QGNN-specific ranking logic was introduced.

## 11. Fresh-Onset / Temporal Analysis

Fresh-onset recall remains at or near zero for GraphSAGE-Reduced/-Full in
every within-dataset condition, consistent with the standing, independently-established
finding (`GRAPH_SAGE_IMPROVEMENT_PLAN.md`'s Q2, replicated repeatedly
since) that this benchmark's generator provides no genuine pre-event
signal. Cross-dataset, QGNN-Reduced showed nominally higher fresh-onset
recall (0.227 ± 0.121, 0.475 ± 0.276) than GraphSAGE-Reduced (0.000,
0.313) — **not interpreted as a QGNN strength**: QGNN's expected
calibration error is 2.5-6x worse in the same conditions, and the
seed44→seed43 recall figure's own standard deviation (0.276) is nearly as
large as its mean, i.e. highly seed-unstable. This reads as a
poorly-calibrated model crossing its decision threshold more liberally
(more true and false positives together) rather than genuine
discriminative skill, and is reported as such rather than overinterpreted.

## 12. Computational Resource Analysis

- **Backend**: PennyLane `default.qubit` (CPU), `diff_method="backprop"` — the practical default for every reported result in this document, including the ablations, which used CPU simulation throughout (recorded explicitly here per the plan's own instruction).
- **Why not GPU**: `pennylane-lightning-gpu` was installed and verified working on the available NVIDIA RTX 5060 Ti (confirmed real GPU memory allocation via `nvidia-smi --query-compute-apps` during execution, correct non-zero gradients via `diff_method="adjoint"`). Direct, controlled measurement at this qubit count found `default.qubit`+backprop **~140-368x faster** than `lightning.gpu`+adjoint for numerically identical results — adjoint differentiation on `lightning.gpu` appears to process a "batched" call as a serialized loop internally rather than a truly vectorized batch execution, while PyTorch's native backprop vectorizes the whole batch as tensor ops. GPU usage was not forced merely to claim GPU acceleration, per the plan's own instruction; the GPU implementation was not deleted and remains available via `device_name="lightning.gpu"`.
- **Circuit size**: 4/6/8 qubits (= reduced dimensionality), 1 or 2 variational layers, 4/6/8 (1 layer) or 8/12/16 (2 layers) trainable quantum parameters, 53-97 total trainable parameters (quantum + MLP) depending on configuration.
- **Training time**: full 100-epoch (with early stopping) run for both GraphSAGE-Reduced and QGNN-Reduced together, on `default.qubit`, completed in under 2 minutes wall-clock for a single seed on the full ~26,700-example primary dataset — the entire 4/6/8 dimension sweep (5 seeds each) and both ablations (5 seeds each) together took well under an hour.
- **No quantum computational advantage is claimed anywhere in this document.** CUDA availability is reported as a simulation-backend fact, not evidence of quantum speedup.

## 13. Leakage and Reproducibility Audit

| Control | Status |
|---|---|
| Reducer (PCA and domain-selected) fit uses training rows only | Verified by test (`test_reducer_fit_never_uses_validation_or_test_rows`) — tampering all non-train rows to `1e9` and refitting produces bit-identical fitted parameters |
| Domain-selected features are label-free | By construction — fixed existing static supplier columns, no label ever read |
| Quantum encoding is label-free | By construction — `angle = tanh(x) * π` is a deterministic function of the already leakage-safe standardized input only |
| Class weights use training data only | Unchanged existing `compute_pos_weight`/`build_loss`, called on `train_ex["target"]` only |
| Threshold selection uses validation data only | Unchanged existing `select_threshold`, called on the validation split's predictions only, for both GraphSAGE-Reduced and QGNN-Reduced |
| Test labels used only for final evaluation | Unchanged existing `_finalize_evaluation` reused verbatim for QGNN |
| Cross-dataset preprocessing/reduction uses source-fitted transformations | `prepare_reduced_for_cross_dataset_eval` takes an already-fit reducer as a parameter and never calls `.fit()` on it |
| No target-world fitting during cross-dataset evaluation | Same function; target benchmark's graph/labels are only ever scored, never used for fitting |
| Dimension/reduction-method/circuit-depth never selected on test performance | Selected on validation PR-AUC throughout (§6); test numbers reported only after selection |

Full test suite re-run fresh (not assumed) before this document was
written: **205/205 passing** (195 pre-existing GraphSAGE tests + 10 QGNN/
reduction-specific tests). No new test was added for the two ablations —
both exercise code paths (`PCASupplierReducer`, `n_layers` parameter)
already covered by existing tests (`test_pca_reducer_output_shape_and_no_nan`,
`test_qgnn_quantum_resource_summary_reports_expected_fields` with
`n_layers=2`), per the plan's own instruction to add a test only for a
genuinely new pathway.

## 14. Findings

### Supported findings (direct measurement, 5 seeds)

- QGNN-Reduced did not outperform GraphSAGE-Reduced at any tested dimension (4/6/8), reduction method (PCA/domain-selected), or circuit depth (1/2 layers).
- Circuit depth (1→2 layers) produced a real, mostly-consistent (4/5 seeds) improvement in QGNN-Reduced's PR-AUC, ROC-AUC, and calibration — but did not close the gap with GraphSAGE-Reduced.
- Reduction method (PCA vs domain-selected) did not produce a reliable, direction-clear difference in QGNN-Reduced's performance — both stayed far below GraphSAGE-Reduced.
- GraphSAGE-Reduced (8 dims) reproduces GraphSAGE-Full's direction-asymmetric cross-dataset behavior (catastrophic below-chance failure one direction, real partial transfer the other) to within ~0.005 PR-AUC of the full 60-dim model's own numbers.
- QGNN-Reduced's calibration is substantially worse than either GraphSAGE variant's in every condition tested (ECE 2.5-9x higher).

### Plausible hypotheses (not established)

- QGNN-Reduced's near-chance within-seed44 performance may stem from an interaction between seed44's different class prevalence and this circuit's shallow optimization landscape — not diagnosed.
- QGNN-Reduced's elevated cross-dataset fresh-onset recall likely reflects poor calibration (a more liberal decision threshold) rather than genuine discriminative skill — consistent with, not proven by, the calibration numbers.

### Unresolved findings

- The mechanism behind the seed44 within-dataset anomaly (§9) remains unknown.
- Whether a 3+ layer circuit would narrow the gap further (as 2 layers did over 1) was not tested — the plan explicitly scoped this ablation to 1 vs 2 layers only, "do not tune layer counts beyond 1 and 2."
- Whether the reduction-method result would clarify with more than 5 seeds (given PCA's much higher seed-to-seed variance) is open.

## 15. Limitations

- QGNN-Reduced does not perform heterogeneous message passing — it is architecturally excluded from using graph-relational information GraphSAGE-Reduced retains, by design (§4), which is itself a partial explanation for the gap, not a flaw in the experiment.
- Only a shallow circuit family was tested: angle encoding, linear CNOT entanglement, 1-2 variational layers.
- Only 4-8 qubit regimes were evaluated — larger circuits were out of scope per the plan's own "do not attempt to encode the entire graph" and "keep the first sweep small" instructions.
- The benchmark is synthetic; all findings are scoped to this generator's specific event/labeling structure (see `GRAPHSAGE_PHASE_A5_ROOT_CAUSE_FINDINGS.md` for the independent evidence that cross-world transfer itself is generator-structure-dependent, now further supported by §9).
- Two independent dataset realizations (seed43, seed44) were used for cross-dataset evaluation — this is not five independent synthetic worlds; direction-specific findings should be read as "observed on two dataset realizations," not as broadly generalizable.
- All quantum execution was simulator-based (`default.qubit`, verified numerically consistent with `lightning.gpu`) — no real quantum hardware was used, and no claim of quantum computational advantage is made anywhere in this document.
- Every QGNN result in this document applies specifically to the evaluated architecture and configuration — it is not a general claim about quantum machine learning, variational circuits at other scales, or other encoding/ansatz choices.

## 16. Final Conclusion

Across the evaluated 4-, 6-, and 8-qubit configurations, QGNN-Reduced did
not outperform the matched GraphSAGE-Reduced control at any point in this
benchmark. The result persisted across five model seeds, across both
reduction methods tested (domain-selected and PCA), and — while
meaningfully narrowed by increasing circuit depth from one to two
variational layers — was not reversed by that increase either. Within the
scope of this benchmark (a shallow hybrid angle-encoded variational
circuit at 4-8 qubits, evaluated in simulation on a synthetic
supply-chain disruption benchmark), **the quantum component did not
provide measurable predictive benefit over the corresponding classical
processing pipeline given the same input.** Circuit depth was shown to be
a real, non-trivial lever on QGNN-Reduced's performance, suggesting the
gap is not necessarily a hard ceiling for this architecture family at
larger depths — a question this benchmark did not test and does not
answer. Separately, this benchmark's cross-dataset experiments add
further, independent support (not proof) to the earlier finding that this
project's synthetic environment induces dataset-specific transfer
instability largely independent of which features or how many dimensions
a model is given.

## 17. Reproducibility

```bash
# Dimension sweep (4/6/8 dims, domain-selected, 1 layer)
python scripts/run_qgnn_experiment.py --config configs/qgnn.yaml --tag dimension_4 --n-components 4 --seeds 42,43,44,45,46
python scripts/run_qgnn_experiment.py --config configs/qgnn.yaml --tag dimension_6 --n-components 6 --seeds 42,43,44,45,46
python scripts/run_qgnn_experiment.py --config configs/qgnn.yaml --tag dimension_8 --n-components 8 --seeds 42,43,44,45,46

# Reduction ablation (8 dims, PCA)
python scripts/run_qgnn_experiment.py --config configs/qgnn.yaml --tag reduction_pca_dim8 --n-components 8 --reduction-method pca --seeds 42,43,44,45,46

# Circuit-depth ablation (8 dims, domain-selected, 2 layers)
python scripts/run_qgnn_experiment.py --config configs/qgnn.yaml --tag depth2_dim8 --n-components 8 --reduction-method domain_selected --n-layers 2 --seeds 42,43,44,45,46

# Cross-dataset evaluation (8 dims, domain-selected, 1 layer), both directions
python scripts/run_qgnn_crossdataset_experiment.py --config configs/qgnn.yaml --source-dataset-id scm_v1_black_swan_seed43 --target-dataset-id scm_v1_black_swan_seed44 --n-components 8 --reduction-method domain_selected --seeds 42,43,44,45,46
python scripts/run_qgnn_crossdataset_experiment.py --config configs/qgnn.yaml --source-dataset-id scm_v1_black_swan_seed44 --target-dataset-id scm_v1_black_swan_seed43 --n-components 8 --reduction-method domain_selected --seeds 42,43,44,45,46
```

All runs write reproducible artifacts (config, model checkpoint,
predictions, metrics, calibration, onset breakdown, training history,
plots, quantum resource summary, run metadata including git commit and
timestamp) under `experiments/qgnn/`, following the existing project's
artifact conventions — no run directory is ever overwritten
(`new_run_dir` raises `FileExistsError` rather than clobber).
