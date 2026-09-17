# QGNN-v2 Data Re-Uploading Benchmark

Controlled follow-up to `QGNN_V2_DEPTH_BENCHMARK.md`'s selected depth-3
configuration. Executed exactly the plan's single-question experiment:
does re-injecting the 8D input between variational layers improve the
already-selected depth-3 QGNN? All training run by the user via
`scripts/run_qgnn_v2_reupload_experiment.py`.

---

## 1. Objective

> Test whether data re-uploading improves the expressive power of the
> existing depth-3 QGNN while keeping the comparison scientifically
> controlled.

Not a hyperparameter search — one architectural change, isolated.

## 2. Architecture

```
input x (8D, identical to the depth-3 baseline)
    -> angle encoding
    -> variational layer 1
    -> data re-upload of x
    -> variational layer 2
    -> data re-upload of x
    -> variational layer 3
    -> measurement
    -> same MLP head (Linear(8,8) -> ReLU -> Linear(8,1))
```

Implemented in `src/scm_dataset/modeling/qgnn_v2_reupload.py`
(`QuantumCircuitLayerReupload`, `QGNNReupload`) — purely additive; `qgnn.py`
and `qgnn_v2.py` were not modified. Verified directly by inspecting the
PennyLane tape (not assumed): the baseline circuit contains exactly 1
`AngleEmbedding` operation regardless of depth; the re-upload circuit
contains exactly `n_layers` (3, for depth-3), positioned strictly between
layers — not before layer 1, not after the last layer — exactly matching
the requested diagram.

## 3. Fairness Controls

Both models, for every seed, shared: the same frozen GraphSAGE-Full
checkpoint for that seed, the same extracted embeddings, the same
train-only-fit PCA (same fitted object, not refit), the same train/
validation/test rows, the same seed for both weight initialization and
data shuffling, the same optimizer/learning rate/weight decay/class
weighting/early-stopping/evaluation code. Confirmed by re-running the
baseline (A) through this new script and finding it bit-identical to the
original depth-3 result from `QGNN_V2_DEPTH_BENCHMARK.md`: validation
PR-AUC 0.6499 ± 0.1022 both times, to 4 decimal places.

## 4. Implementation Details

New files only: `src/scm_dataset/modeling/qgnn_v2_reupload.py`,
`scripts/run_qgnn_v2_reupload_experiment.py`,
`tests/test_qgnn_v2_reupload.py`. Reused unchanged: `qgnn_v2.py`'s
`V2PreparedData`/`build_v2_prepared`/`train_v2_head`/`evaluate_v2`/
`generate_predictions_v2` (all already model-agnostic) and
`graph_embedding_reduction.py`'s frozen-encoder + PCA pipeline.

## 5. Tests

7 new tests, `tests/test_qgnn_v2_reupload.py`, all passing:

| Checklist item | Verified by |
|---|---|
| 8 qubits, depth 3 | `test_reupload_model_has_8_qubits_and_depth_3` |
| Data re-uploaded at each intended layer | `test_data_is_reuploaded_between_every_pair_of_consecutive_layers` — direct PennyLane tape inspection, checked at depths 1/2/3/5, plus explicit ordering check for depth 3 |
| Output shape matches existing QGNN | `test_reupload_output_shape_matches_baseline_qgnn` |
| Classical MLP head unchanged | `test_reupload_classical_mlp_head_is_architecturally_unchanged` (shapes and parameter count compared directly) |
| No gradients into frozen encoder | `test_reupload_end_to_end_through_frozen_encoder_no_leakage` |
| PCA train-only | Same test — reuses the already-tested, unmodified `build_v2_prepared`/reducer |
| No target leakage | Same test |
| Existing QGNN-v2/GraphSAGE tests still pass | Full suite re-run: **224/224 passing** (217 pre-existing + 7 new) |

Also verified: identical trainable quantum parameter count to baseline
at every tested depth (`test_reupload_has_identical_trainable_quantum_parameter_count_to_baseline`)
— re-uploading adds gates, not parameters, and this is checked, not
assumed.

## 6. Smoke-Test Result

1 seed, 5 epochs, both models: completed in ~30s, finite gradients on
every parameter (including the quantum weights), bounded `[0,1]`
predictions, no crashes, no NaNs. Confirmed a real distinction from the
baseline (`data_reuploads: 2`, `total_angle_embedding_calls: 3` in the
saved resource summary). Not treated as a scientific result.

## 7. Five-Seed Results (seeds 42-46)

| | Test PR-AUC | Test ROC-AUC | Val PR-AUC | Brier (test) | ECE (test) |
|---|---:|---:|---:|---:|---:|
| **A. Baseline depth-3** | 0.6032 ± 0.1132 | 0.9369 ± 0.0205 | 0.6499 ± 0.1022 | 0.0875 ± 0.0220 | 0.1585 ± 0.0361 |
| **B. Depth-3 + data re-upload** | 0.5438 ± 0.1574 | 0.9000 ± 0.0456 | 0.6352 ± 0.1297 | 0.0875 ± 0.0133 | 0.1578 ± 0.0237 |

Full metrics (precision/recall/F1/balanced accuracy) recorded in the
saved run artifacts; PR-AUC/ROC-AUC/calibration are the ones that matter
for this comparison and are reported above in full.

## 8. Per-Seed Comparison

Test PR-AUC:

| Seed | A. Baseline | B. Re-upload |
|---:|---:|---:|
| 42 | 0.7216 | 0.5973 |
| 43 | 0.6471 | 0.5476 |
| 44 | 0.3934 | 0.2432 |
| 45 | 0.5884 | 0.6878 |
| 46 | 0.6654 | 0.6432 |

Validation PR-AUC (the split that matters for this comparison, per the
plan's own instruction not to select on test):

| Seed | A. Baseline | B. Re-upload |
|---:|---:|---:|
| 42 | 0.8024 | 0.6972 |
| 43 | 0.6937 | 0.6180 |
| 44 | 0.5058 | 0.3957 |
| 45 | 0.5741 | 0.6901 |
| 46 | 0.6734 | 0.7752 |

## 9. Paired Deltas (data-reupload − baseline)

**Test PR-AUC**: per-seed `[-0.1243, -0.0996, -0.1501, +0.0994, -0.0222]`
— mean **−0.0594**, std 0.0902, **1 seed improving, 4 degrading**.

**Validation PR-AUC**: per-seed `[-0.1052, -0.0757, -0.1101, +0.1160,
+0.1018]` — mean **−0.0146**, std 0.1016, **2 seeds improving, 3
degrading**.

**The validation-based result is meaningfully more ambiguous than the
test-based one.** Test PR-AUC alone would suggest a fairly clear
regression (4/5 seeds, mean −0.059); validation PR-AUC — the only split
this decision should be based on — shows a much smaller mean effect
(−0.015, an order of magnitude smaller) with the seeds nearly split
down the middle (2 up, 3 down). Neither split shows a *positive*,
consistent effect; the honest reading is that re-uploading did not help,
and on the correct (validation) criterion the evidence is closer to "no
clear effect" than to "clearly worse" — but it is not an improvement
either way.

## 10. Calibration Comparison

Nearly identical between the two models: test Brier 0.0875 vs 0.0875
(equal to 4 decimal places), test ECE 0.1585 vs 0.1578, validation
likewise close. Data re-uploading neither helped nor hurt calibration
— unlike the depth sweep (where depth 3 was both the best-performing
*and* best-calibrated point among depths 1-5), this change is calibration-neutral.

## 11. Resource Comparison

| | A. Baseline | B. Data re-upload |
|---|---:|---:|
| Qubits | 8 | 8 |
| Circuit depth (variational layers) | 3 | 3 |
| Trainable quantum parameters | 24 | 24 (identical, verified by test) |
| Total trainable parameters | 105 | 105 |
| `AngleEmbedding` calls per forward pass | 1 | 3 |
| Mean training time (5 seeds) | 108.2s ± 25.9s | 153.3s ± 16.0s (+42%) |
| Mean inference time, full test split (5 seeds) | 1.35s ± 0.10s | 1.85s ± 0.14s (+37%) |

Re-uploading costs real, non-trivial extra wall-clock time (more circuit
evaluations per forward/backward pass) for no compensating benefit in
either predictive performance or calibration.

## 12. Interpretation

Per the plan's own three possible outcomes: this is **Outcome 2 — small/no
improvement**. "The existing depth-3 ansatz may already extract most of
the useful information available to this QNN configuration" is the
best-supported reading: validation PR-AUC's paired mean effect (−0.015)
is small relative to the per-seed spread (std 0.102) and nearly evenly
split in direction (2 up, 3 down) — not a clear win, and not
unambiguously "worse" either, though it is clearly **not an
improvement**. Combined with the real training/inference cost increase
(§11) and calibration neutrality (§10), there is no basis to prefer the
re-upload variant over the existing depth-3 baseline.

## 13. Limitations

- 5 seeds is this project's standing convention, not a large-N study; no formal significance test was run or is implied.
- Only one re-upload pattern was tested (between every pair of consecutive layers, matching the requested diagram exactly) — other re-upload schedules (e.g., every other layer, or re-uploading a transformed rather than identical copy of x) were not tested and are out of scope per the plan's own "at most one carefully justified ablation" framing from the earlier depth work.
- Test-based and validation-based paired deltas disagree in magnitude (though not in overall sign of the mean) — reported transparently (§9) rather than presenting only the more clear-cut test-based number.
- This result is specific to depth 3, 8 qubits, and this benchmark's data — it says nothing about whether re-uploading would behave differently at other depths or qubit counts, which were not tested here per the plan's explicit scope.

## 14. Decision for Next Experiment

Per the plan's own pre-committed stop condition: data re-uploading did
**not** produce a meaningful improvement (§12). **QGNN architecture
optimization stops here.** No further architectural variants (entanglement
topology, alternative ansätze, deeper re-upload schedules) will be
pursued as a follow-up to this specific result — the plan is explicit
that only "did not improve" leads to stopping, and that is what was
measured.

**Next step: the already-planned cross-dataset depth-3 evaluation**
(`43 → 44`, `44 → 43`), using the baseline (non-re-upload) depth-3
configuration selected in `QGNN_V2_DEPTH_BENCHMARK.md`, since re-uploading
did not earn a place as the new primary configuration.
