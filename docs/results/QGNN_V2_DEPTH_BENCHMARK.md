# QGNN-v2 Depth Benchmark

Executes Phase 4 of `QGNN_V2_HYBRID_IMPLEMENTATION_AND_DEPTH_BENCHMARK.md`.
Builds on `QGNN_V2_IMPLEMENTATION_REPORT.md` (Phases 1-3: implementation,
tests, smoke test — all complete, 217/217 tests passing). All training
was executed by the user; this document consolidates the results.

---

## 1. Executive Summary

**Primary finding**: giving QGNN a graph-derived representation (the
same 8D input GraphSAGE-Hybrid-Control receives, via a frozen GraphSAGE-Full
encoder + train-only PCA) substantially improved it over v1's
feature-only QGNN — validation PR-AUC rose from 0.398 ± 0.066 (v1's best,
2-layer, domain-selected raw features) to **0.650 ± 0.102** (v2's best,
3-layer, graph-derived), a ~63% relative increase. **This confirms the
central hypothesis behind building v2**: the v1 fairness gap (no graph
access) was costing QGNN real performance.

**It did not close the gap with the classical control.** GraphSAGE-Hybrid-Control
reaches validation PR-AUC 0.867 ± 0.017 on the identical input — still
roughly 0.22 absolute (≈25% relative) above QGNN-v2's best depth. Per the
plan's own outcome taxonomy, this is **Outcome B — Interesting result**:
lack of graph information was a real limitation in v1, but the quantum
head still does not match the classical control.

**Depth is not monotonic.** Performance rises from depth 1 to a
broad peak around depths 2-3, then declines through depths 4-5 — closest
to the plan's "peaks" pattern (`1 < 2 < 3 > 4 > 5`), though the exact peak
location (2, 3, or 4) is less certain than the overall shape: paired
per-seed deltas show the 1→2 rise and 4→5 decline are consistent across
most seeds, while the middle transitions (2→3, 3→4) are closer to a coin
flip per seed (§6).

## 2. Architecture (recap)

```
existing, unmodified GraphSAGE-Full checkpoint (per seed)
    -> frozen (requires_grad_(False)), encode()["supplier"]
    -> [n_suppliers, 128] embedding
    -> PCA (train-split fit only, frozen), 8D
         /                                    \
  GraphSAGE-Hybrid-Control              Hybrid-QGNN
  Linear(8,8)->ReLU->Linear(8,1)        8 qubits, depth 1-5, existing qgnn.py circuit, unchanged
```

## 3. Fairness Design (recap)

Both heads receive byte-for-byte the same 8D tensor at every example
(verified by test, `test_classical_and_quantum_heads_receive_byte_identical_input`).
The frozen encoder never receives a gradient (verified by test). Only
head type and QNN depth vary across the results below — dataset, split,
target, horizon, class weighting, optimizer, learning rate, weight
decay, early stopping, and evaluation code are all identical to the
existing GraphSAGE/QGNN-v1 conventions, unchanged.

## 4. Training Protocol (recap, as actually implemented)

Per model seed: load that seed's existing GraphSAGE-Full checkpoint →
freeze → extract embeddings for every (supplier, time) row in train/
validation/test → fit PCA on train-split embeddings only → train the
classical head and, separately, the QGNN head (same seed for weight
init and data-shuffling, both, per the reproducibility fix in
`QGNN_V2_IMPLEMENTATION_REPORT.md` §5) → early-stop on validation PR-AUC
→ evaluate, frozen, on test.

## 5. Depth Experiment — Results (5 seeds, 42-46)

**Selected on validation PR-AUC (never test)** — full table:

### Test split

| Model | Depth | PR-AUC | ROC-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Hybrid-QGNN | 1 | 0.5134 ± 0.1256 | 0.8749 ± 0.0531 | 0.1410 ± 0.0523 | 0.2749 ± 0.0928 |
| Hybrid-QGNN | 2 | 0.5570 ± 0.1181 | 0.9121 ± 0.0325 | 0.1352 ± 0.0860 | 0.2295 ± 0.1359 |
| **Hybrid-QGNN** | **3** | **0.6032 ± 0.1132** | 0.9369 ± 0.0205 | **0.0875 ± 0.0220** | **0.1585 ± 0.0361** |
| Hybrid-QGNN | 4 | 0.5722 ± 0.1097 | 0.9235 ± 0.0252 | 0.0954 ± 0.0289 | 0.1722 ± 0.0686 |
| Hybrid-QGNN | 5 | 0.5064 ± 0.1113 | 0.8951 ± 0.0571 | 0.1198 ± 0.0712 | 0.2087 ± 0.1239 |
| GraphSAGE-Hybrid-Control | — | 0.7520 ± 0.0278 | 0.9775 ± 0.0118 | 0.0972 ± 0.0804 | 0.1831 ± 0.1550 |
| GraphSAGE-Full (existing) | — | 0.8070 ± 0.0655 | 0.9871 ± 0.0083 | 0.0418 ± 0.0119 | 0.0493 ± 0.0103 |

### Validation split (the selection criterion)

| Model | Depth | PR-AUC | ROC-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Hybrid-QGNN | 1 | 0.5513 ± 0.1034 | 0.8471 ± 0.0401 | 0.1415 ± 0.0523 | 0.2641 ± 0.0994 |
| Hybrid-QGNN | 2 | 0.5877 ± 0.0769 | 0.8777 ± 0.0265 | 0.1376 ± 0.0840 | 0.2201 ± 0.1331 |
| **Hybrid-QGNN** | **3** | **0.6499 ± 0.1022** | 0.8996 ± 0.0235 | **0.0900 ± 0.0235** | **0.1454 ± 0.0388** |
| Hybrid-QGNN | 4 | 0.6292 ± 0.1026 | 0.8774 ± 0.0384 | 0.0984 ± 0.0278 | 0.1619 ± 0.0661 |
| Hybrid-QGNN | 5 | 0.5708 ± 0.1004 | 0.8751 ± 0.0337 | 0.1199 ± 0.0693 | 0.2000 ± 0.1252 |
| GraphSAGE-Hybrid-Control | — | 0.8674 ± 0.0172 | 0.9272 ± 0.0224 | 0.0877 ± 0.0814 | 0.1627 ± 0.1616 |
| GraphSAGE-Full (existing) | — | 0.8810 ± 0.0127 | 0.9411 ± 0.0082 | 0.0379 ± 0.0125 | 0.0471 ± 0.0092 |

**Depth 3 selected**: highest validation PR-AUC (0.6499), and — reassuringly,
though not the basis for selection — also highest test PR-AUC (0.6032)
and the best calibration on both splits. Depth was never chosen from test
performance; the two happening to agree here is a consistency check, not
the selection method.

`GraphSAGE-Hybrid-Control` is bit-identical across all five depth runs
(it does not depend on `--n-layers`) — confirms reproducibility, not five
independent results.

## 6. Per-Seed Results and Paired Depth-Delta Analysis

Test PR-AUC, Hybrid-QGNN, per seed:

| Seed | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 5 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.5262 | 0.5635 | 0.7216 | 0.7153 | 0.4763 |
| 43 | 0.5545 | 0.6119 | 0.6471 | 0.5367 | 0.5318 |
| 44 | 0.2901 | 0.3301 | 0.3934 | 0.4072 | 0.3292 |
| 45 | 0.6780 | 0.6127 | 0.5884 | 0.5315 | 0.5187 |
| 46 | 0.5182 | 0.6670 | 0.6654 | 0.6700 | 0.6763 |

**Paired per-seed deltas** (plan section 17 — do not declare significance
from aggregate means alone):

| Transition | Mean Δ | Std Δ | Seeds improving | Seeds degrading |
|---|---:|---:|---:|---:|
| depth 1 → 2 | +0.0436 | 0.0681 | 4 | 1 |
| depth 2 → 3 | +0.0461 | 0.0636 | 3 | 2 |
| depth 3 → 4 | −0.0310 | 0.0467 | 2 | 3 |
| depth 4 → 5 | −0.0657 | 0.0915 | 1 | 4 |

No formal significance test was run or is claimed. Reading direction and
magnitude only: the 1→2 rise and 4→5 decline are the most consistent
transitions (4/5 and 4/5 seeds agree respectively); the 2→3 and 3→4
transitions are closer to evenly split across seeds despite non-trivial
mean shifts, meaning the precise peak location is less certain than the
overall rise-then-fall shape. Seed 44 is the weakest performer at every
depth (0.29-0.41) — consistent low performance, not an outlier reversing
direction, so it does not appear to be driving the peak/decline pattern
by itself.

## 7. Depth Interpretation

Per the plan's own categories (section 22): closest to **"peaks"**
(`1 < 2 < 3 > 4 > 5`), both on the aggregate mean and on validation
PR-AUC. Not "improves" (5 is clearly worse than 3, both test and
validation), not "saturates" (there is a real decline, not a plateau),
not "flat" (the spread across depths, ~0.10-0.14 absolute PR-AUC on
validation, is larger than any single depth's own seed-to-seed std).
Increasing depth is not a free improvement for this architecture and
input at this scale — depth 5 (40 quantum parameters) performs worse
than depth 1 (8 quantum parameters) on both splits.

## 8. Classical-Control Comparison (RQ2 — primary)

| | Val PR-AUC | Val ROC-AUC |
|---|---:|---:|
| GraphSAGE-Hybrid-Control | 0.8674 ± 0.0172 | 0.9272 ± 0.0224 |
| Hybrid-QGNN (best: depth 3) | 0.6499 ± 0.1022 | 0.8996 ± 0.0235 |

**QGNN-v2 does not outperform its matched classical control at any
tested depth.** The gap at the best depth (≈0.22 absolute PR-AUC) is
smaller than v1's gap at its best configuration (v1: GraphSAGE-Reduced
0.825 vs QGNN-Reduced 0.301, ≈0.52 absolute — see
`QGNN_FINAL_BENCHMARK.md` §6-8) — narrowed by giving both models the
same graph-derived input, but not closed.

## 9. GraphSAGE-Full Comparison (RQ3)

| | Val PR-AUC |
|---|---:|
| GraphSAGE-Full | 0.8810 ± 0.0127 |
| GraphSAGE-Hybrid-Control | 0.8674 ± 0.0172 |
| Hybrid-QGNN (best: depth 3) | 0.6499 ± 0.1022 |

GraphSAGE-Hybrid-Control sits very close to GraphSAGE-Full (within ~0.014
val PR-AUC) — bottlenecking the 128D embedding to 8D via PCA costs the
classical head almost nothing, consistent with
`QGNN_V2_ARCHITECTURE_ANALYSIS.md` §10's note that GraphSAGE-Reduced and
GraphSAGE-Hybrid-Control need not behave identically, and here the
after-message-passing bottleneck (Hybrid-Control) turned out to preserve
information about as well as the full embedding. Hybrid-QGNN, even at its
best depth, remains well short of both classical references — it does
not approach the GraphSAGE-Full ceiling.

## 10. Calibration Analysis

QGNN-v2's calibration is **substantially better than v1's** at every
depth (v1 QGNN ECE range: 0.30-0.46 per `QGNN_FINAL_BENCHMARK.md`; v2
QGNN ECE range: 0.145-0.275) — another benefit of the graph-derived
input, not just PR-AUC. Depth 3 has the best calibration among the five
depths on both splits (test ECE 0.1585, val ECE 0.1454), coinciding with
its being the best-performing depth — **no PR-AUC-vs-calibration
tradeoff was observed**: the depth that improved discrimination also
improved calibration, rather than trading one for the other. QGNN-v2's
best-depth calibration (ECE ≈0.15) still trails GraphSAGE-Hybrid-Control
(ECE ≈0.16 test, 0.16 val — actually comparable at this specific depth)
and both remain well behind GraphSAGE-Full (ECE ≈0.05).

## 11. Resource Analysis

| | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 5 |
|---|---:|---:|---:|---:|---:|
| Qubits | 8 | 8 | 8 | 8 | 8 |
| Quantum trainable params | 8 | 16 | 24 | 32 | 40 |
| MLP head params | 81 | 81 | 81 | 81 | 81 |
| **Total trainable (QGNN)** | 89 | 97 | 105 | 113 | 121 |
| Classical control trainable | 81 (constant across depths) | | | | |
| Frozen encoder (shared, not trainable) | ≈1.2M | | | | |

Backend: `default.qubit` (CPU, `diff_method="backprop"`) throughout —
the practical default established in v1 (§`QGNN_IMPLEMENTATION_AND_BENCHMARK.md`
§10/12). All 5 depths × 5 seeds, both heads, plus embedding extraction
from 5 separate frozen checkpoints, completed in well under 20 minutes
total wall-clock. No quantum computational advantage is claimed —
simulator-based throughout, as in every prior phase.

## 12. Cross-Dataset Results

**Not run in this phase.** Per the plan's own section 12 instruction
("perform cross-dataset evaluation only for the chosen primary v2
configuration after the within-world depth study... do not automatically
run all five depths cross-dataset"), cross-dataset evaluation is deferred
to a follow-up run now that depth 3 is selected. `qgnn_v2.py`'s
`apply_fitted_reducer`/`evaluate_v2_on_target` functions are implemented
and ready; only the orchestration script
(`scripts/run_qgnn_v2_crossdataset_experiment.py`, not yet written, see
`QGNN_V2_IMPLEMENTATION_REPORT.md` §1) remains.

## 13. Limitations

- Depth was swept 1-5 only, per the plan's own explicit scope; whether a shallower peak (2) or a later one (4+) is the "true" optimum on a different dataset realization is untested.
- The 2→3 and 3→4 paired transitions are close to evenly split across the 5 seeds — the peak's exact location has real seed-to-seed disagreement, not a sharp, unanimous optimum.
- Cross-dataset generalization of the selected configuration is untested in this phase (§12).
- RQ1 (does graph structure help at all?) was not directly re-tested here — no `GraphSAGE-NoGraph` (`num_layers=0`) control was run in Phase 4; this remains available cheaply if wanted (`QGNN_V2_ARCHITECTURE_ANALYSIS.md` §10).
- Only one circuit family (angle encoding, linear CNOT entanglement, PauliZ measurement) was varied by depth; other architectural axes (entanglement topology, data re-uploading, qubit count) were held fixed per the plan's own instruction not to search multiple hyperparameters simultaneously.
- Five seeds is this project's standing convention throughout, not a large-N statistical study; no significance test was performed or is implied by any comparison above.
- Simulator-based only; no real quantum hardware; no claim of quantum advantage.

## 14. Scientific Interpretation

Per the plan's own outcome definitions (§21):

> **Outcome B — Partial improvement.** Hybrid-QGNN substantially improves
> over QGNN-v1 but remains below the classical hybrid control.

This is the outcome observed. The improvement is not marginal — validation
PR-AUC rose ~63% relative to v1's best result, and calibration improved
by roughly 2x — so "lack of graph information was a real limitation in
v1" is well supported, not a weak or borderline reading of the evidence.
At the same time, the remaining ≈0.22 absolute gap to
GraphSAGE-Hybrid-Control at matched input is large relative to any
individual depth's own seed-to-seed variance, so "the quantum head still
does not match the classical control" is equally well supported. Neither
half of Outcome B's description is being softened to make the result
sound more or less favorable than what was measured.

## 15. Final Conclusion

Giving Hybrid-QGNN the same graph-derived representation GraphSAGE-Hybrid-Control
receives substantially improved it over the feature-only QGNN-v1 — both
in ranking quality (validation PR-AUC 0.398 → 0.650, best configurations)
and in calibration (ECE roughly halved) — confirming that v1's lack of
graph access was a real, measurable limitation, not a minor one. Sweeping
circuit depth from 1 to 5 variational layers found a real, non-monotonic
effect: performance rises through depth ≈2-3 and declines through depth
5, so more depth is not automatically better for this architecture at
this scale, and depth 3 (24 quantum parameters) was selected on
validation performance as the primary configuration. Despite this
improvement, **Hybrid-QGNN did not outperform, and did not closely
approach, its matched classical control (GraphSAGE-Hybrid-Control) at any
tested depth**, nor did it approach the GraphSAGE-Full reference. Within
the scope of this benchmark — 8 qubits, depths 1-5, angle encoding with
linear entanglement, simulator-based evaluation on this synthetic
supply-chain benchmark — the quantum component, even once given
information-matched, graph-derived input, did not close the gap with
classical processing of the same representation.

## 16. Reproducibility

```bash
# Depth sweep (8 dims, 5 seeds each)
python scripts/run_qgnn_v2_experiment.py --config configs/qgnn_v2.yaml --tag depth1 --n-layers 1 --seeds 42,43,44,45,46
python scripts/run_qgnn_v2_experiment.py --config configs/qgnn_v2.yaml --tag depth2 --n-layers 2 --seeds 42,43,44,45,46
python scripts/run_qgnn_v2_experiment.py --config configs/qgnn_v2.yaml --tag depth3 --n-layers 3 --seeds 42,43,44,45,46
python scripts/run_qgnn_v2_experiment.py --config configs/qgnn_v2.yaml --tag depth4 --n-layers 4 --seeds 42,43,44,45,46
python scripts/run_qgnn_v2_experiment.py --config configs/qgnn_v2.yaml --tag depth5 --n-layers 5 --seeds 42,43,44,45,46
```

Each run reuses the existing GraphSAGE-Full checkpoints in
`experiments/classical_gnn/` as its frozen encoder (never retrained) and
writes reproducible artifacts (config, model checkpoint, predictions,
metrics, calibration, onset breakdown, training history, plots, quantum
resource summary, encoder-checkpoint provenance, run metadata including
git commit and timestamp) under `experiments/qgnn_v2/` — no run directory
is ever overwritten. Verified bit-identical across independent reruns
(`QGNN_V2_IMPLEMENTATION_REPORT.md` §4).

## 17. Suggested Next Step

Cross-dataset evaluation (§12) for the selected depth-3 configuration,
both directions (seed43→seed44, seed44→seed43), before any further
architecture work — matching this project's standing practice of
checking generalization before investing in refinement.
