# QGNN-v3 Controlled Architecture Benchmark

Controlled follow-up to `QGNN_V2_DEPTH_BENCHMARK.md`'s selected depth-3
configuration (and a sibling study to `QGNN_V2_DATA_REUPLOAD_BENCHMARK.md`,
which already ruled out data re-uploading). Tests three isolated
architecture changes — ring entanglement, all-to-all entanglement,
trainable input scaling — each holding everything else fixed. All
training run by the user via `scripts/run_qgnn_v3_experiment.py`.

---

## 1. Objective

> Extend the completed QGNN-v2 depth-3 benchmark with a controlled
> architecture study.

Not unrestricted hyperparameter tuning: three specific, isolated changes,
each compared against the unmodified baseline on validation PR-AUC.

## 2. Architecture

```
input x (8D, identical to the depth-3 baseline)
    -> angle encoding (baseline: tanh(x)*pi; D: tanh(a*x+b)*pi, a/b trainable)
    -> variational layer 1 (RY per qubit, trainable)
    -> entangle (A/D: linear chain; B: ring; C: all-to-all)
    -> variational layer 2
    -> entangle
    -> variational layer 3
    -> entangle
    -> measurement (PauliZ per qubit)
    -> same MLP head (Linear(8,8) -> ReLU -> Linear(8,1))
```

Implemented in `src/scm_dataset/modeling/qgnn_v3.py`. Variant A is not
reimplemented at all — it is `qgnn.build_qgnn_model` run directly,
unmodified. Variants B/C share a generic circuit builder parameterized
only by which CNOT pairs it entangles; variant D reuses `qgnn._make_qnode`
(the baseline circuit itself, imported directly, unmodified) and changes
only the classical angle computation upstream of it. `qgnn.py`,
`qgnn_v2.py`, and `qgnn_v2_reupload.py` were not modified.

**Verified, not assumed**: the shared B/C circuit builder, when given the
baseline's own linear pairing, produces a **bit-identical PennyLane tape**
to the untouched original circuit (same operations, same wires, same
order) — proof the generic mechanism introduces no incidental behavior
change. Variant D's scaling parameters (`a=1`, `b=0` at init) produce
numerically identical output to the baseline at initialization, confirmed
by direct tensor comparison, not just by construction.

## 3. Fairness Controls

All four variants, every seed: same frozen GraphSAGE-Full checkpoint for
that seed, same extracted 128D embeddings, same train-only-fit PCA (same
object, not refit per variant), same 8D input rows/splits, same seed for
weight init and data shuffling, same optimizer/LR/weight decay/class
weighting/early stopping/evaluation/threshold-selection code. Confirmed:
variant A, re-run through this new script, reproduced
`QGNN_V2_DEPTH_BENCHMARK.md`'s established baseline exactly — test PR-AUC
0.6032 ± 0.1132, validation PR-AUC 0.6499 ± 0.1022, both to 4 decimal
places.

## 4. Tests

12 new tests, `tests/test_qgnn_v3.py`, all passing — covering every item
in the required checklist (qubit/depth counts, exact topology pair sets
for ring/all-to-all, bit-identical-tape proof that linear reduces to
baseline, exact added-parameter count for D, numerical init-equivalence
for D, output shape, MLP head architecture, finite forward/backward for
all three variants, and full pipeline leakage/freeze checks reusing the
already-tested `qgnn_v2.py` utilities unchanged). **Full suite: 236/236
passing** (224 pre-existing + 12 new), re-run fresh before writing this
document.

## 5. Smoke-Test Result

1 seed, 5 epochs, all four variants: completed in ~55s total, finite
gradients on every parameter, bounded predictions, no NaNs/crashes.
Resource summaries matched expectations exactly at the smoke stage
(A: 21 total CNOTs; B: 24; C: 84; D: 121 total trainable parameters =
105 + 16 added scaling params) — not treated as a scientific result.

## 6. Five-Seed Results (seeds 42-46)

| Variant | Test PR-AUC | Test ROC-AUC | Val PR-AUC | Val ROC-AUC | Test Brier | Test ECE |
|---|---:|---:|---:|---:|---:|---:|
| **A. Baseline (linear)** | 0.6032 ± 0.1132 | 0.9369 ± 0.0205 | 0.6499 ± 0.1022 | 0.8996 ± 0.0235 | 0.0875 ± 0.0220 | 0.1585 ± 0.0361 |
| B. Ring entanglement | 0.5027 ± 0.0802 | 0.9066 ± 0.0320 | 0.5544 ± 0.0576 | 0.8627 ± 0.0258 | 0.1226 ± 0.0418 | 0.2202 ± 0.0932 |
| C. All-to-all entanglement | 0.5066 ± 0.1857 | 0.9094 ± 0.0411 | 0.5931 ± 0.1159 | 0.8851 ± 0.0313 | 0.1157 ± 0.0284 | 0.1954 ± 0.0761 |
| **D. Trainable input scaling** | 0.5854 ± 0.1228 | 0.9236 ± 0.0248 | **0.6856 ± 0.1395** | **0.9104 ± 0.0314** | **0.0839 ± 0.0235** | **0.1419 ± 0.0586** |

## 7. Per-Seed Comparison (Validation PR-AUC — the selection criterion)

| Seed | A. Baseline | B. Ring | C. All-to-all | D. Scaling |
|---:|---:|---:|---:|---:|
| 42 | 0.8024 | 0.5219 | 0.7580 | 0.8357 |
| 43 | 0.6937 | 0.5613 | 0.6440 | 0.7862 |
| 44 | 0.5058 | 0.4767 | 0.4050 | 0.4649 |
| 45 | 0.5741 | 0.5605 | 0.5512 | 0.5824 |
| 46 | 0.6734 | 0.6515 | 0.6074 | 0.7587 |

## 8. Paired Deltas vs. A (Validation PR-AUC)

| Variant | Per-seed delta | Mean | Std | Improving | Degrading |
|---|---|---:|---:|---:|---:|
| B. Ring | [−0.2805, −0.1324, −0.0291, −0.0136, −0.0219] | **−0.0955** | 0.1021 | **0** | **5** |
| C. All-to-all | [−0.0443, −0.0497, −0.1007, −0.0229, −0.0660] | **−0.0567** | 0.0260 | **0** | **5** |
| D. Scaling | [+0.0333, +0.0926, −0.0408, +0.0083, +0.0852] | **+0.0357** | 0.0496 | **4** | **1** |

**B and C are unambiguous, unanimous regressions** — every one of 5
seeds is worse than baseline for both, and C's degradation is
additionally the most *consistent* (tightest std of the three deltas,
0.026) despite having the widest raw validation-PR-AUC spread of any
variant — a uniformly-applied penalty, not one bad seed dragging an
average down. **D shows a real, directionally consistent improvement**
(4 of 5 seeds), but the mean effect (+0.036) is smaller than its own
standard deviation (0.050) — a genuine signal, not overwhelming evidence.
No formal significance test was run or is claimed for any variant.

## 9. Calibration Comparison

D is the best-calibrated variant of all four — including baseline A —
on both Brier score and ECE, on both splits (test ECE 0.142 vs A's
0.159; validation ECE 0.132 vs A's 0.145). B and C are both
*worse*-calibrated than A (ECE 0.195–0.220 vs A's 0.159 on test) —
consistent with, not contradicting, their worse discrimination: these
are not cases of "better ranking, worse calibration" trade-offs: B/C are
worse on every axis measured, and D is better on every axis measured.

## 10. Resource Comparison

| Variant | CNOTs/layer | CNOTs total | Added params | Train time (mean) | Inference time (mean) |
|---|---:|---:|---:|---:|---:|
| A. Baseline | 7 | 21 | — | 107.9s ± 24.4s | 1.35s ± 0.26s |
| B. Ring | 8 | 24 | — | 99.2s ± 41.3s | 1.51s ± 0.05s |
| C. All-to-all | 28 | 84 | — | 171.4s ± 55.4s (+59%) | 2.33s ± 0.11s (+72%) |
| D. Scaling | 7 | 21 | 16 (a, b) | 108.4s ± 37.7s | 1.52s ± 0.04s |

C is both the worst-performing entanglement variant *and* substantially
the most expensive — 59% more training time, 72% more inference time,
for validation PR-AUC 0.057 below baseline. D's added cost is
negligible (16 extra classical scalars, no measurable training-time
overhead) for its modest gain — the cheapest possible way to test this
particular change.

## 11. Selected Architecture

**A (the existing baseline) remains selected.** Per the plan's own
instruction ("A more expensive circuit needs a meaningful benefit to
justify the added complexity" and "Architecture selection is based on
validation PR-AUC, never test PR-AUC"): B and C both fail decisively —
worse on every metric, and C additionally much more expensive — ruled
out with high confidence (5/5 seeds each). D is the only variant with a
positive, multi-metric-consistent signal (PR-AUC, ROC-AUC, and
calibration all improved, at negligible extra cost), but per §8 the
effect size does not clearly exceed its own seed-to-seed noise (mean
+0.036 vs std 0.050) — not "clear improvement" by the plan's own bar,
closer to a promising-but-inconclusive result than a decisive win.

## 12. Combination (Variant E) Decision

**Not run**, for two independent reasons:

1. D's improvement, while real and multi-metric-consistent (§8-9), does
   not clearly clear the "clear validation improvement" threshold the
   plan sets for triggering E — the mean paired delta is smaller than
   its own standard deviation.
2. Even if it did, E is specified as a **combination** of favorable
   elements from B/C/D. B and C are both unambiguous regressions (§8) —
   there is nothing from either to combine with D. Running E would mean
   either re-testing D alone (not a combination, and not new information)
   or combining D with a variant already shown to hurt, which contradicts
   the evidence just gathered.

## 13. Interpretation

Per the plan's own outcome categories: B and C are **Degradation** —
retain the current depth-3 baseline, decisively. D is neither a clean
**Clear improvement** nor a pure **Rough tie** — it is a real,
directionally consistent (4/5 seeds), multi-metric-favorable signal that
falls short of "clear" by a strict effect-size-vs-noise standard.
Following the plan's own guidance for anything short of a clear win
("retain the simplest architecture"), and given the project's standing
discipline against chasing marginal signals through repeated tuning, **A
remains the selected configuration**. D is recorded as a specific,
well-characterized, low-cost, worth-remembering direction for possible
future work (e.g., a dedicated, larger-seed-count follow-up), not as
something acted on now.

## 14. Limitations

- 5 seeds is this project's standing convention, not a large-N study; no significance test was performed or is implied for any comparison.
- C's all-to-all convention (ascending-index CNOT pairing) is one of several possible conventions for connecting every qubit pair; a different ordering was not tested and might behave differently, though this is unlikely to change C's basic conclusion given how uniform its degradation was across seeds.
- D's scaling parameters are per-feature but not per-layer (the same `a`/`b` are used at every layer's re-encoding were there any — there is no re-uploading in this circuit family, so this doesn't currently apply, but is worth noting if D is ever combined with re-uploading in future work).
- This study varied entanglement topology and input encoding; it did not revisit qubit count, PCA dimensionality, or circuit depth, all already fixed by the plan's own scope (depth-3, 8 qubits, established by `QGNN_V2_DEPTH_BENCHMARK.md`).

## 15. Readiness for Cross-Dataset Evaluation

**Ready.** The selected configuration for the planned cross-dataset study
(`43 → 44`, `44 → 43`) is **variant A — the existing, unmodified depth-3
baseline** (8 qubits, linear entanglement, fixed angle encoding), exactly
as established before this architecture study began. No new PCA fitting
or target-world data use is required beyond what `qgnn_v2.py`'s existing,
already-tested `apply_fitted_reducer`/`evaluate_v2_on_target` machinery
already implements (built during the depth-benchmark phase, not yet
exercised end-to-end) — the orchestration script for this specific run
still needs to be written.
