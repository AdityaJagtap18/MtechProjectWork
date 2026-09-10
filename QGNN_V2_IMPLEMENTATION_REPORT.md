# QGNN-v2 Implementation Report

Executes Phases 1-3 of
`QGNN_V2_HYBRID_IMPLEMENTATION_AND_DEPTH_BENCHMARK.md` (implementation,
tests, smoke test). Phase 4 (the 5-seed × 5-depth benchmark) is handed
off at the end of this document, per the plan's own instruction: "Only
after the smoke test passes should the 5-seed depth benchmark run."

Builds on `QGNN_V2_ARCHITECTURE_ANALYSIS.md`'s recommended Option C
(frozen shared encoder, two matched heads) — this document implements
exactly that design.

---

## 1. What Was Built

All new, additive files — nothing existing was modified except one
isolated addition to `scripts/_analysis_common.py` (a new function,
`find_latest_graphsage_full_checkpoint`, alongside its existing sibling
finder functions):

- **`src/scm_dataset/modeling/graph_embedding_reduction.py`** (new):
  - `load_frozen_graphsage_encoder(checkpoint_path, prepared)` — loads an existing `model.pt`, sets `requires_grad_(False)` and clears any `.grad` on every parameter, `.eval()`.
  - `prepare_frozen_encoder_input(config, benchmark, preprocessing_dir)` — builds the full (unreduced) `PreparedData` needed to feed the frozen encoder, reusing the checkpoint's own saved `FeaturePreprocessor` (`FeaturePreprocessor.load()`, already existed) rather than refitting.
  - `extract_supplier_embeddings(model, prepared, times)` — runs `model.encode(...)["supplier"]` under `torch.no_grad()` for each timestep, returns a `(supplier_id, time)`-indexed DataFrame, `emb_0..emb_{hidden_dim-1}`.
  - `fit_embedding_pca(embedding_frame, train_examples, n_components)` — reuses `reduction.PCASupplierReducer` **unchanged**; it only needed a DataFrame + boolean fit mask + column list, so it works on embedding columns exactly as it already did on raw feature columns.
- **`src/scm_dataset/modeling/qgnn_v2.py`** (new):
  - `ClassicalHybridHead` — `Linear(d,8)→ReLU→Linear(8,1)`, matched in shape to `qgnn.QGNN`'s own MLP head.
  - `QGNNv2ArchConfig`/`QGNNv2ExperimentConfig`/`load_qgnn_v2_config` — mirrors `qgnn.py`'s own config-wrapping pattern.
  - `V2PreparedData` — flat, graph-free analogue of `PreparedData`; duck-types `.benchmark`/`.config` so `evaluate.py`'s existing `_finalize_evaluation` works unmodified.
  - `build_v2_prepared` / `apply_fitted_reducer` (cross-dataset counterpart) / `train_v2_head` (model-agnostic — same function trains either head) / `generate_predictions_v2` / `evaluate_v2` / `evaluate_v2_on_target`.
- **`configs/qgnn_v2.yaml`** (new).
- **`scripts/run_qgnn_v2_experiment.py`** (new) — the Phase 4 runner.
- **`scripts/_analysis_common.py`** (isolated addition) — `find_latest_graphsage_full_checkpoint`.
- **`tests/test_qgnn_v2.py`** (new, 12 tests).

**Not yet built** (deferred to when cross-dataset evaluation is actually
run, per the plan's own phasing): `scripts/run_qgnn_v2_crossdataset_experiment.py`.
`qgnn_v2.py`'s `apply_fitted_reducer`/`evaluate_v2_on_target` functions
already exist and are ready for it, mirroring the v1 cross-dataset
script's structure closely — writing the script itself was deferred to
avoid building an unused artifact before the within-world depth study
(Phase 4) even has a selected primary configuration, matching plan
section 12's own instruction: "perform cross-dataset evaluation only for
the chosen primary v2 configuration after the within-world depth study."

## 2. Design Confirmation

Matches `QGNN_V2_ARCHITECTURE_ANALYSIS.md`'s recommendation exactly:

```
existing, unmodified GraphSAGE-Full checkpoint (per seed)
    -> loaded, frozen (requires_grad_(False), .grad cleared, .eval())
    -> HeteroGraphSAGE.encode(...)["supplier"]   -- pre-existing method, unmodified
    -> [n_suppliers, 128] embedding per timestep
    -> PCASupplierReducer (reused unmodified), fit on train-split rows only
    -> 8D, identical for both heads
         /                                    \
  ClassicalHybridHead (new, tiny)      qgnn.QGNN (reused unmodified)
```

`qgnn.py`'s `QGNN`/`QuantumCircuitLayer`/`build_qgnn_model` needed **zero
changes** — confirmed by direct reuse, not just analysis: `n_layers` is
already a constructor parameter, so the depth experiment (Phase 4) is a
pure CLI-flag sweep with no new quantum code.

## 3. Tests (Phase 2)

`tests/test_qgnn_v2.py`, 12 tests, run against a freshly-trained-then-frozen
tiny GraphSAGE (not the real checkpoint, for speed):

| Plan's Phase 2 checklist item | Test |
|---|---|
| Checkpoint loading | `test_frozen_encoder_has_no_trainable_parameters` (+ real-checkpoint loading exercised directly in the smoke test, §4) |
| Embedding shape | `test_embedding_extraction_shape` |
| PCA train-only, deterministic, exactly-d-dim | `test_pca_embedding_reducer_is_train_only_deterministic_8d` |
| Input equality (classical vs QNN) | `test_classical_and_quantum_heads_receive_byte_identical_input` |
| Encoder freeze, no gradients | `test_embedding_extraction_uses_no_grad_encoder_receives_nothing` |
| QNN gradients | `test_train_v2_head_end_to_end_classical_and_qgnn` (asserts quantum weights actually move during training) |
| Depth parameter counts | `test_qnn_depth_parameter_counts`, parametrized 1→8, 2→16, 3→24, 4→32, 5→40 — **matches the plan's own table exactly** |
| Leakage (train-only tampering) | `test_pca_embedding_reducer_fit_never_uses_validation_or_test_rows` — corrupts every non-train embedding row to `1e9`, confirms identical fitted PCA components |

**One bug caught by testing, fixed**: the first version of
`test_embedding_extraction_uses_no_grad_encoder_receives_nothing` failed
— not because extraction leaked gradients, but because the test fixture
reused an already-trained model object whose `.grad` tensors were stale
from *before* freezing (freezing stops future gradients, it doesn't erase
past ones). Fixed by explicitly clearing `.grad` after freezing, in both
the test fixture and in `load_frozen_graphsage_encoder` itself (defensive,
matches correct practice regardless of how a model was obtained).

**Full suite**: 205 (pre-v2) → **217/217 passing** after v2's 12 new tests.

## 4. Smoke Test (Phase 3)

Real primary benchmark (`scm_v1_black_swan_seed43`), real existing
GraphSAGE-Full checkpoint (`experiments/classical_gnn/20260905T062728Z_hetero_graphsage_seed42`),
run via the actual CLI (`scripts/run_qgnn_v2_experiment.py --tag smoke
--seeds 42 --n-layers 1 --epochs 5`), not a standalone script.

- Checkpoint loaded correctly; embedding extraction produced exactly `(26700, 128)` = 300 suppliers × 89 distinct prediction times — matches the primary dataset's known example count exactly.
- No NaNs anywhere in predictions or metrics.
- Encoder froze correctly (`assert all(not p.requires_grad ...)` in the script itself, plus the dedicated unit test).
- Both heads trained, evaluated, produced valid `[0,1]`-bounded predictions and complete metrics/calibration.
- Artifacts saved and verified present: `config.yaml`, `model.pt`, `predictions.csv`, `supplier_risk_ranking.csv`, `metrics.json`, `calibration.json`, `onset_breakdown.json`, `training_history.csv`, `run_metadata.json`, `encoder_checkpoint.json`, `quantum_resource_summary.json`, `reduction/`, `preprocessing/`, `plots/` (5 PNGs — initially missing, caught and fixed, §5).
- **Reproducibility verified directly**: two independent runs with `--seeds 42` produced bit-identical PR-AUC/ROC-AUC for both heads (0.5223091982528566 / 0.8273409004769663 classical; 0.48953358767091304 / 0.9077354436672753 QGNN, both digit-for-digit identical across reruns).
- No existing v1 or GraphSAGE artifact was touched (verified — the script only ever *reads* from `experiments/classical_gnn/`, writes only under a fresh `experiments/qgnn_v2/` directory).

**These numbers are a 1-seed, 5-epoch smoke test — not a scientific
result.** Reported here only to confirm the pipeline works end to end;
not to be cited as evidence about whether Hybrid-QGNN outperforms
anything. For calibration: a full-training (not 5-epoch) run at this
configuration will very likely differ substantially from these numbers,
same as every prior phase's smoke-vs-final gap in this project.

## 5. Bugs Found and Fixed During Implementation

1. **Missing plots**: `_save_run` never called `render_all_plots`, so every run's `plots/` directory existed (created by `new_run_dir`) but was empty. Fixed by adding the existing, unmodified `render_all_plots` call.
2. **Reproducibility gap**: the classical head's weight *initialization* happened before `train_v2_head`'s internal `set_seed(seed)` call took effect (the model was constructed by the *caller*, outside the function that seeds) — unlike v1's `train_qgnn`, which constructs its model internally, after seeding. Fixed by calling `set_seed(seed)` in the script immediately before each `build_*` call. Verified fixed via the repeated-run bit-identity check in §4.
3. **Stale-gradient test fixture issue** (§3) — test-only, not a real implementation bug, but the defensive `.grad = None` fix was also applied to the real `load_frozen_graphsage_encoder` function.

## 6. Leakage Audit Status

All items from the plan's section 9 checklist, re-verified against the
actual implementation:

| Control | Status |
|---|---|
| Graph encoder training | None happens in v2 — reuses an existing, already-audited checkpoint |
| PCA fit | Train-split embeddings only, verified by tampering test |
| Domain/label leakage into encoding | None — embedding extraction is a pure forward pass under `torch.no_grad()` |
| Class weighting | Train-split only (`compute_pos_weight` on `train_y`, unchanged function) |
| Threshold selection | Validation-split only (`select_threshold`, unchanged function) |
| Test evaluation | Only after model selection, via unmodified `_finalize_evaluation` |
| Encoder gradient isolation | Verified by test — parameters have `requires_grad=False` and no `.grad` accumulates through a full extract→PCA→train cycle |

## 7. Full Test Suite

**217/217 passing**, re-run fresh immediately before writing this
document (not assumed).

## 8. Hand-off — Phase 4 (5-Seed × 5-Depth Benchmark)

Per the plan's own instruction, this is where implementation stops and
execution hands off (matching every prior phase's division of labor:
code and smoke tests are mine, multi-seed research-record runs are
yours). Five depths × one classical control, 5 seeds each:

```bash
python scripts/run_qgnn_v2_experiment.py --config configs/qgnn_v2.yaml --tag depth1 --n-layers 1 --seeds 42,43,44,45,46
```
```bash
python scripts/run_qgnn_v2_experiment.py --config configs/qgnn_v2.yaml --tag depth2 --n-layers 2 --seeds 42,43,44,45,46
```
```bash
python scripts/run_qgnn_v2_experiment.py --config configs/qgnn_v2.yaml --tag depth3 --n-layers 3 --seeds 42,43,44,45,46
```
```bash
python scripts/run_qgnn_v2_experiment.py --config configs/qgnn_v2.yaml --tag depth4 --n-layers 4 --seeds 42,43,44,45,46
```
```bash
python scripts/run_qgnn_v2_experiment.py --config configs/qgnn_v2.yaml --tag depth5 --n-layers 5 --seeds 42,43,44,45,46
```

Each command also (re-)trains the matched `GraphSAGE-Hybrid-Control` for
all 5 seeds — a bit-identical-reproducible duplicate across the five
commands (it doesn't depend on `--n-layers`), not five separate new
results to track.

Given the smoke test's ~11-second full run (both heads, 5 seeds not yet
included) and this project's established pattern of full-epoch training
completing in low minutes even at 5 seeds, expect each of the five
commands above to complete quickly — but confirm actual timing on your
machine before assuming.

Once all five are back, I will pull validation-split numbers (never
test-based selection — matches every prior phase's discipline), run the
paired per-seed depth-delta analysis the plan requires (section 17), and
write `QGNN_V2_DEPTH_BENCHMARK.md`.
