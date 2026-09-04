# Classical GraphSAGE — Implementation Checkpoint

Status snapshot for resuming work on `CLASSICAL_GRAPHSAGE_IMPLEMENTATION_PLAN.md`.
Read that plan first, then this file for "where things actually stand."

## What's done and verified

**Package** (`src/scm_dataset/modeling/`): `config.py`, `data.py`, `features.py`,
`preprocessing.py`, `hetero_graph.py`, `graphsage.py`, `losses.py`, `metrics.py`,
`calibration.py`, `pipeline.py` (shared setup, factored out so train/evaluate/
baselines/tests all run the identical leakage-audited pipeline), `train.py`,
`evaluate.py`, `baselines.py`, `experiment.py` (run-dir + metadata).

**Scripts**: `scripts/run_graphsage_experiment.py` (main entry point — prepare
once, train + evaluate per seed, optional `--baselines`), `scripts/train_graphsage.py`,
`scripts/evaluate_graphsage.py` (standalone re-evaluation, reloads saved
preprocessing rather than refitting).

**Config**: `configs/graphsage.yaml` — dataset pinned to `scm_v1_black_swan_seed43`
(see its comments for why: has all 3 split types + a severity-5 event + is the
dataset `WORK_SUMMARY.md` already traced end to end).

**Dependencies**: torch 2.14 (CPU wheel), torch_geometric 2.8, scikit-learn 1.9
installed into `.venv` and added to `pyproject.toml`'s `[modeling]` extra +
`requirements.txt`.

**Verified by hand against the real benchmark** (`scm_v1_black_swan_seed43`):
- `prepare()` loads/builds features/fits preprocessing in ~2.3s, 26,700 examples
  (300 suppliers x 89 usable prediction times), split sizes train=18600/val=4500/test=3600,
  positive rates 2.6%/8.1%/6.7%.
- A short training run (2-3 epochs) trains, evaluates, and produces sane output:
  GraphSAGE test PR-AUC ~0.72 vs. majority-baseline 0.067 (= positive rate, as
  expected) vs. logistic-regression 0.30 — **not a final result** (only 2-3
  epochs), but confirms the graph/pipeline is wired correctly and that the
  expected baseline ordering (majority < logreg < graphsage) already shows up.
- Risk ranking, early-warning, calibration, and all 5 required plots render
  without errors.

**Tests**: `tests/conftest.py` provides `tiny_benchmark`/`tiny_prepared_data`
fixtures — a small in-memory benchmark (20 suppliers, 60 periods, seed=3,
empirically chosen for non-degenerate positive rates in all 3 splits: train
6.8%/val 17.2%/test 20%) built through the **real** generator/simulation/
labels/splits code and run through the **real** `pipeline.prepare_from_benchmark`,
not a reimplementation. Written and passing so far: `test_model_data.py` (4),
`test_temporal_features.py` (12, including a causality regression test that
injects a future record and asserts a past feature value is unchanged),
`test_preprocessing.py` (7). **Full suite: 113/113 passing** (90 original +
23 new), confirmed via `pytest -q`.

## Key decisions made (documented in code, summarized here)

- `supplier_risk_score` **excluded** from model inputs (plan §7/§72) —
  `feature_audit.csv` already marks all of `labels/supplier_labels.csv`
  `allowed=False`; confirmed via `labels/risk_labels.py` that it's a realized
  fulfillment shortfall from the same simulation outcomes as the target.
- `ProcurementOrder` graph-node fields (e.g. `actual_lead_time`) **are**
  treated as safe static features — confirmed via `generator/procurement.py`
  that they're generated at Phase 2 (topology time), independent of the
  simulation timeline, unlike the *operational* `procurement.csv` log.
- Region/procurement identifier fields (`region_id`, `country_or_region`,
  procurement FK columns) excluded from feature vectors — that relationship
  already reaches the model structurally through graph edges.
- Reverse edges (`rev_<type>`) added in the PyG representation only, per
  plan §50, so 2-layer message passing can flow both directions.
- Scenario/severity split assignment for prediction *examples* is derived
  from the target window (t, t+H], not a plain join on `t` — documented
  deviation in `pipeline.py`'s `_apply_generalization_split` docstring, since
  the plan's split files are period-level but the task is windowed-future
  prediction. A validation slice is carved from the tail of the resulting
  train pool (these split files are train/test-only).
- Risk ranking snapshot uses the latest TEST-split prediction time (the
  plan's example table is one row per supplier, implying "as of now," not
  one row per supplier x time).

## What's NOT done yet — pick up here

1. **Remaining test files** (plan §67/§87 list) — `conftest.py` fixture is
   ready and verified; these should be quick to write against it:
   - `tests/test_hetero_graph.py` — node index determinism, edge validation
     raising on a bad reference, snapshot shape/dtype correctness.
   - `tests/test_graphsage.py` — forward pass shape, logits are NOT
     pre-sigmoided, dropout/hidden_dim honored.
   - `tests/test_training.py` — pos_weight from train-only targets, early
     stopping triggers, best checkpoint = best val PR-AUC epoch (construct a
     scenario where a later epoch is worse and confirm the earlier one wins).
   - `tests/test_evaluation.py` — metric formulas against hand-computed
     values, threshold selected from validation only, single-class ROC-AUC
     handled safely, calibration ECE/Brier sanity.
   - One integration test using `tiny_prepared_data`: benchmark -> features
     -> HeteroData -> GraphSAGE -> prediction -> metrics, end to end.
2. **Run the actual "first experiment"** (plan §69/§75/§76) at full config
   (100 epochs, seed 42) — only 2-3 epoch smoke tests have been run so far:
   ```bash
   .venv/bin/python scripts/run_graphsage_experiment.py \
       --config configs/graphsage.yaml --seeds 42 --baselines
   ```
   Then manually verify against plan §92's checklist (files exist,
   `risk_probability` in [0,1], `actual_disruption` in {0,1}, no future
   timestamps) before trusting the result.
3. **Multi-seed run** (42-46) once the single-seed run looks correct:
   `--seeds 42,43,44,45,46` (writes a `multiseed_summary_*.json` automatically).
4. Optional, after the above are stable (plan §70/§77): severity/scenario
   generalization runs (`config.split.strategy = "severity" | "scenario"` in
   a copy of `configs/graphsage.yaml`) and ablations A-E (plan §55/§58).
5. Write `experiment_report.md` inside the frozen run's directory (plan §64).
6. Final pass against plan §91's Definition of Done checklist.

## Environment note

`torch`/`torch_geometric`/`scikit-learn` were installed into `.venv` this
session (CPU-only torch wheel). If resuming in a fresh environment, run:
```bash
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install torch_geometric scikit-learn
```
