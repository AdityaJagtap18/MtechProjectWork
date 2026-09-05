# Classical GraphSAGE — Status

Companion to `CLASSICAL_GRAPHSAGE_IMPLEMENTATION_PLAN.md`. Read that plan
first, then this file for "where things actually stand" and the frozen
first-run numbers.

## Status: first frozen classical baseline complete

All of plan §91's Definition of Done items are satisfied. The classical
GraphSAGE benchmark is correct, reproducible, leakage-audited (structurally
+ by automated check + by tests), and ready to be the baseline a later QGNN
is compared against under the same protocol (plan §71/§72).

## Headline results (test split, threshold = 0.5, `scm_v1_black_swan_seed43`)

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Majority | 0.067 | 0.500 | 0.000 | 0.000 | 0.000 | 0.064 |
| Logistic Regression | 0.304 | 0.821 | 0.181 | 0.632 | 0.282 | 0.145 |
| **GraphSAGE (mean ± std, 5 seeds)** | **0.807 ± 0.066** | **0.987 ± 0.008** | **0.522 ± 0.042** | **0.968 ± 0.016** | **0.677 ± 0.035** | **0.042 ± 0.012** |

Full narrative (dataset, features, architecture, split, calibration, early
warning, limitations, reproducibility check) in
`experiments/classical_gnn/20260905T041715Z_hetero_graphsage_seed42/experiment_report.md`
(the `experiments/` tree is gitignored — regenerable byte-for-byte by
rerunning the command below with the same seed, which was verified: two
independent seed-42 runs matched to 15 decimal digits).

Reproduce with:
```bash
.venv/bin/python scripts/run_graphsage_experiment.py \
    --config configs/graphsage.yaml --seeds 42,43,44,45,46 --baselines
```

## Package layout

`src/scm_dataset/modeling/`: `config.py`, `data.py`, `features.py`,
`preprocessing.py`, `hetero_graph.py`, `graphsage.py`, `losses.py`,
`metrics.py`, `calibration.py`, `pipeline.py` (shared setup so train/
evaluate/baselines/tests all run the identical leakage-audited pipeline),
`train.py`, `evaluate.py`, `baselines.py`, `experiment.py` (run-dir +
reproducibility metadata).

`scripts/`: `run_graphsage_experiment.py` (main entry point — prepare once,
train+evaluate per seed, optional `--baselines`, auto-writes a multi-seed
summary), `train_graphsage.py`, `evaluate_graphsage.py` (standalone
re-evaluation of a saved run — reloads its preprocessing artifacts rather
than refitting).

`configs/graphsage.yaml`: dataset pinned to `scm_v1_black_swan_seed43` (see
its comments for why — has all 3 split types, a severity-5 event, and is
the dataset `WORK_SUMMARY.md` already traced end to end).

**Dependencies**: torch 2.14 (CPU wheel), torch_geometric 2.8, scikit-learn
1.9 — in `.venv` and in `pyproject.toml`'s `[modeling]` extra /
`requirements.txt`. Fresh-environment install:
```bash
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install torch_geometric scikit-learn
```

## Tests

`tests/conftest.py` provides `tiny_benchmark`/`tiny_prepared_data` fixtures
— a small in-memory benchmark (20 suppliers, 60 periods, seed=3, chosen for
non-degenerate positive rates in all 3 splits) built through the **real**
generator/simulation/labels/splits code and run through the **real**
`pipeline.prepare_from_benchmark`. Covers data loading, the leakage audit,
target/temporal alignment (incl. a causality regression test that injects a
future record and asserts a past feature is unchanged), preprocessing
(train-only fit, missing-value handling, categorical encoding), hetero
graph construction (node id determinism, edge validation), the GraphSAGE
model (forward pass, no internal sigmoid — checked architecturally),
training (pos_weight, early stopping selects the best-val-PR-AUC
checkpoint, never builds a test-split snapshot), evaluation/metrics/
calibration, and two integration tests (in-process, and a subprocess run of
the actual CLI script against a tiny on-disk benchmark).

**Full suite: 158/158 passing** (90 original dataset-framework tests + 68 new).

## Key decisions (documented in code, summarized here)

- `supplier_risk_score` **excluded** from model inputs (plan §7/§72) —
  `feature_audit.csv` already marks all of `labels/supplier_labels.csv`
  `allowed=False`; confirmed via `labels/risk_labels.py` that it's a
  realized fulfillment shortfall from the same simulation outcomes as the
  target.
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
  deviation in `pipeline.py`'s `_apply_generalization_split` docstring,
  since the plan's split files are period-level but the task is
  windowed-future prediction. A validation slice is carved from the tail of
  the resulting train pool (these split files are train/test-only).
- Risk ranking snapshot uses the latest TEST-split prediction time (the
  plan's example table is one row per supplier, implying "as of now," not
  one row per supplier x time).
- Default 0.5 threshold gives high recall (~0.97) / moderate precision
  (~0.52) — a direct consequence of `class_weighting: balanced`'s
  `pos_weight`, appropriate for an early-warning framing. A
  validation-selected threshold policy (`f1_optimal` /
  `precision_constrained` / `recall_constrained`) is implemented in
  `metrics.select_threshold` and configurable via `configs/graphsage.yaml`'s
  `threshold.policy` if a different operating point is wanted.

## Not yet done (all optional next steps, not blockers)

The plan explicitly sequences these *after* a stable single/multi-seed
run (§70/§77) — none were required for "definition of done," and none are
started yet:

1. **Severity generalization experiment** (plan §29/§39): copy
   `configs/graphsage.yaml`, set `split.strategy: severity`, rerun. Tests
   whether a model trained mostly on severity 1-3 events generalizes to
   held-out severity 4-5 periods.
2. **Scenario generalization experiment** (plan §30/§40): same, with
   `split.strategy: scenario` (default held-out type: cyberattack).
3. **Ablations B-E** (plan §55/§58) — static-only, dynamic-only, remove
   region risk, single- vs multi-source. Ablation A (no graph) is already
   covered by the Logistic Regression baseline.
4. A robustness rerun against `scm_v1_black_swan_seed44` (higher 6.1%
   positive rate, 9 events) by changing `configs/graphsage.yaml`'s
   `dataset.dataset_id`.
5. Once the above are judged sufficient, freeze this configuration
   explicitly (plan §80) before starting the QGNN stage.
