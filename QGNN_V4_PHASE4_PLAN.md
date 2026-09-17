# QGNN-v4 Phase 4: Representation, Architecture and Quantum Optimization Search — Plan

Establishes what already exists, what is genuinely missing, and the exact
staged experimental order **before any Phase 4 run happens** — per this
phase's own "Required Output: create `QGNN_V4_PHASE4_PLAN.md` before
running the experiments" instruction. Nothing in this document has been
run yet. `QGNN_V4_PHASE4_RESULTS.md` will follow after Stage 1 completes.

---

## 0. Inputs carried forward from Phase 3 (unchanged, not re-derived)

| Config | Primary PR-AUC | Severity PR-AUC |
|---|---|---|
| Classical GraphSAGE-Full (extracted, not retrained) | 0.807 ± 0.066 | 0.449 ± 0.013 |
| 6q / 2L / StronglyEntangling / LayerNorm-no-affine (**Phase-3 reference**) | 0.811 ± 0.074 | 0.398 ± 0.057 |
| 4q / 2L | 0.847 ± 0.102 | 0.396 ± 0.030 |
| 8q / 2L | 0.810 ± 0.091 | 0.364 ± 0.054 |
| 6q / 1L | 0.693 ± 0.198 | 0.386 ± 0.017 |
| 6q / 3L (primary only; severity killed mid-run, no usable data) | 0.804 ± 0.186 | — |

Still outstanding from Phase 3, not part of Phase 4's critical path: 6q/4L,
`hardware_efficient_ring`, `reduced_entanglement` (both splits each), and
6q/3L severity. Phase 4 does not require these to proceed — Track E below
re-specifies a smaller, controlled ansatz comparison instead of resuming
the full leftover matrix.

## 1. Required code audit — what exists vs. what must be built

Read in full before writing this plan: `src/scm_dataset/modeling/{data,
features,hetero_graph,graphsage,graph_embedding_reduction,preprocessing,
reduction,losses,evaluate,metrics,qgnn_v2,config}.py`,
`src/scm_dataset/labels/risk_labels.py`, `src/scm_dataset/benchmark/
splits.py`, `src/scm_dataset/modeling/quantum/*.py`,
`scripts/run_qgnn_v4_experiment.py`, `scripts/_analysis_common.py`,
`configs/qgnn_v4*.yaml`, and the actual dataset on disk
(`data/benchmark/scm_v1_black_swan_seed43`).

### 1.1 Already exists — reuse unchanged

| Need | Existing piece |
|---|---|
| Frozen encoder load + freeze | `graph_embedding_reduction.load_frozen_graphsage_encoder` |
| Raw 128D embedding extraction, per (supplier, t) | `graph_embedding_reduction.extract_supplier_embeddings` |
| Wrapping raw embeddings as a head-ready prepared dataset (same `examples`/`split`/`target` the QGNN itself trains on) | `quantum.model.build_v4_prepared` → `qgnn_v2.V2PreparedData` |
| PCA fit **train-split only**, transform | `reduction.PCASupplierReducer`, `graph_embedding_reduction.fit_embedding_pca` |
| Classification metrics (PR-AUC, ROC-AUC, F1, MCC, Brier, ECE, confusion matrix, specificity) | `metrics.compute_classification_metrics`, `_analysis_common.py`'s `mcc_from_confusion`/`specificity_from_confusion`/`max_calibration_error` |
| Fresh-onset vs. already-ongoing breakdown (recall + PR-AUC/ROC-AUC, per split) | `evaluate.disruption_onset_breakdown` |
| Severity-level reconstruction from raw `events.csv` (verified zero-mismatch in Phase 2b) | `_analysis_common.load_period_severity` |
| Weighted BCE loss (`pos_weight` = neg/pos on train) | `losses.build_loss`/`compute_pos_weight` |
| Checkpoint resolution (primary vs. severity encoder) | `_analysis_common.find_latest_graphsage_full_checkpoint(..., split_suffix=...)` |
| Rolling-window dynamic features (order volume, fulfillment ratio, inventory, production, demand, backlog — windows 4/8/12) | `features.build_dynamic_panels`, `_supplier_panel`/`_material_panel`/`_plant_panel`/`_product_panel` |
| Leakage audit gate | `features.audit_feature_sources` / `build_feature_audit` |
| All 4 quantum ansätze (`strongly_entangling`, `basic_entangler`, `hardware_efficient_ring`, `reduced_entanglement`), arbitrary qubit/layer count | `quantum.circuit.build_quantum_layer` |
| LayerNorm output head, arbitrary `n_qubits`/`n_layers`/`ansatz` | `quantum.heads.HybridQuantumHeadLayerNorm` |
| Matched-capacity classical control | `quantum.heads.MatchedCapacityClassicalHead` |
| Diagnostics training loop (per-epoch grad norms, PR-AUC) | `quantum.train.train_v4_head_with_diagnostics` |

### 1.2 Genuinely missing — must be built this phase

| Track | What's missing |
|---|---|
| A1 | A diagnostic script running sklearn `LogisticRegression`/`LinearSVC`/`MLPClassifier`/`RandomForestClassifier` on the frozen embedding (train-fit only), scored on primary + severity val/test. Does not exist. |
| A2 | PCA explained-variance sweep {4,6,8,16,32,64,128} + a diagnostic classifier re-run at each level. `PCASupplierReducer` exists; the sweep script does not. |
| A3 | Per-dimension (128) train-only point-biserial correlation with target/severity/fresh-onset, plus mutual information (`sklearn.feature_selection.mutual_info_classif`). Does not exist as a script, though Phase 2d's `inspect_seed45_quantum_features.py` already established the point-biserial-on-a-channel pattern this can reuse. |
| A4 | Graph/event/label distributional statistics (node/edge counts by type, degree distribution, density, connected components, avg path length, class/risk/severity/temporal distribution). Does not exist as a script; raw counts pulled ad hoc for this plan (§2 below) but not yet a saved, reusable diagnostic. |
| B2/B3/B4 | Nonlinear projection head (`Linear(128,32)→act→Linear(32,N)`), pre-projection LayerNorm variant, PCA-informed projection variant. None exist — new `HybridQuantumHead*` subclasses needed in `quantum/heads.py`. |
| C | Encoding-scale variants (π/2, π, 2π · tanh). Currently hardcoded to `π · tanh` inside each head's `forward` — needs a parameter, not a new class. |
| D | Data re-uploading (repeat `AngleEmbedding` + variational layer twice). New ansatz/circuit variant in `circuit.py`. |
| F | Init-strategy control on `TorchLayer` weights (small-Gaussian, identity-like). `qml.qnn.TorchLayer` accepts a `weight_init` config currently unset (PennyLane default init used implicitly) — needs an explicit param threaded through `build_quantum_layer`. |
| G | Focal loss. `losses.py` currently only has weighted BCE (`build_loss`) — focal loss does not exist. |
| K | GCN/GAT encoder variant trained under the identical protocol as `HeteroGraphSAGE`. Does not exist — `graphsage.py` only implements SAGEConv. |

### 1.3 Environment constraint discovered

`xgboost` is **not installed** in `.venv` (not in `requirements.txt`,
`pip show xgboost` returns nothing). Per this phase's own instruction
("XGBoost if already available in the environment"), **A1's tree-based
diagnostic uses `RandomForestClassifier` only** — XGBoost is skipped
rather than installed as a new dependency, since Phase 4 is diagnostic,
not a request to expand the project's dependency surface. This will be
stated plainly in the results, not silently substituted.

## 2. What the audit already tells us (facts, not yet experiments)

These follow directly from reading the code/data, not from running
anything new — worth stating up front because they sharpen Stage 1's
hypotheses rather than starting from nothing.

- **The dataset used by every QGNN-v4 run so far** (`data/benchmark/
  scm_v1_black_swan_seed43`, per `configs/qgnn_v4*.yaml`) **has only 6
  events total, and only 1 of them reaches severity 5** (severities: two
  at 1, three at 2, one at 5). The severity split's entire test set is
  therefore built from one event's active window (plus its recovery
  tail), not a diverse population of severe disruptions. This is a
  direct, concrete candidate explanation for why the severity split has
  stayed far below the classical baseline (0.36–0.45 vs. 0.449) across
  every Phase-2/3 configuration tried — worth confirming empirically in
  Track H/A4 before concluding anything about quantum vs. classical
  capability on severity generalization specifically.
- **The graph is a fixed bipartite-ish DAG-like topology**: 300
  suppliers / 100 materials / 50 plants / 200 products / 20 regions /
  2000 procurement-order nodes, 7675 directed edges across 9 relation
  types (plus 9 reverse relations added only in the PyG view). Multi-
  source (2–5 suppliers/material) and multi-plant material sourcing
  exist (`materials_per_plant` 5–20), so the graph is not a simple tree —
  worth confirming actual degree/density/component numbers in A4 rather
  than reasoning from generator defaults alone.
- **No explicit "own disruption history" or "neighbor disruption
  history" feature is fed to GraphSAGE.** `features._supplier_panel`
  builds only rolling procurement/delivery aggregates (order volume,
  orders count, delivery volume, fulfillment ratio, windows 4/8/12) —
  `supplier_disrupted` itself is read **only** by `build_prediction_
  examples` to build the *target*, never merged into the feature frame
  (`features.py`'s own docstring: "`supplier_risk_score` is deliberately
  never read by this module"). So the model's only route to "this
  supplier was recently in trouble" is an indirect proxy (a dip in
  `fulfillment_ratio`), not a direct memory signal, and GraphSAGE's
  2-layer message passing only aggregates *current-snapshot* neighbor
  features, not neighbors' own disruption history. This is a strong,
  code-grounded starting hypothesis for Track I (weak fresh-onset
  signal) and Track J (temporal features) — to be confirmed against
  actual MI/correlation numbers in A3, not assumed as already proven.
- **The training loss is already exactly "weighted BCE"** (`losses.
  build_loss` with `pos_weight = negatives/positives`, computed from
  train targets only) — Track G1 (current loss) needs no new code, only
  inspection (done, this section). Track G2 (focal loss) is new.
- **PCA-on-embeddings infrastructure already exists and is exactly
  train-fit-only** (`fit_embedding_pca` → `supplier_fit_mask` →
  `PCASupplierReducer.fit`), so A2/B4 need no new leakage-safety code,
  only a sweep driver.

## 3. Experiment order (unchanged from the phase spec) and what each stage needs

| Stage | Tracks | New code required | Involves QGNN/quantum training? |
|---|---|---|---|
| 1 — Representation audit | A1, A2, A3, A4, H, I, J | Yes (table 1.2) | **No** — diagnostic classifiers on frozen embeddings (seconds, CPU) + pure data/graph analysis. No quantum circuit, no GraphSAGE retraining. |
| 2 — Bottleneck | B1 (control, already have), B2, B3, B4 | Yes | Yes — new head variants, trained under frozen protocol |
| 3 — Quantum encoding | C1/C2 (control, already have), C3, D1 (control), D2 | Yes | Yes |
| 4 — Initialization | F1 (control, already have), F2, F3 | Yes | Yes |
| 5 — Ansatz | E1 (control, already have — reference StronglyEntangling), E2 (new, controlled: 4q+2L and 6q+2L only, both new ansätze) | No (circuit.py already has both) | Yes |
| 6 — Loss | G1 (control, already have), G2 | Yes (focal loss) | Yes |
| 7 — Encoder architecture | K | Yes (GCN/GAT variant) | Yes — **only attempted if Stage-1 evidence indicates GraphSAGE itself is limiting performance** |

Per the phase's own **Rule 1** (one changed factor per experiment) and
**Experiment Budget** (small controlled run → analyze → keep/drop →
expand only promising configs to all 5 seeds × 2 splits), each stage
after Stage 1 will itself get a short recap document only once Stage 1's
findings are in — Stage 2 onward isn't detailed further here because its
exact shape (e.g. whether B2's hidden width is 32 vs. something else)
should be informed by A1–A3's actual signal-concentration findings, not
fixed in advance of them.

## 4. Stage 1 — concrete plan (next to execute)

All seven Stage-1 tracks reuse `build_v4_prepared`'s exact `(embedding,
target, split)` triples the QGNN itself is scored on — so any diagnostic
number is directly comparable to the QGNN/classical PR-AUC numbers
already in hand, not a different train/test partition.

One new script, `scripts/run_phase4_stage1_diagnostics.py`, covering
A1–A4/H/I/J in one pass per split (primary, severity), producing:

- `experiments/qgnn_v4/phase4_stage1/representation_diagnostics.csv` (A1: one row per {model × split × metric})
- `experiments/qgnn_v4/phase4_stage1/pca_sweep.csv` (A2: one row per {n_components × split × metric}, plus explained-variance-ratio column)
- `experiments/qgnn_v4/phase4_stage1/dimension_correlations.csv` (A3: one row per {embedding dim × target variant (target/severity/fresh_onset) × correlation/MI})
- `experiments/qgnn_v4/phase4_stage1/graph_diagnostics.json` (A4)
- `experiments/qgnn_v4/phase4_stage1/severity_level_breakdown.csv` (H, reusing `load_period_severity`)
- `experiments/qgnn_v4/phase4_stage1/fresh_onset_audit.csv` (I, reusing `disruption_onset_breakdown` against saved GraphSAGE/QGNN prediction tables already on disk)
- `QGNN_V4_PHASE4_STAGE1_TEMPORAL_FEATURE_AUDIT.md` section inside the results doc (J — a written finding + a proposal only, no dataset change, per the phase's explicit "do not immediately modify the dataset" rule)

This is diagnostic analysis on already-extracted, already-frozen
embeddings and already-existing CSVs — no new GraphSAGE or QGNN training
run is needed for any of A1–A4/H/I/J, so (consistent with how prior
analysis-only scripts in this project were handled, e.g. `extract_
classical_gnn_baseline.py`, `inspect_seed45_quantum_features.py`) this
stage will be built and executed directly, with results reported back
rather than handed off as a command to run manually. Stage 2 onward —
anything that trains a new head variant end-to-end — stays handed off
per the project's standing convention.
