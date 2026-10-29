# GraphSAGE Next-Phase Specification

**Status of this document**: pure specification. No model, feature, benchmark,
label, split, or hyperparameter was changed to produce it. All numbers below
were measured directly from the repository (`data/benchmark/scm_v1_black_swan_seed43`,
`src/scm_dataset/`, `experiments/classical_gnn/`) during this investigation,
not recalled from prior summary documents. Where a prior summary's claim is
repeated here, it has been re-verified against source.

Evidence is marked **Observed** (a fact read directly from code or data),
**Measurement** (a number computed during this investigation), **Hypothesis**
(a plausible but unconfirmed explanation), or **Recommendation** (an action
this document proposes). These are never mixed silently.

---

## 1. Executive Summary

The classical GraphSAGE baseline is fully built, tested (180/180 passing),
and has produced two rounds of results: a primary temporal-split run
(test PR-AUC 0.807 ± 0.066 over 5 seeds) and a feature-information ablation
sweep (Phase A/B of `GRAPH_SAGE_IMPROVEMENT_PLAN.md`). The ablation sweep
answered its central question **definitively and negatively**: no feature
group tested (dynamic operational history, static supplier attributes, or
background risk exposure) contains usable pre-event signal for a fresh
disruption onset — recall is exactly 0.0000 and ranking quality is *below
chance* (ROC-AUC 0.34–0.46) in every case, on the severity-holdout split.

This investigation adds one new, load-bearing finding that the prior phases
did not surface: **the "static features beat dynamic features" result is,
on direct inspection, substantially attributable to which specific regions
this one random dataset draw happened to place its 4 region-targeting
events in, not to those regions carrying genuinely higher risk scores.**
Four regions (region_1, region_3, region_8, region_19) contain 44 of the
benchmark's 46 ever-disrupted suppliers; their mean risk-score percentile
across the dataset's 20 regions is 0.46, 0.43, 0.74, and 0.76 respectively —
average to moderately above average, not the top of the distribution two
of them sit *below* the median. A model that performs well by learning
"these four regions are risky" is very plausibly memorizing this dataset's
specific random draw of which regions got hit, not learning a genuinely
risk-proportional signal that would transfer to a re-generated benchmark
with a different draw. Section 14 gives the exact, cheap experiment
(rerun the existing `static_only` ablation against `scm_v1_black_swan_seed44`,
already generated, no new dataset needed) that would confirm or refute this
before any further architecture or feature work is invested on the strength
of the current static-feature result.

This document's roadmap (§25) is accordingly reordered from the prior
plan's: **disambiguate the static-feature finding first** (cheap, uses an
existing dataset, directly gates whether "improve static/structural
features" is worth doing at all), **then** proceed to the temporal/
architecture/training improvements that only matter for the persistence-
detection task, since fresh-onset prediction has already been shown to be
unattainable from any input this benchmark provides.

---

## 2. Repository Findings

**Observed** — documents read in full: `WORK_SUMMARY.md`,
`GRAPHSAGE_WORK_SUMMARY.md`, `GRAPHSAGE_PROGRESS.md`,
`GRAPH_SAGE_IMPROVEMENT_PLAN.md`, `GRAPHSAGE_IMPROVEMENT_PROGRESS.md`,
`GRAPHSAGE_IMPROVEMENT_WORK_SUMMARY.md`, `CLASSICAL_GRAPHSAGE_IMPLEMENTATION_PLAN.md`,
`README.md`. Source inspected: all of `src/scm_dataset/modeling/`, all of
`src/scm_dataset/events/`, `src/scm_dataset/schema/`, `src/scm_dataset/labels/risk_labels.py`,
`src/scm_dataset/simulation/engine.py`, `src/scm_dataset/generator/procurement.py`,
all of `scripts/*.py`, all of `configs/*.yaml`, all of `tests/`. Data
inspected directly (not summarized secondhand): every file under
`data/benchmark/scm_v1_black_swan_seed43/` (graph, operations, labels,
events, splits) plus `data/benchmark/feature_audit.csv`. Experiment
artifacts inspected: `metrics.json`, `calibration.json`, `training_history.csv`,
`run_metadata.json`, `onset_breakdown.json` from the primary frozen run and
the severity-split ablation runs.

**Observed** — no implementation bug was found that the plan's leakage
rules fail to catch. One naming/documentation issue was found and is noted
in §7 (not a leakage bug — a discrepancy between a docstring and the
literal audit-file content, already handled correctly by the parsing code).

---

## 3. Current GraphSAGE Architecture

Traced directly from `src/scm_dataset/modeling/graphsage.py`,
`hetero_graph.py`, `train.py`, `evaluate.py`, `losses.py`, `metrics.py`.

```text
raw benchmark (data/benchmark/scm_v1_black_swan_seed43/)
    |
    v
data.py: load_benchmark() -- reads graph/, operations/, labels/, splits/,
         feature_audit.csv; validates schema; raises if anything expected
         is missing (never silently adapts).
    |
    v
features.py: build_static_features() + build_dynamic_panels() + build_feature_frames()
         -- static: one row per node, straight from graph/nodes.csv.
         -- dynamic: rolling-window aggregates over operations/*.csv,
            windows [4, 8, 12] periods, causal by construction (full
            entity x time grid, rolling(min_periods=1) only ever looks
            backward from its own row).
    |
    v
features.py: audit_feature_sources() -- checks every (table, column) this
         module reads against feature_audit.csv; raises if anything is
         marked disallowed or is unrecognized. Runs BEFORE any feature is
         built, every single run.
    |
    v
features.py: apply_feature_mode() -- optional column restriction for the
         Phase B ablations (full/dynamic_only/static_only/region_risk_only/
         static_plus_graph). Identity function when feature_mode="full".
    |
    v
features.py: build_prediction_examples() -- one row per (supplier_id, t),
         Y(t) = max(supplier_disrupted[t+1..t+H]), read directly from
         labels/supplier_labels.csv.
    |
    v
pipeline.py: build_split_assignment() -- joins the benchmark's own
         temporal_split.csv (direct join on t) or scenario/severity_split.csv
         (joined via the example's target window (t, t+H]; a validation
         slice is then carved from the tail of the resulting train pool,
         since those two files are train/test-only).
    |
    v
preprocessing.py: FeaturePreprocessor.fit() -- per node type, numeric
         columns get {value, missing indicator} (missing imputed with the
         TRAIN-only median; scaled by TRAIN-only mean/std), categorical
         columns get one-hot + explicit UNKNOWN bucket, vocabulary fit on
         TRAIN-only rows. Supplier's fit rows = exactly its train-split
         (supplier_id, t) prediction examples; material/plant/product's fit
         rows = all entities at train-split t values (they have no
         examples of their own); region/procurement's fit rows = ALL rows
         (pure generation-time constants, identical in every split).
    |
    v
hetero_graph.py: HeteroGraphSnapshotBuilder -- node id -> index maps
         (deterministic natural sort), static topology built once
         (build_edge_index_dict adds a rev_<type> relation for every one
         of the benchmark's 9 edge types -- 18 relations total, PyG
         representation only, never written back to the benchmark).
         build(t) assembles one torch_geometric.data.HeteroData per
         prediction time: static node types (region, procurement) get the
         same tensor at every t; supplier/material/plant/product get a
         fresh row sliced from the preprocessed dynamic panel at t.
    |
    v
graphsage.py: HeteroGraphSAGE
         -- per-node-type nn.Linear input projection to hidden_dim (128
            default), independent weights per node type.
         -- num_layers (2 default) x HeteroConv({relation: SAGEConv(hidden,
            hidden, aggr="mean") for relation in 18 relations}, aggr="mean")
            -- "mean" twice: SAGEConv's own neighbor aggregation within one
            relation, and HeteroConv's aggregation across the several
            relations that can feed one node type.
         -- F.relu + nn.Dropout(0.20 default) after every conv layer.
         -- readout: only the supplier node type's embedding is used.
            nn.Sequential(Linear(hidden, hidden//2), ReLU, Dropout,
            Linear(hidden//2, 1)) -> one raw logit per supplier.
         -- no sigmoid inside the model (checked architecturally in
            tests/test_graphsage.py, not just by output range).
    |
    v
losses.py: BCEWithLogitsLoss(pos_weight = negatives/positives, computed
         from TRAIN-split targets only every run -- e.g. 37.67 on the
         primary temporal run).
    |
    v
train.py: train_graphsage() -- Adam(lr=0.001, weight_decay=0.0001,
         config-driven), one forward/backward pass per distinct train-split
         prediction time per epoch (not per example -- one snapshot scores
         all 300 suppliers at once), up to 100 epochs, early stopping on
         validation PR-AUC with patience 10, restores the best-val-PR-AUC
         epoch's weights (verified by a dedicated test that scripts a
         later, worse epoch and confirms the earlier one is what's
         returned). set_seed(seed) resets Python/NumPy/PyTorch RNGs at the
         start of every call -- reproducibility verified empirically:
         independent reruns (including by a different user, from a fresh
         terminal) matched to full floating-point precision.
    |
    v
metrics.py: select_threshold() -- fixed 0.5 by default; f1_optimal /
         recall_constrained / precision_constrained implemented and
         available via config, computed ONLY from validation-split
         predictions (never test).
    |
    v
evaluate.py: evaluate_experiment() -- one forward pass per distinct
         prediction time across ALL splits, threshold applied, then:
         compute_classification_metrics() (precision/recall/F1/PR-AUC/
         ROC-AUC/balanced accuracy/Brier/confusion matrix, with safe
         single-class handling), compute_calibration() (Brier + ECE +
         10-bin reliability curve), disruption_onset_breakdown() (recall
         AND ranking quality split by already-ongoing vs fresh-onset,
         added this phase), temporal_variation_summary() (fraction of
         suppliers whose prediction varies at all across time -- added
         this phase after catching a degenerate ablation mode), risk
         ranking (latest test-split snapshot, sorted descending), and
         compute_warning_times() (secondary descriptive early-warning
         metric on raw label transitions).
```

**Measurement**: 1,221,377 trainable parameters (`hidden_dim=128`,
`num_layers=2`, full feature mode). Input dimensionality per node type
after preprocessing: supplier 60, material 48, plant 38, product 45,
region 12, procurement 16 (these counts already include the {value,
missing-indicator} doubling for numeric columns and one-hot expansion for
categoricals — see §6 for the raw feature list each number is built from).

---

## 4. Dataset Structure

All numbers below are from `data/benchmark/scm_v1_black_swan_seed43`
(the frozen primary/severity dataset), measured directly.

### 4.1 Graph (`graph/nodes.csv`, `graph/edges.csv`)

**Measurement**: 2,670 nodes total — supplier 300, material 100, plant 50,
product 200, region 20, procurement 2,000. `nodes.csv` is one wide file
(61 columns), `node_type` distinguishes rows; a node's non-applicable
columns are blank (become `NaN` on read, coerced by `load_graph`'s type
hints back to each field's declared type).

**Measurement**: 7,675 edges — `supplier_procurement` 2,000,
`procurement_material` 2,000, `procurement_plant` 2,000, `material_plant`
649, `supplier_region` 300, `supplier_material` 276, `plant_product` 200,
`product_region` 200, `plant_region` 50. Zero isolated nodes of any type
(every node has at least one edge). Full structural detail (degree
distributions, single/multi-source counts) in §11.

### 4.2 Operations (`operations/*.csv`) — all 104 weekly periods (0-103)

| File | Rows | Columns | Time range |
|---|---:|---|---|
| `demand.csv` | 20,800 | product_id, time, demand | 0-103 |
| `production.csv` | 20,800 | plant_id, product_id, time, production | 0-103 |
| `inventory.csv` | 67,496 | plant_id, material_id, time, inventory | 0-103 |
| `backlog.csv` | 20,800 | product_id, time, backlog | 0-103 |
| `deliveries.csv` | 11,472 | plant_id, material_id, supplier_id, time, quantity | 3-103 |
| `procurement.csv` | 11,662 | plant_id, material_id, supplier_id, time_ordered, quantity_ordered, quantity_fulfilled, expected_delivery_time | 2-103 |

**Observed**: `deliveries.csv`/`procurement.csv` start at t=2/t=3, not t=0
— the first 2-3 periods have no completed order/delivery cycle yet (lead
time). `features.py`'s full-grid construction (`_full_grid`) fills these
early, genuinely-inactive periods with 0, not `NaN` — a real zero, not a
missing observation (except `fulfillment_ratio_*`, deliberately left `NaN`
when the rolling window sum of `quantity_ordered` is 0, since a 0/0 ratio
is genuinely undefined, not zero).

### 4.3 Labels (`labels/*.csv`)

| File | Rows | Columns |
|---|---:|---|
| `supplier_labels.csv` | 31,200 | supplier_id, time, supplier_disrupted, supplier_risk_score |
| `material_labels.csv` | 67,496 | plant_id, material_id, time, material_shortage, material_risk_score |
| `plant_labels.csv` | 5,200 | plant_id, time, production_loss, production_loss_fraction, plant_disruption |
| `product_labels.csv` | 20,800 | product_id, time, product_shortage, revenue_impact |

**Measurement**: `supplier_disrupted` is 1 in 1,039 of 31,200 rows
(3.33%), across exactly 46 of 300 suppliers (never more, never fewer —
every disrupted supplier has exactly one 0->1 onset transition in this
dataset; none recover and get hit again).

### 4.4 Events (`events/events.csv`, `events/event_impact.csv`)

**Observed**: 6 events. `event_0` logistics/sev2/region_8, `event_1`
supplier_failure/sev1/supplier_53 (no region), `event_2`
geopolitical/sev2/region_1, `event_3` logistics/sev5/region_19, `event_4`
cyberattack/sev1/supplier_240 (no region), `event_5` logistics/sev2/region_3.
`event_impact.csv` (5 columns: event_id, cascade_severity,
total_affected_nodes, time_to_impact, recovery_time) is a **post-hoc
summary computed after the fact** — `feature_audit.csv` marks the entire
table disallowed as a model input (confirmed correct: it can only exist
once an event's consequences have already played out).

### 4.5 Splits (`splits/*.csv`)

**Measurement**: `temporal_split.csv` — train 73/validation 15/test 16
periods. `scenario_split.csv` — train 50/test 54 (held-out types
`{cyberattack, geopolitical}`, per `scripts/build_benchmark.py`'s
`SCENARIO_SPLIT_TEST_TYPES`). `severity_split.csv` — train 67/test 37
(train severity <=3, test severity >=4; only `event_3`, sev5, meets the
test threshold in this dataset — its long recovery tail drives the entire
37-period test window).

### 4.6 Feature audit (`data/benchmark/feature_audit.csv`)

**Observed**: 14 rows. `graph/nodes.csv,*` allowed (static, generation-time).
`operations/{demand,production,inventory,backlog}.csv`'s value column and
`deliveries.csv`'s `quantity` allowed (available at t, safe for target > t).
`procurement.csv`'s `quantity_ordered, quantity_fulfilled` allowed (note:
**this one row's `column` field is a single comma-joined string covering
two columns** — `features.py::audit_feature_sources` splits on `,` before
matching; a naive exact-string match against this file would have silently
rejected both columns, which is exactly the bug this session's smoke test
caught and fixed early in the classical-baseline phase). `procurement.csv`'s
`expected_delivery_time` explicitly disallowed (a future timestamp).
`events/events.csv,*` and `events/event_impact.csv,*` disallowed.
`labels/{supplier,material,plant,product}_labels.csv,*` disallowed (targets
only, including `supplier_risk_score` — see §7).

---

## 5. Target Definition

**Observed**, from `labels/risk_labels.py::compute_supplier_labels` and
`modeling/features.py::build_prediction_examples`:

`supplier_disrupted[supplier, t] = 1` iff some event's `affected_suppliers`
list contains `supplier` AND `t` falls in that event's active window
`[start_time, start_time + duration + recovery_delay + recovery_periods)`.
This is a direct, unambiguous read of the event's own targeting — not a
threshold on any simulated quantity.

`supplier_risk_score[supplier, t]` = `1 - quantity_fulfilled/quantity_ordered`
summed over that supplier's orders placed at exactly `time_ordered = t`
(`None` if no orders that period). **This is a realized outcome of the
same simulation dynamics the target reflects** — `feature_audit.csv`
correctly marks the entire label table disallowed as an input.

**Prediction target**: `modeling/features.py::build_prediction_examples`:
`Y(supplier, t) = max(supplier_disrupted[t+1 .. t+H])`, `H` = 4 (config).
Usable `t` range: `[min_history_periods - 1, horizon_periods - 1 - H]` =
`[11, 99]` for this benchmark (89 values) — bounded below by the largest
rolling window (12) needing full history, above by `t+H` needing to fit
inside the recorded horizon.

**Measurement — overlap between examples**: one raw onset (46 total)
generates up to `H` positive prediction examples (one for each `t` in
`[onset-H, onset-1]` that is itself a usable prediction time and the
supplier was not already disrupted at that `t`) — this is why example-level
positive counts (fresh-onset: 40 in temporal-validation, 84 in
severity-test) exceed the raw onset count. Once a supplier is disrupted,
every subsequent `t` while `supplier_disrupted[t]=1` also produces a
(different, "already-ongoing") positive example for every `t` in that
window whose look-ahead still finds a 1 — this is why total positive
example rows (e.g. 481 in the temporal-split train set) are far larger
than the raw label positive count would suggest per-supplier, and why a
single long event (event_3, active for up to 54 weeks for the suppliers
`event_2` hit) can dominate an entire split's positive population (see §9).

**Observed**: prediction-time alignment is correct and tested
(`tests/test_temporal_features.py::test_target_is_max_over_future_horizon_not_current_period`
and `..._never_uses_current_or_past_periods`) — the disruption period
itself is never counted as part of its own "future" window.

---

## 6. Feature Provenance

Every column that can reach the model, read directly from
`modeling/features.py`'s `STATIC_NUMERIC_FIELDS`/`STATIC_CATEGORICAL_FIELDS`
dicts and the `_supplier_panel`/`_material_panel`/`_plant_panel`/`_product_panel`
functions (dynamic panels).

| Feature | Source | Node | Static/Dynamic | Available at t? | Target-derived? | Safe? | Notes |
|---|---|---|---|---|---|---|---|
| tier, capacity, capacity_utilization, reliability, financial_health, lead_time_mean, lead_time_variability, quality_score, inventory_buffer, substitution_availability, geopolitical_exposure, disaster_exposure, cyber_exposure, criticality | graph/nodes.csv | Supplier | Static | Yes (generation-time) | No | Yes | 14 numeric fields |
| industry | graph/nodes.csv | Supplier | Static (categorical) | Yes | No | Yes | one-hot + UNKNOWN |
| order_volume/orders_count/delivery_volume/fulfillment_ratio, x windows [4,8,12] | operations/procurement.csv, deliveries.csv | Supplier | Dynamic | Yes (rolling, causal) | No | Yes | 12 numeric; fulfillment_ratio is NaN (not 0) when window has zero orders |
| criticality, substitutability, demand, unit_cost, inventory_level, safety_stock, supplier_count, concentration, required_quantity_per_product | graph/nodes.csv | Material | Static | Yes | No | Yes | supplier_count/concentration are derived at *generation* time from topology, not from operations |
| material_category, procurement_type | graph/nodes.csv | Material | Static (categorical) | Yes | No | Yes | procurement_type nullable (MTS/OTS/None) |
| inventory mean/std/trend, x windows [4,8,12] | operations/inventory.csv | Material | Dynamic | Yes | No | Yes | summed across all plants holding that material |
| production_capacity, utilization, operating_cost, inventory_capacity, resilience_score, downtime_cost, recovery_rate | graph/nodes.csv | Plant | Static | Yes | No | Yes | |
| production mean/std/trend + inventory_mean, x windows [4,8,12] | operations/production.csv, inventory.csv | Plant | Dynamic | Yes | No | Yes | 12 numeric |
| demand, revenue_per_unit, margin, material_dependency, criticality, substitution_score, backlog | graph/nodes.csv | Product | Static | Yes | No | Yes | `backlog` here is the *static* generation-time field, distinct from the dynamic operations/backlog.csv series below |
| product_category | graph/nodes.csv | Product | Static (categorical) | Yes | No | Yes | |
| demand mean/std/trend + backlog_mean, x windows [4,8,12] | operations/demand.csv, backlog.csv | Product | Dynamic | Yes | No | Yes | 12 numeric |
| geopolitical_risk, natural_disaster_risk, infrastructure_risk, trade_risk, cyber_risk, transport_reliability | graph/nodes.csv | Region | Static | Yes | No | Yes | no dynamic component exists for region in this benchmark |
| order_quantity, order_value, order_frequency, promised_lead_time, actual_lead_time, urgency, contract_duration, priority | graph/nodes.csv | Procurement | Static | Yes | No | Yes | **generated at Phase-2/topology time** (confirmed by reading `generator/procurement.py` — derives from supplier.reliability/material.criticality, not from any simulated period), independent of the simulation timeline despite field names like `actual_lead_time` sounding like a realized outcome |
| **supplier_risk_score** | labels/supplier_labels.csv | Supplier | n/a | n/a | **Yes** | **No — excluded** | never read by `features.py` at all; `feature_audit.csv` marks the whole table disallowed |
| **supplier_disrupted[t]** | labels/supplier_labels.csv | Supplier | n/a | n/a | **Yes (it IS the target)** | **No — never a feature** | read only by `build_prediction_examples`, never merged into the feature frame; `disruption_onset_breakdown` reads it for a POST-hoc diagnostic only, never as a model input |
| expected_delivery_time | operations/procurement.csv | — | — | No (future timestamp) | No | **No — excluded** | explicitly disallowed in feature_audit.csv |
| events/events.csv, events/event_impact.csv | — | — | — | No | No | **No — excluded** | ground truth about the disruption being predicted / a post-hoc summary |
| region_id, country_or_region, plant_id/supplier_id/material_id FK columns | graph/nodes.csv | (identifiers) | — | Yes | No | **Excluded by design choice, not audit rule** | that relationship reaches the model structurally via graph edges instead — see §14 for why this design choice turned out to matter more than expected |

**Recommendation (documentation only, not a code change)**: the last row's
exclusion (`region_id` as a direct feature) was made to avoid duplicating
information the graph structure already carries. §14 shows this exclusion
does not actually prevent the model from learning "which region" — it
still gets that information through the `SUPPLIER_REGION` edge and the
region's own static features. The design choice was reasonable but its
premise (that graph structure alone dilutes identity-level memorization)
does not hold; this is noted for the record, not proposed as something to
undo.

---

## 7. Leakage Analysis

**Observed**: `audit_feature_sources` (called at the start of every
`prepare()`/`prepare_from_benchmark()` call, before any feature is built)
cross-checks every `(table, column)` pair in `FEATURE_SOURCE_COLUMNS`
against `feature_audit.csv`, raising `ValueError` if a source is
unrecognized or marked `allowed=False`. This is fail-closed: an unaudited
column cannot silently enter the model.

**Observed** — the one genuine ambiguity in the audit file itself:
`operations/procurement.csv`'s row lists `column` as the literal string
`"quantity_ordered, quantity_fulfilled"` (comma-joined in one cell), not
two separate rows. The current parsing code (`audit_feature_sources`)
correctly splits this on `,` before building its lookup table — verified
by `tests/test_temporal_features.py::test_audit_splits_comma_joined_column_cell`.
This is not a live bug; it is flagged here only because a naive re-
implementation of this check (e.g., in a future refactor) could easily
reintroduce the literal-string-match failure mode that this session's
first smoke test against the real benchmark actually hit and had to fix.

**Observed**: no dynamic feature construction path reads any column not
declared in `FEATURE_SOURCE_COLUMNS`. `supplier_risk_score` is not read
anywhere in `features.py`. `expected_delivery_time`, `events/*.csv`, and
`event_impact.csv` are not read anywhere in `features.py` or `pipeline.py`.

**Measurement**: `tests/test_temporal_features.py` includes a causality
regression test that injects a large future record into a supplier's
procurement history and asserts the feature value at an earlier `t` is
unaffected — this is an executable, ongoing guarantee, not a one-time
check.

**Conclusion**: no leakage was found. The audit correctly enforces the
plan's rules as written, and the implementation correctly respects them.

---

## 8. Temporal Analysis

**Observed** rolling-window construction (`features.py::_rolling`,
`_rolling_sum`): for value column `v` and window `w`, computes
`mean_w = rolling(w, min_periods=1).mean()`, `std_w = rolling(w,
min_periods=1).std(ddof=0)` (filled 0 if undefined), and
`trend_w = (v[t] - v[t-w+1]) / max(w-1, 1)` — a simple **endpoint slope**
(first-to-last difference divided by elapsed periods), not an OLS
regression slope over the whole window. This is a documented,
deliberate simplification (`features.py` docstring cites "small
interpretable feature set" per the original plan's §13 instruction), but
it means `trend_w` is sensitive to noise at exactly the two endpoints and
ignores the shape of everything in between — a genuine limitation worth
testing against an OLS-slope alternative (see §15).

**Observed**: windows are `[4, 8, 12]` periods (config-driven,
`FeaturesConfig.rolling_windows`), applied uniformly to every entity type
that has a dynamic component (supplier, material, plant, product). No
entity type currently gets a different window set even though their
underlying processes have different natural timescales (e.g., inventory
turns over faster than a multi-week backlog).

**Measurement**: usable prediction times span `t in [11, 99]` (89 values)
for `horizon_periods=104`, `min_history_periods=12`, `H=4`.

---

## 9. Fresh-Onset vs Persistence Analysis

**Definitions** (as implemented in `evaluate.py::disruption_onset_breakdown`,
confirmed against source):
- **Ongoing**: `supplier_disrupted[t] == 1` — the supplier is already
  visibly disrupted at the prediction time itself.
- **Fresh onset**: `supplier_disrupted[t] == 0` at the prediction time,
  but the target `Y(t)=1` because a disruption starts somewhere in `(t, t+H]`.

**Measurement** (temporal split): train 88 fresh-onset positives / 393
ongoing (of 481 total); validation 40 fresh-onset / 325 ongoing (of 365);
**test 0 fresh-onset / 242 ongoing (of 242)** — the entire test-split
positive population is ongoing cases, because `event_3`'s (severity 5)
long recovery tail (54+ periods for the suppliers `event_2` hit,
similarly long for `event_3` itself) spans almost the entire back half of
the 104-period horizon.

**Measurement** (severity split): train 191 total positives (breakdown
not separately re-measured this session, available in the run's own
`onset_breakdown.json`); validation 80 positive, 0 fresh-onset; **test 817
total, 84 fresh-onset / 733 ongoing**.

**Measurement — distribution across raw onset events** (46 total 0->1
transitions across the dataset): weeks `{11: 14, 19: 1, 46: 10, 67: 11,
86: 10}`. Attributable per event: `event_0` (logistics, region_8) 14
onsets, `event_2` (geopolitical, region_1) 10, `event_3` (logistics,
region_19, sev5) 11, `event_5` (logistics, region_3) 9, `event_1`
(supplier_failure) 1, `event_4` (cyberattack) 1. **All 46 raw onsets trace
to exactly 6 events** — there is no scenario in this benchmark with a
larger, more diverse population of independent onset instances; the
"population" of fresh-onset examples is fundamentally a handful of
repeated draws from 6 underlying events, not 46 independent trials.

**Measurement — per-supplier disruption duration**: the 10 suppliers hit
by `event_2` (geopolitical, 14-period duration + 1 delay + 39 recovery =
54-period active window) are disrupted for 54 of 104 weeks each — over
half the entire simulation. This single event alone generates the large
majority of "ongoing" positive examples in later splits.

**Implication**: because so few underlying events exist, and because their
active windows are long relative to the horizon, the temporal split's
test set can end up with **zero fresh-onset examples purely by chance of
which weeks land in the test tail** — this happened here, and would
happen differently (possibly not at all) with a different temporal split
boundary or a different dataset seed. This is why the ablation sweep in
§1 needed the severity split specifically to test the fresh-onset
question at all, and why any future robustness claim about fresh-onset
performance should be checked against `disruption_onset_breakdown`'s
per-split counts before being trusted, not assumed present just because
the overall split has positives.

**Recommendation**: report fresh-onset population size explicitly in
every future experiment's headline metrics (not buried in a JSON file) —
a metric computed over zero examples is not evidence of anything, and this
benchmark's small underlying event count makes that an easy trap to fall
into by accident (as happened with this session's own first ablation
sweep, run against a split whose test set had zero fresh onsets).

---

## 10. Event-Generation Analysis

**Observed**, from `events/base.py::event_intensity`:
```python
def event_intensity(event, t):
    if t < event.start_time:
        return 0.0
    ...
```
Effect is **exactly** `0.0` (not "small," not "ramping" — the literal
float value zero) for every period strictly before `start_time`.
`capacity_multiplier`/`lead_time_multiplier` are therefore exactly `1.0`
(no effect whatsoever) until the instant the event starts. This was
checked against **all five** event-type modules
(`supplier_failure.py`, `natural_disaster.py`, `geopolitical.py`,
`cyberattack.py`, `logistics.py`) — none of them override or add to this
shared intensity function; they only set `start_time`, `duration`,
`capacity_reduction_fraction`, and `lead_time_increase_fraction`
differently per type. **No event type has a ramp-up or precursor phase.**

**Observed**, from `events/generate.py::maybe_generate_events`: whether an
event of a given type starts in period `t` is drawn as
`rng.random() < probability` — a fixed, per-type, per-period Bernoulli
draw, entirely independent of the current simulation state (inventory,
backlog, demand, or any other operational variable). Severity is drawn
independently from a fixed categorical distribution
(`severity_weights`). **Neither the occurrence nor the severity of a
future event depends on anything in the simulated history.**

**Conclusion, stated precisely**: for any period `t-4, t-3, t-2, t-1`
before an event's `start_time = t`, the simulator's own generation process
guarantees these periods contain **zero** information about the event
about to start — not "a weak signal a model might miss," but a
provably-absent one, in both the intensity profile and the occurrence
process. This is the root cause of §1/§9's "no pre-event signal" finding;
it is a property of `events/generate.py` and `events/base.py`, not of any
one feature set tried against the data. **No feature engineering on top of
this benchmark's operations tables can recover a signal that was never
written into them by the generator.** The only theoretically available
channel is a supplier's/region's *static* attributes correlating with
being targeted in a way that would generalize beyond one dataset draw —
which is exactly what §14 investigates and finds is likely NOT what is
currently happening (the correlation observed is dataset-draw-specific,
not risk-magnitude-specific).

**Recommendation (not to be acted on without user sign-off — this is
explicitly out of scope per this document's own §27/§32 boundary, listed
here only because it directly explains the negative Q2 result)**: if a
genuine early-warning research question is a hard requirement, the only
way to get it is to change the event generator to include a real,
graded pre-onset phase (e.g., `event_intensity` ramping from a small
nonzero value starting several periods before `start_time`, or an
occurrence probability that depends on a state variable). This is
generator work, is explicitly prohibited by this phase's scope, and
should not be done "to make GraphSAGE perform better" — only if the
research question itself is revised to require it, as a deliberate,
documented decision.

---

## 11. Graph Structure Analysis

**Measurement**, from `graph/edges.csv`:

| Relation | Count | Notes |
|---|---:|---|
| supplier_material | 276 | 32/100 materials single-sourced; distribution over supplier-count-per-material: {1: 32, 2: 15, 3: 18, 4: 15, 5: 20} |
| material_plant | 649 | materials-per-plant (BOM size): mean 6.49, std 6.40, min 1, 25%ile 1, median 4, 75%ile 9.25, max 28 |
| plant_product | 200 | products-per-plant: mean 4.35, max 11; **every product is single-plant-sourced (100/100 products, exactly 1 plant each)** — no redundancy at the product level anywhere in this benchmark |
| supplier_region | 300 | suppliers-per-region: mean 15, std 3.76, min 8, max 21 — fairly even |
| supplier_procurement, procurement_material, procurement_plant | 2,000 each | every procurement node has exactly one material edge and one plant edge (checked directly — 0 procurement nodes have >1 material edge) |

**Measurement — a structurally important fact not previously documented
in any prior summary**: **147 of 300 suppliers (49%) have zero direct
`SUPPLIER_MATERIAL` edges.** These suppliers connect to the graph
*exclusively* through the 2-hop `SUPPLIER_PROCUREMENT -> PROCUREMENT_MATERIAL`
/ `PROCUREMENT_PLANT` path via whichever procurement (contract) nodes they
happen to be tied to. For these suppliers, the direct supplier->material
topological relationship the plan's conceptual diagrams describe does not
exist in the actual generated graph; their only route to
material/plant-level context is through procurement nodes' own (static,
Phase-2-generated) attributes. This matters directly for §14's
supplier-279/supplier-0 contrast noted in the prior phase's work summary:
supplier_0 (used as an "already-ongoing" example) has zero direct
procurement.csv order records AND zero direct SUPPLIER_MATERIAL edges in
this dataset — its only signal pathway to the model is through whatever
procurement nodes and their neighbors it does connect to.

**Measurement**: 0 isolated nodes of any type. Every node participates in
at least one edge.

**Implication for structural features (§14)**: "downstream plant count"/
"downstream product count" computed via `SUPPLIER_MATERIAL -> MATERIAL_PLANT
-> PLANT_PRODUCT` graph traversal would be **zero or undefined for the 147
suppliers with no direct SUPPLIER_MATERIAL edge**, unless the traversal is
redefined to go through `SUPPLIER_PROCUREMENT -> PROCUREMENT_MATERIAL`
instead (which reaches a materials/plants for essentially every supplier,
since procurement participation is comprehensive at 2,000/2,000 edges).
Any structural feature proposal must specify which path it traverses and
verify coverage before being implemented, not assume the plan's conceptual
`Supplier -> Material -> Plant -> Product` diagram matches the literal
edge set (see also `CLASSICAL_GRAPHSAGE_IMPLEMENTATION_PLAN.md`'s own note
that this exact assumption should not be taken for granted).

---

## 12. Current Experimental Results

**Measurement**, primary frozen run (`experiments/classical_gnn/20260905T041715Z_hetero_graphsage_seed42`,
temporal split, full features, seed 42): test PR-AUC 0.7874, ROC-AUC
0.9872, F1 0.7236, precision 0.5796, recall 0.9628, Brier 0.0333, ECE
0.0441, confusion matrix TP=233/FP=169/FN=9/TN=3189. Best epoch 11 of 21
(early-stopped, patience 10). Training duration 50.2s (CPU). `pos_weight`
= 37.67 (computed from train-split targets only, as designed).

**Measurement — calibration curve** (10 bins, test split): predicted
probability bins near 0 are well-calibrated (bin mean pred 0.0007, observed
0.0, n=3085 — the large majority of examples). Mid-range bins are **poorly
and non-monotonically calibrated**: bin pred=0.54 (n=64) has observed
frequency 0.031, *lower* than bin pred=0.35's observed 0.036 and bin
pred=0.24's observed 0.118 — the model's mid-confidence predictions do not
track observed frequency in a consistent order. High bins trend upward
(pred=0.77 -> obs=0.89, pred=0.83 -> obs=0.93) then dip at the very top
(pred=0.99 -> obs=0.67, n=163) — some overconfidence at the extreme end.

**Measurement — training/validation loss divergence** (same run,
`training_history.csv`): `train_loss` falls monotonically 0.972 -> 0.121
over 21 epochs; `validation_loss` rises 2.315 -> 8.590 over the same
window (a ~3.7x increase) despite `validation_pr_auc` still fluctuating
usefully (peaking at epoch 11, 0.875). **This is a textbook overfitting
signature on the loss scale specifically** — the model becomes
increasingly overconfident (large BCE magnitude on validation) even while
its *ranking* quality doesn't collapse in the same way, which is exactly
consistent with the calibration curve's mid-range noise and top-end
overconfidence above. Early stopping on PR-AUC (not loss) protects the
*ranking* metric from this but does nothing for probability quality.

**Measurement — 5-seed variance** (from `GRAPHSAGE_WORK_SUMMARY.md`,
re-verified against the underlying per-seed `metrics.json` files this
session): test PR-AUC 0.807 ± 0.066 (range 0.689-0.874 across seeds 42-46)
— a coefficient of variation of ~8%, driven by training instability (see
above), not by data variance (same train/val/test data every seed).

**Measurement — ablation sweep results**: full tables reproduced in §14
and in `GRAPHSAGE_IMPROVEMENT_WORK_SUMMARY.md` §4. Headline: severity-split
fresh-onset recall = 0.0000 and ROC-AUC < 0.5 for all three tested feature
groups (dynamic-only, static-only, region-risk-only).

**Observed**: `static_plus_graph` (already retired per the prior phase) is
excluded from the tables above — it is confirmed structurally degenerate
(0/300 suppliers show any prediction variance across time), not a
legitimate comparison point.

---

## 13. Current Model Weaknesses

Ranked by scientific importance x expected impact, per §16's requested
categories. Each entry: Observed/Evidence -> Implication -> Recommendation
(pointer to the relevant experiment in §20/§21).

### 13.1 Generalization / identity memorization (Data representation + Generalization) — **HIGHEST PRIORITY**
**Observed**: 4 of 20 regions (region_1, region_3, region_8, region_19)
contain 44 of 46 ever-disrupted suppliers; these are precisely the regions
`event_0`, `event_2`, `event_3`, `event_5` targeted. **Evidence**: these 4
regions' mean risk-score percentile across all 20 regions is 0.46, 0.74,
0.43, 0.76 — average to moderately-above-average, not concentrated at the
top of the risk distribution (two sit *below* the median). **Implication**:
a model that performs well by learning "these 4 regions/46 suppliers are
risky" may be memorizing this dataset's one random draw of who got hit,
not learning a risk-magnitude relationship that would hold under a
different draw. This directly undermines trusting the `static_only`/
`region_risk_only` ablation results as evidence of "structural signal"
without further checking. **Recommendation**: Experiment D1 (§20).

### 13.2 Fresh-onset prediction (Temporal modeling + Data representation) — answered, not a gap to fix
**Observed/Evidence**: §10's generator analysis + §1's ablation results.
**Implication**: no further feature engineering on the current input
channels will recover fresh-onset signal; this is a property of the
benchmark, confirmed three independent ways (event intensity function,
event occurrence process, and empirical ablation across three feature
groups). **Recommendation**: do not spend further effort chasing this
without first revising the research question (out of scope here) — see §19.

### 13.3 Overfitting on the loss/calibration scale (Training + Calibration)
**Observed/Evidence**: §12's training-history and calibration-curve
measurements. **Implication**: the model's *ranking* quality (PR-AUC/
ROC-AUC) is reasonably stable across seeds, but its *probability quality*
degrades over training in a way early stopping (on PR-AUC) does not catch.
Anyone using the raw probabilities for risk-ranking purposes (plan's own
`supplier_risk_ranking.csv` deliverable) is currently trusting numbers with
demonstrated mid-range miscalibration. **Recommendation**: Experiments O5,
O6 (§21).

### 13.4 Rolling-feature trend definition (Feature engineering)
**Observed**: `trend_w` is an endpoint slope, not a regression slope; noisy
by construction. **Implication**: plausible (not yet measured) source of
some dynamic-feature noise contributing to dynamic-only's weaker
performance. **Recommendation**: Experiment O1 (§21).

### 13.5 Single aggregation scheme, no attention (Message passing + Architecture)
**Observed**: mean aggregation only, both within-relation (SAGEConv) and
across-relation (HeteroConv) — deliberate per the original plan
("clean classical baseline," no attention). **Implication**: mean
aggregation treats every neighbor identically regardless of how
informative it is; with 49% of suppliers reaching materials/plants only
via procurement nodes (§11), some neighbors are structurally more load-
bearing than others. **Recommendation**: Experiment O3 (§21) — but LOW
priority, since §13.1/§13.2 gate whether more architecture investment on
persistence-detection is even worth it.

### 13.6 Reproducibility (already strong — noted for completeness, not a weakness)
**Observed**: seed-controlled, empirically verified bit-identical across
independent reruns including a different user's machine. No action needed.

### 13.7 Interpretability (Interpretability) — not yet attempted
**Observed**: no permutation importance, gradient attribution, or
graph-explanation method has been run. **Implication**: the region/
supplier-identity hypothesis in §13.1 is currently supported by a targeted
manual correlation check (§14), not by a systematic attribution method.
**Recommendation**: low priority relative to §13.1's direct experiment,
which is cheaper and more decisive; revisit only if D1 (§20) confirms
genuine (not memorized) structural signal and a paper-quality explanation
of *which* structural features matter becomes necessary.

---

## 14. Structural Feature Opportunities

**The central finding of this investigation, in full.**

**Observed**: `modeling/features.py` deliberately excludes `region_id` and
other identifier columns from the SUPPLIER node's own feature vector,
reasoning (per its own docstring) that "that relationship already reaches
the model structurally through the graph's own edges."

**Measurement**: this reasoning is correct that the information reaches
the model, but does not prevent memorization. The `SUPPLIER_REGION` edge
plus the region node's own static feature vector (`geopolitical_risk`,
`natural_disaster_risk`, `infrastructure_risk`, `trade_risk`, `cyber_risk`,
`transport_reliability`) are exactly as sufficient for a GraphSAGE layer
to learn "which region is this" as a raw `region_id` one-hot would have
been — a 20-region, 6-real-valued-feature space is easily separable by a
128-hidden-dim network. And the previous section already showed *which*
regions are memorable in this one dataset draw are not the highest-risk
ones by the region's own static scores.

**Hypothesis** (explicitly not yet confirmed — this is the load-bearing
open question for the rest of this document's roadmap): the `static_only`
and `region_risk_only` ablations' persistence-detection success (§1's
temporal-split table: 0.818 and 0.633 PR-AUC respectively, both above
`dynamic_only`'s 0.583) is substantially driven by the model learning
"suppliers in regions {1, 3, 8, 19} / with static attribute values
correlated with membership in those regions" rather than a risk-magnitude
relationship. Both explanations are consistent with everything measured so
far; they are **not yet distinguished**, because both would produce the
same PR-AUC on this one dataset.

**The distinguishing experiment (Experiment D1, detailed in §20)**: rerun
`static_only` (and ideally `region_risk_only`) against `scm_v1_black_swan_seed44`
— a different, already-generated dataset with a different random seed,
hence a different random draw of which regions/suppliers get hit (per
`benchmark_manifest.json`: 9 events, 6.1% overall positive rate, vs. seed43's
6 events/3.3%). Under the memorization hypothesis, persistence-detection
PR-AUC on `seed44` using `seed43`-derived intuitions should be no better
than a model has any right to expect from chance-level region matching
(since seed44's specific hit regions will almost certainly differ from
seed43's); under the genuine-signal hypothesis, if the underlying static
risk features that correlate with being targeted are consistent in
*direction* across independently-generated datasets (not guaranteed, since
`maybe_generate_events` per §10 targets regions/suppliers uniformly at
random regardless of their risk scores — there is no reason to expect this
direction to hold), performance should degrade less severely. **Given
§10's confirmation that event targeting is uniform-random and independent
of risk scores, the a priori expectation is that this experiment will
support the memorization hypothesis** — but it has not been run, and this
document does not assert a result it does not have.

### Proposed structural features, ranked

| Feature | Computable from | Time | Leakage risk | Duplicates existing? | Static/Temporal | Rank |
|---|---|---|---|---|---|---|
| Region-identity indicator (implicit, already reachable) | graph/nodes.csv + SUPPLIER_REGION edge | generation-time | None (already reaches the model) | N/A — already present via message passing | Static | **N/A — this is the mechanism already responsible for §14's finding, not a new feature to add** |
| Single-source material indicator (supplier serves a material with `concentration >= 0.5`, i.e. <=2 suppliers) | graph/nodes.csv (Material.concentration, already computed at generation time) + SUPPLIER_MATERIAL edges | generation-time | None | Partially — `Material.concentration` already exists on the material node and reaches the supplier via message passing; a supplier-side indicator would make it explicit/direct rather than requiring the network to learn to use it | Static | MEDIUM |
| Downstream plant/product count (2-3 hop traversal from supplier) | SUPPLIER_MATERIAL -> MATERIAL_PLANT -> PLANT_PRODUCT, or SUPPLIER_PROCUREMENT -> PROCUREMENT_PLANT for the 147 suppliers with no direct SUPPLIER_MATERIAL edge (§11) | generation-time | None (pure topology) | No | Static | MEDIUM — must handle the two-path split explicitly or it will silently undercount for half the suppliers |
| Two-hop dependency count via procurement path | SUPPLIER_PROCUREMENT -> PROCUREMENT_MATERIAL / PROCUREMENT_PLANT | generation-time | None | No | Static | MEDIUM — this is the path that actually covers all suppliers (§11) |
| Regional supplier concentration (how many suppliers share this supplier's region) | SUPPLIER_REGION edges, already-known counts (mean 15, std 3.76) | generation-time | None | No, but low variance across regions (8-21) limits how discriminative it can be | Static | LOW |
| **A dataset-seed-44 rerun of the existing static/region-risk ablations (not a new feature — a new experiment)** | scm_v1_black_swan_seed44 (already generated) | n/a | None | n/a | n/a | **HIGH VALUE — do this before any of the above** |
| Raw `region_id` as an explicit one-hot supplier feature | graph/nodes.csv | generation-time | None | Fully duplicates what message passing already provides (per this section's finding) | Static | **DO NOT USE** — would not add information, and would make the memorization risk more visible/tempting to rely on rather than fixing it |

---

## 15. Temporal Feature Opportunities

**Observed**: current rolling features are `{mean, std, trend}` at windows
`{4, 8, 12}`, uniform across all dynamic entity types. `trend` is an
endpoint slope (§8).

### O1 — OLS trend vs. endpoint-slope trend
**Specific proposal**: for each dynamic value column, add (or replace)
`trend_w` with the slope of an ordinary-least-squares fit over the `w`
most recent periods (`numpy.polyfit(range(w), values, 1)[0]`), computed
identically causally (only periods `<= t`). Compare against the current
endpoint-slope `trend_w` using validation PR-AUC on `dynamic_only` mode
specifically (since that mode isolates the effect — static features would
otherwise dominate the comparison and hide any real difference). Fit
train-only where any normalization is involved (it already is, via
`preprocessing.py`'s existing train-only scaling — no leakage risk from
this change).

### O2 — Deviation-from-normal features
**Specific proposal**: add `{value_t - mean_w} / std_w` ("z-score vs. own
recent history") for `fulfillment_ratio` and `inventory` specifically
(the two series most directly tied to the supplier_279 example that showed
a clear, legitimate visible-distress signal in the prior phase's
investigation) at window 8. Test whether this recovers any of the
sub-chance fresh-onset ranking from §1 — **expected result, per §10's
generator analysis: no**, since the underlying series contain no
pre-onset information regardless of how it's transformed; this experiment
is included for completeness/rigor (to rule out "the transformation was
the problem" as an explanation) rather than because it is expected to
succeed.

### O3 — Per-entity-type window tuning
**Specific proposal**: currently all entity types share windows `[4, 8,
12]`. Test independently varying material's window set (inventory turns
over on a different natural timescale than a multi-week product backlog)
to `[2, 4, 8]` vs. product's `[8, 12, 20]`, holding supplier/plant windows
fixed, using validation PR-AUC to select — only after O1 is resolved,
since window choice and trend definition interact.

**All three are OPTIMIZATION experiments (§18), not diagnostic** — they
should only be run after §14's D1 experiment resolves whether the
persistence-detection task is worth optimizing further at all, and only
ever selected via validation, never test (per the existing, already-
enforced `select_threshold`/preprocessing train-only discipline).

---

## 16. Architecture Opportunities

Per §19.C, considered but explicitly gated on §14's D1 result:

- **2 vs. 3 layers**: 3 layers would let a supplier's embedding reach
  2-hop-further neighbors (e.g., reach a plant's *products*, not just the
  plant itself) — with the caveat that GraphSAGE's own literature
  documents oversmoothing risk past ~2-3 layers on graphs this dense
  (mean supplier-region degree 15, mean plant BOM size 6.5). Test 2 vs. 3
  with everything else fixed (hidden_dim=128, dropout=0.20, same
  optimizer/seeds), holding out the possibility that 3 layers primarily
  amplifies the identity-memorization channel (§14) rather than adding
  genuine signal, given how few underlying distinct entities are involved
  (46 disrupted suppliers, 6 events).
- **Hidden dimension 64/128/256**: current parameter count is 1.22M at
  128; 256 would roughly quadruple it against the same ~26,700 training
  examples (18,600 for temporal-split train) — a real overfitting risk
  given §12's already-observed loss divergence. Test only with weight
  decay tuned jointly (§17), not in isolation.
- **Residual/skip connections**: would help gradient flow if layers are
  increased to 3+; not obviously useful at the current 2-layer depth.
  Only test alongside the 3-layer variant above, not standalone.
- **Attention (GAT-style) instead of mean aggregation**: would let the
  model weight the 147-suppliers'-worth of procurement-only neighbors
  differently from the direct-SUPPLIER_MATERIAL suppliers (§11) — a
  structurally motivated reason to try it, not a generic "attention is
  usually better" argument. Genuinely useful only if §14's D1 experiment
  confirms real (not memorized) structural signal exists to weight.
- **Edge-aware message passing**: the benchmark's `graph/edge_features.csv`
  is empty for every edge in this dataset (checked: `export_graph`'s own
  docstring notes it is written but unused when no edge carries attributes)
  — there is currently nothing to make message passing edge-aware
  *with*. Not actionable without first adding edge attributes at the
  dataset-generation layer, which is out of scope for this phase.

**All architecture experiments should be run only after §14's D1**, since
an architecture change cannot distinguish memorization from genuine
signal — it can only make whichever one is present easier or harder to
fit.

---

## 17. Training Opportunities

- **Focal loss vs. weighted BCE**: current `pos_weight` (37.67 on the
  primary run) already strongly upweights the rare positive class. Focal
  loss would additionally downweight *easy* examples (already-confident
  correct predictions) — plausibly relevant given §12's calibration
  finding (overconfidence at the top bin), but adds a hyperparameter
  (focal `gamma`) that itself needs validation-only tuning. Test only
  after confirming (via O5/O6 below) whether the calibration issue is a
  weighting problem or a training-duration problem.
- **Weight decay**: currently 0.0001, unchanged across all experiments so
  far. Given §12's clear overfitting-on-loss-scale evidence, a sweep of
  `{0.0001, 0.001, 0.01}` on validation PR-AUC *and* validation Brier
  score (not PR-AUC alone, since the problem is specifically probability
  quality, not ranking) is directly motivated by measured evidence, not
  a generic hyperparameter-sweep instinct.
- **Learning rate**: current 0.001, Adam. No evidence yet either way;
  lower priority than weight decay given the specific overfitting
  signature observed.
- **Early stopping metric**: currently validation PR-AUC only. Given
  §12's loss/calibration divergence, consider **also** tracking validation
  Brier score and reporting both the PR-AUC-best and Brier-best checkpoints
  in a future experiment's artifacts (not necessarily changing which one
  is selected by default — that would need its own justification per
  §22's model-selection protocol) so their difference can be quantified.

---

## 18. Calibration/Threshold Opportunities

**Observed**: calibration is currently *measured* (Brier, ECE, 10-bin
curve) but never *corrected*. `metrics.select_threshold` supports
`f1_optimal`/`recall_constrained`/`precision_constrained`, computed from
validation only — this machinery already exists and is already leakage-
safe; it is simply not exercised by default (`configs/graphsage.yaml`
uses `policy: fixed, value: 0.5`).

**Specific proposal**: fit a validation-only Platt scaling (logistic
regression on validation logits vs. validation labels) or isotonic
regression (`sklearn.isotonic.IsotonicRegression`, fit on validation
predictions/labels) as a post-hoc correction, applied to test predictions
only after being fit exclusively on validation — directly analogous to
the existing threshold-selection discipline, and directly motivated by
§12's measured mid-range calibration noise. Report Brier/ECE before and
after on the SAME test predictions to quantify the improvement, rather
than assuming isotonic regression helps.

**Specific proposal**: run the existing `f1_optimal` threshold policy
(already implemented, not yet used in the frozen results) on the primary
run's validation split and report the resulting test precision/recall/F1
alongside the current fixed-0.5 numbers — this requires zero new code,
only a config change and a rerun, and would directly show whether the
current 0.58 precision / 0.96 recall operating point is an artifact of
the arbitrary 0.5 default or reflects a genuine precision-recall tradeoff
at the model's actual best operating point.

---

## 19. Generalization Strategy

**Observed status**: temporal-split and severity-split generalization
experiments are both done (prior phases). Scenario-split (held-out
`{cyberattack, geopolitical}`) has not been run. Dataset-seed robustness
(`scm_v1_black_swan_seed44`) has not been run for either the primary model
or any ablation.

**Recommendation, in priority order**:
1. **§14's D1 experiment (dataset-seed robustness on the static/region-risk
   ablations) is now the single highest-value generalization experiment**,
   because it directly tests the memorization-vs-signal question that
   currently has no other evidence resolving it, and it requires no new
   dataset generation (`scm_v1_black_swan_seed44` already exists).
2. Scenario-split rerun of the full model (existing capability,
   `config.split.strategy: scenario`, never executed) — lower priority
   than (1) because the scenario split's held-out event types
   (cyberattack, geopolitical) contribute only 1 and 10 of the 46 raw
   onsets respectively (§9); a scenario-holdout result would be
   statistically thin (small held-out event population) compared to what
   D1 can show.
3. Only after (1) and (2): consider whether the *persistence-detection*
   task (not fresh-onset, already answered) generalizes across dataset
   seeds — i.e., rerun the *full* model (not just ablations) on `seed44`
   and compare test PR-AUC distributions.

**Explicit non-goal**: do not attempt to "fix" generalization by
engineering features that specifically target `seed43`'s known hit
regions — that would be curve-fitting to the one dataset this document
was written against, precisely the failure mode §14 is trying to detect
and avoid propagating further.

---

## 20. Diagnostic Experiment Matrix

Diagnostics answer "what does the data/model actually contain," not "how
do we make the number bigger." None of these should proceed to
optimization-experiment status without their result being interpreted
first.

### D1 — Dataset-seed robustness check on static/region-risk signal
**Research question**: is the `static_only`/`region_risk_only` persistence-
detection advantage over `dynamic_only` a genuine risk-magnitude signal,
or memorization of which specific regions this one dataset draw targeted?
**Hypothesis**: performance advantage will substantially shrink or vanish
on `scm_v1_black_swan_seed44`, since §10 confirms event targeting is
uniform-random and independent of any risk score.
**Input features**: `static_only` and `region_risk_only` feature modes
(already implemented, `config.features.feature_mode`).
**Excluded features**: dynamic operational history (already excluded by
these modes).
**Graph structure**: unchanged (full heterogeneous graph, both modes only
restrict the supplier's own feature columns per the existing, documented
`apply_feature_mode` semantics).
**Model architecture**: unchanged (2-layer HeteroGraphSAGE, hidden_dim=128,
exactly as frozen).
**Training configuration**: unchanged (Adam, lr=0.001, weight_decay=0.0001,
pos_weight from seed44's own train split, early stopping patience=10).
**Split**: temporal split of `scm_v1_black_swan_seed44` (need to confirm
this dataset has a `temporal_split.csv` — per `benchmark_manifest.json`
it does; severity/scenario splits for seed44 have not been confirmed to
exist and should be checked before assuming they can be reused).
**Seeds**: 42, 43, 44, 45, 46 (model seeds, distinct from the dataset seed).
**Metrics**: overall PR-AUC/ROC-AUC, plus `disruption_onset_breakdown`'s
already-ongoing recall/PR-AUC/ROC-AUC (fresh-onset metrics too, if that
split's test set has any — check via `onset_breakdown.json`'s `n_fresh_onset`
before interpreting, per §9's own lesson).
**Expected interpretation**: a large drop in `static_only`'s advantage
over `dynamic_only` (not just a drop in absolute PR-AUC, which is expected
given seed44's different event mix — the *relative ordering and gap*
between modes is the signal) supports memorization; a preserved gap of
similar relative magnitude would be a genuine, surprising, and important
counter-finding worth investigating further.
**Success criterion**: a clear, interpretable answer either way — this is
a diagnostic, not something with a "good" or "bad" outcome.
**Failure criterion**: `scm_v1_black_swan_seed44` lacks a usable severity/
scenario split or its temporal split's test window has degenerate class
balance (e.g., near-zero positives) making the comparison statistically
meaningless — check this before running the full sweep, not after.

### D2 — Two-hop dependency count via the procurement path (not the SUPPLIER_MATERIAL path)
**Research question**: does an explicit "how many plants/products does
this supplier structurally reach" feature, computed via the path that
actually covers all 300 suppliers (§11's SUPPLIER_PROCUREMENT path, not
the 51%-coverage SUPPLIER_MATERIAL path), add signal beyond what message
passing already implicitly encodes?
**Hypothesis**: small or no improvement, since a 2-layer GraphSAGE should
already be able to approximate a 2-hop count implicitly — this experiment
is valuable specifically because a null result would be informative (it
would show the network is already capturing this without hand-engineering).
**Input features**: full feature set + one new static supplier feature
(downstream plant count, downstream product count via the procurement
path). **Excluded features**: none beyond the existing frozen exclusions.
**Graph structure**: unchanged. **Model/training/split/seeds**: identical
to the frozen primary configuration. **Metrics**: overall + onset-breakdown,
compared against the existing frozen `full` result on the *same* split/seeds
(no re-randomization). **Success criterion**: PR-AUC improvement
exceeding the existing 5-seed standard deviation (0.066 on the temporal
split) — anything smaller is not distinguishable from seed noise.
**Failure criterion**: improvement within noise, or a regression — either
way, informative, not a failure of the investigation.

### D3 — Onset-cohort attribution (which of the 6 events drives which result)
**Research question**: are the ablation sweep's results dominated by one
event (e.g., `event_2`'s 54-week-long disruption of 10 suppliers), such
that the "5-seed mean" is really closer to a 1-scenario measurement with
5 replicates of model-fitting noise, not 5 independent looks at the
underlying phenomenon?
**Method**: recompute `disruption_onset_breakdown`-style metrics stratified
by which event generated each positive example (already partially
supported — `onset_breakdown.json`'s per-example already-ongoing/fresh
split can be joined against §9's event-attribution mapping). No new
training required — pure re-analysis of already-saved `predictions.csv`
files from existing runs.
**Success criterion**: understanding whether the reported ± standard
deviations across model seeds actually reflect 5 independent
measurements or are inflated/deflated by one dominant event's contribution.

---

## 21. Optimization Experiment Matrix

Optimization experiments assume D1-D3 have been interpreted and the
persistence-detection task has been confirmed worth optimizing further.
**Do not run these to try to move the fresh-onset numbers — §1/§9/§10
already show that is not possible with the current inputs.**

### O1 — Trend feature definition
See §15. Compare endpoint-slope vs. OLS-slope `trend_w`, `dynamic_only`
mode, validation PR-AUC selection, same split/seeds as frozen.

### O2 — Deviation-from-normal (z-score) dynamic features
See §15. Included for completeness/rigor given §10's expectation of a
null result on fresh-onset specifically.

### O3 — Layer depth (2 vs. 3)
See §16. Gate on D1's outcome first.

### O4 — Hidden dimension x weight decay joint sweep
`{64, 128, 256} x {0.0001, 0.001, 0.01}`, validation PR-AUC AND validation
Brier score both reported (not PR-AUC alone — see O5/O6's calibration
motivation), same split/seeds otherwise. 9 configurations x 5 seeds = 45
runs; each run is ~50s on CPU per the measured primary-run duration, so
~37.5 minutes total — computationally trivial, no need to subsample.

### O5 — Weight decay targeted at the observed overfitting signature
A narrower, faster version of O4 focused specifically on validation loss/
Brier trajectory (not just PR-AUC), directly motivated by §12's measured
train/validation loss divergence.

### O6 — Post-hoc calibration correction
See §18. Platt scaling and isotonic regression, both fit on validation
only, evaluated once on the frozen test predictions (does not require
retraining — pure post-processing of already-saved `predictions.csv`).

---

## 22. Model-Selection Protocol

**Explicit rule** (extending, not replacing, the already-implemented
train-only-preprocessing / validation-only-threshold discipline):

```text
1. PRIMARY selection criterion during training: validation PR-AUC
   (already implemented, unchanged).
2. Among checkpoints/configurations within one standard deviation of the
   best validation PR-AUC (estimated from the 5-seed spread already
   measured, e.g. ±0.066 on the temporal split): break ties using
   validation Brier score (calibration quality), not test performance of
   any kind.
3. Robustness check BEFORE final selection, not after: any configuration
   under consideration must have already been run through D1 (dataset-seed
   check) if it involves static/region-risk features specifically, per
   this document's central finding. A configuration that only "works" on
   seed43 is not eligible to be called an improvement.
4. FINAL test evaluation: exactly once, after 1-3 are settled, on the
   frozen test split. If test results are inspected and any change is
   made afterward, that is a new experimental iteration and must be
   documented as such (per the original plan's own §63 rule, unchanged
   here).
5. Multi-seed reporting: mean +/- std over the same 5 seeds (42-46) used
   throughout this phase, never a single best-seed number.
```

**What "better" explicitly does NOT mean**: a higher single-seed test
PR-AUC, a result that has not been checked against D1 if it depends on
static/region-risk features, or any result obtained by looking at test
performance before finalizing the configuration.

---

## 23. Computational Considerations

**Measurement**: 2,670 nodes / 15,350 edges (including the 18-relation
reverse-edge doubling) per graph snapshot; 89 distinct snapshots needed
per full sweep over the primary dataset's usable prediction times; 1.22M
trainable parameters at the frozen hidden_dim=128/2-layer configuration.
Training one seed to completion (with early stopping) took 46-50 seconds
on CPU in every measured run — no GPU was used or is currently required.

**Observed**: all graph construction uses PyTorch Geometric's sparse
`edge_index` representation throughout (`hetero_graph.py`); no dense N×N
adjacency matrix is ever constructed, confirmed by direct code inspection
(`build_edge_index_dict` builds a `[2, E]` tensor per relation, never a
dense matrix).

**Recommendation for scaling**: the current per-snapshot cost is
dominated by the number of distinct prediction times (89), not by graph
size — each snapshot's forward pass is cheap given the graph's small
absolute size (2,670 nodes). If a future dataset scales node/edge counts
by 10x while keeping a similar number of periods, training cost would
scale roughly linearly with node/edge count per the sparse GraphSAGE
implementation (no quadratic terms present) — the current architecture
should remain practical. The O4 sweep's 45-run, ~37-minute total cost
estimate (§21) confirms hyperparameter search at this scale is not a
computational bottleneck; iteration speed is not a reason to skip any of
the diagnostic experiments in §20.

---

## 24. Research/Paper Implications

**Suggested structure**, adapted from the request's template to reflect
what has actually been established:

```text
Problem: supplier disruption prediction on a synthetic SCM benchmark
    |
    v
Heterogeneous graph representation (6 node types, 9 edge types + PyG-only
reverse relations)
    |
    v
Classical GraphSAGE baseline (frozen, reproducible, leakage-audited)
    |
    v
Diagnostic finding #1: persistence detection is real and strong;
fresh-onset prediction is provably unattainable from this benchmark's
inputs (three independent lines of evidence: generator design, occurrence
process, empirical ablation)
    |
    v
Diagnostic finding #2 (this document): the apparent value of static/
structural features is confounded with dataset-draw-specific region
memorization; disambiguating experiment (D1) specified but not yet run
    |
    v
[Gate: D1's result determines whether Phase 3+ (structural/temporal/
architecture optimization) is worth pursuing for a paper claim, or
whether the paper's contribution is instead the diagnostic methodology
itself]
    |
    v
Final frozen classical GraphSAGE (pending D1 + selected optimizations)
    |
    v
Generalization/robustness (temporal + severity done; scenario +
dataset-seed pending)
    |
    v
Later QGNN comparison (§27's protocol)
```

**What is novel**: the systematic separation of persistence-detection from
fresh-onset prediction via `disruption_onset_breakdown`, and — pending D1 —
potentially the region-memorization diagnostic methodology itself
(checking whether an apparent structural-risk finding survives a
dataset-seed swap is a generally applicable check for any synthetic
benchmark with a small number of underlying rare events, not specific to
this one).

**What is baseline**: GraphSAGE itself, the heterogeneous graph
construction, the training/evaluation protocol — all standard, correctly
applied, not claimed as novel.

**What is a methodological contribution**: the `temporal_variation_summary`
check (catches a model that has silently degenerated to a time-invariant
score) and the onset-breakdown ranking-quality split are both generically
reusable diagnostics for any temporal graph-learning benchmark with rare,
bursty positive classes — worth stating explicitly as a contribution if a
paper is written, since they are not specific to supply chains.

**Limitations that must be acknowledged**: only 6 underlying black-swan
events exist in the primary dataset (§9) — every reported statistic is a
function of a small number of independent scenarios, however many
supplier-week examples or model seeds are used to measure it. This is not
fixable by more careful evaluation; it is a property of the benchmark's
scale, and any claim of statistical significance should account for this
effective sample size (closer to "6 events" than "1,039 positive
supplier-weeks") rather than treating supplier-week rows as independent
observations.

---

## 25. Final Recommended Roadmap

Reordered from the prior plan's generic sequence to reflect this
document's actual evidence — specifically, moving the dataset-seed
disambiguation (D1) before any further structural/temporal/architecture
investment, since it gates whether that investment has a sound premise.

```text
Phase 1 (already done)
Persistence vs. fresh-onset diagnostic framework + feature-mode ablation
infrastructure
        |
        v
Phase 2 (already done)
Feature-information ablation sweep (temporal split, then corrected to
severity split) -- definitive negative answer on fresh-onset signal
        |
        v
Phase 3 (NEW, highest priority, not in the prior plan) -- Experiment D1
Dataset-seed robustness check on the static/region-risk finding
        |
        +---- Confirms memorization?
        |             |
        |             +---- YES -> do not build further structural
        |             |            features chasing this signal; static-
        |             |            feature value is dataset-draw-specific,
        |             |            not a generalizable finding. Proceed
        |             |            directly to Phase 6 (persistence-only
        |             |            optimization), skipping Phase 4.
        |             |
        |             +---- NO (signal survives) -> proceed to Phase 4,
        |                    now with a confirmed premise worth building on
        v
Phase 4 (conditional on D1) -- Experiments D2, D3
Structural dependency features (via the procurement path, which covers
all suppliers) + onset-cohort attribution analysis
        |
        v
Phase 5 -- Experiments O1, O2
Temporal representation improvements (trend definition, deviation
features) -- diagnostic value regardless of D1's outcome, since these
also inform whether dynamic-only's weaker persistence-detection result
is a feature-engineering artifact or a genuine information gap
        |
        v
Phase 6 -- Experiments O3, O4, O5
Architecture and training tuning for the persistence-detection task
specifically (never fresh-onset, per Phase 2's finding)
        |
        v
Phase 7 -- Experiment O6
Calibration correction, given the measured overfitting-on-probability-
quality signature (Section 12/13.3)
        |
        v
Phase 8
Scenario-split execution (lower priority than D1 per Section 19's
reasoning about held-out-event-type population size) + full-model
dataset-seed-44 rerun
        |
        v
Phase 9
Model selection per Section 22's explicit protocol; single untouched
test evaluation
        |
        v
Phase 10
Freeze the classical GraphSAGE benchmark (Section 27's requirements);
QGNN stage may begin against the frozen protocol
```

---

## 26. Definition of Done (for this specification document only)

- [x] Repository inspected directly (source code, actual generated
  benchmark files), not only prior summaries.
- [x] Complete pipeline traced from raw benchmark through calibration,
  with exact implementation details (aggregation, activation, dropout,
  loss, optimizer, early stopping, threshold, seeds) confirmed against code.
- [x] Actual dataset files inspected with row counts, dtypes, ranges,
  temporal coverage, and entity coverage measured directly.
- [x] Feature provenance table built for every feature currently reaching
  the model, with static/dynamic, availability, target-derivation, and
  safety determined from source, not inferred.
- [x] Feature audit rules read and cross-checked against the enforcing
  code; one documentation-level ambiguity (comma-joined audit column)
  identified and confirmed already handled correctly (not a live bug).
- [x] Target construction explained exactly, including the overlap/
  persistence mechanics that make one raw onset generate many prediction
  examples.
- [x] Fresh-onset vs. persistence quantified across splits, event types,
  severities, and weeks, with exact counts.
- [x] Event generator inspected for all 5 event types plus the shared
  intensity function and the occurrence-probability process; precursor
  absence confirmed as a property of the generator, not a modeling gap.
- [x] Graph structure analyzed with exact degree distributions,
  single/multi-source counts, and a previously-undocumented structural
  fact (49% of suppliers have no direct SUPPLIER_MATERIAL edge) surfaced.
- [x] Current experimental results inspected directly from saved
  artifacts (metrics.json, calibration.json, training_history.csv), not
  only from prior summary prose.
- [x] The static-vs-dynamic finding investigated for its cause, with a
  concrete, evidence-backed hypothesis (region-draw memorization) and a
  specific, already-executable disambiguating experiment (D1) — not
  merely repeating the existing PR-AUC numbers.
- [x] The `static_plus_graph` degeneracy's mechanism explained and its
  existing automated safeguard (`temporal_variation_summary`) documented.
- [x] Weaknesses ranked by scientific importance, with each tied to
  Observed/Evidence/Implication/Recommendation.
- [x] Structural and temporal feature opportunities specified concretely
  enough to implement (exact source columns, exact computation, exact
  leakage check) — not "improve feature engineering."
- [x] Architecture, training, and calibration opportunities specified
  with exact configurations to compare and the specific evidence
  motivating each, gated appropriately on D1.
- [x] Diagnostic and optimization experiments kept in separate matrices,
  each experiment fully specified per the requested template.
- [x] An explicit model-selection protocol defined, rejecting "highest
  test score wins."
- [x] Computational considerations measured directly (parameter count,
  graph size, training duration), not estimated.
- [x] Research-paper framing provided without overclaiming novelty, with
  the benchmark's small effective sample size (6 events) stated as an
  explicit limitation.
- [x] QGNN boundary respected — this document only defines what the
  final classical benchmark must record, does not design the QGNN.
- [x] Final roadmap reordered based on actual evidence (D1 promoted to
  Phase 3), not a copy of the prior plan's generic sequence.
- [x] No model, feature, dataset, label, split, or hyperparameter was
  modified in the process of producing this document. No experiment was
  run beyond read-only inspection and pure re-analysis of already-saved
  artifacts.
