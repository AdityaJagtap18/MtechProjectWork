# SCM Dataset Generation

Reproducible synthetic supply-chain dataset generation framework for
supplier/procurement risk analysis under rare, high-impact disruptions, and
a benchmark for comparing classical GNNs against QGNNs.

Full specification: [SCM_DATASET_GENERATION_PLAN.md](SCM_DATASET_GENERATION_PLAN.md).
Reconnaissance and design decisions made before implementation began:
[DATASET_DESIGN_REVIEW.md](DATASET_DESIGN_REVIEW.md).

## Status

**All 8 phases of plan §42 complete.** The dataset framework is
implemented end-to-end: schema → topology → operations → real-data
calibration → black-swan events → cascade labels → validation → benchmark
export. Only after all of §55's "Definition of Done" checklist is
satisfied should the project proceed to the GNN/QGNN comparison itself
(plan §54) — that comparison has not been started.

**Phase 1 — schema:**
- Six node types (`Supplier`, `ProcurementOrder`, `Material`, `Plant`,
  `Product`, `Region`) with field-level validation — `src/scm_dataset/schema/nodes.py`
- Nine typed edges with fixed endpoint-type pairs and documented operational
  meaning — `src/scm_dataset/schema/edges.py`
- A heterogeneous graph container with referential-integrity validation —
  `src/scm_dataset/schema/graph.py`
- CSV serialization (`graph/nodes.csv`, `graph/edges.csv`,
  `graph/edge_features.csv`) round-tripping back to typed dataclasses —
  `src/scm_dataset/export/csv.py`

**Phase 2 — topology generator** (`src/scm_dataset/generator/`):
- Config-driven node generation for all six types, with dependent (not
  independently sampled) attributes per plan §13
- Non-uniform, preferential-attachment-based edge generation for all nine
  edge types (plan §9): single- vs multi-source materials, supplier
  concentration, capacity-weighted plant→product assignment
- `material.supplier_count`/`concentration` and `product.material_dependency`
  are derived from the generated topology, not sampled
- Deterministic given a seed; produces a graph that passes Phase 1's
  validator
- Runnable standalone: `python scripts/generate_topology.py --config configs/base.yaml --output data/generated/scm_v1_topology_seed42`

**Phase 3 — normal operations** (`src/scm_dataset/operations/`,
`src/scm_dataset/simulation/`):
- Discrete-time simulation (weekly by default, plan §10) of demand,
  production, inventory, and procurement on top of the Phase 2 graph,
  enforcing plan §11's constraints (bounded production, no negative
  inventory, lead-time-gated delivery, capacity-limited suppliers) and
  §11.5 (production responds to demand/backlog, not run independently of it)
- Inventory/reorder buffers and supplier delivery capacity are derived from
  each (plant, material) pair's own implied consumption rate
  (`plant.production_capacity * material.required_quantity_per_product`),
  not from the Material/Supplier nodes' standalone Phase 2 attributes —
  see `simulation/engine.py`'s docstring for why that distinction matters
  (an earlier version that skipped it collapsed production to ~0.4% of
  demand)
- Runnable end-to-end: `python scripts/generate_normal.py --config configs/normal.yaml --output data/generated/scm_v1_normal_seed42`
  — full plan §8 baseline (2,670 nodes) plus 104 periods (~2 years) of
  operations in well under a second; ~98.5% demand fill rate at defaults

**Phase 4 — real-data calibration** (`src/scm_dataset/real_data/`):
- Region risk (4 of 6 dimensions) calibrated from real public data instead
  of synthetic archetypes: `geopolitical_risk`/`trade_risk` from World Bank
  WGI (Political Stability, Regulatory Quality), `natural_disaster_risk`/
  `infrastructure_risk` from the EU JRC INFORM Risk Index — both CC-BY-4.0,
  191 countries covered by both sources. `cyber_risk`/`transport_reliability`
  stay synthetic but correlated with the real risk level (no clean public
  source found — DATASET_DESIGN_REVIEW.md decision 3)
- Opt-in via `real_data.use_real_region_calibration` (off by default —
  `configs/base.yaml`/`normal.yaml` are unaffected; `configs/real_calibration.yaml`
  turns it on) so Phase 2/3 behavior stays backward compatible
- NIST sample data loader + field mapping + structural stats (multi-source
  rate, BOM-level/hierarchy-depth distributions) — kept to a schema/
  structural reference per DATASET_DESIGN_REVIEW.md §3.1, not injected as
  literal node values (NIST's own content is small-N synthetic placeholder
  text, not something to calibrate distributions against)
- All three external sources (NIST, WGI, INFORM) downloaded and tracked in
  `data/raw/` with a `provenance.json` per source (plan §3.3); pipeline:
  `python scripts/download_real_data.py` then `python scripts/calibrate.py`
  → `data/processed/region_risk_calibration.csv` (191 countries) +
  `data/processed/nist_structural_stats.json`
- Example result: generating with real calibration gives Japan
  `geopolitical_risk=0.25` (politically stable) but
  `natural_disaster_risk=0.83` (real earthquake/tsunami exposure) — a
  differentiated pattern no synthetic archetype produces on its own

**Phase 5 — black-swan event engine** (`src/scm_dataset/events/`,
`src/scm_dataset/schema/events.py`):
- Five event types (plan §15: Supplier Failure, Natural Disaster,
  Geopolitical Disruption, Cyberattack, Logistics Disruption), each
  targeting the entities plan §19's cascade diagram says they should
  (a single supplier, or every supplier/plant in a region) and weighting
  capacity-reduction vs. lead-time-increase differently per type (e.g.
  Logistics is lead-time-dominant with minimal capacity effect; Natural
  Disaster is the reverse and the only type that directly hits plants)
- A 1-5 severity scale (plan §16) that moves *multiple* variables together
  from one shared intensity value — affected-entity count, capacity cut,
  lead-time increase, duration, and recovery time all scale with severity,
  not just one number
- A recovery profile (plan §21): full effect for `duration` periods, held
  for `recovery_delay` more, then a linear ramp back to normal over
  `recovery_periods` — verified end-to-end on a real generated event: a
  severity-5 Logistics event roughly doubled mean lead time (2.1→4.5
  periods) and dropped fulfillment ratio (0.93→0.78) for exactly its 11
  affected suppliers, no others
- **No new cascade machinery was needed**: events just multiply the
  capacity/lead-time inputs Phase 3's engine already uses, so procurement
  disruption → material shortage → production reduction → backlog growth
  (plan §19) falls out of the existing conservation constraints. Caught and
  fixed one real calibration bug this way: Geopolitical events' recovery
  time averaged ~125 periods at severity 5 — longer than the entire
  104-period horizon, making "recovery" unobservable in the exported
  dataset — before rebalancing its duration/recovery scale to match
  Natural Disaster's (~82 periods)
- Opt-in via `events.enabled` (off by default — `configs/base.yaml`/
  `normal.yaml` are event-free, matching plan §25 Dataset A);
  `configs/black_swan.yaml` turns it on with plan §38's example
  probabilities
- Pipeline: `scripts/generate_black_swan.py --input <existing graph dir> --config configs/black_swan.yaml`
  loads an **already-generated** graph (doesn't regenerate topology) and
  re-simulates it with events on — so a normal run and its black-swan
  counterpart share the identical underlying supply chain, only differing
  in which disruptions occurred (needed for plan §32 Experiment 3: normal
  vs. normal+black-swan training data)
- Events exported to `events/events.csv` (plan §24), including for
  event-free datasets (empty file, for a consistent directory structure)

**Phase 6 — cascade ground-truth labels** (`src/scm_dataset/labels/`):
- Four entity-level label tables (plan §22): `supplier_labels.csv`
  (`supplier_disrupted`, `supplier_risk_score`), `material_labels.csv`
  (`material_shortage`, `material_risk_score`), `plant_labels.csv`
  (`production_loss`, `production_loss_fraction`, `plant_disruption`),
  `product_labels.csv` (`product_shortage`, `revenue_impact`) — every one
  derived from simulated time series (production/inventory/fulfillment/
  backlog), never copied from the static input risk/criticality attributes
  (plan §13, §43.2)
- `production_loss` nets out *demand*, not just capacity — a plant idle
  because nobody wants more output isn't "disrupted"; only output lost to a
  binding capacity/material constraint counts, which is what
  `plant_disruption` thresholds on (plan §11.5's demand-responsiveness,
  extended to labels)
- **New logic beyond reading off Phase 3/5's time series**: `events/event_impact.csv`
  (`cascade_severity`, `total_affected_nodes`, `time_to_impact`,
  `recovery_time`) traces each *specific* event's downstream reach through
  the graph (Supplier→Material→Plant→Product) — the simulator itself only
  tracks the union of everything happening at once, not which event caused
  which consequence. `time_to_impact` counts only *purely downstream*
  plants (reached via material dependency, not directly targeted) to
  isolate genuine cascade-propagation lag rather than trivially reporting
  ~0 for event types that hit plants directly. No counterfactual
  re-simulation is performed to isolate one event's marginal effect from
  concurrent events/ordinary demand noise — documented as an accepted
  imprecision, not a hidden one
- Verified on the real generated dataset: mild, single-period severity-1
  events (a lone supplier failure, a cyberattack) show no measurable
  downstream cascade at 300-supplier scale (`time_to_impact` undefined,
  `recovery_time=0`) — exactly the expected behavior, since not every
  event should visibly propagate
- Labels computed for every dataset variant, including event-free ones
  (`generate_normal.py` too — supplier_disrupted etc. are then mostly 0,
  itself a legitimate label set for Dataset A)

**Phase 7 — validation** (`src/scm_dataset/validation/`):
- `scripts/validate_dataset.py --input <dataset dir>` validates the
  *exported* CSVs a downstream researcher would actually receive — not
  just re-checking in-memory simulator state, which Phases 3-6's own unit
  tests already do — catching any export/round-trip regression those
  tests would miss. Writes four reports to `<dataset>/validation/`:
  - `constraints_report.json` (plan §35 operational validation, §36 sanity
    tests): no negative inventory, no production over capacity, every
    delivery follows a real preceding order, every event references real
    nodes/regions — 0 violations on the generated black-swan dataset
  - `topology_report.json` (§35 structural validation): supplier
    concentration (Gini, top-decile share), multi-source rate, graph
    connectivity (weakly-connected components, clustering, via networkx),
    and the one real structural comparison available — NIST's observed
    multi-source rate (Phase 4) vs. ours (0.20 vs. 0.68 — a real,
    honestly-reported gap, not hidden, given NIST's small-N caveat)
  - `statistical_report.json` (§12's template: real/synthetic mean, std,
    KS statistic, Wasserstein distance — via scipy): real WGI/INFORM
    region risk vs. the *default* (uncalibrated) synthetic archetype
    generator — a deliberately uncalibrated baseline comparison, since
    that's what motivates `configs/real_calibration.yaml` existing at all
  - `event_validation_report.json` (§35 event validation): severity
    monotonicity (severity 5 averages 0.95 capacity reduction vs.
    severity 1's 0.15 — matches `SEVERITY_INTENSITY` almost exactly),
    geographic isolation (region-scoped events never affect entities
    outside their targeted region — 0 violations), single-source
    concentration consistency (materials with exactly one supplier always
    show `concentration == 1.0`)
- No `validation/plots/` directory is generated — plan §24 lists one, but
  §42's Phase 7 deliverable is specifically "a validation report," and
  visual inspection is already covered by `notebooks/walkthrough.ipynb`
  (matplotlib stays a notebook-only dependency, not a core one)

**Phase 8 — benchmark export** (`src/scm_dataset/benchmark/`,
`src/scm_dataset/export/metadata.py`):
- `scripts/build_benchmark.py --input data/generated --output data/benchmark`
  assembles one or more already-generated dataset directories into a final
  package: copies each, computes all three plan §26 split strategies into
  `<dataset>/splits/*.csv`, writes a shared `feature_audit.csv` (plan §27),
  and a `benchmark_manifest.json` listing every included dataset — so
  multiple seeds of the same scenario (plan §28: report mean ± std, never
  just the best run) are easy to enumerate together
- **Temporal split**: train/validation/test on disjoint period ranges — no
  random row shuffling of a simulated trajectory (plan §26's explicit
  warning against that)
- **Scenario split** (plan §49's black-swan generalization experiment):
  periods touched by a held-out event type (Cyberattack, Geopolitical) go
  to test, everything else (including normal baseline periods) to train —
  tests whether a model learned general disruption propagation rather than
  memorizing per-type signatures
- **Severity split**: train on severity ≤3, test on severity 4-5 — tests
  generalization from moderate to extreme disruptions
- **Feature audit** (plan §27, preventing data leakage): every exported
  table/column is marked allowed/disallowed as a model input feature with
  a stated reason — static graph attributes and realized operational
  observations (demand, production, inventory, backlog, deliveries) are
  allowed; event records, the post-hoc cascade summary, and all four label
  tables are explicitly disallowed, since those are the ground truth being
  predicted, not something a model would legitimately see beforehand
- **Reproducibility metadata** (plan §39, written by every `generate_*.py`
  script at generation time, not reconstructed after the fact):
  `dataset_id`, `generator_version`, `git_commit`, `seed`, `config_hash`,
  `generation_timestamp`, `real_data_sources`, `software_versions` in
  `metadata/provenance.json`, plus `dataset_info.json`,
  `generation_config.yaml` (the exact config used), and
  `feature_schema.json` (every table's columns)
- Verified end-to-end at full scale: assembled 4 real generated datasets
  (topology-only, normal, and two different-seed black-swan runs) into
  `data/benchmark/` in ~1 second

Every phase of plan §42 is now implemented — see
[DATASET_DESIGN_REVIEW.md](DATASET_DESIGN_REVIEW.md) for the design
decisions made along the way, and `notebooks/walkthrough.ipynb` for a
step-by-step visual walkthrough of Phases 1-3.

## Installation

```bash
pip install -e ".[dev]"
```

`data/raw/` (NIST, WGI, INFORM) is already downloaded and tracked in this
repo. To refresh it or to build `data/processed/` (needed for
`configs/real_calibration.yaml`):

```bash
python scripts/download_real_data.py
python scripts/calibrate.py
```

## Tests

```bash
pytest
```

A handful of `tests/test_real_data.py` tests are skipped if `data/raw/`
isn't present.

## Notebook walkthrough

[notebooks/walkthrough.ipynb](notebooks/walkthrough.ipynb) builds a tiny
graph by hand (Phase 1), runs the full generator at `configs/base.yaml`
scale (Phase 2) with the degree concentration and attribute dependency
checks described above, then runs the Phase 3 simulator and plots demand
vs. production and an inventory reorder cycle. Requires the `notebook`
extra:

```bash
pip install -e ".[notebook]"
jupyter notebook notebooks/walkthrough.ipynb
```
