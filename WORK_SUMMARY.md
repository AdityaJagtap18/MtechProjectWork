# Work Summary

Everything done in this session, implementing `SCM_DATASET_GENERATION_PLAN.md` end to end.

## Session narrative

1. Read `SCM_DATASET_GENERATION_PLAN.md`; per its own §56 instruction, did reconnaissance before writing any generator code
2. Cloned `snap-stanford/supply-chains` (SupplySim), read its simulator, checked its license (none — reuse ideas only, not code)
3. Downloaded and inspected the NIST Sample Purchasing/Supply Chain dataset (found: schema exemplar, too small/synthetic for statistical calibration)
4. Wrote `DATASET_DESIGN_REVIEW.md`
5. Resolved 4 open design decisions (NIST role, SupplySim reuse policy, region-risk calibration source, Procurement Type field)
6. Built Phase 1 — schema (6 node types, 9 edge types, graph container, CSV export)
7. Built and executed `notebooks/walkthrough.ipynb` (Phases 1–2), sent to user
8. Built Phase 2 — topology generator (non-uniform/preferential-attachment edges)
9. Built Phase 3 — operations simulator; found and fixed a production-collapse bug; extended the notebook to cover it
10. Built Phase 4 — downloaded real WGI + INFORM Risk Index + NIST data, built region-risk calibration
11. Built Phase 5 — black-swan event engine (5 event types); found and fixed a recovery-time-exceeds-horizon bug
12. Built Phase 6 — cascade ground-truth labels
13. Built Phase 7 — validation suite (constraints, topology, statistical, event-engine checks)
14. Built Phase 8 — benchmark export (splits, metadata, provenance)
15. Committed everything to git (`cef0443`)
16. Explained how to run and verify the framework; user ran it themselves end to end on their own machine, confirmed 90/90 tests and all scripts working
17. Clarified a "where is the data" question — proved data exists with real file sizes/row counts/sample rows
18. Explained the full generation methodology (graph → simulate → disrupt → label → validate → package)
19. Traced one real black-swan event through actual generated data: `event_3` (severity-5 logistics disruption) → `supplier_279`'s fulfillment ratio → `material_75`'s inventory at `plant_4` → `plant_4`'s aggregate production
20. Built and published the **[Supply Chain Data Dictionary](https://claude.ai/code/artifact/0c9613f1-4452-41bd-9c7a-ab09b013d0cc)** artifact (full field reference); found and fixed a character-encoding bug during verification
21. Wrote this summary

## Dataset schema — every field, every table

### Graph — `graph/nodes.csv` (2,670 rows, 6 node types in one wide file, `node_type` column distinguishes)

**Supplier** (300 rows)
`supplier_id`, `tier` (int 1–3), `region_id` (FK), `industry` (categorical), `capacity` (float), `capacity_utilization` (0–1), `reliability` (0–1), `financial_health` (0–1), `lead_time_mean` (days), `lead_time_variability`, `quality_score` (0–1), `inventory_buffer`, `substitution_availability` (0–1), `geopolitical_exposure` (0–1), `disaster_exposure` (0–1), `cyber_exposure` (0–1), `criticality` (0–1)

**Material** (100 rows)
`material_id`, `material_category`, `criticality` (0–1), `substitutability` (0–1), `demand`, `unit_cost`, `inventory_level`, `safety_stock`, `supplier_count` (int, derived), `concentration` (0–1, derived = 1/supplier_count), `required_quantity_per_product`, `procurement_type` (MTS/OTS/null)

**Plant** (50 rows)
`plant_id`, `region_id` (FK), `production_capacity`, `utilization` (0–1), `operating_cost`, `inventory_capacity`, `resilience_score` (0–1), `downtime_cost`, `recovery_rate`

**Product** (200 rows)
`product_id`, `product_category`, `demand`, `revenue_per_unit`, `margin` (−1 to 1), `material_dependency` (0–1), `criticality` (0–1), `substitution_score` (0–1), `backlog`

**Region** (20 rows)
`region_id`, `country_or_region` (real country name if calibrated, else synthetic label), `geopolitical_risk` (0–1, real: WGI Political Stability), `natural_disaster_risk` (0–1, real: INFORM "Natural"), `infrastructure_risk` (0–1, real: INFORM "Infrastructure"), `trade_risk` (0–1, real: WGI Regulatory Quality), `cyber_risk` (0–1, synthetic), `transport_reliability` (0–1, synthetic)

**ProcurementOrder** (2,000 rows)
`procurement_id`, `supplier_id`, `material_id`, `plant_id` (FKs), `order_quantity`, `order_value`, `order_frequency`, `promised_lead_time`, `actual_lead_time`, `urgency` (0–1), `priority` (0–1), `contract_duration`

### Graph — `graph/edges.csv` (~7,600 rows)

`edge_id`, `source_id`, `target_id`, `edge_type` — one of: `SUPPLIER_MATERIAL`, `SUPPLIER_PROCUREMENT`, `PROCUREMENT_MATERIAL`, `PROCUREMENT_PLANT`, `MATERIAL_PLANT`, `PLANT_PRODUCT`, `PRODUCT_REGION`, `SUPPLIER_REGION`, `PLANT_REGION`

`graph/edge_features.csv`: `edge_id` + optional per-edge attributes (unused/empty in this generator)

### Operations — 104 weekly rows per entity

- `demand.csv`: `product_id`, `time`, `demand`
- `production.csv`: `plant_id`, `product_id`, `time`, `production`
- `inventory.csv`: `plant_id`, `material_id`, `time`, `inventory`
- `backlog.csv`: `product_id`, `time`, `backlog`
- `deliveries.csv`: `plant_id`, `material_id`, `supplier_id`, `time`, `quantity`
- `procurement.csv`: `plant_id`, `material_id`, `supplier_id`, `time_ordered`, `quantity_ordered`, `quantity_fulfilled`, `expected_delivery_time`

### Events

- `events.csv`: `event_id`, `event_type` (supplier_failure / natural_disaster / geopolitical / cyberattack / logistics), `start_time`, `duration`, `severity` (1–5), `probability_regime`, `affected_regions`, `affected_suppliers`, `affected_plants` (semicolon-joined), `capacity_reduction_fraction` (0–1), `lead_time_increase_fraction` (≥0), `recovery_delay`, `recovery_periods`, `cascade_mechanism` (constant)
- `event_impact.csv`: `event_id`, `cascade_severity` (0–1), `total_affected_nodes` (int), `time_to_impact` (int or null), `recovery_time` (int)

### Labels (ground truth, computed post-simulation)

- `supplier_labels.csv`: `supplier_id`, `time`, `supplier_disrupted` (binary), `supplier_risk_score` (float or null)
- `material_labels.csv`: `plant_id`, `material_id`, `time`, `material_shortage` (binary), `material_risk_score` (0–1)
- `plant_labels.csv`: `plant_id`, `time`, `production_loss` (float), `production_loss_fraction` (0–1), `plant_disruption` (binary)
- `product_labels.csv`: `product_id`, `time`, `product_shortage` (binary), `revenue_impact` (float)

### Metadata (per dataset, not per-row)

- `dataset_info.json`: node/edge counts, horizon, event count
- `generation_config.yaml`: exact config used
- `feature_schema.json`: machine-readable column list
- `provenance.json`: `dataset_id`, `generator_version`, `git_commit`, `seed`, `config_hash`, `generation_timestamp`, `real_data_sources`, `software_versions`

### Splits and audit (in `data/benchmark/`)

- `splits/temporal_split.csv`, `scenario_split.csv`, `severity_split.csv`: `time`, `split` (train/validation/test)
- `feature_audit.csv`: `table`, `column`, `available_time`, `target_time_rule`, `allowed`, `reason`

## Files created, by phase

### Reconnaissance
- `DATASET_DESIGN_REVIEW.md`

### Phase 1 — Schema
- `src/scm_dataset/schema/nodes.py`, `edges.py`, `graph.py`
- `src/scm_dataset/export/csv.py` (`export_graph`/`load_graph`)
- `tests/test_schema.py` — 8 tests

### Phase 2 — Topology generator
- `src/scm_dataset/generator/config.py`, `regions.py`, `suppliers.py`, `materials.py`, `plants.py`, `products.py`, `procurement.py`, `topology.py`, `utils.py`
- `configs/base.yaml`
- `scripts/generate_topology.py`
- `tests/test_topology.py` — 7 tests

### Phase 3 — Normal operations
- `src/scm_dataset/operations/demand.py`, `production.py`, `inventory.py`, `procurement.py`
- `src/scm_dataset/simulation/engine.py`
- `configs/normal.yaml`
- `scripts/generate_normal.py`
- `tests/test_simulation.py` — 9 tests
- Bug found+fixed: production collapsing to ~0.4% of demand (units mismatch between `Plant.production_capacity` and `Material.safety_stock`)

### Phase 4 — Real-data calibration
- `src/scm_dataset/real_data/loader.py`, `cleaner.py`, `mapper.py`, `calibrator.py`
- `scripts/download_real_data.py`, `scripts/calibrate.py`
- `configs/real_calibration.yaml`
- `data/raw/{nist,wgi,inform}/` — downloaded and tracked with `provenance.json` per source
- `tests/test_real_data.py` — 12 tests

### Phase 5 — Black-swan event engine
- `src/scm_dataset/schema/events.py`
- `src/scm_dataset/events/base.py`, `supplier_failure.py`, `natural_disaster.py`, `geopolitical.py`, `cyberattack.py`, `logistics.py`, `generate.py`
- `configs/black_swan.yaml`
- `scripts/generate_black_swan.py`
- `tests/test_events.py` — 17 tests
- Bug found+fixed: Geopolitical events' mean recovery time at severity 5 exceeded the 104-period horizon

### Phase 6 — Cascade ground-truth labels
- `src/scm_dataset/labels/risk_labels.py`
- `operations/backlog.csv` export added
- `tests/test_labels.py` — 10 tests

### Phase 7 — Validation
- `src/scm_dataset/validation/constraints.py`, `topology.py`, `statistical.py`, `report.py`
- `scripts/validate_dataset.py`
- `tests/test_validation.py` — 17 tests
- New dependencies: `networkx`, `scipy`

### Phase 8 — Benchmark export
- `src/scm_dataset/benchmark/splits.py`
- `src/scm_dataset/export/metadata.py`
- `scripts/build_benchmark.py`
- `tests/test_benchmark.py` — 9 tests

## Datasets generated (local, gitignored)

- `data/generated/scm_v1_topology_seed42`
- `data/generated/scm_v1_normal_seed42`
- `data/generated/scm_v1_black_swan_seed43`
- `data/generated/scm_v1_black_swan_seed44`
- `data/benchmark/` — all four assembled with splits + manifest

## Other deliverables

- `notebooks/walkthrough.ipynb` — executed, Phases 1–3
- **[Supply Chain Data Dictionary](https://claude.ai/code/artifact/0c9613f1-4452-41bd-9c7a-ab09b013d0cc)** — published artifact
- `pyproject.toml`, `requirements.txt` — pandas, numpy, pyyaml, openpyxl, networkx, scipy, pytest (+ jupyter/matplotlib as `notebook` extra)
- `.gitignore`

## Test suite

90 tests total, all passing (`pytest -q`).

## Git

- Repo initialized, initial commit `cef0443`: "Implement full SCM dataset generation framework (plan §42 Phases 1-8)" — 96 files, 10,000 insertions
