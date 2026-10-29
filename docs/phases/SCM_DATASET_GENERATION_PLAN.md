# SCM Dataset Generation & Benchmark Specification

## Supplier Procurement Risk Analysis Under Black-Swan Events

**Document purpose:** Implementation specification for an AI coding
agent (e.g., Codex) to build a reproducible supply-chain dataset
generation framework suitable for GNN/QGNN research.

**Primary research direction:** Build a realism-constrained synthetic
supply-chain environment, optionally calibrated from public real SCM
data, generate controlled rare/black-swan disruption scenarios and
cascading outcomes, and use the resulting benchmark to compare classical
GNNs against QGNNs.

------------------------------------------------------------------------

# 1. Executive Research Goal

Build a reusable dataset/simulation framework for:

> **Supplier and procurement risk analysis under rare, high-impact
> supply-chain disruptions, with a controlled comparison between
> classical Graph Neural Networks and Quantum Graph Neural Networks.**

The dataset framework must be useful independently of the final GNN/QGNN
experiment.

The project should therefore NOT be implemented as a single flat CSV
generator. It should produce:

1.  A supply-chain graph.
2.  Node and edge attributes.
3.  Normal operational time-series observations.
4.  Black-swan event definitions.
5.  Simulated event propagation/cascades.
6.  Ground-truth risk and impact labels.
7.  Multiple scenario datasets.
8.  Metadata describing exactly how each dataset was generated.
9.  Validation reports showing whether synthetic data is structurally
    and statistically plausible.
10. Reproducible seeds/configurations.

The resulting benchmark should allow another researcher to train a
different model without needing to understand or modify the simulator.

------------------------------------------------------------------------

# 2. Research Positioning

The strongest framing is not:

> "Generate fake supply-chain data and compare GNN and QGNN."

Instead, frame the work as:

> **A controlled, reproducible synthetic supply-chain benchmark for
> rare-event risk prediction and cascading disruption analysis.**

The GNN/QGNN comparison becomes an application and benchmark on top of
the environment.

Potential research contributions:

### Contribution A --- SCM synthetic-data framework

A configurable generator for a multi-tier procurement/supply-chain
graph.

### Contribution B --- Black-swan event simulator

A mechanism for injecting rare, high-impact disruptions into the graph
and propagating their operational consequences.

### Contribution C --- Ground-truth cascade labels

Labels are derived from simulated consequences rather than directly
copied from an input risk score.

### Contribution D --- GNN/QGNN benchmark

A controlled comparison of classical and quantum graph-learning
approaches using identical data splits, features and targets.

### Contribution E --- Synthetic-data validation

Demonstrate that the synthetic environment preserves important
real-world SCM distributions, dependencies and topology characteristics.

------------------------------------------------------------------------

# 3. Existing Resources to Incorporate

## 3.1 Stanford Supply Chains / SupplySim --- primary implementation reference

Repository:

https://github.com/snap-stanford/supply-chains

The Stanford repository supports research on learning production
functions for supply-chain networks. It contains code for SupplySim and
synthetic datasets. The repository states that SupplySim generates
static graphs, exogenous supply/demand schedules and time-varying
transactions. It also provides synthetic settings including standard
supply, supply shocks and missing transactions.

This should be treated as a **reference implementation and source of
ideas**, not blindly copied.

Important repository areas to inspect:

-   `synthetic_data`
-   `TGB/modules/synthetic_data.py`
-   `register_data`
-   released synthetic datasets
-   paper methodology and Appendix B.2

The repository also describes pipelines for converting transaction-level
data into standardized graph/hypergraph representations.

**Implementation requirement:** Before writing a new simulator, inspect
the Stanford implementation carefully and document which concepts are
reused, adapted or independently redesigned.

## 3.2 NIST Sample Purchasing / Supply Chain Data

Official source:

https://catalog.data.gov/dataset/sample-purchasing-supply-chain-data

NIST provides public sample purchasing data containing information about
suppliers, products they provide and projects those products are used
for. The dataset is public and is explicitly tagged for supply chain,
purchasing and supplier use.

Use this as a potential **real-data anchor** for validating the
supplier/product/procurement portion of the synthetic environment.

Do NOT assume it contains every attribute required by the proposed
graph. Missing attributes should be treated as calibration targets to be
modeled, not invented as if they were observed.

## 3.3 Additional public network data

If a suitable public supplier/customer network is available, use it only
for topology validation and clearly distinguish it from
transaction-level observations.

The project must maintain provenance metadata for every external
dataset:

``` yaml
source_name:
source_url:
access_date:
license:
data_type:
observed_or_synthetic:
fields_used:
fields_not_used:
transformations:
```

## 3.4 Synthetic disruption datasets

Synthetic disruption datasets found online may be useful for comparison
or scenario-design inspiration, but they must NOT automatically be
described as real-world evidence.

The project should maintain a strict distinction:

-   observed real data
-   public derived data
-   synthetic data from external projects
-   our generated synthetic data

------------------------------------------------------------------------

# 4. Recommended Data Strategy

Use a hybrid strategy.

``` text
                 PUBLIC REAL SCM DATA
                         |
             +-----------+-----------+
             |                       |
       NIST purchasing       Other public SCM
             |                 topology data
             +-----------+-----------+
                         |
                         v
              REALISM / CALIBRATION
                         |
                         v
                OUR SCM GENERATOR
                         |
                         v
                  NORMAL SCM
                         |
                  +------+------+
                  |             |
                  v             v
             Normal data   Black-swan engine
                                |
                                v
                         Cascade simulator
                                |
                                v
                         Ground truth labels
                                |
                                v
                         Benchmark datasets
                                |
                    +-----------+-----------+
                    |                       |
                   GNN                    QGNN
```

Real data is primarily used to constrain and validate the synthetic
environment.

The simulator is responsible for creating the rare-event cases that real
datasets usually lack.

------------------------------------------------------------------------

# 5. Six Node Types

The benchmark should use exactly six node types for the initial research
version.

## 5.1 Supplier

Represents a firm providing materials/components.

Suggested attributes:

-   supplier_id
-   tier
-   region_id
-   industry
-   capacity
-   capacity_utilization
-   reliability
-   financial_health
-   lead_time_mean
-   lead_time_variability
-   quality_score
-   inventory_buffer
-   substitution_availability
-   geopolitical_exposure
-   disaster_exposure
-   cyber_exposure
-   criticality

## 5.2 Procurement Order

Represents a procurement relationship/order rather than a physical
organization.

Attributes:

-   procurement_id
-   supplier_id
-   material_id
-   plant_id
-   order_quantity
-   order_value
-   order_frequency
-   promised_lead_time
-   actual_lead_time
-   urgency
-   contract_duration
-   priority

The procurement node is useful because the research focus is
supplier/procurement risk rather than only supplier-to-plant
connectivity.

## 5.3 Material

Represents a raw material, component or intermediate input.

Attributes:

-   material_id
-   material_category
-   criticality
-   substitutability
-   demand
-   unit_cost
-   inventory_level
-   safety_stock
-   supplier_count
-   concentration
-   required_quantity_per_product

## 5.4 Plant

Represents a manufacturing/production facility.

Attributes:

-   plant_id
-   region_id
-   production_capacity
-   utilization
-   operating_cost
-   inventory_capacity
-   resilience_score
-   downtime_cost
-   recovery_rate

## 5.5 Product

Represents a finished or intermediate product.

Attributes:

-   product_id
-   product_category
-   demand
-   revenue_per_unit
-   margin
-   material_dependency
-   criticality
-   substitution_score
-   backlog

## 5.6 Region

Represents a geographic area.

Attributes:

-   region_id
-   country_or_region
-   geopolitical_risk
-   natural_disaster_risk
-   infrastructure_risk
-   trade_risk
-   cyber_risk
-   transport_reliability

------------------------------------------------------------------------

# 6. Graph Schema

The graph must support heterogeneous relationships.

Recommended edge types:

``` text
Supplier -> Material
Supplier -> Procurement
Procurement -> Material
Procurement -> Plant
Material -> Plant
Plant -> Product
Product -> Region
Supplier -> Region
Plant -> Region
```

Do not force all information into one homogeneous adjacency matrix
during data generation.

Maintain a heterogeneous representation first.

A later preprocessing layer may convert the graph into the
representation required by:

-   GCN
-   GraphSAGE
-   GAT
-   heterogeneous GNN
-   QGNN

------------------------------------------------------------------------

# 7. Graph Semantics

Every edge must have a clear operational meaning.

Examples:

### Supplier -\> Material

Supplier provides material.

### Supplier -\> Procurement

Supplier participates in procurement order.

### Procurement -\> Material

Order requests material.

### Procurement -\> Plant

Order is intended for plant.

### Material -\> Plant

Plant consumes material.

### Plant -\> Product

Plant produces product.

### Supplier -\> Region

Supplier operates in region.

### Product -\> Region

Product demand/market is associated with region.

------------------------------------------------------------------------

# 8. Graph Size

The first benchmark should remain computationally manageable.

Recommended baseline:

``` text
Suppliers:       300
Procurement:    2,000
Materials:       100
Plants:           50
Products:        200
Regions:          20
```

Total possible nodes: approximately 2,670.

This is deliberately moderate.

The generator should support scaling parameters:

``` yaml
suppliers: 300
procurement_orders: 2000
materials: 100
plants: 50
products: 200
regions: 20
```

Do not hard-code these numbers.

------------------------------------------------------------------------

# 9. Network Topology Generation

Do not generate edges uniformly at random.

The topology should reflect plausible SCM properties:

-   supplier concentration
-   multi-tier relationships
-   preferential dependence on important suppliers
-   single-source and multi-source procurement
-   geographic clustering
-   material specialization
-   plant-material compatibility
-   product-material BOM relationships

The generator should support:

### Single-source suppliers

A material depends on one supplier.

### Multi-source suppliers

A material has several possible suppliers.

### Critical suppliers

A supplier serves many downstream nodes.

### Bottleneck materials

A material is difficult to substitute.

### Regional concentration

Several suppliers are located in the same region.

These structures are essential for meaningful cascade behavior.

------------------------------------------------------------------------

# 10. Normal Operational Data

Generate a time-varying normal operating state.

Recommended temporal resolution:

-   daily for detailed simulation, OR
-   weekly for the first prototype

Recommended initial horizon:

``` text
2–3 simulated years
```

For each time step, maintain:

``` text
inventory
demand
orders
deliveries
lead_time
capacity
production
backlog
delivery_delay
quality
```

Do not generate every feature independently.

Features must have dependencies.

Example:

``` text
production <= available_capacity
production <= available_material
inventory[t+1] =
    inventory[t]
    + deliveries[t]
    - consumption[t]
```

------------------------------------------------------------------------

# 11. Realism Constraints

The simulator must enforce physical/operational constraints.

## 11.1 Capacity constraint

``` text
production_t <= production_capacity_t
```

## 11.2 Inventory constraint

``` text
consumption_t <= inventory_available_t
```

## 11.3 Procurement constraint

A procurement order cannot exceed supplier available capacity unless the
simulation explicitly represents backlog.

## 11.4 Lead-time constraint

Delivered quantity should occur after the modeled lead time.

## 11.5 Demand constraint

Production should respond to demand/backlog rather than being completely
independent.

## 11.6 Supplier dependency

If a material has one supplier, its disruption should have a stronger
effect than if it has many substitute suppliers.

------------------------------------------------------------------------

# 12. Statistical Calibration from Real Data

When a real dataset provides an observable field, estimate its
distribution rather than inventing arbitrary distributions.

Examples:

``` text
supplier count
orders per supplier
order quantity
supplier concentration
product/supplier degree
material frequency
geographic distribution
```

Possible techniques:

-   empirical distributions
-   kernel density estimation
-   log-normal/gamma fitting
-   mixture distributions
-   copulas for correlated variables

Do not fit distributions blindly.

For each calibrated variable, create a validation report:

``` text
feature
real_mean
synthetic_mean
real_std
synthetic_std
KS_statistic
wasserstein_distance
```

For graph structure:

``` text
real_degree_mean
synthetic_degree_mean
real_degree_distribution
synthetic_degree_distribution
clustering_difference
component_difference
```

------------------------------------------------------------------------

# 13. Synthetic Data Must Preserve Dependencies

Avoid independent random sampling for correlated SCM attributes.

Example:

A supplier with high utilization should generally have less spare
capacity.

A supplier with low financial health may have elevated disruption
probability.

A single-source material should increase downstream vulnerability.

A region with high geopolitical exposure should influence affected
suppliers.

Use dependency functions rather than arbitrary independent random
numbers.

Example conceptual model:

``` text
supplier_risk =
    f(
        financial_health,
        reliability,
        capacity_utilization,
        region_risk,
        dependency,
        event_exposure
    )
```

This score should be used carefully and should NOT simply become the
target label.

------------------------------------------------------------------------

# 14. Black-Swan Event Engine

Black-swan scenarios should be generated separately from normal
operational noise.

The event object should contain:

``` yaml
event_id:
event_type:
start_time:
duration:
affected_regions:
affected_suppliers:
severity:
probability_regime:
direct_effects:
cascade_rules:
recovery_model:
```

------------------------------------------------------------------------

# 15. Event Types

Initial benchmark: five event families.

## 15.1 Supplier Failure

Directly reduces or eliminates supplier capacity.

Possible effects:

``` text
capacity reduction
delivery delay
quality degradation
order backlog
```

## 15.2 Natural Disaster

Examples:

-   flood
-   earthquake
-   cyclone
-   extreme weather

Effects:

``` text
regional infrastructure reduction
supplier capacity reduction
transport delay
temporary shutdown
```

## 15.3 Geopolitical Disruption

Examples:

-   trade restriction
-   sanctions
-   border closure
-   regional conflict

Effects:

``` text
supplier accessibility reduction
lead-time increase
transport capacity reduction
cost increase
material availability reduction
```

## 15.4 Cyberattack

Effects:

``` text
order-processing disruption
supplier information blackout
delivery delays
temporary capacity reduction
```

## 15.5 Logistics/Transportation Disruption

Effects:

``` text
delivery lead time increase
transport capacity reduction
inventory arrival delay
```

------------------------------------------------------------------------

# 16. Severity Model

Use a controlled severity scale.

``` text
1 = Minor
2 = Moderate
3 = Major
4 = Extreme
5 = Black Swan
```

The event generator must not make severity 5 simply equal to a large
numerical multiplier.

Severity should affect multiple operational variables coherently.

Example:

``` text
severity ↑
    |
    +--> affected nodes ↑
    +--> capacity reduction ↑
    +--> lead-time increase ↑
    +--> duration ↑
    +--> recovery time ↑
```

------------------------------------------------------------------------

# 17. Defining Black-Swan Scenarios

For the benchmark, define black-swan scenarios as:

-   low probability in the generated scenario population
-   high operational impact
-   significant deviation from normal operating conditions
-   ability to propagate through dependencies

Avoid claiming that synthetic events are statistically identical to
historical black swans.

Use the term as an experimental regime.

------------------------------------------------------------------------

# 18. Cascade Simulator

This is a core component.

The simulator takes:

``` text
normal graph
+
normal operational state
+
event
```

and generates:

``` text
post-event graph/state
+
time-series consequences
+
ground-truth labels
```

------------------------------------------------------------------------

# 19. Cascade Propagation

Recommended propagation sequence:

``` text
EVENT
  |
  v
DIRECTLY AFFECTED REGION
  |
  v
AFFECTED SUPPLIERS
  |
  v
PROCUREMENT DISRUPTION
  |
  v
MATERIAL SHORTAGE
  |
  v
PLANT PRODUCTION REDUCTION
  |
  v
PRODUCT SHORTAGE
  |
  v
DOWNSTREAM IMPACT
```

Each step should be governed by explicit rules.

------------------------------------------------------------------------

# 20. Example Cascade Equations

## Supplier available capacity

``` text
C_supplier(t) =
    C_base *
    (1 - disruption_factor)
```

## Delivery quantity

``` text
delivery(t) =
    min(
        order_quantity,
        supplier_available_capacity,
        transport_capacity
    )
```

## Inventory

``` text
inventory(t+1) =
    inventory(t)
    + delivery(t)
    - consumption(t)
```

## Shortage

``` text
shortage(t) =
    max(
        0,
        required_material(t) - available_material(t)
    )
```

## Plant production

``` text
production(t) =
    min(
        plant_capacity(t),
        material_available(t) / material_required_per_unit
    )
```

## Product impact

``` text
product_shortage =
    demand - available_product
```

These are initial models and should be configurable.

------------------------------------------------------------------------

# 21. Recovery Model

Every disruption should have a recovery process.

Possible model:

``` text
initial disruption
       |
       v
maximum impact
       |
       v
partial recovery
       |
       v
normal operation
```

Parameters:

``` yaml
recovery_rate:
recovery_delay:
maximum_downtime:
```

Different event types should have different recovery profiles.

------------------------------------------------------------------------

# 22. Ground Truth Labels

Ground truth must be generated by the simulation.

Do NOT label data using the same feature that the ML model receives.

Suggested labels:

## Supplier risk

``` text
supplier_disrupted
supplier_risk_score
```

## Material risk

``` text
material_shortage
material_risk_score
```

## Plant risk

``` text
production_loss
plant_disruption
```

## Product impact

``` text
product_shortage
revenue_impact
```

## Overall cascade

``` text
cascade_severity
total_affected_nodes
time_to_impact
recovery_time
```

------------------------------------------------------------------------

# 23. Primary ML Target

The first research experiment should have one clearly defined target.

Recommended:

> **Predict whether a supplier/material/plant node will experience
> significant disruption within a future prediction window.**

Example:

``` text
target = 1 if:
    downstream operational loss > threshold
within next H time steps
```

The threshold and horizon must be fixed before final evaluation.

Secondary targets can include:

-   severity
-   time-to-impact
-   recovery time
-   product shortage

------------------------------------------------------------------------

# 24. Dataset Structure

Do not only produce one CSV.

Recommended output:

``` text
dataset/
├── metadata/
│   ├── dataset_info.json
│   ├── generation_config.yaml
│   ├── feature_schema.json
│   └── provenance.json
│
├── graph/
│   ├── nodes.csv
│   ├── edges.csv
│   └── edge_features.csv
│
├── operations/
│   ├── procurement.csv
│   ├── inventory.csv
│   ├── demand.csv
│   ├── production.csv
│   └── deliveries.csv
│
├── events/
│   ├── events.csv
│   └── event_impact.csv
│
├── labels/
│   ├── supplier_labels.csv
│   ├── material_labels.csv
│   ├── plant_labels.csv
│   └── product_labels.csv
│
└── validation/
    ├── statistical_report.json
    ├── topology_report.json
    └── plots/
```

------------------------------------------------------------------------

# 25. Scenario Dataset Versions

Generate several benchmark regimes.

## Dataset A --- Normal

No major disruption.

Purpose:

-   baseline graph learning
-   normal operations

## Dataset B --- Minor disruptions

Mostly severity 1--2.

Purpose:

-   ordinary resilience

## Dataset C --- Major disruptions

Severity 3--4.

Purpose:

-   significant disruption prediction

## Dataset D --- Black-swan

Rare severity 5 events.

Purpose:

-   rare-event detection

## Dataset E --- Mixed

Normal + minor + major + black-swan.

Purpose:

-   realistic mixed training environment

------------------------------------------------------------------------

# 26. Train/Validation/Test Design

Avoid simply randomly splitting rows from the same simulated trajectory.

Preferred split strategies:

## Scenario split

Training:

``` text
flood
supplier failure
```

Testing:

``` text
cyberattack
geopolitical disruption
```

## Severity generalization

Training:

``` text
severity 1–3
```

Testing:

``` text
severity 4–5
```

## Temporal split

Training:

``` text
t0 -> tN
```

Validation:

``` text
tN -> tN+k
```

Test:

``` text
future period
```

The final paper should report which split is used and why.

------------------------------------------------------------------------

# 27. Prevent Data Leakage

Do not include:

-   future risk labels
-   post-event values
-   future inventory
-   future production
-   event outcome
-   target-derived risk scores

in model input features.

For a prediction at time `t`, only information available at or before
`t` should be used.

This must be enforced programmatically.

Create a feature audit:

``` text
feature
available_time
target_time
allowed
reason
```

------------------------------------------------------------------------

# 28. Multiple Random Seeds

Every benchmark must support seeds.

Example:

``` text
42
43
44
45
46
```

Run every important model using multiple seeds.

Report:

``` text
mean ± standard deviation
```

Do not report only the best run.

------------------------------------------------------------------------

# 29. Baseline Models

At minimum:

### Non-graph baseline

One of:

-   Logistic Regression
-   Random Forest
-   XGBoost
-   MLP

### Classical graph baseline

At minimum:

-   GCN or GraphSAGE

Recommended:

-   GraphSAGE
-   GAT

### Quantum model

Use the selected QGNN architecture.

The classical and quantum models must receive equivalent information.

------------------------------------------------------------------------

# 30. Fair GNN vs QGNN Comparison

Keep the following identical:

``` text
dataset
node features
edge structure
target
train/validation/test split
seed policy
evaluation metrics
prediction horizon
class weighting policy
```

Do not intentionally weaken the classical model.

Do not assume QGNN superiority.

A result where GNN outperforms QGNN is scientifically valid.

------------------------------------------------------------------------

# 31. QGNN Feasibility Constraint

The full SCM graph should NOT necessarily be loaded directly into a
quantum circuit.

Quantum hardware/simulation has encoding and qubit constraints.

Therefore create a preprocessing layer:

``` text
large heterogeneous SCM
        |
        v
target subgraph / neighborhood
        |
        v
feature reduction
        |
        v
quantum-compatible representation
        |
        v
QGNN
```

The classical GNN can operate on the full graph or the same selected
subgraph depending on the experiment.

If comparing computational efficiency, report the preprocessing cost
separately.

------------------------------------------------------------------------

# 32. Core Experiments

## Experiment 1 --- Classical baselines

Compare:

``` text
MLP
GCN
GraphSAGE
GAT
```

## Experiment 2 --- QGNN comparison

Compare:

``` text
best classical GNN
vs
QGNN
```

## Experiment 3 --- Black-swan augmentation

Compare:

``` text
normal training data
vs
normal + synthetic black-swan data
```

## Experiment 4 --- Severity generalization

Train on:

``` text
severity 1–3
```

Test on:

``` text
severity 4–5
```

## Experiment 5 --- Unseen event generalization

Train on selected event families and test on another.

## Experiment 6 --- Synthetic realism ablation

Compare:

``` text
random synthetic data
vs
constraint-based synthetic data
vs
realism-calibrated synthetic data
```

This experiment can become one of the strongest parts of the paper.

------------------------------------------------------------------------

# 33. Evaluation Metrics

For rare-event classification, accuracy must NOT be the primary metric.

Report:

-   Precision
-   Recall
-   F1
-   ROC-AUC
-   PR-AUC
-   balanced accuracy
-   confusion matrix

For operational prediction:

-   mean absolute error
-   root mean square error
-   calibration error
-   early warning time

For cascade behavior:

-   affected-node count error
-   cascade severity error
-   recovery-time error

------------------------------------------------------------------------

# 34. Statistical Evaluation

Use multiple seeds.

Where appropriate, use statistical tests or confidence intervals.

Report:

``` text
model
mean
std
confidence interval
```

Do not make claims of superiority from a tiny numerical difference
without uncertainty analysis.

------------------------------------------------------------------------

# 35. Synthetic-vs-Real Validation

The synthetic generator should have its own evaluation independent of
the GNN.

## Statistical validation

Compare:

-   mean
-   variance
-   quantiles
-   distributions
-   correlations

## Structural validation

Compare:

-   degree distribution
-   supplier concentration
-   component sizes
-   graph density
-   clustering
-   path lengths where applicable

## Operational validation

Check:

-   capacity feasibility
-   inventory feasibility
-   order feasibility
-   lead-time consistency
-   production constraints

## Event validation

Check that:

-   severity increases impact
-   higher dependency increases vulnerability
-   multi-source supply reduces vulnerability
-   recovery returns systems toward normal
-   geographically isolated events do not affect unrelated regions
    without a modeled path

------------------------------------------------------------------------

# 36. Sanity Tests

Write automated tests before trusting generated data.

Examples:

``` text
No negative inventory unless explicitly allowed
No production above capacity
No delivery before lead time
No impossible edge types
Every procurement references valid supplier/material/plant
Every product references valid plant/material relationships
Every event references valid nodes/regions
```

Also test monotonic behaviors.

Example:

If all else is equal:

``` text
supplier failure severity 5
```

should not result in less disruption than:

``` text
supplier failure severity 1
```

unless a documented stochastic mechanism explains it.

------------------------------------------------------------------------

# 37. Code Architecture

Recommended repository structure:

``` text
scm_dataset/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
│
├── configs/
│   ├── base.yaml
│   ├── real_calibration.yaml
│   ├── normal.yaml
│   ├── black_swan.yaml
│   └── experiments.yaml
│
├── src/
│   └── scm_dataset/
│       ├── __init__.py
│       │
│       ├── schema/
│       │   ├── nodes.py
│       │   ├── edges.py
│       │   ├── events.py
│       │   └── labels.py
│       │
│       ├── real_data/
│       │   ├── loader.py
│       │   ├── cleaner.py
│       │   ├── mapper.py
│       │   └── calibrator.py
│       │
│       ├── generator/
│       │   ├── topology.py
│       │   ├── suppliers.py
│       │   ├── procurement.py
│       │   ├── materials.py
│       │   ├── plants.py
│       │   ├── products.py
│       │   └── regions.py
│       │
│       ├── operations/
│       │   ├── demand.py
│       │   ├── inventory.py
│       │   ├── procurement.py
│       │   ├── production.py
│       │   └── logistics.py
│       │
│       ├── events/
│       │   ├── base.py
│       │   ├── supplier_failure.py
│       │   ├── natural_disaster.py
│       │   ├── geopolitical.py
│       │   ├── cyberattack.py
│       │   └── logistics.py
│       │
│       ├── simulation/
│       │   ├── engine.py
│       │   ├── cascade.py
│       │   ├── recovery.py
│       │   └── state.py
│       │
│       ├── labels/
│       │   └── risk_labels.py
│       │
│       ├── validation/
│       │   ├── statistical.py
│       │   ├── topology.py
│       │   ├── constraints.py
│       │   └── report.py
│       │
│       └── export/
│           ├── csv.py
│           ├── parquet.py
│           └── metadata.py
│
├── scripts/
│   ├── download_real_data.py
│   ├── calibrate.py
│   ├── generate_normal.py
│   ├── generate_black_swan.py
│   ├── validate_dataset.py
│   └── build_benchmark.py
│
├── tests/
│   ├── test_schema.py
│   ├── test_topology.py
│   ├── test_constraints.py
│   ├── test_events.py
│   ├── test_cascade.py
│   └── test_reproducibility.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── generated/
│   └── benchmark/
│
└── reports/
    ├── validation/
    └── figures/
```

------------------------------------------------------------------------

# 38. Configuration-First Design

Do not hard-code assumptions.

Example:

``` yaml
seed: 42

network:
  suppliers: 300
  procurement_orders: 2000
  materials: 100
  plants: 50
  products: 200
  regions: 20

simulation:
  days: 1095
  timestep: daily

events:
  supplier_failure:
    probability: 0.02
  natural_disaster:
    probability: 0.01
  geopolitical:
    probability: 0.005
  cyberattack:
    probability: 0.01
  logistics:
    probability: 0.015
```

Exact probabilities must be treated as experimental parameters, not
factual real-world probabilities.

------------------------------------------------------------------------

# 39. Reproducibility

Every generated dataset must save:

``` yaml
dataset_id:
generator_version:
git_commit:
seed:
config_hash:
generation_timestamp:
real_data_sources:
software_versions:
```

A dataset generated from the same:

``` text
code version
+
config
+
seed
```

should be reproducible.

------------------------------------------------------------------------

# 40. Data Versioning

Recommended naming:

``` text
scm_v1_normal_seed42
scm_v1_black_swan_seed42
scm_v1_mixed_seed42
```

Do not overwrite previous benchmark versions.

------------------------------------------------------------------------

# 41. Dataset Generation Pipeline

The main command should conceptually support:

``` bash
python scripts/generate_normal.py --config configs/normal.yaml --seed 42
```

Then:

``` bash
python scripts/generate_black_swan.py \
    --input data/generated/scm_v1_normal_seed42 \
    --config configs/black_swan.yaml \
    --seed 42
```

Then:

``` bash
python scripts/validate_dataset.py \
    --input data/generated/scm_v1_black_swan_seed42
```

Finally:

``` bash
python scripts/build_benchmark.py \
    --input data/generated \
    --output data/benchmark
```

------------------------------------------------------------------------

# 42. Development Order

Do NOT implement everything at once.

## Phase 1 --- Schema

Implement:

-   six node types
-   edge types
-   validation
-   serialization

Deliverable:

``` text
valid SCM graph
```

## Phase 2 --- Topology generator

Implement:

-   supplier generation
-   material generation
-   plant generation
-   product generation
-   region generation
-   procurement generation

Deliverable:

``` text
heterogeneous SCM graph
```

## Phase 3 --- Normal operations

Implement:

-   demand
-   procurement
-   inventory
-   delivery
-   production

Deliverable:

``` text
time-varying normal SCM
```

## Phase 4 --- Real-data calibration

Add:

-   NIST loader
-   preprocessing
-   distribution estimation
-   topology comparison

Deliverable:

``` text
calibrated synthetic generator
```

## Phase 5 --- Black-swan events

Implement one event first:

``` text
supplier failure
```

Then add:

-   natural disaster
-   geopolitical disruption
-   cyberattack
-   logistics disruption

Deliverable:

``` text
event trajectories
```

## Phase 6 --- Cascade simulation

Implement:

-   material shortage
-   production reduction
-   product shortage
-   recovery

Deliverable:

``` text
ground-truth cascade labels
```

## Phase 7 --- Validation

Implement:

-   statistical tests
-   topology tests
-   operational constraints
-   sanity checks

Deliverable:

``` text
validation report
```

## Phase 8 --- Benchmark export

Create:

-   train
-   validation
-   test
-   scenario-specific splits

Deliverable:

``` text
reproducible benchmark
```

Only after these phases should the GNN/QGNN models be integrated.

------------------------------------------------------------------------

# 43. What the Coding Agent Must NOT Do

The implementation agent must NOT:

1.  Generate arbitrary independent random values for every feature.
2.  Create labels directly from the same risk feature used as model
    input.
3.  Randomly split correlated rows from the same future trajectory into
    train and test.
4.  Claim synthetic data is real data.
5.  Claim black-swan probabilities are empirically measured unless
    supported by evidence.
6.  Claim quantum advantage before experimental evidence exists.
7.  Copy external code without checking its license.
8.  silently modify the six-node ontology.
9.  remove operational constraints just to make generation easier.
10. optimize the generator for a particular model's expected result.
11. create an unrealistically easy classification target.
12. leak event outcomes into pre-event features.

------------------------------------------------------------------------

# 44. Research Integrity Rules

The benchmark should be designed so that the answer is not
predetermined.

For example, the generator must NOT intentionally create graph patterns
that make QGNN perform better.

Both GNN and QGNN must receive the same underlying information.

The simulator should represent the operational logic independently of
the ML model.

------------------------------------------------------------------------

# 45. Suggested Research Questions

Primary:

> **RQ1: Can a realism-constrained synthetic supply-chain environment
> generate useful rare-event scenarios for supplier disruption risk
> prediction?**

Secondary:

> **RQ2: Does synthetic black-swan augmentation improve detection of
> rare supply-chain disruptions?**

> **RQ3: How does a QGNN compare with classical GNN architectures for
> cascading supplier-risk prediction under rare-event conditions?**

> **RQ4: How well do models trained on moderate disruptions generalize
> to extreme/black-swan disruptions?**

> **RQ5: How sensitive are model results to the realism and diversity of
> the synthetic generator?**

------------------------------------------------------------------------

# 46. Hypotheses

Possible hypotheses:

### H1

Realism-calibrated synthetic data produces more realistic SCM
graph/operational statistics than unconstrained random synthetic data.

### H2

Adding synthetic rare-event scenarios improves recall/PR-AUC for rare
disruption prediction.

### H3

Graph-based models outperform non-graph baselines when downstream
dependency structure is important.

### H4

QGNN performance differs from classical GNN performance under rare-event
conditions; the direction and magnitude must be empirically determined.

### H5

Training with a diverse set of disruption scenarios improves
generalization to unseen severe scenarios.

------------------------------------------------------------------------

# 47. Important Ablation Study

One of the strongest possible experiments:

``` text
A: Random synthetic
B: Constraint-based synthetic
C: Realism-calibrated synthetic
D: Realism-calibrated + black-swan augmentation
```

Evaluate the same GNN/QGNN models on all four.

This separates:

-   benefit of graph structure
-   benefit of realism
-   benefit of black-swan augmentation
-   benefit of quantum/classical architecture

------------------------------------------------------------------------

# 48. Another Important Ablation

Dependency structure:

``` text
single-source
vs
multi-source
```

Hypothesis:

Single-source materials should be more vulnerable to supplier
disruption.

Test whether the models actually learn this.

------------------------------------------------------------------------

# 49. Black-Swan Generalization Experiment

Create a held-out event family.

Example:

\`\`\`text TRAIN: supplier failure natural disaster logistics disruption

TEST: cyberattack geopolitical disruption


    This tests whether the model learned general disruption propagation rather than memorizing event labels.

    ---

    # 50. Expected Final Dataset

    A benchmark release should contain:

    ```text
    1. graph structure
    2. node features
    3. edge features
    4. temporal operational data
    5. event definitions
    6. event timelines
    7. cascade states
    8. ground-truth labels
    9. train/validation/test splits
    10. generation configuration
    11. random seeds
    12. data provenance
    13. validation reports
    14. documentation

Prefer CSV/Parquet for tabular data and a graph-friendly format for
graph structures.

------------------------------------------------------------------------

# 51. Expected Figures for the Paper

The generator should automatically produce figures useful for
publication.

## Figure 1 --- SCM graph

Six node types and their relationships.

## Figure 2 --- Dataset-generation pipeline

``` text
Real data
   ↓
Calibration
   ↓
Synthetic SCM
   ↓
Black-swan injection
   ↓
Cascade
   ↓
Ground truth
```

## Figure 3 --- Real vs synthetic distributions

For major calibrated variables.

## Figure 4 --- Black-swan cascade

Show one event propagating through the network.

## Figure 5 --- Model architecture

GNN vs QGNN.

## Figure 6 --- Performance comparison

GNN/QGNN/baselines.

## Figure 7 --- Generalization

Performance versus event severity.

------------------------------------------------------------------------

# 52. Publication-Oriented Requirements

Before claiming the benchmark is complete, ensure:

-   generation process is documented
-   external data sources are cited
-   licenses are recorded
-   synthetic data is clearly labeled
-   random seeds are available
-   code can regenerate benchmark
-   train/test splits are reproducible
-   evaluation metrics are predefined
-   multiple runs are reported
-   limitations are explicitly documented
-   no proprietary dataset is redistributed
-   no external code is copied without appropriate licensing

------------------------------------------------------------------------

# 53. Limitations to Acknowledge

The final paper should explicitly acknowledge:

1.  Synthetic data cannot reproduce every real SCM behavior.
2.  Rare-event definitions are operational modeling assumptions.
3.  Public SCM datasets may have limited temporal information.
4.  Supplier networks can be proprietary and incomplete.
5.  Event propagation rules are simplified representations of real
    organizations.
6.  Quantum simulation results may not translate directly to practical
    quantum hardware.
7.  QGNN comparisons must distinguish algorithmic performance from
    hardware/runtime performance.
8.  Synthetic-data realism does not automatically imply real-world
    predictive validity.

------------------------------------------------------------------------

# 54. Final Implementation Philosophy

The system should follow this principle:

> **Generate realistic structure first, simulate realistic operations
> second, inject rare events third, derive consequences fourth, and only
> then train ML models.**

Never reverse this order to make the model perform better.

The generator is an independent scientific artifact.

The ML model is a consumer of the generated benchmark.

------------------------------------------------------------------------

# 55. Definition of Done

The dataset framework is considered ready for research experiments only
when:

### Graph

-   [ ] Six node types implemented.
-   [ ] All edge types validated.
-   [ ] No invalid references.
-   [ ] Configurable graph size.
-   [ ] Multi-source and single-source relationships supported.

### Normal operations

-   [ ] Demand implemented.
-   [ ] Procurement implemented.
-   [ ] Inventory implemented.
-   [ ] Delivery implemented.
-   [ ] Production implemented.
-   [ ] Capacity constraints enforced.

### Real data

-   [ ] NIST dataset ingestion implemented where useful.
-   [ ] Data provenance recorded.
-   [ ] Calibration process documented.
-   [ ] Real-vs-synthetic validation implemented.

### Black swans

-   [ ] Supplier failure implemented.
-   [ ] Natural disaster implemented.
-   [ ] Geopolitical disruption implemented.
-   [ ] Cyberattack implemented.
-   [ ] Logistics disruption implemented.
-   [ ] Severity levels implemented.
-   [ ] Recovery implemented.

### Labels

-   [ ] Supplier disruption labels.
-   [ ] Material shortage labels.
-   [ ] Plant impact labels.
-   [ ] Product impact labels.
-   [ ] Cascade severity labels.
-   [ ] Future-information leakage checked.

### Validation

-   [ ] Statistical validation.
-   [ ] Topology validation.
-   [ ] Constraint validation.
-   [ ] Event sanity tests.
-   [ ] Reproducibility tests.

### Benchmark

-   [ ] Scenario splits.
-   [ ] Temporal splits.
-   [ ] Black-swan test set.
-   [ ] Multiple seeds.
-   [ ] Versioned configs.
-   [ ] Complete metadata.

Only after all of these are satisfied should the project proceed to the
final GNN/QGNN comparison.

------------------------------------------------------------------------

# 56. Immediate Next Task for Codex

**Do not start by implementing the complete generator.**

First perform a repository and research reconnaissance:

1.  Inspect the existing repository.
2.  Inspect Stanford SupplySim and relevant code.
3.  Identify which ideas can be reused conceptually.
4.  Inspect the NIST sample purchasing dataset structure.
5.  Determine the exact fields available.
6.  Map available real fields to the six-node ontology.
7.  Identify missing fields that must be simulated.
8.  Propose distributions/dependencies for missing fields.
9.  Produce a short `DATASET_DESIGN_REVIEW.md`.
10. Wait for approval before implementing the generator.

After approval, implement the generator incrementally according to
Section 42.

------------------------------------------------------------------------

# 57. Key External References

Stanford Supply Chains / SupplySim:

https://github.com/snap-stanford/supply-chains

NIST Sample Purchasing / Supply Chain Data:

https://catalog.data.gov/dataset/sample-purchasing-supply-chain-data

The Stanford repository describes both its released synthetic SupplySim
datasets and the preprocessing pipeline for transaction-level
supply-chain data. The NIST dataset is a public purchasing dataset
covering suppliers, products and projects.

For literature review, also examine recent work on graph-based
supply-chain disruption prediction and rare-event augmentation,
including:

https://doi.org/10.1145/3801228.3801250

This recent work is relevant because it explicitly studies scarce
black-swan samples using a spatio-temporal graph adversarial approach
followed by dynamic graph attention risk prediction.

------------------------------------------------------------------------

# 58. Final Architecture

The intended end-to-end architecture is:

``` text
                         PUBLIC REAL DATA
                                |
                +---------------+---------------+
                |                               |
         Procurement data                Network data
                |                               |
                +---------------+---------------+
                                |
                                v
                    REALISM CALIBRATION LAYER
                                |
                                v
                     SYNTHETIC SCM GENERATOR
                                |
                 +--------------+--------------+
                 |                             |
                 v                             v
          Normal operations             Graph topology
                 |                             |
                 +--------------+--------------+
                                |
                                v
                      BLACK-SWAN EVENT ENGINE
                                |
            +-------------------+-------------------+
            |                   |                   |
            v                   v                   v
        Disaster          Geopolitical          Cyber
            |                   |                   |
            +-------------------+-------------------+
                                |
                                v
                       CASCADE SIMULATOR
                                |
             +------------------+------------------+
             |                  |                  |
             v                  v                  v
        Supplier risk      Plant impact       Product impact
             |                  |                  |
             +------------------+------------------+
                                |
                                v
                       RESEARCH BENCHMARK
                                |
                 +--------------+--------------+
                 |                             |
                 v                             v
             Classical GNN                  QGNN
                 |                             |
                 +--------------+--------------+
                                |
                                v
                       FAIR EVALUATION
                                |
              +-----------------+-----------------+
              |                 |                 |
              v                 v                 v
           Accuracy        Robustness       Generalization
```

**Primary principle:** the dataset/simulator must remain model-agnostic.
GNN and QGNN are downstream consumers of the benchmark, not components
that determine how the data is generated.
