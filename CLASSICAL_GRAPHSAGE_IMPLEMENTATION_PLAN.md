# Classical GraphSAGE Implementation Plan
## Supply-Chain Black-Swan Disruption Prediction Benchmark

> **Implementation target:** Classical Graph Neural Network using **GraphSAGE**
>
> **Primary task:** Predict future supplier disruption from the existing heterogeneous supply-chain benchmark.
>
> **Later comparison:** The resulting frozen classical GraphSAGE benchmark will be compared with a QGNN under the same research protocol.

---

# 1. Executive Summary

The supply-chain dataset generation framework is already complete.

The existing framework provides:

- 6 node types
- 9 edge types
- graph topology
- 104 weekly operational periods
- normal operations
- black-swan events
- cascade simulation
- ground-truth labels
- validation
- temporal/scenario/severity benchmark splits
- feature leakage audit
- dataset provenance

The next stage is **not dataset generation**.

The next stage is:

```text
Existing SCM Benchmark
        ↓
Leakage-Safe Feature Construction
        ↓
Heterogeneous Graph Representation
        ↓
GraphSAGE
        ↓
Supplier Embeddings
        ↓
Supplier Disruption Probability
        ↓
Classification + Risk Evaluation
        ↓
Frozen Classical Baseline
        ↓
Later QGNN Comparison
```

The first classical model should be deliberately understandable and reproducible.

Use:

> **2-layer Heterogeneous GraphSAGE**

Do not begin with GAT, Graph Transformer, or a complex multi-task architecture.

---

# 2. Research Question

The initial research question is:

> **Can a GraphSAGE model learn supply-chain graph and operational patterns that provide early prediction of future supplier disruptions under normal and black-swan conditions?**

The later comparative question becomes:

> **How does a classical GraphSAGE model compare with a QGNN for rare-event supplier disruption prediction under identical supply-chain benchmark conditions?**

Do not assume that the QGNN will outperform GraphSAGE.

The research result may favor:

- GraphSAGE
- QGNN
- or show comparable performance.

---

# 3. Primary Prediction Task

## Supplier Disruption Prediction

At a prediction time `t`, use all information that is legitimately available at `t` to predict whether supplier disruption occurs within a future horizon `H`.

Recommended first experiment:

```text
H = 4 weeks
```

This must remain configurable.

Example:

```yaml
prediction:
  horizon_periods: 4
```

Conceptually:

```text
Historical information
weeks 1 ─────────────── t
                         │
                         ▼
                    GraphSAGE
                         │
                         ▼
              P(disruption in
                 next 4 weeks)
```

---

# 4. What Is the Model Output?

The model should produce a **continuous probability**, not just a binary label.

For supplier `SUP_279`:

```text
risk_probability = 0.91
```

Then:

```text
risk_probability >= threshold
            ↓
predicted_disruption = 1
```

Example final prediction:

```text
supplier_id: supplier_279
time: 2025-W40
risk_probability: 0.913
predicted_disruption: 1
actual_disruption: 1
```

The raw probability must always be preserved.

---

# 5. Why Supplier Disruption Is the First Target

The existing benchmark contains labels for:

```text
Supplier → supplier_disrupted
Material → material_shortage
Plant → plant_disruption
Product → product_shortage
```

The first model should focus on:

```text
Supplier → supplier_disrupted
```

because this provides a clean binary classification task for establishing the classical baseline.

After the supplier task is stable, additional tasks can be investigated.

Do not implement all tasks simultaneously in the first version.

---

# 6. Existing Graph

The existing graph contains 6 node types.

```text
Supplier
Material
Plant
Product
Region
ProcurementOrder
```

Existing edge types:

```text
SUPPLIER_MATERIAL
SUPPLIER_PROCUREMENT
PROCUREMENT_MATERIAL
PROCUREMENT_PLANT
MATERIAL_PLANT
PLANT_PRODUCT
PRODUCT_REGION
SUPPLIER_REGION
PLANT_REGION
```

The model must preserve these semantic node and edge types.

---

# 7. Conceptual Supply-Chain Graph

The graph should represent relationships such as:

```text
                    REGION
                   /      \
                  /        \
            SUPPLIER       PLANT
               │             │
               │             │
               ▼             ▼
           MATERIAL       PRODUCT
               ▲
               │
        PROCUREMENT ORDER
```

A more complete dependency path is:

```text
SUPPLIER
    │
    │ supplies
    ▼
MATERIAL
    │
    │ required by
    ▼
PLANT
    │
    │ produces
    ▼
PRODUCT
    │
    │ associated with
    ▼
REGION
```

Procurement relationships provide operational transaction context:

```text
SUPPLIER
    │
    ▼
PROCUREMENT ORDER
    ├──────────────► MATERIAL
    │
    └──────────────► PLANT
```

---

# 8. Why Graph Structure Matters

A traditional tabular model may only see:

```text
Supplier 279
capacity = 0.72
reliability = 0.81
financial_health = 0.64
```

GraphSAGE can additionally learn from the supplier's neighborhood:

```text
Supplier 279
      │
      ├── Material 75
      │       │
      │       └── Plant 4
      │               │
      │               └── Product 42
      │
      ├── Material 31
      │       │
      │       └── Plant 8
      │
      └── Region 4
```

This allows the model to use dependency structure.

For example, a supplier may become more important because:

```text
high criticality material
+
few alternative suppliers
+
many dependent plants
+
high regional exposure
```

The GNN can learn these patterns through neighborhood aggregation.

---

# 9. Recommended Framework

Use:

> **PyTorch Geometric (PyG)**

Preferred graph representation:

```text
torch_geometric.data.HeteroData
```

Do not flatten the heterogeneous graph unless implementation constraints make that unavoidable.

The graph should preserve:

```text
node type
edge type
edge direction
node features
```

---

# 10. Heterogeneous Graph Representation

Conceptual PyG structure:

```text
HeteroData(
    supplier={
        x=...
    },
    material={
        x=...
    },
    plant={
        x=...
    },
    product={
        x=...
    },
    region={
        x=...
    },
    procurement={
        x=...
    },

    edge_index_dict={
        ...
    }
)
```

The exact implementation must use the actual repository schema rather than assumed column names.

---

# 11. Node Feature Design

The model must use features that are available at prediction time.

Do not automatically use every column in the CSV.

First inspect:

```text
feature_audit.csv
```

Only permitted features may enter the model.

---

# 12. Supplier Features

Existing supplier fields include:

```text
tier
industry
capacity
capacity_utilization
reliability
financial_health
lead_time_mean
lead_time_variability
quality_score
inventory_buffer
substitution_availability
geopolitical_exposure
disaster_exposure
cyber_exposure
criticality
region_id
```

Potential historical features can include:

```text
rolling fulfillment ratio
rolling delivery delay
rolling lead time
rolling order volume
rolling order frequency
utilization trend
recent operational volatility
```

All historical features must be constructed using information available through `t`.

---

# 13. Material Features

Existing fields include:

```text
material_category
criticality
substitutability
demand
unit_cost
inventory_level
safety_stock
supplier_count
concentration
required_quantity_per_product
procurement_type
```

Potential historical features:

```text
inventory trend
inventory volatility
recent shortage frequency
recent demand trend
supplier reliability aggregate
```

Again, only use information available through the prediction time.

---

# 14. Plant Features

Existing fields:

```text
production_capacity
utilization
operating_cost
inventory_capacity
resilience_score
downtime_cost
recovery_rate
```

Potential historical features:

```text
recent production
production volatility
utilization trend
inventory pressure
backlog pressure
production-loss history
```

---

# 15. Product Features

Existing fields:

```text
product_category
demand
revenue_per_unit
margin
material_dependency
criticality
substitution_score
backlog
```

Potential historical features:

```text
demand trend
backlog trend
recent shortage history
```

---

# 16. Region Features

Existing contextual fields:

```text
geopolitical_risk
natural_disaster_risk
infrastructure_risk
trade_risk
cyber_risk
transport_reliability
```

These provide geographic context.

Do not provide:

```text
future affected_regions
future event_type
future event severity
future cascade information
```

to the model.

---

# 17. Procurement Features

Existing fields:

```text
order_quantity
order_value
order_frequency
promised_lead_time
actual_lead_time
urgency
priority
contract_duration
```

For supplier prediction, useful historical aggregations may include:

```text
orders_last_4_weeks
orders_last_8_weeks
mean_order_quantity
mean_order_value
mean_actual_lead_time
mean_delivery_delay
fulfillment_ratio
urgent_order_ratio
```

All aggregation windows must end at `t`.

---

# 18. Temporal Feature Construction

This is one of the most important implementation details.

Do NOT create:

```text
one row = one weekly observation
```

and randomly split rows.

That would cause temporal leakage and correlated samples across train/test.

Instead define:

```text
prediction example =
supplier + prediction_time
```

Example:

```text
Supplier:
SUP_279

Prediction time:
week 60

Features:
information from weeks 1–60

Target:
disruption during weeks 61–64
```

---

# 19. Historical Windows

Make history windows configurable.

Recommended:

```yaml
features:
  rolling_windows:
    - 4
    - 8
    - 12
```

For a 4-week window:

```text
mean_4w
std_4w
trend_4w
```

For an 8-week window:

```text
mean_8w
std_8w
trend_8w
```

For a 12-week window:

```text
mean_12w
std_12w
trend_12w
```

Do not automatically generate a huge number of features.

---

# 20. Target Definition

The preferred target is:

```text
Will the supplier be disrupted at any point during the next H periods?
```

For horizon `H`:

```text
Y(t) =
    1 if disruption occurs in t+1 ... t+H
    0 otherwise
```

Equivalent:

```text
Y(t) = max(
    supplier_disrupted[t+1],
    ...
    supplier_disrupted[t+H]
)
```

This must be implemented consistently.

The exact target construction should be validated against the existing label semantics before coding.

---

# 21. Label Alignment

For every training example:

```text
supplier_id
prediction_time
features_up_to_prediction_time
future_target
```

Example:

```text
SUP_279
week_60
X(weeks <= 60)
Y(weeks 61–64)
```

Never accidentally align:

```text
X(week 60)
Y(week 60)
```

unless the research task explicitly defines same-time classification.

The intended task is **future prediction**.

---

# 22. Leakage Audit

Implement an automated test that verifies:

```text
max(feature_timestamp) <= prediction_time
```

for every dynamic feature.

Also verify:

```text
target_timestamp > prediction_time
```

for the prediction horizon.

The model pipeline should fail loudly if a feature violates these rules.

---

# 23. Static vs Dynamic Features

Separate features into:

### Static

Examples:

```text
industry
tier
criticality
capacity
resilience_score
```

### Dynamic

Examples:

```text
inventory
demand
production
delivery
lead time
backlog
utilization
fulfillment
```

Dynamic features require timestamp-aware construction.

This separation should exist in the implementation.

---

# 24. Categorical Features

Potential categorical fields:

```text
industry
material_category
product_category
procurement_type
tier
```

Use a deterministic encoding strategy.

Do not fit categorical mappings using test data.

Recommended approach:

```text
fit encoder on training data
apply unchanged to validation/test
```

Unknown categories should have an explicit handling mechanism.

---

# 25. Numerical Scaling

Continuous features may have dramatically different ranges:

```text
capacity
0–1000

reliability
0–1

lead time
1–100

order value
0–100000
```

Scale appropriate continuous features.

Fit scalers using the training split only.

Then apply:

```text
train scaler → validation
train scaler → test
```

Never fit the scaler on the complete dataset.

---

# 26. Missing Values

Never insert arbitrary random values.

Use deterministic handling.

Example:

```text
missing_value_indicator = 1
imputed_value = training_median
```

For categorical values:

```text
UNKNOWN
```

should be handled explicitly if necessary.

Document every imputation rule.

---

# 27. GraphSAGE Architecture

The first model should be:

> **2-layer Heterogeneous GraphSAGE**

Conceptual architecture:

```text
                Input Graph
                    │
       ┌────────────┴────────────┐
       │                         │
Node features              Typed edges
       │                         │
       └────────────┬────────────┘
                    ▼
        Type-specific projection
                    │
                    ▼
        Heterogeneous GraphSAGE
               Layer 1
                    │
                    ▼
        Heterogeneous GraphSAGE
               Layer 2
                    │
                    ▼
            Supplier embeddings
                    │
                    ▼
                 MLP
                    │
                    ▼
                 Logit
                    │
                    ▼
               Sigmoid
                    │
                    ▼
         Disruption probability
```

---

# 28. GraphSAGE Concept

GraphSAGE learns an embedding by combining a node's own features with information aggregated from neighboring nodes.

Conceptually:

```text
Node representation
        +
Neighbor representations
        ↓
Aggregation
        ↓
Transformation
        ↓
New node representation
```

With two layers, information can propagate approximately two hops through the graph.

For example:

```text
Supplier
   ↓
Material
   ↓
Plant
```

can allow the supplier representation to incorporate plant-level information through the graph.

---

# 29. Aggregation

The initial GraphSAGE implementation should use a standard aggregation such as:

```text
mean aggregation
```

Do not immediately introduce complicated attention mechanisms.

The goal is to establish a clear classical baseline.

---

# 30. Initial Model Hyperparameters

Recommended initial configuration:

```yaml
model:
  name: hetero_graphsage
  hidden_dim: 128
  num_layers: 2
  dropout: 0.20
```

Training:

```yaml
training:
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 10
  class_weighting: balanced
```

These are starting values, not claimed optimal values.

---

# 31. Type-Specific Input Projection

Different node types have different feature dimensions.

Therefore use type-specific projections:

```text
Supplier features
      ↓
Supplier projection
      ↓
128 dimensions

Material features
      ↓
Material projection
      ↓
128 dimensions

Plant features
      ↓
Plant projection
      ↓
128 dimensions

...
```

This allows GraphSAGE layers to operate on a shared hidden representation while preserving node-type-specific input structure.

---

# 32. Supplier Readout

Only supplier nodes are classified in the first task.

After message passing:

```text
supplier_embedding
        ↓
MLP
        ↓
logit
        ↓
sigmoid
        ↓
risk probability
```

Example:

```text
embedding = [0.21, -0.13, ..., 0.44]

MLP
 ↓

logit = 2.17

sigmoid
 ↓

0.897
```

Output:

```text
P(disruption) = 0.897
```

---

# 33. Loss Function

Use:

```text
BCEWithLogitsLoss
```

Do not apply sigmoid before passing values to the loss if using `BCEWithLogitsLoss`.

Conceptually:

```text
model → logits
             ↓
BCEWithLogitsLoss
```

For prediction:

```text
logits
  ↓
sigmoid
  ↓
probability
```

---

# 34. Class Imbalance

Black-swan disruption is expected to be a rare event.

Before training calculate:

```text
positive_count
negative_count
positive_rate
```

If class imbalance is substantial, support:

```text
pos_weight
```

Example concept:

```text
pos_weight =
negative_training_examples /
positive_training_examples
```

Calculate this using **training labels only**.

Do not calculate class weights using validation/test data.

---

# 35. Training Procedure

The complete training procedure should be:

```text
Load benchmark
      ↓
Validate schema
      ↓
Construct prediction examples
      ↓
Leakage audit
      ↓
Apply benchmark split
      ↓
Fit preprocessing on train
      ↓
Construct HeteroData
      ↓
Initialize GraphSAGE
      ↓
Train
      ↓
Validate
      ↓
Early stopping
      ↓
Restore best checkpoint
      ↓
Evaluate frozen test set
```

---

# 36. Early Stopping

Use:

```text
Validation PR-AUC
```

as the primary early-stopping metric.

Example:

```text
Epoch 1  PR-AUC = 0.61
Epoch 2  PR-AUC = 0.65
Epoch 3  PR-AUC = 0.68
...
Epoch 17 PR-AUC = 0.79
Epoch 18 PR-AUC = 0.78
...
```

Save the best checkpoint:

```text
best_model.pt
```

Do not select the model using test performance.

---

# 37. Train / Validation / Test

Use the existing benchmark split files.

Do not randomly split trajectory rows.

Required split types:

```text
Temporal
Scenario
Severity
```

---

# 38. Temporal Experiment

Example:

```text
Earlier weeks
      ↓
TRAIN

Later weeks
      ↓
VALIDATION

Latest weeks
      ↓
TEST
```

This approximates real deployment:

> learn from the past → predict the future.

---

# 39. Severity Generalization Experiment

One of the strongest experiments for the paper:

```text
TRAIN:
severity 1–3

TEST:
severity 4–5
```

Question:

> Can GraphSAGE trained primarily on moderate disruptions identify risk under extreme events?

This should be a separate reported experiment.

---

# 40. Scenario Generalization

Use held-out scenarios to test whether the model generalizes beyond the exact scenarios seen during training.

Do not leak scenario identifiers that directly reveal target outcomes.

---

# 41. Primary Evaluation Metrics

Calculate:

```text
PR-AUC
ROC-AUC
Precision
Recall
F1
Balanced Accuracy
Confusion Matrix
```

Primary metrics:

```text
PR-AUC
F1
```

Why:

Rare-event datasets can produce misleadingly high accuracy when the model predicts the majority class.

---

# 42. Precision

Precision:

```text
TP / (TP + FP)
```

Interpretation:

> Of suppliers predicted as disrupted, how many actually became disrupted?

High precision means fewer false alarms.

---

# 43. Recall

Recall:

```text
TP / (TP + FN)
```

Interpretation:

> Of suppliers that actually became disrupted, how many did the model detect?

For risk management, recall can be particularly important because missing a true disruption can be costly.

---

# 44. F1

F1 combines precision and recall:

```text
F1 = 2 × Precision × Recall /
     (Precision + Recall)
```

Use it as a balanced classification metric.

---

# 45. PR-AUC

PR-AUC is especially important for rare-event detection.

Report it as a primary metric.

Do not rely solely on ROC-AUC.

---

# 46. Probability Calibration

The model outputs:

```text
0.10
0.40
0.75
0.95
```

These should be meaningful probabilities rather than arbitrary ranking scores.

Evaluate:

```text
Brier Score
Expected Calibration Error
Calibration curve
```

Save:

```text
calibration.json
```

and:

```text
calibration_curve.png
```

---

# 47. Threshold Selection

Keep the raw probability.

Default evaluation:

```text
threshold = 0.5
```

Optionally select an operating threshold using validation data.

Possible rules:

```text
F1-optimal threshold
recall-constrained threshold
precision-constrained threshold
```

The threshold must be chosen before final test evaluation.

Never optimize the threshold on test data.

---

# 48. Prediction Output

Create:

```text
predictions.csv
```

with:

```text
supplier_id
time
risk_probability
predicted_disruption
actual_disruption
split
```

Example:

```text
supplier_279,2025-W40,0.913,1,1,test
supplier_031,2025-W40,0.184,0,0,test
supplier_142,2025-W40,0.781,1,0,test
```

---

# 49. Risk Ranking Output

Create an additional analysis file:

```text
supplier_risk_ranking.csv
```

Fields:

```text
rank
supplier_id
time
risk_probability
predicted_disruption
criticality
region_id
```

Sort descending by:

```text
risk_probability
```

This is useful for demonstrations and downstream analysis.

---

# 50. Cascade Analysis

The model's primary output is:

```text
supplier disruption probability
```

After prediction, perform post-hoc cascade analysis.

For high-risk suppliers:

```text
Supplier
    ↓
Materials
    ↓
Plants
    ↓
Products
```

Use existing simulation outcomes to determine:

```text
affected_materials
affected_plants
affected_products
production_loss
material_shortage
revenue_impact
cascade_severity
recovery_time
```

These are **evaluation/analysis outputs**, not model inputs.

---

# 51. Example End-to-End Prediction

Suppose GraphSAGE predicts:

```text
SUP_279
risk_probability = 0.94
```

Graph neighborhood:

```text
SUP_279
 ├── MAT_75
 │    └── PLANT_4
 │         └── PROD_42
 │
 ├── MAT_31
 │    └── PLANT_8
 │
 └── REGION_4
```

Post-hoc simulation ground truth may show:

```text
Supplier disruption: YES
Material shortage: YES
Affected plants: 3
Affected products: 7
Production loss: 38%
Recovery time: 16 weeks
```

The model did not receive those future outcomes.

They are used to evaluate what happened after the prediction point.

---

# 52. Do Not Confuse Prediction With Simulation

The simulator determines:

```text
What actually happens after an event
```

The GNN determines:

```text
What disruption risk can be predicted from information available before it happens
```

Therefore:

```text
Simulation
    ↓
Ground truth

GNN
    ↓
Prediction
```

This separation is essential.

---

# 53. Non-Graph Baseline

Before claiming GraphSAGE adds value, implement:

```text
Majority classifier
Logistic Regression
```

Optional:

```text
MLP
XGBoost
Random Forest
```

The first comparison should be:

```text
Majority
   ↓
Logistic Regression
   ↓
GraphSAGE
```

All should use the same prediction target and test set.

---

# 54. Why Logistic Regression Matters

Suppose results are:

```text
Logistic Regression:
PR-AUC = 0.62

GraphSAGE:
PR-AUC = 0.78
```

This supports the argument that graph structure provides useful predictive information.

If:

```text
Logistic Regression:
PR-AUC = 0.76

GraphSAGE:
PR-AUC = 0.77
```

then the graph provides little additional predictive value under that experiment.

That is still a legitimate research result.

---

# 55. Required Ablations

After the primary model works:

### Ablation A — Remove graph

Supplier-level tabular model.

### Ablation B — Static features only

Remove dynamic operational history.

### Ablation C — Dynamic features only

Remove static/contextual features.

### Ablation D — Remove region risk

Test geographic-risk contribution.

### Ablation E — Single-source vs multi-source

Compare performance on different dependency structures.

---

# 56. Graph Statistics

For every benchmark, record:

```text
node count by type
edge count by type
average degree
degree distribution
isolated nodes
single-source material fraction
multi-source material fraction
supplier concentration
```

This helps explain model performance.

---

# 57. Model Explainability

The first version should not make unsupported causal claims.

Useful post-hoc analysis can include:

```text
supplier risk ranking
neighbor inspection
feature ablation
feature permutation
```

If later using an explanation method, document it separately.

Do not claim:

> "GraphSAGE says this supplier caused the disruption."

Instead:

> "GraphSAGE assigned a high disruption probability based on learned patterns in the supplier's features and graph neighborhood."

---

# 58. Experiment Directory

Use:

```text
experiments/
└── classical_gnn/
    └── <run_id>/
        ├── config.yaml
        ├── model.pt
        ├── predictions.csv
        ├── supplier_risk_ranking.csv
        ├── metrics.json
        ├── classification_report.json
        ├── confusion_matrix.csv
        ├── calibration.json
        ├── training_history.csv
        ├── run_metadata.json
        ├── preprocessing/
        │   ├── numerical_scaler.*
        │   └── categorical_encoder.*
        └── plots/
            ├── pr_curve.png
            ├── roc_curve.png
            ├── calibration_curve.png
            ├── confusion_matrix.png
            └── training_curve.png
```

Never overwrite previous experiments.

---

# 59. Suggested Repository Changes

Keep modeling separate from data generation.

Add:

```text
src/scm_dataset/
└── modeling/
    ├── __init__.py
    ├── data.py
    ├── features.py
    ├── preprocessing.py
    ├── hetero_graph.py
    ├── graphsage.py
    ├── losses.py
    ├── train.py
    ├── evaluate.py
    ├── metrics.py
    └── calibration.py
```

Scripts:

```text
scripts/
├── train_graphsage.py
├── evaluate_graphsage.py
└── run_graphsage_experiment.py
```

Tests:

```text
tests/
├── test_model_data.py
├── test_temporal_features.py
├── test_preprocessing.py
├── test_hetero_graph.py
├── test_graphsage.py
├── test_training.py
└── test_evaluation.py
```

---

# 60. Configuration

Create:

```text
configs/graphsage.yaml
```

Example:

```yaml
dataset:
  benchmark_path: data/benchmark

prediction:
  target: supplier_disrupted
  horizon_periods: 4

features:
  rolling_windows:
    - 4
    - 8
    - 12

model:
  name: hetero_graphsage
  hidden_dim: 128
  num_layers: 2
  dropout: 0.20
  aggregation: mean

training:
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 10
  class_weighting: balanced

experiment:
  seeds:
    - 42
    - 43
    - 44
    - 45
    - 46
```

The actual benchmark path must be confirmed from the repository.

---

# 61. Reproducibility

Every run must record:

```text
dataset_id
dataset version
generator version
dataset git commit
model git commit
seed
config hash
feature list
prediction horizon
split
model architecture
hyperparameters
software versions
hardware
training duration
```

Save this to:

```text
run_metadata.json
```

---

# 62. Multiple Seeds

Do not report only one lucky run.

Recommended:

```text
Seed 42
Seed 43
Seed 44
Seed 45
Seed 46
```

Report:

```text
mean ± standard deviation
```

Example format:

```text
PR-AUC = 0.781 ± 0.018
F1     = 0.742 ± 0.021
```

The numbers above are only examples; calculate actual values from experiments.

---

# 63. Statistical Reporting

For each model, report:

```text
mean
standard deviation
```

across seeds.

For the final GNN vs QGNN comparison, use the same test examples and, where appropriate, paired statistical testing.

Do not call a small numerical difference a meaningful improvement without statistical evidence.

---

# 64. Training History

Save:

```text
epoch
train_loss
validation_loss
validation_pr_auc
validation_f1
validation_roc_auc
```

Example:

```text
1,0.691,0.682,0.61,0.42,0.71
2,0.661,0.645,0.65,0.48,0.75
...
```

This allows training behavior to be reproduced and inspected.

---

# 65. Required Visualizations

Generate at minimum:

## PR Curve

Most important classification visualization.

## ROC Curve

Secondary classification visualization.

## Confusion Matrix

Show:

```text
TP FP
FN TN
```

## Calibration Curve

Show:

```text
predicted probability
vs
observed frequency
```

## Training Curve

Show:

```text
training loss
validation loss
validation PR-AUC
```

---

# 66. Sanity Check — Majority Baseline

If:

```text
95% = no disruption
5% = disruption
```

a model predicting:

```text
no disruption
```

for everything obtains 95% accuracy.

That does not mean the model is useful.

Therefore always compare:

```text
Accuracy
Precision
Recall
F1
PR-AUC
```

and emphasize the rare-event metrics.

---

# 67. Sanity Check — Label Distribution

Before training:

```text
Train positive rate
Validation positive rate
Test positive rate
```

must be printed.

Also report:

```text
number of supplier-time examples
number of positive examples
number of negative examples
```

If the positive class is extremely small, investigate before training.

---

# 68. Sanity Check — Graph Alignment

Verify:

```text
every supplier label maps to exactly one supplier node
```

and:

```text
supplier IDs in labels
==
supplier IDs in graph
```

Do not silently discard unmatched IDs.

---

# 69. Sanity Check — Edge Alignment

Verify every edge:

```text
source node exists
target node exists
edge type is valid
node types match edge semantics
```

Example:

```text
SUPPLIER_MATERIAL
source = Supplier
target = Material
```

If the existing edge file stores reversed directions, adapt to the actual repository semantics and document the decision.

---

# 70. Sanity Check — Temporal Alignment

For a prediction at week 60:

```text
features:
≤ week 60

target:
weeks 61–64
```

Automate this check.

---

# 71. Sanity Check — Future Event Leakage

The model must not receive:

```text
event_type
event severity
affected_regions
affected_suppliers
affected_plants
capacity_reduction_fraction
lead_time_increase_fraction
recovery_delay
cascade_severity
total_affected_nodes
recovery_time
```

if those fields describe the future event being predicted.

These fields belong to:

```text
event generation
ground truth
post-hoc analysis
```

not the predictive input.

---

# 72. Important Issue: `supplier_risk_score`

The existing supplier label table contains:

```text
supplier_disrupted
supplier_risk_score
```

Do not automatically use:

```text
supplier_risk_score
```

as an input feature.

First determine how it is generated.

If it is derived from the event/target/outcome, it is leakage.

If it is independently available before the prediction point, it may be considered, but this must be documented.

The safest initial benchmark is:

> exclude target-derived supplier risk scores from model inputs.

---

# 73. Important Issue: Procurement Order Nodes

Procurement orders are temporal entities.

Do not create artificial future knowledge through their final outcome.

For example, if:

```text
actual_lead_time
quantity_fulfilled
```

is only known after delivery, it cannot be used for a prediction made before delivery.

Only use it in historical aggregates when the actual outcome was already known at time `t`.

This must be enforced by timestamp.

---

# 74. Full Graph vs Temporal Graph

The model needs access to the relevant graph structure.

For static relationships:

```text
supplier → material
material → plant
plant → product
supplier → region
plant → region
```

the graph topology can remain stable.

For temporal operational information:

```text
inventory
production
demand
deliveries
procurement
```

features must change by prediction time.

The implementation should therefore distinguish:

```text
static topology
+
time-indexed features
```

rather than rebuilding arbitrary future graphs.

---

# 75. Recommended First Run

Run exactly one small controlled experiment first.

Configuration:

```text
Model:
2-layer Heterogeneous GraphSAGE

Hidden dimension:
128

Aggregation:
mean

Dropout:
0.20

Learning rate:
0.001

Weight decay:
0.0001

Loss:
BCEWithLogitsLoss

Class weighting:
balanced

Prediction horizon:
4 weeks

Early stopping:
validation PR-AUC

Seed:
42
```

Use the actual benchmark dataset available in the repository.

Do not perform a large hyperparameter search yet.

---

# 76. First Run Acceptance

The first run is successful if:

```text
dataset loads
        ↓
features build
        ↓
graph builds
        ↓
GraphSAGE trains
        ↓
validation predictions work
        ↓
test predictions work
        ↓
metrics are produced
        ↓
plots are produced
        ↓
model checkpoint is saved
        ↓
metadata is saved
```

The result does not need to be excellent in the first run.

The priority is:

> **correctness before performance.**

---

# 77. After First Run

Only after correctness is established:

```text
multiple seeds
      ↓
baseline models
      ↓
ablations
      ↓
hyperparameter tuning
      ↓
severity generalization
      ↓
scenario generalization
```

Do not tune everything at once.

---

# 78. Hyperparameter Experiments

Potential later search:

```text
hidden_dim:
64
128
256

layers:
2
3

dropout:
0.1
0.2
0.3

learning_rate:
0.0001
0.001
0.01
```

Do not conduct a massive search initially.

Use validation performance only for model selection.

---

# 79. What Not to Optimize Against

Never optimize hyperparameters using:

```text
test PR-AUC
test F1
test Recall
test ROC-AUC
```

The test set is the final evaluation.

If you repeatedly inspect test results and modify the model, the test set becomes part of training/selection.

---

# 80. Final Classical Benchmark

After experiments, freeze:

```text
feature definition
target definition
prediction horizon
preprocessing
graph construction
GraphSAGE architecture
training configuration
threshold policy
test split
```

This becomes the **classical benchmark**.

Then the QGNN stage can be developed against the same frozen benchmark.

---

# 81. GNN vs QGNN Fairness Requirements

The later comparison must use:

```text
same dataset
same supplier labels
same prediction horizon
same test examples
same train/validation/test split
same allowed information
same target
same evaluation metrics
```

Do not give the QGNN extra information.

Do not give the classical GNN less information intentionally.

---

# 82. QGNN Input Reduction

The QGNN may not be able to encode the entire SCM graph directly.

A later QGNN implementation may therefore use:

```text
supplier-centered local subgraph
```

such as:

```text
Supplier
  ↓
Materials
  ↓
Plants
  ↓
Products
```

with a limited neighborhood.

This is acceptable only if:

1. the reduction rule is predefined,
2. it does not use future labels,
3. the same target/test examples are preserved,
4. the limitation is reported.

Do not modify the benchmark to make it easier for the QGNN.

---

# 83. Recommended Research Table

The final paper can contain a table like:

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Majority | — | — | — | — | — | — |
| Logistic Regression | — | — | — | — | — | — |
| GraphSAGE | — | — | — | — | — | — |
| QGNN | — | — | — | — | — | — |

Populate with actual experimental values.

Do not invent values.

---

# 84. Severity Generalization Table

A second useful table:

| Model | Train Severity | Test Severity | PR-AUC | F1 | Recall |
|---|---|---|---:|---:|---:|
| GraphSAGE | 1–3 | 4–5 | — | — | — |
| QGNN | 1–3 | 4–5 | — | — | — |

This directly addresses extreme-event generalization.

---

# 85. Cascade-Oriented Analysis

For high-risk supplier predictions, calculate:

```text
mean downstream affected nodes
mean production loss
mean material shortage
mean product shortage
mean recovery time
```

Possible analysis:

```text
Predicted high-risk suppliers
             ↓
Observed downstream impact
```

This connects the ML task back to the supply-chain research problem.

---

# 86. Research Integrity

The following claims are not allowed without evidence:

### Do not say:

> GraphSAGE predicts black-swan events perfectly.

### Say:

> GraphSAGE was evaluated for supplier disruption prediction under the defined synthetic black-swan benchmark.

Do not say:

> The synthetic dataset represents real-world black-swan probabilities.

Say:

> The simulator generates controlled rare-event scenarios calibrated using available reference data where applicable.

Do not say:

> QGNN is better because it is quantum.

Only report measured results.

---

# 87. Testing Requirements

Add unit tests for:

```text
feature timestamp validation
target alignment
node-ID alignment
edge-ID alignment
graph construction
categorical encoding
numerical scaling
missing-value handling
class-weight calculation
GraphSAGE forward pass
loss calculation
prediction generation
metric calculation
```

Add integration test:

```text
benchmark
→ features
→ graph
→ GraphSAGE
→ prediction
→ metrics
```

---

# 88. Existing Test Suite

The dataset framework currently has:

```text
90 tests
```

All existing tests must continue to pass.

After adding the modeling stage:

```text
pytest -q
```

must still pass.

Do not break dataset-generation functionality.

---

# 89. Performance and Scalability

Do not assume the final graph is small.

The current generated dataset is a manageable research graph, but the architecture should support larger future configurations.

Avoid unnecessary:

```text
dense adjacency matrices
```

Use sparse graph representations.

Do not construct:

```text
N × N
```

dense matrices for a large graph.

Use PyTorch Geometric's sparse/edge-index based representation.

---

# 90. Full Experiment Flow

The complete implementation should follow:

```text
                  EXISTING BENCHMARK
                         │
                         ▼
                Schema Validation
                         │
                         ▼
                Feature Audit Check
                         │
                         ▼
             Temporal Feature Builder
                         │
                         ▼
                Train/Val/Test Split
                         │
                         ▼
                Train-only Preprocess
                         │
                         ▼
                  HeteroData Graph
                         │
                         ▼
               GraphSAGE Layer 1
                         │
                         ▼
               GraphSAGE Layer 2
                         │
                         ▼
                Supplier Embeddings
                         │
                         ▼
                       MLP
                         │
                         ▼
                      Logits
                         │
                         ▼
                     Sigmoid
                         │
                         ▼
                Risk Probabilities
                         │
                         ▼
              Thresholded Predictions
                         │
                         ▼
                     Metrics
                         │
                         ▼
                Cascade Post-analysis
```

---

# 91. Definition of Done

The GraphSAGE implementation is complete only when:

- [ ] Existing benchmark is used without unnecessary modification.
- [ ] Actual repository schema has been inspected.
- [ ] Feature audit has been inspected.
- [ ] Target definition is explicitly documented.
- [ ] Prediction horizon is configurable.
- [ ] Temporal features are leakage-safe.
- [ ] Train-only preprocessing is implemented.
- [ ] Heterogeneous graph loads correctly.
- [ ] All six node types are handled.
- [ ] Existing edge types are handled correctly.
- [ ] Two-layer GraphSAGE is implemented.
- [ ] Supplier-level readout is implemented.
- [ ] BCEWithLogitsLoss is implemented.
- [ ] Class weighting is supported.
- [ ] Validation PR-AUC is calculated.
- [ ] Test PR-AUC is calculated.
- [ ] Precision is calculated.
- [ ] Recall is calculated.
- [ ] F1 is calculated.
- [ ] ROC-AUC is calculated.
- [ ] Calibration is calculated.
- [ ] Predictions are saved.
- [ ] Model checkpoint is saved.
- [ ] Training history is saved.
- [ ] Risk ranking is saved.
- [ ] Plots are generated.
- [ ] Run metadata is saved.
- [ ] Multiple seeds are supported.
- [ ] Majority and Logistic Regression baselines are available.
- [ ] Existing 90 tests remain passing.
- [ ] New model tests pass.
- [ ] One complete end-to-end experiment succeeds.
- [ ] Classical benchmark can be frozen for later QGNN comparison.

---

# 92. Instructions to Codex

## BEFORE CODING

First inspect the repository recursively.

Read:

```text
WORK_SUMMARY.md
```

Then inspect:

```text
src/scm_dataset/
data/benchmark/
data/generated/
configs/
tests/
```

Specifically inspect:

```text
graph/nodes.csv
graph/edges.csv
operations/
labels/supplier_labels.csv
data/benchmark/feature_audit.csv
data/benchmark/splits/
```

Confirm the actual filenames and schemas.

Do not assume the documentation and runtime data are perfectly identical.

---

## DURING CODING

Implement the model in a separate modeling layer.

Do not rewrite the generator.

Do not change existing labels.

Do not change benchmark splits.

Do not create arbitrary features.

Do not use future information.

Do not optimize for future QGNN performance.

Keep all model configuration explicit.

Add tests before declaring the implementation complete.

---

## AFTER CODING

Run:

```text
pytest -q
```

Then run one complete GraphSAGE experiment.

Verify:

```text
model.pt exists
predictions.csv exists
metrics.json exists
run_metadata.json exists
plots exist
```

Inspect the generated prediction rows manually.

Verify:

```text
risk_probability ∈ [0, 1]
```

Verify:

```text
actual_disruption ∈ {0, 1}
```

Verify no future feature timestamps exist.

---

# 93. Final Goal

The desired end state is:

```text
              SUPPLY-CHAIN BENCHMARK
                       │
                       ▼
             CLASSICAL GraphSAGE
                       │
                       ▼
          Supplier disruption probability
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       Binary decision       Risk ranking
             │
             ▼
       Evaluation metrics
             │
             ▼
      Cascade analysis
             │
             ▼
      FROZEN BASELINE
             │
             ▼
           QGNN
             │
             ▼
      Fair comparison
```

The classical GraphSAGE model is therefore not just a preliminary model.

It is the **controlled classical baseline** that establishes whether graph-based learning can predict rare supplier disruptions before the quantum model is introduced.

The most important principle is:

> **Get the classical GraphSAGE benchmark correct, reproducible, leakage-free, and statistically defensible before attempting to demonstrate any QGNN advantage.**
