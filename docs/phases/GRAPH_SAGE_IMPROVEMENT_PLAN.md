# GraphSAGE Improvement & Robust Early-Warning Plan

## 1. Purpose

This document defines the next phase of work for improving the classical GNN/GraphSAGE component of the supply-chain risk project.

The goal is **not** to blindly tune GraphSAGE for a higher headline score. The goal is to determine what the GNN can genuinely learn, improve its ability to identify meaningful supplier risk, and produce a scientifically defensible classical-GNN benchmark that can later be compared with a QGNN under the same protocol.

The existing GraphSAGE implementation is already functional, tested, reproducible, and leakage-audited. This phase should build on it rather than rewrite it.

---

# 2. Current Baseline

The current task is:

> Given information available at week `t`, predict whether a supplier will be disrupted at any point during weeks `t+1..t+4`.

Current implementation:

- Heterogeneous PyTorch Geometric graph
- 6 node types
- 9 edge types
- 2 GraphSAGE layers
- Mean aggregation
- Type-specific projections
- Hidden dimension 128
- Dropout 0.20
- Weighted BCEWithLogitsLoss
- Validation PR-AUC early stopping
- Temporal, severity, and scenario split infrastructure
- Majority and Logistic Regression baselines
- Train-only preprocessing
- Leakage audit
- Multi-seed experiments
- Calibration and risk-ranking outputs

The current temporal experiment performs strongly, but deeper analysis shows that much of the performance comes from recognizing suppliers that are already disrupted at prediction time.

The existing analysis found:

- Fresh-onset positives receive essentially zero predicted probability.
- Fresh-onset recall is 0%.
- The simulator currently applies event effects as a hard step at event start.
- Therefore, dynamic operational variables may contain little or no information immediately before an exogenous black-swan event.

**Do not remove or hide this finding. Preserve it as an important diagnostic result.**

---

# 3. Core Objective of This Phase

We need to answer four questions in order:

### Q1. Is GraphSAGE learning useful information beyond current operational disruption?

Determine whether the model is primarily detecting persistence/current state or whether it can exploit structural supplier vulnerability.

### Q2. Does the existing dataset contain any genuine pre-event signal?

Test static and structural information separately from dynamic operational information.

### Q3. Can GraphSAGE be improved for the useful prediction task without introducing leakage?

Improve architecture/features/training only after establishing what information is actually available.

### Q4. What should the final classical-GNN benchmark be?

Select a frozen configuration and evaluation protocol based on evidence, not on whichever experiment gives the highest metric.

---

# 4. Do NOT Do This

The coding agent must NOT:

- rebuild the synthetic dataset from scratch;
- alter event-generation behavior just to make GraphSAGE perform better;
- introduce future information;
- use `supplier_disrupted[t]` as a model feature;
- use `supplier_risk_score` as a feature;
- tune directly on the test set;
- repeatedly inspect test performance and then modify the model;
- cherry-pick the best seed;
- weaken the Logistic Regression baseline;
- artificially weaken GraphSAGE;
- assume GraphSAGE should outperform other models;
- assume QGNN should outperform GraphSAGE;
- optimize for accuracy because the task is highly imbalanced;
- claim that high recall means successful early warning unless fresh-onset performance supports that claim.

The benchmark and its existing leakage rules remain the source of truth.

---

# 5. Phase A — Establish a Clean Diagnostic Framework

Before changing the main architecture, create a robust evaluation breakdown.

Every experiment should report at least:

1. Overall test performance
2. Already-disrupted-at-`t` performance
3. Fresh-onset performance
4. Negative examples
5. Positive prevalence
6. PR-AUC
7. ROC-AUC
8. Precision
9. Recall
10. F1
11. Balanced accuracy
12. Brier score
13. Calibration/ECE
14. Confusion matrix

For fresh-onset analysis, define:

> A fresh-onset positive is a supplier that is not disrupted at prediction time `t` but becomes disrupted during the prediction horizon.

Make this definition explicit in code and experiment metadata.

Do not use the current-period target as a model input. It may only be used after prediction for diagnostic evaluation.

---

# 6. Phase B — Feature-Information Experiments

The most important improvement is to understand where the signal comes from.

Implement controlled feature groups.

## B1. Dynamic-only

Use only information derived from recent operational history.

Examples:

- fulfillment ratio
- delivery behavior
- inventory
- backlog
- procurement activity
- rolling means
- rolling variability
- recent trends

Purpose:

> Determine how much the GNN relies on current operational behavior.

---

## B2. Static-only

Use information known before the prediction period and not dependent on future outcomes.

Examples:

Supplier:

- tier
- region
- industry
- capacity
- capacity utilization only if legitimately known at `t`
- reliability
- financial health
- lead-time mean
- lead-time variability
- quality score
- inventory buffer
- substitution availability
- geopolitical exposure
- disaster exposure
- cyber exposure
- criticality

Region:

- geopolitical risk
- natural-disaster risk
- infrastructure risk
- trade risk
- cyber risk
- transport reliability

Purpose:

> Determine whether structural supplier/environment characteristics contain pre-event risk information.

---

## B3. Region/Risk-only

Use only explicit environmental and geographic risk information.

Examples:

- region geopolitical risk
- natural-disaster risk
- infrastructure risk
- trade risk
- cyber risk
- transport reliability
- supplier exposure variables

Purpose:

> Test whether higher-risk environments produce a measurable baseline vulnerability signal.

This is a diagnostic, not necessarily the final model.

---

## B4. Static + Graph Structure

Use static features while retaining the heterogeneous graph.

Purpose:

> Determine whether neighboring entities and supply-chain topology provide information beyond individual supplier features.

---

## B5. Full Model

Use the existing approved feature set:

- static features
- leakage-safe dynamic features
- graph structure

Purpose:

> Establish the complete GraphSAGE benchmark.

---

# 7. Phase C — Important Structural Risk Features

Before inventing complex new features, inspect whether the existing graph already exposes the following concepts.

## Supplier concentration

Measure whether a material depends heavily on one supplier.

Examples:

- number of suppliers per material
- supplier share of material sourcing
- single-source indicator
- concentration index

## Supplier criticality

Measure how many downstream entities depend on a supplier.

Examples:

- number of materials served
- number of plants reachable
- number of products reachable
- downstream dependency count

## Substitution risk

Measure whether a supplier/material has alternatives.

Examples:

- supplier substitution availability
- material substitutability
- alternate supplier count

## Regional exposure

Measure whether multiple important entities depend on the same region.

## Graph centrality

Experiment carefully with structural metrics such as:

- degree
- weighted degree
- two-hop downstream count
- supplier-to-material dependency count

Centrality features must be computed only from information available at prediction time.

Do not use target outcomes to construct them.

---

# 8. Phase D — Improve Temporal Representation

If dynamic features are retained, improve the representation of trends rather than simply adding more columns.

For each relevant operational signal, consider:

- recent level
- recent mean
- recent standard deviation
- minimum
- maximum
- slope/trend
- change from previous period
- change from longer-term baseline
- volatility

Use only data from:

`t-k ... t`

Never use:

`t+1 ... t+4`

for prediction at `t`.

Compare several history lengths, for example:

- 1 week
- 2 weeks
- 4 weeks
- 8 weeks
- 12 weeks

Treat the history length as an experimental hyperparameter.

Do not choose the best history length using the test set.

---

# 9. Phase E — Architecture Improvements

Only after the information-ablation experiments are complete should architecture tuning begin.

Keep the baseline model intact as the control.

Test a small number of principled variants:

### Architecture A — Current baseline

2-layer heterogeneous GraphSAGE.

### Architecture B — Larger representation

Hidden dimension:

- 64
- 128
- 256

### Architecture C — Depth

Compare:

- 2 layers
- 3 layers

Avoid unnecessary depth because excessive message passing can cause oversmoothing.

### Architecture D — Aggregation

Compare:

- mean
- max
- mean + max if implementation remains fair and simple

### Architecture E — Residual connections

Test residual/skip connections for deeper or wider variants.

### Architecture F — Edge-aware information

If the existing edge schema supports meaningful edge attributes, evaluate whether incorporating them improves the model.

Do not fabricate edge attributes.

Every architecture variant must use the same information budget and target.

---

# 10. Phase F — Training Improvements

Keep the existing training protocol as the baseline.

Evaluate only controlled alternatives.

Possible experiments:

- learning rate: `1e-4`, `5e-4`, `1e-3`
- weight decay: `0`, `1e-4`, `1e-3`
- dropout: `0.0`, `0.2`, `0.4`
- hidden dimension: `64`, `128`, `256`

For severe class imbalance, compare:

1. weighted BCE
2. focal loss, only if justified and implemented correctly

Any hyperparameter search must use:

**training → validation**

only.

The test set must remain untouched until the final frozen configuration is evaluated.

---

# 11. Phase G — Threshold Selection

Do not assume `0.5` is the optimal operational threshold.

Select thresholds using validation data only.

Evaluate:

- threshold maximizing F1
- threshold achieving a chosen recall target
- threshold chosen from a precision-recall tradeoff

Then apply the frozen threshold once to the test set.

Report both:

- threshold-independent metrics: PR-AUC, ROC-AUC
- threshold-dependent metrics: precision, recall, F1, balanced accuracy

---

# 12. Phase H — Calibration

Because this is a risk-prediction system, probability quality matters.

Evaluate:

- Brier score
- Expected Calibration Error
- reliability/calibration curve

If calibration is poor, optionally test:

- Platt scaling
- isotonic regression

Calibration parameters must be learned from training/validation data only.

Never calibrate on the test set.

---

# 13. Phase I — Robustness and Generalization

The final GNN should not depend on one lucky synthetic scenario.

Run:

### I1. Temporal generalization

Existing temporal split.

### I2. Severity generalization

Train on moderate events and evaluate on more severe events.

Existing severity split should be retained.

### I3. Scenario generalization

Hold out event types.

Use the existing scenario split.

### I4. Dataset-seed robustness

Run the final model on another generated benchmark seed, such as `seed44`, without changing the model based on its results.

### I5. Multi-seed model robustness

Use seeds:

`42, 43, 44, 45, 46`

Report:

`mean ± standard deviation`

Do not report only the best seed.

---

# 14. Phase J — Ablation Matrix

Create a clear final ablation table.

At minimum:

| Model/Input | Overall PR-AUC | Overall ROC-AUC | Fresh-Onset PR-AUC | Fresh-Onset ROC-AUC |
|---|---:|---:|---:|---:|
| Majority | | | | |
| Logistic Regression | | | | |
| Dynamic-only GraphSAGE | | | | |
| Static-only GraphSAGE | | | | |
| Region/Risk-only GraphSAGE | | | | |
| Static + Graph Structure | | | | |
| Full GraphSAGE | | | | |

If useful, add:

- no-region-risk
- single-source vs multi-source
- dynamic history length variants

This table is more scientifically valuable than simply showing the best GraphSAGE score.

---

# 15. Phase K — Interpretation

For the best frozen GraphSAGE model, analyze:

### What it gets right

- examples of correctly identified high-risk suppliers
- structural characteristics
- current operational signals
- downstream dependencies

### What it gets wrong

Especially:

- false negatives
- fresh-onset positives
- suppliers with high structural risk but no operational deterioration

### Risk ranking

Produce ranked suppliers with:

- supplier ID
- predicted probability
- predicted class
- relevant risk/context features
- downstream dependency information

Do not expose target-derived information in the prediction artifact.

---

# 16. Optional Explainability

If practical, add explainability after the benchmark is stable.

Potential methods:

- feature ablation
- permutation importance
- gradient-based attribution
- GraphMask or another graph explanation method if technically appropriate

The goal is not to create a flashy explainability component.

The goal is to answer:

> Which supplier characteristics and neighboring entities are contributing to GraphSAGE's risk prediction?

Any explanation method must be validated and documented.

---

# 17. What Counts as an Improvement?

Do NOT define improvement as:

> “The new model has a higher accuracy.”

A meaningful improvement should satisfy one or more of:

1. Higher PR-AUC on unseen data.
2. Better fresh-onset discrimination.
3. Better severity generalization.
4. Better scenario generalization.
5. Better calibration.
6. Better robustness across seeds.
7. Better risk ranking.
8. Better performance without increasing leakage or complexity unnecessarily.

A tiny gain on one split with worse generalization should NOT automatically replace the baseline.

---

# 18. Recommended Decision Tree

Follow this order.

```text
START
  |
  v
Validate current GraphSAGE
  |
  v
Separate persistence vs fresh onset
  |
  v
Run feature-information ablations
  |
  +---- Static/structural signal exists?
  |             |
  |             +---- YES
  |             |      |
  |             |      v
  |             |  Improve structural representation
  |             |
  |             +---- NO
  |                    |
  |                    v
  |              Do not force early-warning claims
  |
  v
Improve temporal representation
  |
  v
Controlled architecture tuning
  |
  v
Training/threshold/calibration tuning
  |
  v
Scenario/severity/seed robustness
  |
  v
Select frozen configuration
  |
  v
Final GraphSAGE benchmark
```

---

# 19. Recommended Priority

Do not spend equal effort everywhere.

Priority order:

### Priority 1 — Very high

Fresh-onset vs persistence evaluation.

### Priority 2 — Very high

Static/dynamic/region-risk/graph-structure ablations.

### Priority 3 — High

Structural dependency features.

### Priority 4 — High

Temporal feature/history improvements.

### Priority 5 — Medium

Architecture and hyperparameter tuning.

### Priority 6 — Medium

Calibration and threshold optimization.

### Priority 7 — Medium

Scenario/severity/dataset-seed robustness.

### Priority 8 — Low

Explainability and additional advanced architecture variants.

---

# 20. Implementation Requirements

The coding agent must preserve the existing package structure wherever practical.

Add reusable configuration options rather than hard-coding experimental variants.

For example:

```yaml
feature_mode: full
history_window: 4
hidden_dim: 128
num_layers: 2
aggregation: mean
dropout: 0.2
learning_rate: 0.001
weight_decay: 0.0001
loss: weighted_bce
threshold_mode: validation_f1
```

Support experiment identifiers so every run records exactly:

- dataset
- dataset seed
- model seed
- feature mode
- history window
- architecture
- optimizer
- loss
- threshold method
- split
- software environment
- git commit

---

# 21. Experiment Artifact Requirements

Each experiment should produce:

```text
experiments/classical_gnn/<run_id>/
    config.yaml
    metadata.json
    model.pt
    metrics.json
    predictions.csv
    risk_ranking.csv
    training_history.csv
    onset_breakdown.json
    calibration.json
    plots/
    experiment_report.md
```

Multi-seed experiments should additionally produce an aggregate summary.

Do not allow filenames from different split strategies to collide.

---

# 22. Testing Requirements

Every new feature must have tests.

At minimum test:

- feature-mode selection
- leakage safety
- temporal window correctness
- static/dynamic feature separation
- graph construction
- structural feature computation
- training reproducibility
- validation-only threshold selection
- calibration
- fresh-onset definition
- experiment configuration
- artifact generation

Run the complete existing test suite.

The existing **161 passing tests must remain passing** unless a test is demonstrably incorrect and the change is documented.

---

# 23. Final Deliverable

At the end of this phase, produce:

## A. Improved GraphSAGE implementation

A clean, configurable, reproducible model.

## B. Diagnostic report

Explain:

- what information the GNN uses;
- how much comes from dynamic operations;
- how much comes from static risk;
- how much comes from graph structure;
- persistence performance;
- fresh-onset performance.

## C. Ablation report

Show the contribution of each feature group.

## D. Robustness report

Show temporal, severity, scenario, and dataset-seed performance.

## E. Frozen benchmark

Select one final GraphSAGE configuration.

Document it clearly.

Do not continue tuning it after the benchmark is frozen.

---

# 24. Scientific Interpretation Rules

Use language carefully.

If the model performs well mainly on already-disrupted suppliers, say:

> “GraphSAGE effectively identifies persistent/ongoing supplier disruption.”

Do NOT say:

> “GraphSAGE predicts black-swan events with 97% recall.”

If static features provide useful fresh-onset discrimination, say:

> “Structural and environmental supplier characteristics provide measurable pre-event risk information.”

If they do not, say:

> “The benchmark provides limited evidence of pre-event predictive signal for fresh disruption.”

Do not manufacture a positive conclusion.

The objective is a correct scientific result.

---

# 25. Definition of Done

This phase is complete only when:

- [ ] Current GraphSAGE baseline remains reproducible.
- [ ] Persistence vs fresh-onset evaluation is automatic.
- [ ] Dynamic-only experiment exists.
- [ ] Static-only experiment exists.
- [ ] Region/risk-only experiment exists.
- [ ] Static + graph experiment exists.
- [ ] Full model experiment exists.
- [ ] Structural dependency features are evaluated.
- [ ] Temporal history experiments are evaluated.
- [ ] Controlled architecture tuning is completed.
- [ ] Threshold selection uses validation only.
- [ ] Calibration is evaluated.
- [ ] Temporal generalization is evaluated.
- [ ] Severity generalization is evaluated.
- [ ] Scenario generalization is evaluated.
- [ ] Dataset-seed robustness is evaluated.
- [ ] Five model seeds are used for final comparisons.
- [ ] All important results use mean ± standard deviation.
- [ ] Test set is not used for model selection.
- [ ] No future leakage is introduced.
- [ ] Existing 161 tests remain green.
- [ ] New tests pass.
- [ ] Final configuration is frozen.
- [ ] Final GraphSAGE report is generated.
- [ ] Claims in the report match what the experiments actually demonstrate.

---

# 26. Final Principle

**Do not optimize GraphSAGE for a prettier number.**

First determine what information the graph and features actually contain.

Then improve the model where improvement is scientifically meaningful.

The strongest outcome is not necessarily:

> “GraphSAGE reached 99% recall.”

The strongest outcome is:

> “We systematically established what GraphSAGE can learn from the supply-chain graph, isolated persistence detection from genuine fresh-onset prediction, quantified the contribution of static risk and graph structure, improved the model where justified, and produced a reproducible frozen classical-GNN benchmark.”

That is the benchmark that should eventually be used for any later model comparison.
