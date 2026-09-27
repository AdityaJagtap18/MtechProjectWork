# References — DataCo real-world benchmark track

Sources actually read/verified during this work (not a general literature
survey) -- for the paper-writing literature review and related-work section.
Each entry notes what we actually checked, not just what the source claims.

## Dataset

Constante, Fabian; Silva, Fernando; Pereira, Antonio (2019). **"DataCo SMART
SUPPLY CHAIN FOR BIG DATA ANALYSIS."** Mendeley Data, V5.
DOI: [10.17632/8gx2fvg2k6.5](https://doi.org/10.17632/8gx2fvg2k6.5).
CC-BY-4.0. 180,519 order-item rows, 53 columns, 2015-2018 (the download's
actual date range -- some secondary sources claim 2015-2019).

## Peer-reviewed, credible

**EAGLE: Edge-Aware Graph Learning for Proactive Delivery Delay Prediction in
Smart Logistics Networks.** arXiv:2604.05254. Transformer patch encoder +
Edge-Aware Graph Attention Network on DataCo. Reports F1=0.876,
AUC-ROC>0.97 (stabilizes after 15 epochs), F1 std=0.0089 across 4 seeds. The
strongest credible result found on this exact dataset -- likely because its
graph encodes actual logistics/hub-route structure, which our
customer/product/region graph (notebooks/dataco_exploration.ipynb section 6)
does not attempt to reconstruct.

**Deep learning framework for interpretable supply chain forecasting using
SOM, ANN and SHAP.** Scientific Reports (Nature). Reports 96% accuracy,
96.22% F1 on DataCo, outperforming RF/XGBoost/Decision Tree baselines.
Peer-reviewed, but this number is an outlier next to every other result
found on this dataset (including EAGLE, and our own reproduction) --
plausible explanations not confirmed here: a different/looser split, or
additional preprocessing not fully specified in the paper. Cite with an
explicit caveat, don't treat as a settled number.

## Not peer-reviewed / discredited -- do not cite as a result

**"Integrated Machine Learning for Enhanced Supply Chain Risk Prediction"**
(Jin, T., 2025). Preprints.org, doi:10.20944/preprints202501.1019.v1.
Explicitly marked "Not peer-reviewed version." Claims an RF+GBM+NN ensemble
on DataCo achieving 85.4% accuracy / F1=0.85 (full table below), but its
Figure 4 lists features -- "Lead Time," "Demand Variation," "Supplier
Reliability," "Inventory Level," "Order Cost" -- that **do not exist
anywhere in DataCo's actual 53-column schema** (verified directly against
the downloaded CSV header). Either the author used different data while
citing DataCo, or the figure is unconnected filler. Safe to mention in a
literature review as "a claimed result that could not be corroborated
against the public dataset," not as a number to compare against.

Their reported table, for reference only:

| Model | Accuracy | F1 | MSE |
|---|---|---|---|
| Logistic Regression | 0.765 | 0.74 | 0.024 |
| SVM | 0.782 | 0.76 | 0.022 |
| Random Forest | 0.825 | 0.81 | 0.019 |
| XGBoost | 0.841 | 0.82 | 0.017 |
| Deep Neural Network | 0.837 | 0.83 | 0.018 |
| Proposed Ensemble | 0.854 | 0.85 | 0.015 |

## Informal (not a paper, useful as an independent sanity check)

[fatmaaboamra/Supply-Chain-Late-Delivery-Prediction](https://github.com/fatmaaboamra/Supply-Chain-Late-Delivery-Prediction)
(GitHub). Plain Random Forest on DataCo: 68.75% accuracy, 81.88% precision,
63.19% F1. Not a citable academic source, but its RF numbers land close to
our own from-scratch RF result (67.3% accuracy, F1=0.671) -- used in this
work as a second independent check that our leakage-audited feature set and
temporal split weren't producing an inflated or deflated number.

## This work's own results

See `QGNN_DATACO_ABLATION_BENCHMARK.md` for full methodology and results
(classical baselines, GraphSAGE, QGNN ablation across qubit count, circuit
depth, data efficiency, and simulated noise, plus the mode-collapse
diagnosis/fix). Not an external reference, listed here so the two documents
are easy to find together.
