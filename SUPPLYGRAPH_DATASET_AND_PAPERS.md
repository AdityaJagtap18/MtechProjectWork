# SupplyGraph: the open dataset, and the papers/models built on it

SupplyGraph is the only openly downloadable real-world supply-chain *graph*
dataset found in the IEEE/Springer GNN risk literature search
(`LITERATURE_GNN_SUPPLIER_RISK.md`). This file records what the data actually
contains (from downloading and inspecting it) and every paper or model found
that uses it.

## 1. Where to get it

| Source | Link | Notes |
|---|---|---|
| GitHub (original) | https://github.com/ciol-researchlab/SupplyGraph | Homogeneous raw data + `Code/coding_directions.md` + a heterogeneous-graph notebook |
| GitHub (extended, "SCG") | https://github.com/ciol-researchlab/SCG | Same data, adds `raw/heterogeneous/`; LGPL-2.1; Zenodo DOI 10.5281/zenodo.13652826 |
| Kaggle | https://www.kaggle.com/datasets/azminetoushikwasi/supplygraph-supply-chain-planning-using-gnns | Mirror |
| Hugging Face | https://huggingface.co/datasets/azminetoushikwasi/SupplyGraph | Mirror |

Files load directly from `raw.githubusercontent.com`, e.g.
`https://raw.githubusercontent.com/ciol-researchlab/SupplyGraph/main/Raw%20Dataset/Homogenoeus/Temporal%20Data/Unit/Sales%20Order.csv`
(note the upstream spelling `Homogenoeus`, and the trailing space in
`Production .csv`).

## 2. What is in it (inspected, not taken from the paper)

Source: a Bangladeshi FMCG company. Layout under `Raw Dataset/Homogenoeus/`:

```
Nodes/      Nodes.csv, NodesIndex.csv (41 products -> index 0..40),
            Node Types (Product Group and Subgroup).csv,
            Nodes Type (Plant & Storage).csv (6,545 rows, 277 unique product/plant/storage combos)
Edges/      Edges (Plant|Product Group|Product Sub-Group|Storage Location).csv  (product codes)
Edges/EdgesIndex/  same four files with integer node indices
Temporal Data/Unit/    Production .csv, Sales Order.csv, Delivery To distributor.csv, Factory Issue.csv
Temporal Data/Weight/  Production .csv, Factory Issue.csv, ... (weight units)
```

| Property | Value |
|---|---|
| Nodes | **41 products** (the only node type in the homogeneous graph) |
| Categorical attributes | 5 product groups (A, E, M, P, S), 19 sub-groups, 25 plants, 13 storage locations |
| Edge types (unique undirected pairs) | Plant 360 · Storage location 665 · Product group 179 · Sub-group 48 · **union 684** (matches the paper) |
| Edge semantics | Two products are linked if they share a plant / storage location / group / sub-group, so edges mean *co-location / similarity*, not material flow |
| Temporal node features | Production, Sales Order, Delivery to Distributor, Factory Issue, in units and weight |
| Time span | **221 daily steps**, 2023-01-01 to 2023-08-09 |
| Sparsity | Zero entries: production 62%, sales order 46%, factory issue 44%, delivery 36%; **10 of 41 products have zero production for the whole period** |
| Labels | **None shipped.** Tasks (forecasting, classification, anomaly detection) are formulated by the user from the temporal columns |

### Fit for this project (supplier procurement risk under rare disruptions)

- **No supplier, procurement-order or material nodes.** Nodes are finished
  products, and plants and storage locations appear only as edge types. The
  heterogeneous SCG version turns plants and storage into nodes, but still has
  no suppliers.
- **No disruption / risk label.** A proxy such as "delivery < 80% of sales
  order" fires on **46.8%** of the 4,880 active product-days. That is ordinary
  day-to-day shortfall, not a rare black-swan event, so it cannot stand in for
  this project's imbalanced disruption labels.
- **Tiny graph** (41 nodes), so GraphSAGE neighbour sampling and QGNN
  scalability claims cannot be meaningfully tested on it.
- **Realistic use:** at most an external sanity check of the classical
  GraphSAGE encoder on real data (e.g. next-day sales / production regression,
  or plant-code classification as in paper 6 below). It is not a replacement
  for the synthetic benchmark. This also supports the Stage 1 gap argument: the
  one public real supply-chain graph has no supplier nodes and no risk labels.

## 3. Papers and models built on SupplyGraph

Full texts were not openly readable from this environment (arXiv, Springer,
MDPI and Medium were blocked). Numbers are from search-engine extracts, so
verify them against the PDFs before citing.

| # | Paper | Venue | Task on SupplyGraph | Models | Reported result | Code |
|---|---|---|---|---|---|---|
| 1 | Wasi, Islam, Akib, **"SupplyGraph: A Benchmark Dataset for Supply Chain Planning using GNNs"** | AAAI 2024 workshop (GCLR), arXiv 2401.15299 | 6 tasks: demand prediction, production forecasting, product classification, relation classification/detection, anomaly detection | Statistical ML, deep learning, GCN, GAT and others (full list not retrieved) | GNNs beat ML/DL by ~10–30% (regression), ~10–30% (classification/detection), ~15–40% (anomaly detection) | Data + usage guide (GCLSTM example with PyTorch Geometric Temporal) |
| 2 | Wasi, Islam, Akib, Bappy, **"GNNs in Supply Chain Analytics and Optimization: Concepts, Perspectives, Dataset and Benchmarks"** (SCG) | arXiv 2411.08550 (2024, updated 2025) | Same six tasks, homogeneous **and heterogeneous** graphs | GCN, GAT and others | Heterogeneous product-relation detection: **GCN 92.12%, GAT 91.27% accuracy** | https://github.com/ciol-researchlab/SCG |
| 3 | K. Han, **"Applying graph neural network to SupplyGraph for supply chain network"** | arXiv 2408.14501 (2024) | Demand forecasting | MLP vs GCN vs GAT, matched hyperparameters | Test median squared error **MLP 0.3946, GCN 0.3125, GAT 0.1439**; median oversupply 94 / 79 / 7 items; differences significant at α = 0.05 after multiple-comparison correction | Not found |
| 4 | C.-S. Chen, Y.-J. Chen, **"Optimizing Supply Chain Networks with the Power of GNNs"** | arXiv 2501.06221 (2025) | Demand forecasting (single-node and multi-node) | GNN variants vs MLP and GCN | GNNs beat MLP/GCN, especially single-node forecasting (numbers not retrieved) | Not found |
| 5 | X. Liu, Q. Wang, X. Wei, H. Liang, **"Hierarchical Attention-Driven Dynamic GNNs for Accurate Supply Chain Demand Forecasting"** | **Springer**, ICIC 2025, LNCS vol. 15863, doi 10.1007/978-981-95-0009-3_40 | Demand forecasting | Hierarchical attention, dynamic graph-structure learning, probabilistic head; compared with several GNN baselines | Outperforms GNN baselines (numbers not retrieved) | Not found |
| 6 | **"Smart Logistics Model for Supply Chain Management via Brain-Inspired Geometric Deep Networks"** | MDPI *Biomimetics* 11(6):440 (2026), doi 10.3390/biomimetics11060440 | **25-class plant-code classification** using SupplyGraph's pre-defined edge index (plus 4 other datasets) | Hybrid **LSTM + CNN + GraphSAGE** layers | Numbers for the SupplyGraph task not retrieved | Not found |
| 7 | **"An Intelligent Multi-Task Supply Chain Model Based on Bio-Inspired Networks"** (Ch-EGN) | MDPI *Biomimetics* 11(2):123 (2026); arXiv 2510.26203 | Risk / logistics tasks on **SupplyGraph + DataCo** | Chebyshev graph convolution + GAT ensemble | **98.95%** average accuracy (ensemble, risk management) | Not found |

Related, but not confirmed to use SupplyGraph: "Graph Neural Network for Daily
Supply Chain Problems" (Preprints.org 202409.2376) cites it in a survey of GNN
supply-chain tasks. A hybrid GraphSAGE logistics paper (arXiv 2511.11753) uses
DataCo, Shipping and Smart Logistics rather than SupplyGraph.

### Takeaways

- Every published model on SupplyGraph targets **forecasting or
  classification of planning quantities**. None does supplier or procurement
  **risk**, and none uses rare-event labels.
- Paper 5 is the only **Springer** paper found on it. Papers 6 and 7 (MDPI) are
  the only ones that pair it with **GraphSAGE** or a risk-management framing.
- Paper 3 is the cleanest controlled comparison (MLP vs GCN vs GAT with matched
  hyperparameters and a significance test). Its protocol is a good template if
  SupplyGraph is used as a real-data sanity check here.
