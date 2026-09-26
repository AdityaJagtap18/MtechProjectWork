# Literature: GNNs for Supplier / Procurement / Supply-Chain Risk (IEEE & Springer)

Companion to the Stage 1 literature survey (`MSE_Phase1/TemplatePPT/Stage1.tex`,
Table 1, and `refs.bib` on the `qgnn-package-restructure` branch). That survey
covers classical supplier-risk ML [1, 2, 5], QML surveys [3, 4, 8] and QML/QGNN
fraud work [6, 7], but its only GNN-for-supply-chain-risk entry is a survey
(Yadav 2021 [10]) with an incomplete citation. This file adds IEEE and Springer
papers that apply GNNs to supply-chain risk, and for each one records:

- whether it has a **real implementation** with **reported metrics** (AUC, F1, ...);
- whether its **dataset / code is public**.

## How this was checked (read before citing numbers)

IEEE Xplore, Springer Link, arXiv, ResearchGate and Crossref were all blocked
from this environment, so no full text was opened directly. Numbers below come
from web-search extracts of each paper's abstract or results section. Where a
value is quoted it is what the search surfaced. "Not retrieved" means the paper
reports that metric but the value did not appear in any extract. **Verify
every number against the PDF before it goes into slides or the report.**

"Public dataset: not found" means no data or code link turned up in the
abstract, the preprint or the search results. It does not prove that none exists.

---

## Tier 1: IEEE / Springer, GNN on supply-chain risk, numbers found

### 1. HKTGNN (IEEE, 2023). Closest match to this project

- **Citation:** Z. Zhou, K. Bi, Y. Zhong, C. Tang, D. Li, S. Ying, R. Wang,
  "HKTGNN: Hierarchical Knowledge Transferable Graph Neural Network-based
  Supply Chain Risk Assessment," *21st IEEE Int. Symp. on Parallel and
  Distributed Processing with Applications (ISPA)*, 2023.
  IEEE Xplore: https://ieeexplore.ieee.org/document/10491606/ ·
  preprint: https://arxiv.org/abs/2311.04244
- **Task:** supply-chain (corporate investment) risk classification of
  enterprises on an integrated-circuit supply chain graph.
- **Method:** graph embedding collapses each product's supply chain into a
  product network, then a centrality-based knowledge-transfer module handles
  domain difference and data hunger.
- **Results:** **F1 = 83.13 ± 2.76, AUC = 88.41 ± 1.25** (%).
  Baselines: MLP, GGNN, **GraphSAGE**, GraphSAGE2, GATv2, KTGNN.
- **Dataset:** IC supply chain with 430 products, 1,732 listed companies and
  46,273 related enterprise / natural-person nodes, 1,875 product-product links.
- **Public dataset / code:** not found.
- **Why it matters here:** peer-reviewed IEEE, reports AUC and F1 with ± std,
  and uses a **GraphSAGE baseline**, the same classical baseline this project uses.

### 2. HG-DRA (Springer, *J. Supercomputing*, 2026)

- **Citation:** J. Wang, Q. Zhao, Y. Liu, P. Li, Y. Zhang *et al.*, "Supply
  chain disruption risk prediction based on hypergraph representation and
  dynamic relational-attentive," *The Journal of Supercomputing*, 2026.
  DOI: 10.1007/s11227-026-08342-7 ·
  free preprint: https://www.researchsquare.com/article/rs-6843359/v1
- **Task:** disruption-risk prediction for listed companies. Labels are
  regulatory violation categories and the classes are imbalanced (violation vs
  compliance).
- **Method:** hypergraph representation for N-ary enterprise groups plus
  hierarchical dynamic relational attention.
- **Implementation:** PyTorch + PyTorch Geometric, lr 5e-4, 300 epochs,
  dropout 0.2.
- **Metrics reported:** Accuracy, Recall, F1, **G-mean, AUC** (chosen because
  of class imbalance). Numeric values not retrieved.
- **Dataset:** CSMAR financial database plus public disclosures from
  CSRC / SZSE / SSE.
- **Public dataset / code:** no. CSMAR is a paid subscription database, and
  no code link was found.
- **Note:** `DATASET_DESIGN_REVIEW.md` §6 already flagged HG-DRA as "worth a
  closer look during literature review".

### 3. ML-GCNPS (Springer Nature, *Scientific Reports*, 2025)

- **Citation:** Y. Wang, Y. Sun, "Low-carbon supply chain logistics risk
  prediction using meta-learning-based graph convolutional network on
  prototype space," *Scientific Reports*, Dec. 2025.
  DOI: 10.1038/s41598-025-32184-7 (open access)
- **Task:** firm-level supply-chain logistics risk early warning (imbalanced).
- **Method:** meta-learning GCN on a prototype space with an adaptive
  vertex-to-edge network that builds the graph topology.
- **Results:** **AUPRC = 0.850, FNR = 0.080**, weighted average cost = 45.
- **Dataset:** real multi-modal firm-level carbon emissions plus supply-chain
  relationships.
- **Public dataset / code:** *Scientific Reports* requires a Data Availability
  statement. It was not readable from here, so check the article page.
- **Why it matters here:** its headline metric is AUPRC, the same primary
  metric this project uses.

### 4. BM-GNN (IEEE Access, 2025)

- **Citation:** K. F. Mojdehi, B. Amiri, A. Haddadi, "A Novel Hybrid Model for
  Credit Risk Assessment of Supply Chain Finance Based on Topological Data
  Analysis and Graph Neural Network," *IEEE Access*, vol. 13,
  pp. 13101–13127, 2025. DOI: 10.1109/ACCESS.2025.3528373
- **Task:** SME credit risk in energy-sector supply-chain finance.
- **Method:** BallMapper (topological data analysis) plus GNN, with
  network-based features alongside financial ratios.
- **Metrics reported:** Accuracy and F1, stated to beat traditional ML.
  Numeric values not retrieved.
- **Public dataset / code:** not found.

### 5. Hybrid GNN for cross-border e-commerce supply-chain risk (IEEE conference)

- **Citation:** "Real-Time Risk Prediction of Cross-Border E-Commerce Supply
  Chain Based on Hybrid Graph Neural Network," IEEE conference publication.
  https://ieeexplore.ieee.org/document/11232765/ (authors and year not retrieved)
- **Method:** hybrid GNN with dynamic gating for spatio-temporal-knowledge
  co-evolution.
- **Results:** accuracy 0.81–0.99, latency 38.7–142.5 ms, prediction variance
  0.007. **No AUC or F1 was found.**
- **Public dataset / code:** not found.
- **Caveat:** accuracy-only reporting is weak for an imbalanced risk task, so
  use this one as a lower-priority citation.

---

## Tier 2: IEEE / Springer, relevant, metrics named but values not retrieved

| Paper | Venue | Task / method | Metrics reported | Dataset |
|---|---|---|---|---|
| **GTARA:** "Graph topology-aware attention for industrial chain risk assessment" | Springer, *Int. J. Machine Learning & Cybernetics*, 2026. DOI 10.1007/s13042-026-03027-2 | Heterogeneous graph, meta-paths, risk-injection perturbation, node- and graph-level multi-task risk | AUC, F1, Accuracy + 2 more (values not retrieved) | 2 real industrial-chain datasets (integrated circuits, electronic information). Public: not found |
| **MTHG-PA:** "Multi-temporal heterogeneous graph learning with pattern-aware attention for industrial chain risk detection" | Springer, *World Wide Web* 27, art. 38, 2024. DOI 10.1007/s11280-024-01280-5. Li Z., Sun Y., Bi X. *et al.* | Temporal heterogeneous GNN for industrial-chain risk detection | Values not retrieved | Not retrieved |
| **DHGN-Credit:** "Default prediction and contagion risk assessment in corporate guarantee networks via dynamic heterogeneous graph neural networks" | Springer, *J. King Saud Univ. Comp. & Info. Sci.*, 2026. DOI 10.1007/s44443-026-00904-2 | Dynamic heterogeneous GNN with metapath attention and continuous-time memory; default contagion | AUC, F1, Precision, Recall, **MCC**; 5 seeds, 95% CI, 1,000× bootstrap; about 4% positives, about 30 defaulters in test | China bond-market guarantee networks. Public: not found |
| **Graph-based risk inference expert system:** "An intelligent expert system for logistics disruption prediction and mitigation in global supply chains" | Springer Nature, *Scientific Reports*, 2026. DOI 10.1038/s41598-026-61617-0 | Multi-level supply-chain graph, graph-driven risk propagation plus ensemble forecasting | AUC **+5.3% to +8.7%** over no-risk-inference models | 36 months, >500 nodes, >1,000 logistics paths. Check the Data Availability statement |
| **KGR-HATA:** "Knowledge Graph Reasoning with Hierarchical Attention-Based Temporal Aggregation for Industrial Chain Risk Prediction" | Springer book chapter, 2026 (978-981-95-3462-3_17) | Temporal knowledge-graph reasoning for industrial-chain risk | Values not retrieved | Not retrieved |
| **Supply-chain GNN for financial statement fraud:** "Detecting and Interpreting Financial Statement Fraud via Supply Chain-Based Graph Neural Network Models" | IEEE conference. https://ieeexplore.ieee.org/document/11362543/ | Supply-chain-graph GNN for fraud | A search summary gave AUC ≈ 0.99, but it is not clearly from this paper, so verify it | Not retrieved |

## Tier 3: IEEE, relevant framing, no experimental numbers found

- Y. Yang, C. Peng, E.-Z. Cao, W. Zou, "Building Resilience in Supply Chains:
  A Knowledge Graph-Based Risk Management Framework," *IEEE Trans.
  Computational Social Systems*, 11(3):3873–3881, 2024.
  https://ieeexplore.ieee.org/document/10371335/. This is a knowledge-graph
  framework and no GNN component or metrics were confirmed. Useful for
  motivation, not for results.
- "A Graph Neural Network Framework for Governance, Risk, and Compliance
  Classification and Unified GRC Scoring ...," IEEE conference.
  https://ieeexplore.ieee.org/document/11323587/. This is supplier GRC scoring
  with a GNN, but no metrics or dataset were found.

## Already in the Stage 1 survey: add the number

- **[6] Innan et al. (2024)**, "Financial fraud detection using quantum graph
  neural networks," *Quantum Machine Intelligence* (Springer) 6(1), art. 7.
  DOI 10.1007/s42484-024-00143-6. Reports **AUC ≈ 0.85** for the QGNN, above
  the classical GNN. Adding that number to Table 1 makes the "≈3% gain" row
  concrete.

---

## Public datasets and code (mostly not IEEE / Springer)

None of the IEEE or Springer GNN risk papers above was found to release both
data and code. Everything open that turned up is below.

| Resource | What it is | Risk label? | Access | Venue |
|---|---|---|---|---|
| **SupplyGraph** (Wasi, Islam, Akib) | Real FMCG supply-chain graph from Bangladesh: 40 products, 221 daily time points (Jan–Aug 2023), with production, sales, delivery and factory-issue signals | No (planning / forecasting tasks) | https://github.com/ciol-researchlab/SupplyGraph · https://huggingface.co/datasets/azminetoushikwasi/SupplyGraph | AAAI 2024 GCLR workshop, arXiv 2401.15299 |
| **SupplySim / Stanford supply-chains** | Synthetic supply-chain transaction simulator used in "Learning production functions for supply chains with GNNs" | No | https://github.com/snap-stanford/supply-chains (already in plan §57) | AAAI 2025 |
| **DataCo Smart Supply Chain** | 180k order lines, 52 columns | **Yes:** `late_delivery_risk` | Kaggle (public) | Used by Springer chapters, e.g. "Late Delivery Supply Chain Risk Prediction: A Comparative Study" (10.1007/978-981-96-3358-6_48), but with classical ML, not GNNs |
| Ch-EGN (Chebyshev ensemble graph network) on SupplyGraph + DataCo | GNN risk model on public data; 98.95% average accuracy | Yes (DataCo) | Public data | MDPI *Biomimetics* 11(2):123, 2026 (**not** IEEE/Springer) |

Other widely cited papers with numbers that are **not** IEEE or Springer:

- Kosasih & Brintrup, "A machine learning approach for predicting hidden links
  in supply chain with graph neural networks," *IJPR* 60(17), 2022 (Taylor &
  Francis). Hidden-link prediction reaches **AUC 0.76** on a real automotive
  network. The data is proprietary.
- Zhang *et al.*, "Credit Risk Analysis for SMEs Using Graph Neural Networks in
  Supply Chain," ACM BDAIE 2025 / arXiv 2507.07854. Reports **AUC 0.995**
  (supply-chain link mining) and **0.701** (default prediction) on Discover and
  Ant Credit data, with 23.4M and 8.6M nodes.

---

## What this means for the Stage 1 survey

1. **Replace or supplement [10] Yadav (2021).** It is a survey, the citation is
   still incomplete in `refs.bib`, and the closest match found (an IJRAI
   article titled "Graph Neural Networks for Supply Chain Risk Propagation")
   is neither IEEE nor Springer. **HKTGNN** is the best replacement: it is
   IEEE, reports AUC and F1, and uses GraphSAGE as a baseline.
2. **Strongest additions for Table 1:** HKTGNN (IEEE), HG-DRA (Springer),
   ML-GCNPS (Springer Nature), BM-GNN (IEEE Access), DHGN-Credit (Springer).
3. **The gap is sharper than the current slide says.** Beyond "no QGNN on
   supplier procurement risk":
   - every IEEE or Springer GNN supply-chain-risk paper found trains on
     **proprietary or subscription data** (CSMAR, Chinese industrial-chain
     data, energy-sector SCF firms) with **no released code**, so none of them
     can be reproduced;
   - the public supply-chain graph datasets (SupplyGraph, SupplySim) have **no
     disruption-risk labels**, and the public dataset that has a risk label
     (DataCo) is not a supplier graph.

   Both points support building a seeded, public, labelled synthetic benchmark
   (this repo's Phases 1–8).
4. **Numbers are not comparable across papers.** HKTGNN's AUC of 88.4% and
   this project's GraphSAGE ROC-AUC of 0.987 / PR-AUC of 0.807 come from
   different data and labels. Cite them as evidence that GNNs are used and
   reported with AUC and F1, not as a leaderboard.

### Suggested Table 1 rows (same columns as Stage1.tex)

| Author (Year) | Focus / Method | Limitation |
|---|---|---|
| Zhou et al. (2023), IEEE ISPA | HKTGNN: knowledge-transfer GNN for supply-chain risk; F1 83.1, AUC 88.4, beats GraphSAGE / GATv2 | Proprietary IC-industry data, no code; classical only |
| Wang et al. (2026), Springer J. Supercomputing | HG-DRA: hypergraph + dynamic relational attention for disruption risk; AUC / G-mean under imbalance | CSMAR subscription data; enterprise violations as proxy labels; classical only |
| Wang & Sun (2025), Sci. Reports (Springer Nature) | ML-GCNPS: meta-learning GCN for SC logistics risk; AUPRC 0.850, FNR 0.080 | Single domain (low-carbon logistics); classical only |
| Mojdehi et al. (2025), IEEE Access | BM-GNN: TDA + GNN for supply-chain-finance credit risk; higher accuracy and F1 than ML | Credit (not procurement) risk; energy-sector data not public |
| DHGN-Credit (2026), Springer JKSU-CIS | Dynamic heterogeneous GNN for default contagion; AUC, F1, MCC with CIs at about 4% positives | Guarantee network, not supplier network; data not public |

## BibTeX

Entries in the same style as `refs.bib`. Missing fields are marked in `note`,
as in the existing `jiang2022` / `yadav2021` entries.

```bibtex
@inproceedings{zhou2023,
  author    = {Zhou, Zhanting and Bi, Kejun and Zhong, Yuyanzhen and Tang, Chao and Li, Dongfen and Ying, Shi and Wang, Ruijin},
  title     = {{HKTGNN}: Hierarchical Knowledge Transferable Graph Neural Network-based Supply Chain Risk Assessment},
  booktitle = {IEEE International Symposium on Parallel and Distributed Processing with Applications (ISPA)},
  publisher = {IEEE},
  year      = {2023},
  note      = {IEEE Xplore doc. 10491606; arXiv:2311.04244}
}

@article{wang2026hgdra,
  author    = {Wang, Jinlong and Zhao, Qixin and Liu, Yingmin and Li, Pengjun and Zhang, Yuanyuan and others},
  title     = {Supply chain disruption risk prediction based on hypergraph representation and dynamic relational-attentive},
  journal   = {The Journal of Supercomputing},
  publisher = {Springer},
  year      = {2026},
  doi       = {10.1007/s11227-026-08342-7},
  note      = {volume/pages to be completed}
}

@article{wang2025mlgcnps,
  author    = {Wang, Yueqi and Sun, Yonghe},
  title     = {Low-carbon supply chain logistics risk prediction using meta-learning-based graph convolutional network on prototype space},
  journal   = {Scientific Reports},
  publisher = {Springer Nature},
  year      = {2025},
  doi       = {10.1038/s41598-025-32184-7},
  note      = {article number to be completed}
}

@article{mojdehi2025,
  author    = {Mojdehi, Kosar Farajpour and Amiri, Babak and Haddadi, Amirali},
  title     = {A Novel Hybrid Model for Credit Risk Assessment of Supply Chain Finance Based on Topological Data Analysis and Graph Neural Network},
  journal   = {IEEE Access},
  volume    = {13},
  pages     = {13101--13127},
  year      = {2025},
  doi       = {10.1109/ACCESS.2025.3528373}
}

@article{dhgncredit2026,
  author    = {{Author list to be completed}},
  title     = {Default prediction and contagion risk assessment in corporate guarantee networks via dynamic heterogeneous graph neural networks},
  journal   = {Journal of King Saud University Computer and Information Sciences},
  publisher = {Springer},
  year      = {2026},
  doi       = {10.1007/s44443-026-00904-2}
}

@article{gtara2026,
  author    = {{Author list to be completed}},
  title     = {Graph topology-aware attention for industrial chain risk assessment},
  journal   = {International Journal of Machine Learning and Cybernetics},
  publisher = {Springer},
  year      = {2026},
  doi       = {10.1007/s13042-026-03027-2}
}

@article{li2024mthgpa,
  author    = {Li, Z. and Sun, Y. and Bi, X. and others},
  title     = {Multi-temporal heterogeneous graph learning with pattern-aware attention for industrial chain risk detection},
  journal   = {World Wide Web},
  volume    = {27},
  note      = {art. 38},
  publisher = {Springer},
  year      = {2024},
  doi       = {10.1007/s11280-024-01280-5}
}

@article{yang2024,
  author    = {Yang, Y. and Peng, C. and Cao, E.-Z. and Zou, W.},
  title     = {Building Resilience in Supply Chains: A Knowledge Graph-Based Risk Management Framework},
  journal   = {IEEE Transactions on Computational Social Systems},
  volume    = {11},
  number    = {3},
  pages     = {3873--3881},
  year      = {2024}
}
```
