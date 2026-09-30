# Literature Review

*Grounded strictly in the papers listed in `LR_filled1.xlsx`. Citation
keys refer to `MSE_Phase1/TemplatePPT/refs.bib`. Unrelated papers from
that sheet are named once at the end and not cited.*

---

## 1. Classical AI/ML and Graph-Based Methods for Supply Chain Risk

Graph-structured approaches already treat supply chain risk as
relational rather than per-node. A heterogeneous supply chain graph read
by a meta-path attention GNN with time encoding jointly identifies key
nodes and disruption probability, improving hidden-key-node recall 14
points over centrality methods and rerouting a real disruption scenario
36 hours ahead of the event \cite{GNNArticle}. A pre-2020 survey frames
GCN/GAT-based risk propagation at the conceptual level, flagging
scalability, interpretability, and data availability as unsolved
\cite{GNNSCM}, and a broader architecture review names temporal,
heterogeneous, federated, and explainable GNNs as open directions, not
settled ones \cite{liu2026}.

Outside graph methods, classical ML on real operational data is already
production-deployed: two XGBoost models at HPE predict component
failures (70-80% detection) and quarterly part demand (90-95% accuracy)
\cite{larbi2025}. On real ERP data, an MLP-based supplier delay-risk and
sustainability framework reaches 97-99% accuracy with SHAP explanations
\cite{supplychainDelayDetection}. On the public DataCo dataset, an
LSTM+XGBoost pipeline reaches F1 0.91 / AUC 0.943 on late-delivery risk
under a temporal split \cite{rahman2026a}, and CatBoost reaches F1 89%
on e-commerce delivery-time classification \cite{sayyad2024}. A
systematic review of ten AI/ML supply-chain studies explicitly names
"the need for deep and hybrid models" as the field's open gap
\cite{edhrabooh2024}.

**Gap.** All of this is single-architecture classical ML or purely
classical GNNs — none combines graph-structured supply chain modeling
with a quantum component, and none runs a controlled classical-vs-quantum
comparison. This project uses the same heterogeneous-graph representation
these papers establish \cite{GNNArticle, liu2026}, read by two competing
heads (classical MLP, variational quantum circuit) instead of one.

## 2. Quantum Computing for Supply Chain Management (Non-ML)

Quantum computing's foothold in this domain is on *optimization*, not
prediction: a review surveys QAOA/annealing/QUBO approaches to routing,
scheduling, and inventory, naming implementation difficulty as the real
bottleneck \cite{QCinSupplyChian}, and a quantized policy-iteration
algorithm on a quantum linear-system solver is applied to an inventory
MDP as a feasibility study, not a performance result
\cite{jiang2022}.

**Gap.** Quantum methods here never touch prediction/classification and
never combine with a GNN — this literature shows domain receptiveness to
quantum methods, not a disruption-prediction precedent.

## 3. Quantum and Hybrid Quantum-Classical Models for Fraud/Financial Risk

The closest literature to this project's actual architecture. A QGNN
built from topological-data-analysis graphs plus a variational circuit
(6-16 qubits, 1-2 layers) beats classical GraphSAGE by ~3% PR-AUC on
credit-card fraud using ~200 parameters, but is simulator-only with heavy
undersampling \cite{GNNFraudDetection} — the single closest
methodological ancestor to QGNN-v4. On real bank data, a VQC beats
classical baselines (LR/RF/XGBoost/SVM/NN) on precision/recall but
collapses entirely on real 127-qubit hardware \cite{tudisco2024} — direct
evidence quantum gains are simulator-bound today. A feature-map/ansatz
sensitivity study (VQC/SQNN/EQNN × 3 feature maps × 4 ansätze) finds
architecture choice itself drives performance (F1 0.88 best vs. 0.52-0.59
worst) \cite{QCinFraudDetection}, mirroring this project's own six-factor
ablation. A classical-vs-quantum-plus-economics study finds hybrid
QSVM/QSVC reach near-perfect F1 at ~600× classical training time, and
cautions that noiseless results are optimistic \cite{bhutta2025} — a
caution equally true of this project's own noise-free simulation.

On class imbalance specifically: a SMOTE-assisted QNN reaches 92%
accuracy but only 65-68% minority-class precision/recall \cite{kamble2024}
— why this project reports PR-AUC, not accuracy; a quantum autoencoder
beats a classical one on imbalanced fraud (AUC 0.947 vs. 0.800)
\cite{huot2024}; QSVM with quantum-SMOTE reaches 98.8% vs. classical
92.4% \cite{ren2025}; but an 8-qubit QSVM trained on only 750 samples
falls behind classical XGBoost at production scale \cite{vuppala2024} —
small-scale quantum imbalance gains don't automatically hold at volume.

Results are mixed elsewhere in this cluster: a quantum federated NN
reaches 93-95% accuracy with most (not all) noise robustness
\cite{innan2025}; a QGNN/QSVM pair beats classical DNN/XGBoost on
real-time payment fraud, with the gap widening at 0.05% fraud prevalence
— close to this project's own 3.33% base rate \cite{gurajada2025}; a
quantum-verification QSVC reaches only 41.6% balanced accuracy against
classical's 77%+, an honest negative result \cite{majumder2025}; a
banking pattern-recognition study reports ~97% accuracy with ambiguous
metrics, cited with caution \cite{chaudhary2025}; and a QNN for network
anomaly detection, tuned via a noise-susceptibility "certainty factor,"
reaches F1 0.86 on real IonQ hardware \cite{kukliansky2024}.

**Gap.** No paper here targets supply chain disruption; the one
GNN+quantum precedent \cite{GNNFraudDetection} tests only a small
qubit/depth grid, not a systematic ablation; and none pairs a *frozen*
classical encoder with an independently trained quantum head. This
project's contribution is exactly that combination.

## 4. General QML Surveys and Background

A QCNN survey catalogs toolkits and names scalability/fault-tolerance as
open \cite{rahman2026}; a broader QML survey names hybrid
classical-quantum models as the practical near-term approach given
noise/scalability limits \cite{lamichhane2025} — the same reasoning
behind this project's frozen-encoder design; a review of 23 QML
classification papers finds 3-10% accuracy gains over classical typical,
limited by hardware noise \cite{mohammadisavadkoohi2025} — consistent
with this project's own near-parity (not clear superiority) result; and
an earlier foundational survey frames the field's theoretical motivations
\cite{ramezani2020}.

**Gap.** These surveys flag hybrid architectures as promising but supply
no evidence on *why* a hybrid model's performance decouples from its
classical counterpart on specific properties like calibration — the gap
this project's ablation and diagnostic work targets.

## 5. Hybrid Architectures in Adjacent, Non-Financial Domains

Two papers use the same architectural pattern in different domains. A
quantum-enhanced ANN for medical image compression runs quantum-first,
classical-second (the reverse of this project's ordering) and reports
strong PSNR gains \cite{subbiyan2025}. A GNN+VQC hybrid for
space-data-center routing — the same combination as this project — finds
the VQC backend *underperforms* both classical and photonic baselines at
2-3 orders of magnitude higher latency \cite{ganguly2026}: independent,
cross-domain confirmation that a GNN+VQC hybrid failing to beat its
classical counterpart is not specific to this project's dataset.

**Gap.** Neither paper explains *why* the quantum component under/over-performs — no representation-level diagnostic, as this project runs to explain its own calibration gap.

---

## Summary

Three non-overlapping bodies of work — classical/graph SC risk (§1),
quantum SC optimization (§2), quantum/hybrid fraud classifiers (§3) —
framed by general QML surveys (§4) and cross-domain GNN+VQC precedents
(§5). Nothing reviewed here sits at their intersection: graph-structured
supply-chain risk prediction, under a controlled classical-vs-quantum
protocol, with a frozen shared encoder. That intersection is this
project's contribution — and §3, not the supply-chain literature, is
where the closest comparison-protocol precedents come from.

---

### Excluded

CUDA-Q/GPU engineering (`brown2026`, `kim2026`, `stein2024`, `kim2023`,
`kulkarni2026`, `marevac2026`, `rubinshtein2025`, `bayraktar2023`),
cryptography/security (`ylmaz2026`, `madje2024`, `mahmood2024`), and
off-domain QNN applications (`kan2024`, `siddiqui2025`) — no tie to
supply chain risk, fraud classification, or the GNN+VQC pattern. Row 40
(IJHIT paper) held out pending author name from you.
