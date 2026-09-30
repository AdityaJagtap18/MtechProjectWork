# Literature Review

*Grounded strictly in the papers listed in `LR_filled1.xlsx`. Citation
keys refer to `MSE_Phase1/TemplatePPT/refs.bib`. Unrelated papers from
that sheet are named once at the end and not cited.*

---

## 1. Classical AI/ML and Graph-Based Methods for Supply Chain Risk

Graph-structured approaches already treat supply chain risk as
relational, not a per-node property. A heterogeneous supply chain graph
read by a meta-path attention GNN jointly identifies key nodes and
disruption probability, improving hidden-key-node recall 14 points over
centrality methods \cite{GNNArticle} — a pre-2020 survey frames this same
GCN/GAT risk-propagation idea conceptually, naming scalability and data
availability as unsolved \cite{GNNSCM}.

Classical (non-graph) ML on real operational data is already
production-deployed at comparable or higher accuracy: XGBoost predicting
component failures and part demand at HPE \cite{larbi2025}, an
LSTM+XGBoost pipeline reaching F1 0.91 on late-delivery risk under a
temporal train/test split — the same split discipline this project uses
\cite{rahman2026a} — and similar deployments elsewhere
\cite{supplychainDelayDetection, sayyad2024}. A systematic review of ten
such studies explicitly names "the need for deep and hybrid models" as
the field's open gap \cite{edhrabooh2024}, and a broader GNN-architecture
review agrees, listing hybrid and heterogeneous GNNs as future work
rather than demonstrated results \cite{liu2026}.

**Gap.** All of this is single-architecture classical ML or purely
classical GNNs. None combines graph-structured supply chain modeling
with a quantum component under a controlled comparison — this project
uses the same graph representation these papers establish, read by two
competing heads instead of one.

## 2. Quantum Computing for Supply Chain Management (Non-ML)

Quantum computing's foothold here is on *optimization*, not prediction: a
review surveys QAOA/annealing/QUBO approaches to routing and inventory,
naming implementation difficulty as the real bottleneck
\cite{QCinSupplyChian}, and a quantized policy-iteration algorithm for an
inventory MDP is presented as a feasibility study, not a performance
result \cite{jiang2022}.

**Gap.** Quantum methods here never touch prediction or combine with a
GNN — domain receptiveness to quantum methods, not a disruption-prediction
precedent.

## 3. Quantum and Hybrid Quantum-Classical Models for Fraud/Financial Risk

The closest literature to this project's actual architecture. A QGNN
built on topological-data-analysis graphs plus a variational circuit
beats classical GraphSAGE by ~3% PR-AUC on credit-card fraud, but is
simulator-only with heavy undersampling \cite{GNNFraudDetection} — the
single closest methodological ancestor to QGNN-v4. Two findings from this
cluster matter most for framing this project's own results: a VQC beats
classical baselines in simulation but collapses entirely on real
127-qubit hardware \cite{tudisco2024}, and a feature-map/ansatz
sensitivity study finds architecture choice itself is the dominant
performance driver \cite{QCinFraudDetection} — directly mirroring this
project's own six-factor ablation. A classical-vs-quantum comparison with
economic analysis adds a third caution: near-perfect quantum F1 scores
came at ~600× classical training time, and noiseless results are flagged
as optimistic \cite{bhutta2025} — equally true of this project's own
noise-free simulation.

On the class-imbalance problem this project also faces (3.33% positive
rate), quantum methods show real but inconsistent gains: SMOTE-assisted
and autoencoder-based approaches beat classical baselines on minority-class
metrics \cite{kamble2024, huot2024, ren2025}, but a QSVM trained at
realistic sample sizes falls behind classical XGBoost at production scale
\cite{vuppala2024} — small-scale gains don't automatically hold at volume.
Elsewhere in this cluster, results are genuinely mixed rather than
uniformly favorable: some report strong accuracy with partial noise
robustness \cite{innan2025, gurajada2025, kukliansky2024}, while at least
one is an honest negative result — a quantum-verification QSVC reaching
only 41.6% balanced accuracy against classical's 77%+
\cite{majumder2025} — and one is cited with caution over ambiguous metric
reporting \cite{chaudhary2025}.

**Gap.** No paper here targets supply chain disruption; the one
GNN+quantum precedent \cite{GNNFraudDetection} tests only a small
qubit/depth grid, not a systematic ablation; and none pairs a *frozen*
classical encoder with an independently trained quantum head — this
project's actual contribution.

## 4. General QML Surveys and Background

Survey-level work frames this project's architecture without proposing
new methods: a QCNN survey and a broader QML survey both name hybrid
classical-quantum models as the practical near-term approach given noise
and scalability limits \cite{rahman2026, lamichhane2025} — the same
reasoning behind this project's frozen-encoder design. A review of 23 QML
classification papers finds typical gains of 3-10% over classical,
limited by hardware noise \cite{mohammadisavadkoohi2025} — consistent
with this project's own near-parity, not clear-superiority, result. An
earlier foundational survey frames the field's theoretical motivations
\cite{ramezani2020}.

**Gap.** These surveys flag hybrid architectures as promising but supply
no evidence on *why* a hybrid model's performance decouples from its
classical counterpart on a specific property like calibration — the gap
this project's diagnostic work targets.

## 5. Hybrid Architectures in Adjacent, Non-Financial Domains

Two papers use the same architectural pattern elsewhere. A
quantum-enhanced ANN for medical image compression runs quantum-first,
classical-second — the reverse of this project's ordering — with strong
results \cite{subbiyan2025}. More tellingly, a GNN+VQC hybrid for
space-data-center routing — the same combination as this project — finds
the VQC backend *underperforms* both classical and photonic baselines at
orders-of-magnitude higher latency \cite{ganguly2026}: independent,
cross-domain confirmation that a GNN+VQC hybrid failing to beat its
classical counterpart is not specific to this project's dataset.

**Gap.** Neither paper investigates *why* the quantum component under-
or over-performs — no representation-level diagnostic, which is exactly
what this project runs to explain its own calibration gap.

---

## Summary

Three non-overlapping bodies of work — classical/graph supply chain risk
(§1), quantum supply chain optimization (§2), quantum/hybrid fraud
classifiers (§3) — framed by QML surveys (§4) and cross-domain GNN+VQC
precedents (§5). Nothing reviewed sits at their intersection:
graph-structured supply-chain risk prediction, under a controlled
classical-vs-quantum protocol, with a frozen shared encoder. That
intersection is this project's contribution, and §3 — not the
supply-chain literature — is where the closest comparison-protocol
precedents actually come from.

---

### Excluded

CUDA-Q/GPU engineering (`brown2026`, `kim2026`, `stein2024`, `kim2023`,
`kulkarni2026`, `marevac2026`, `rubinshtein2025`, `bayraktar2023`),
cryptography/security (`ylmaz2026`, `madje2024`, `mahmood2024`), and
off-domain QNN applications (`kan2024`, `siddiqui2025`) — no tie to
supply chain risk, fraud classification, or the GNN+VQC pattern. Row 40
(IJHIT paper) held out pending author name from you.
