# Literature Review

*Grounded strictly in the papers listed in `LR_filled1.xlsx`. Citation keys
refer to `MSE_Phase1/TemplatePPT/refs.bib`. Papers from that sheet with no
tie to this project's actual problem — supply chain risk prediction and
hybrid quantum-classical architectures for classification — are named once
at the end and excluded from citation, per instruction not to cite
unrelated work.*

---

## 1. Classical AI/ML and Graph-Based Methods for Supply Chain Risk

Classical machine learning is already well established for supply chain
risk and disruption prediction, and graph-structured approaches in
particular have converged on the same intuition this project starts
from: risk in a supply chain is relational, not a property of any single
node in isolation.

At the graph-modeling end, a heterogeneous supply chain graph combining
enterprise, logistics, finance, and regulation nodes, read by a meta-path
attention GNN with time encoding, is shown to jointly identify hidden
key nodes and supply-interruption probability — recall for hidden key
nodes improves 14 percentage points over centrality-based methods
(0.68 to 0.82), structure-recognition AUC improves 11 points, and a
real disruption scenario (the Baltimore bridge collapse) is rerouted
36 hours ahead of the actual event \cite{GNNArticle}. A pre-2020
survey of GCN/GAT approaches to supply chain risk propagation frames the
same idea at the conceptual level — firms as nodes, dependencies as
edges, multi-hop propagation — while flagging scalability,
interpretability, and data availability as the field's standing
challenges, not yet solved ones \cite{GNNSCM}. A broader review of GNN
architectures (GCN, GAT, graph recurrent networks) for multi-tier
visibility, demand forecasting, and resilience modeling reaches a similar
conclusion, naming temporal, heterogeneous, federated, and explainable
GNNs as open future directions rather than settled ground
\cite{liu2026}.

Outside graph methods specifically, classical ML on real operational
data performs strongly and is already deployed in production: a
Hewlett Packard Enterprise study builds two XGBoost models directly on
field sensor data — one predicting imminent component failures
(70-80% detection, roughly two-thirds reduction in multi-failure
outages) and one forecasting quarterly failed-part demand
(90-95% accuracy) — and reports this as a live production system, not
a benchmark exercise \cite{larbi2025}. On real ERP data, a supplier
delay-risk and sustainability-scoring framework compares SVM, decision
trees, random forest, XGBoost, and MLP with SHAP explanations, finding
MLP dominant across delay-risk (97% accuracy, F1 0.98) and combined
sustainability scoring (99% accuracy) \cite{supplychainDelayDetection}.
On the public DataCo Smart Supply Chain dataset, a pipeline combining
LSTM demand forecasting, XGBoost late-delivery-risk classification, and
SHAP explainability reports F1 0.91 and AUC-ROC 0.943 for late-delivery
prediction under a temporal (not random) train/test split — the same
split discipline this project's own primary evaluation follows — and
claims a simulated 21.4% inventory-cost reduction from the resulting
policy changes \cite{rahman2026a}. CatBoost-based e-commerce supply
chain prediction reaches comparably strong numbers on delivery-time
classification (F1 89.1%) using only classical gradient boosting
\cite{sayyad2024}. A systematic review of ten Scopus-indexed studies
(2019-2023) on AI/ML in supply chain management groups this body of work
into demand forecasting and risk management, and explicitly names the
gap this project sits in: limited data access, and — the review's own
words — "the need for deep and hybrid models" beyond what classical
tabular ML currently offers \cite{edhrabooh2024}.

**Research gap.** Every classical or graph-based approach reviewed here
is either a single-architecture classifier (XGBoost, MLP, CatBoost) or a
purely classical GNN. None combines graph-structured supply chain
modeling with a quantum computational component, and none is evaluated
under a controlled classical-vs-quantum comparison protocol — a
methodological gap this project addresses directly, using the same
kind of heterogeneous graph representation these papers establish as
appropriate for supply chains \cite{GNNArticle, liu2026}, but reading it
with two competing downstream heads (classical MLP and a variational
quantum circuit) rather than a single classical model.

## 2. Quantum Computing for Supply Chain Management (Non-ML)

A separate thread applies quantum computing to supply chain
*optimization* problems rather than prediction. A review of quantum
computing in supply chain management surveys routing, maintenance,
scheduling, and inventory problems addressed via QAOA, quantum
annealing, and QUBO formulations, and is explicit that implementation
difficulty — not algorithmic promise — is the field's current bottleneck
\cite{QCinSupplyChian}. Concretely, a quantized policy-iteration
algorithm built on a quantum linear system solver is applied to a
classical inventory-control Markov decision process and run on IBM
Qiskit and qBraid hardware, with the paper itself reporting this as a
feasibility study bounded by solver precision (~10⁻³), not a
performance result \cite{jiang2022}.

**Research gap.** This thread demonstrates that quantum computing has a
foothold in supply chain management, but exclusively on the
*optimization* side (routing, inventory policy) — never on the
*prediction/classification* side this project addresses, and never
combined with a graph neural network. It establishes that the domain is
receptive to quantum methods in principle, without providing a
disruption-prediction precedent this project can build on directly.

## 3. Quantum and Hybrid Quantum-Classical Models for Fraud and Financial Risk Detection

This is the literature closest to this project's actual architecture,
even though the application domain (financial fraud, not supply chain
disruption) differs. It is also the literature where a genuine, direct
head-to-head classical-vs-quantum comparison protocol — the same kind
of comparison this project runs — is most established.

The closest single precedent converts credit-card transactions into
graphs via topological data analysis, then classifies them with a QGNN
(angle encoding, two-local variational layers, average pooling), tested
at 6 and 16 qubits and 1-2 layers, and compared directly against a
classical GraphSAGE GNN on the same European credit-card dataset. The
best QGNN configuration (6 qubits, 1 layer) reaches PR-AUC ≈0.85 against
classical GraphSAGE's 0.77 — roughly a 3% accuracy gain — using only
~200 trainable parameters, but the paper's own scope is limited to
simulator-only evaluation with heavy class undersampling
\cite{GNNFraudDetection}. This is the one prior work combining a GNN
architecture with a variational quantum circuit for a classification
task, making it the single closest methodological ancestor to this
project's own QGNN-v4 design — with the qubit-count and gain-magnitude
findings directly comparable to this project's own ablations.

Several papers run the more general comparison — variational quantum
circuits against a full suite of classical baselines — on real bank
data. On a real Intesa Sanpaolo dataset (500,000 transactions), a VQC
(multiple encodings, data re-uploading) beats logistic regression,
random forest, XGBoost, SVM, and a neural network on precision/recall at
several sample sizes, but degrades completely when actually run on IBM's
127-qubit hardware rather than simulated — direct evidence that today's
quantum advantage claims are simulator-bound \cite{tudisco2024}. A
feature-map/ansatz sensitivity study compares VQC, Sampler-QNN, and
Estimator-QNN architectures across three feature maps and four ansätze
on two fraud datasets, finding architecture choice itself is a major
performance driver (best VQC F1 0.88 vs. worst EQNN F1 0.52-0.59) and
that the strongest configurations remain competitive under simulated
noise \cite{QCinFraudDetection} — the same kind of feature-map/ansatz
sensitivity this project's own six-factor ablation (Section 4.1)
investigates for QGNN-v4. A classical-vs-quantum comparison including an
economic-impact analysis finds hybrid QSVM/QSVC configurations reach
near-perfect F1 (~1.000) but at roughly 600× the training time of a
classical SVM, and explicitly cautions that noiseless, randomly-split
results are optimistic relative to production conditions
\cite{bhutta2025} — a caution this project's own noise-free,
`default.qubit`-simulated results (Section 4.1) are equally subject to.

A cluster of papers apply quantum classifiers to fraud detection with
class-imbalance handling, the same core difficulty this project's
3.33%-positive-rate benchmark presents. A QNN with angle encoding and
SMOTE/oversampling reaches 92% overall accuracy but only 65-68%
precision/recall on the minority fraud class specifically
\cite{kamble2024} — a reminder that aggregate accuracy hides minority-class
performance, exactly why this project reports PR-AUC rather than
accuracy (Section 4.3). A quantum autoencoder trained to flag fraud by
reconstruction fidelity, rather than direct classification, reaches
AUC 0.947 at its best threshold and explicitly outperforms a classical
autoencoder baseline (0.800), showing anomaly-style quantum models can
suit highly imbalanced data \cite{huot2024}. A QSVM combined with a
purpose-built quantum-SMOTE oversampling method reaches accuracy 98.8%
against a classical baseline's 92.4% on the same imbalanced credit-card
dataset \cite{ren2025}. By contrast, an 8-qubit angle-encoded QSVM
trained on only 750 samples (a real hardware-scale constraint) falls
well behind classical XGBoost (F1 0.80 vs. 0.87-0.93) on a large
real-world fraud suite, illustrating that quantum data-imbalance gains
reported at small experimental scale do not automatically hold at
production data volumes \cite{vuppala2024}.

Other work in this cluster demonstrates architectural variety without
uniformly positive results, which matters for setting this project's own
expectations honestly. A quantum federated neural network for fraud
detection reaches accuracy/F1 in the 93-95% range while preserving
data privacy across simulated bank clients, with graceful degradation
under most (not all) tested noise models \cite{innan2025}. A QGNN and
QSVM applied to real-time payment fraud reach accuracy 97.3% and 96.5%
respectively against classical DNN/XGBoost baselines in the low-90s, at
millisecond-scale latency, and the QGNN's advantage widens specifically
at very low fraud prevalence (0.05%) — the regime closest to this
project's own 3.33% base rate \cite{gurajada2025}. A quantum
state-verification scheme embedding a QSVM inside an auditable
blockchain is honest about a negative result: its QSVC reaches only
41.6% balanced accuracy against classical RF/XGBoost's 77%+, with the
paper's actual contribution being verifiability rather than
classification accuracy \cite{majumder2025} — a useful counterexample to
any assumption that quantum methods are uniformly competitive. A
QML-based banking fraud pattern-recognition study reports approximately
97% accuracy but its own metrics reporting is ambiguous, and is
therefore cited here with that caveat rather than as strong evidence
\cite{chaudhary2025}. Finally, a QNN for network anomaly detection —
methodologically close to fraud detection — is tuned via a
noise-susceptibility "certainty factor" and run on real IonQ Aria-1
hardware, reaching F1 0.86, an improvement over a prior hardware result
of 0.838 \cite{kukliansky2024}.

**Research gap.** Across this entire cluster, the comparison is
consistently classical-model-vs-quantum-model for *financial fraud*, and
the one GNN+quantum-circuit precedent \cite{GNNFraudDetection} does not
run the systematic, multi-factor architectural ablation (qubit count,
depth, ansatz, projection, initialization, encoding) this project
conducts — it reports two qubit counts and two depths as a small grid,
not a controlled one-factor-at-a-time study. No paper in this cluster
targets supply chain disruption prediction specifically, and none pairs
a *frozen, pre-trained* classical GNN encoder with an independently
trained quantum head — the design choice this project treats as its
central methodological control (Section 2). This project's contribution
relative to this literature is exactly that combination: the domain
(supply chain, not fraud) and the frozen-encoder-plus-independent-head
protocol together.

## 4. General Quantum Machine Learning: Surveys and Foundational Background

A smaller set of survey-level papers frame the broader QML landscape
this project's architecture sits inside, without proposing new methods
themselves. A systematic survey of quantum convolutional neural network
variants (fully quantum, variational, hybrid, graph-based) across
physics, imaging, and time-series applications catalogs the available
toolkits (Qiskit ML, PennyLane, TensorFlow Quantum) and names
scalability, fault tolerance, and security as open challenges
\cite{rahman2026}. A comprehensive QML survey covering cybersecurity,
finance, healthcare, and drug discovery highlights hybrid
classical-quantum models specifically as the practical near-term
approach, citing noise, qubit scalability, and qRAM cost as the reasons
purely quantum models remain impractical \cite{lamichhane2025} — the
same reasoning this project's frozen-classical-encoder design already
follows. A systematic review of 23 QML classification papers
(2013-2023) finds QML classifiers generally match or modestly exceed
classical ones (3-10% accuracy gains reported across the reviewed
studies) but remain limited by noisy hardware
\cite{mohammadisavadkoohi2025} — directly consistent with this project's
own primary-split finding that QGNN-v4 matches, but does not clearly
exceed, its classical counterpart (Section 4.4). An earlier foundational
survey of ML algorithms implemented on quantum computers (quantum SVM,
QNN, quantum clustering) frames the theoretical complexity and speedup
motivations behind the field, without experimental results of its own
\cite{ramezani2020}.

**Research gap.** These surveys consistently identify hybrid
quantum-classical architectures as the practically relevant approach
today, and consistently flag noise and scalability as unresolved —
but none of them, being surveys, contributes new experimental evidence
on *why* a hybrid model's performance decouples from its classical
counterpart (e.g., on calibration specifically, this project's own
central finding in Section 4.4) or on which architectural factors
actually govern that decoupling. That evidence gap is what this
project's own ablation program is designed to fill.

## 5. Hybrid Quantum-Classical Architectures in Adjacent, Non-Financial Domains

Two papers outside both the supply chain and fraud-detection domains are
included because they use the same *architectural pattern* this project
does — pairing a classical component with a quantum circuit — and are
informative by contrast. A quantum-enhanced ANN for medical image
compression encodes classical data into quantum states, processes it
with a parameterized quantum circuit, and feeds the resulting features
into a classical neural network — the reverse ordering of this project's
own design (classical encoder first, quantum head second) — and reports
strong compression gains (PSNR 1.5-15 dB above classical baselines)
\cite{subbiyan2025}, showing the quantum-classical pairing is not
order-dependent in principle, only in this project's specific choice. A
hybrid classical-quantum study of a message-passing GNN combined with a
variational quantum circuit for space-data-center routing — architecturally
the same GNN-plus-VQC combination as this project's own comparison,
just in a different application — finds the VQC backend actually
*underperforms* both the classical and a photonic-sampling backend on
routing accuracy, at two to three orders of magnitude higher latency
\cite{ganguly2026}. This is a second, independent report (beyond this
project's own findings) of a GNN+VQC hybrid failing to outperform its
classical counterpart, in a domain with no direct connection to fraud or
supply chain risk — evidence that this project's own "no reproducible
improvement" finding (Section 4.4) is not an artifact specific to the
supply chain domain or to this project's dataset.

**Research gap.** Both papers confirm that pairing GNNs (or GNN-like
message passing) with a variational quantum circuit is an active,
cross-domain architectural pattern, but neither investigates *why* the
quantum component under- or over-performs — neither runs the kind of
representation-level diagnostic (e.g., probing output-channel
redundancy) this project uses to explain its own calibration gap
(Section 4.4). This project's contribution is the mechanistic
explanation this pattern's cross-domain literature currently lacks.

---

## Summary: How This Project Positions Itself

Taken together, this literature shows three separate, non-overlapping
bodies of work: classical/graph-based ML for supply chain risk
(Section 1), quantum optimization for supply chain logistics
(Section 2), and quantum/hybrid classifiers for financial fraud
(Section 3) — with general QML surveys (Section 4) and a small number of
cross-domain GNN+VQC precedents (Section 5) framing the architectural
choice itself. No paper reviewed here sits at the intersection of all
three: a graph-structured supply chain risk-prediction task, evaluated
under a controlled classical-vs-quantum comparison protocol, with a
frozen classical encoder shared by both heads. That intersection is
this project's actual contribution, and the fraud-detection cluster
(Section 3) in particular — not the supply chain literature — is where
the closest methodological precedents for *how* to run that comparison
actually come from.

---

### Excluded from this review

Per the instruction to cite only what is genuinely relevant, the
following papers from `LR_filled1.xlsx` are **not** cited above, because
their content is CUDA-Q/GPU simulation engineering, quantum
cryptography/security, or an application domain (image classification,
wireless communications) with no connection to supply chain risk,
fraud/financial classification, or the GNN+VQC architectural pattern
this project investigates: the CUDA-Q platform and performance papers
(`brown2026`, `kim2026`, `stein2024`, `kim2023`, `kulkarni2026`,
`marevac2026`, `rubinshtein2025`, `bayraktar2023`), a cryptographic
attack-simulation paper (`ylmaz2026`), two quantum-cryptography/security
papers (`madje2024`, `mahmood2024`), and two off-domain QNN application
papers (`kan2024` — distributed QNN for image classification;
`siddiqui2025` — QNN for wireless communications retry-time
optimization). One paper from the sheet (row 40, "Graph Neural Networks
for Real-Time Supply Chain Risk," IJHIT) is held out pending author
verification — you're supplying the author name(s) separately.
