# Experimental Results

## 1. Experimental Setup

The experiments in this chapter evaluate a hybrid quantum-classical graph
neural network (QGNN-v4) against a classical graph neural network
baseline on a synthetic supply-chain disruption-prediction benchmark
(`scm_v1_black_swan_seed43`). The benchmark graph contains 2,670 nodes of
six types (300 suppliers, 100 materials, 50 plants, 200 products, 20
regions, and 2,000 procurement-order nodes) connected by 7,675 directed
edges across nine relation types, forming a single connected component
with a mean node degree of 5.75 and an average shortest path length of
3.82. The prediction target is whether a supplier will be disrupted
within a four-period-ahead horizon, and the benchmark is deliberately
imbalanced: only 3.33% of supplier-period observations are positive. Two
evaluation conditions were used throughout. The primary (temporal) split
trains on earlier periods and tests on later ones, evaluating a model's
ability to generalize forward in time under conditions similar to
training. The severity (out-of-distribution) split trains only on
periods whose most severe active disruption reaches severity level 3 or
below, and tests exclusively on periods reaching severity level 4 or 5 —
a genuine distribution shift, since the model never observes a
severity-4-or-5 pattern during training. Both models were trained with
Adam (learning rate 0.001, weight decay 0.0001), a maximum of 100 epochs
with early stopping on validation PR-AUC (patience 10, empirically
validated against patience 25 with no material difference), balanced
class weighting, and a threshold fixed at 0.5 for every classification
metric reported.

Every quantum-circuit computation in this project was performed using
PennyLane's `default.qubit` simulator with the `backprop` differentiation
method — an ideal, noise-free classical simulation of the quantum
circuit's behavior. No physical quantum hardware, shot-based sampling, or
noise model was used at any point in this investigation, and no claim
made in this chapter should be read as a claim about how the same
architecture would behave on physical quantum hardware.

## 2. Classical Baseline

The classical baseline is `HeteroGraphSAGE`, a two-layer heterogeneous
GraphSAGE model (mean-aggregation `SAGEConv`, `hidden_dim=128`) trained
end-to-end on the full graph, followed by a two-layer MLP classifier
head. Averaged over five independent training seeds, the classical model
reaches a primary PR-AUC of 0.807 (standard deviation 0.066) and a
severity PR-AUC of 0.449 (standard deviation 0.013). Its severity recall
is exactly 0.4406 in every one of the five seeds — a genuine, previously
documented property of the benchmark (the model consistently catches the
same population of already-visibly-disrupted suppliers) rather than a
measurement artifact. Full per-seed values and the complete extended
metric set (ROC-AUC, F1, precision, recall, specificity, balanced
accuracy, MCC, Brier score, ECE) are given in Section 9 and in
`QGNN_V4_MASTER_RESULTS.csv`.

## 3. QGNN Baseline

The QGNN-v4 architecture reduces the classical GraphSAGE-Full model's own
frozen 128-dimensional supplier embedding — extracted with no gradient
ever flowing back into the encoder — through a trainable
`Linear(128, 6)` layer, bounds the result with `π · tanh(·)`, encodes it
into six qubits via angle embedding, applies a two-layer
`StronglyEntanglingLayers` variational circuit (36 trainable quantum
parameters), measures the Pauli-Z expectation value of each qubit, and
passes the six resulting values through a `LayerNorm` (no learnable
affine parameters) before a final `Linear(6, 1)` classification layer.
The complete head has 817 trainable parameters. The earliest version of
this architecture, without the output-side `LayerNorm`, reached a primary
PR-AUC of 0.833 (standard deviation 0.140) and a severity PR-AUC of 0.392
(standard deviation 0.024) — a nominally higher primary mean than
classical, but with more than double the cross-seed variance and
substantially worse calibration (expected calibration error 0.285 versus
classical's 0.049). Adding `LayerNorm` to the output stage — adopted as
the standing QGNN-v4 reference from this point onward — reduced this
instability considerably: primary PR-AUC of 0.811 (standard deviation
0.074) and severity PR-AUC of 0.398 (standard deviation 0.057), with
severity recall and F1 standard deviations cut by roughly an order of
magnitude relative to the un-normalized baseline. This change was
adopted specifically for its stability and calibration benefit, not
because it closed the performance gap to the classical baseline, which
it did not.

## 4. Primary Temporal Evaluation

On the primary split, the classical and QGNN-v4 reference models are
close: 0.807 versus 0.811 PR-AUC, a difference well within either
model's own cross-seed variance and not read as a directional finding in
either model's favor. The two models diverge more clearly on secondary
metrics. Classical achieves meaningfully higher recall at the fixed 0.5
threshold (0.968 versus 0.898) and is substantially better calibrated,
with an expected calibration error roughly 3.7 times smaller (0.049
versus 0.182). Across the broader set of architectural and training
modifications investigated in this project — projection architecture,
quantum-parameter initialization and its scale, circuit ansatz, and
quantum encoding — no single change produced a primary PR-AUC
improvement that held consistently across all five seeds where a full
evaluation was run; several 2-seed pilot results appeared to show a
primary-PR-AUC gain (most notably small-Gaussian initialization, and the
Gaussian-scale sweep more generally), but these gains were consistently
accompanied by materially higher cross-seed variance than the reference,
and in the one case that was subsequently expanded to five seeds
(identity-like initialization), the apparent gain and its accompanying
stability benefit both failed to hold at the larger seed count.

## 5. Severity/OOD Evaluation

The severity split is substantially harder for both models than the
primary split, and the gap between classical and QGNN-v4 is wider here
than on primary: classical reaches 0.449 PR-AUC against QGNN-v4's 0.398,
alongside a similarly sized gap in ROC-AUC (0.728 versus 0.657) and a
larger gap in calibration (expected calibration error 0.090 versus
0.241). None of the architectural modifications investigated —
projection architecture, initialization or its scale, ansatz, or
encoding — closed this gap in a way that held across seeds; several made
it distinctly worse, most notably small-Gaussian-family initializations,
which improved primary performance while degrading severity performance
in every configuration where this trade-off was tested. A dedicated
diagnostic in the representation-audit phase (Section 7) offers the
clearest available evidence for why this gap persists: when the
identical severity-5 periods used as the severity split's test set are
instead evaluated as part of the primary (temporal) split's test window
— where they are in-distribution relative to training, rather than held
out — the same frozen representation supports a PR-AUC of 0.81, close to
primary's ordinary performance level, rather than the 0.38–0.45 range
observed under the genuine out-of-distribution condition. This result
strongly supports the interpretation that the severity gap is
predominantly a distribution-shift generalization problem rather than an
inherent information deficiency in the input representation, though it
should be read as evidence specific to this benchmark's severity
condition — which is itself dominated by a single underlying severity-5
event (Section 13) — rather than as an established general claim about
generalization across many independent severe disruptions, a question
this dataset cannot directly answer.

## 6. Ablation Studies

Nine distinct architectural and training factors were investigated as
controlled ablations, summarized in `QGNN_V4_ABLATION_TABLE.csv` and, at
the level of individual experiments, in `QGNN_V4_MASTER_RESULTS.csv`.
Training patience (10 versus 25 epochs) produced no material change:
eighteen of twenty runs selected an identical early-stopping checkpoint
regardless of the patience setting, and a positive control confirmed
that validation performance genuinely had nothing further to find in the
additional epochs patience 25 allowed. Output-scale reparameterizations
(a trainable or fixed multiplicative scale, with or without an additive
bias, on the pre-classification representation) are mathematically
redundant with the head's own final linear layer and, correspondingly,
produced no result distinguishable from the un-normalized baseline on
either split. Projection-architecture alternatives — a small nonlinear
bottleneck, a pre-projection normalization layer, and a train-fit PCA
reduction at several component counts — all underperformed the existing
single linear bottleneck in a two-seed pilot, with the PCA-based variant
underperforming despite far fewer trainable parameters, an anomaly
discussed further in Section 12 and left unresolved. Quantum-parameter
initialization (a small Gaussian distribution and an exactly-zero,
identity-like initialization of every rotation gate, generalized to a
five-point standard-deviation sweep) consistently traded primary
performance for severity performance, and the one apparent stability
improvement identified in a pilot — a roughly 82% reduction in
primary-PR-AUC standard deviation under identity-like initialization —
did not survive expansion to the full five-seed evaluation, where the
same configuration's standard deviation rose to within a comparable
range of the reference and its severity-split variance became worse
than the reference's own. Qubit count (four, six, and eight qubits) and
variational circuit depth (one, two, and three layers) both showed
configuration-dependent effects — four qubits reached the highest
primary mean observed anywhere in this project, and one layer reached
the tightest severity standard deviation observed anywhere — but both
ablations are confounded by a parameter-count change in the classical
reduction layer that scales with qubit count, and neither produced an
unambiguous, parameter-matched improvement. A controlled, five-seed
comparison of three circuit ansätze at matched qubit count and depth
(the reference `StronglyEntanglingLayers`, a hardware-efficient ring
topology, and a reduced-entanglement chain topology) found all three
within roughly one percent of each other on primary mean PR-AUC and four
percent on severity mean PR-AUC; the reference ansatz showed the
tightest primary-split variance of the three, and the reduced-entanglement
ansatz showed a reproducible specificity collapse on two of five severity
seeds together with the most mutually correlated quantum feature
channels of the three ansätze tested. Finally, quantum encoding — the
scale and functional form of the mapping from the classical projection
into rotation angles, and whether the classical features are re-encoded
before every variational layer or only once — was investigated across
five configurations in a two-seed pilot; a doubled encoding scale
produced a clear, multiply corroborated regression (the widest angle
spread, the weakest channel-target correlation, and a distinctly delayed
training convergence on severity, observed consistently across both
pilot seeds), a bounded-linear alternative to the existing hyperbolic-
tangent mapping regressed with a directly measured mechanism (nearly
half of all encoded angles saturating exactly at the boundary value),
and neither a reduced encoding scale nor data re-uploading produced a
seed-consistent improvement; none of the five encoding configurations
was judged to show sufficient evidence to justify expansion to a full
five-seed evaluation.

## 7. Representation Analysis

A dedicated representation-audit phase investigated the frozen
GraphSAGE embedding directly, independent of any quantum or classical
downstream head, using several complementary diagnostics. Lightweight
classifiers — logistic regression, a small multilayer perceptron, a
linear support-vector machine, and a random forest — were trained
directly on the frozen 128-dimensional embedding, using the identical
train/validation/test partition every end-to-end model in this project
uses, purely as a measure of how much predictive signal the
representation itself contains; these are diagnostic probes, not
competing end-to-end models, and were not trained under the same
protocol (optimizer, regularization, or early-stopping regime) as either
full model. On the primary split, the strongest of these diagnostics (a
small multilayer perceptron) reached a PR-AUC of 0.733, below both full
models — expected, since the diagnostics received no architecture-
specific tuning. On the severity split, however, a plain logistic
regression reached a PR-AUC of 0.465 (standard deviation 0.005 across
five seeds) — numerically above both the classical end-to-end reference
(0.449) and every QGNN-v4 configuration evaluated in this project (0.36
to 0.41), with markedly tighter cross-seed variance than either. This
finding does not establish that logistic regression is the strongest
model in this project's own end-to-end sense; it establishes that the
frozen embedding already carries more severity-relevant signal than
either full downstream head currently extracts from it, and that the
severity gap observed throughout this project is not solely attributable
to insufficient representational capacity in either the classical or the
quantum head. A principal-component analysis of the same embedding
showed that primary-split predictive signal is concentrated in a small
number of components — four principal components retained 90.9% of
variance and reached the single highest PR-AUC observed across the
entire dimensionality sweep tested (0.781) — while severity-split signal
was more evenly distributed across the full 128 dimensions, with
downstream classifier performance improving monotonically as more
components were retained. This asymmetry offers an independent
explanation for why a four-qubit quantum bottleneck performed
comparably to or better than wider bottlenecks on the primary split in a
separate ablation (Section 6), without requiring any assumption about
the quantum circuit's own behavior. A target-aware correlation analysis
found individual raw embedding dimensions correlating with the
disruption target at up to r ≈ 0.87, confirming the representation
encodes strongly task-relevant structure, while correlation with a
fresh-onset indicator specifically was markedly weaker (up to r ≈ 0.25),
consistent with the fresh-onset weakness documented in Section 10. A
graph-level audit found no meaningful correlation between a supplier's
network degree and its empirical disruption rate (Pearson r = 0.054),
ruling out a trivial structural explanation for the labels. Finally, an
inspection of the feature-construction pipeline confirmed that no
explicit historical-disruption feature — a supplier's own past
disruption state, time since a previous disruption, a rolling disruption
frequency, or a neighboring supplier's historical disruption state — is
computed or supplied to the model anywhere in the current pipeline; the
existing dynamic features are limited to rolling procurement and
delivery aggregates. This absence was identified and documented as a
concrete, testable proposal for future work (Section 22); no such
feature was implemented or evaluated in this project, and no claim is
made about what effect adding one would have.

## 8. Quantum Resource Analysis

The standing QGNN-v4 reference uses six qubits and two variational
layers under PennyLane's `StronglyEntanglingLayers` ansatz, contributing
36 trainable quantum parameters to a 817-parameter total head (the
remaining 781 parameters belong to the classical `Linear(128,6)`
reduction and `Linear(6,1)` output layers, shared in form with the
project's matched-capacity classical control). The two alternative
ansätze evaluated in the controlled ansatz comparison used fewer quantum
parameters at the same qubit count and depth — 24 for the
hardware-efficient ring topology and 12 for the reduced-entanglement
chain topology — and reached comparable mean PR-AUC on both splits
despite this reduced parameter budget, though the comparison is not
parameter-matched and this pattern should not be read as demonstrating
that fewer quantum parameters are inherently sufficient. Every circuit in
this project was executed on PennyLane's `default.qubit` simulator using
exact, analytic backpropagation (`diff_method="backprop"`) — there is no
shot noise, no hardware noise model, and no physical quantum device
anywhere in this investigation's results.

## 9. Seed Sensitivity

Table 1 reports the primary and severity per-seed results for the two
standing references, re-verified directly against the raw per-seed
metric artifacts during this validation pass (not retyped from an
earlier summary).

**Table 1 — Per-seed results, standing references (test split)**

| Model | Seed | Primary PR-AUC | Primary F1 | Primary MCC | Severity PR-AUC | Severity F1 | Severity MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical | 42 | 0.7874 | 0.7236 | 0.7255 | 0.4430 | 0.3838 | 0.3312 |
| Classical | 43 | 0.6888 | 0.6382 | 0.6478 | 0.4560 | 0.4431 | 0.3991 |
| Classical | 44 | 0.8745 | 0.7032 | 0.7055 | 0.4378 | 0.4228 | 0.3753 |
| Classical | 45 | 0.8491 | 0.6865 | 0.6988 | 0.4358 | 0.4022 | 0.3518 |
| Classical | 46 | 0.8354 | 0.6340 | 0.6449 | 0.4708 | 0.4045 | 0.3544 |
| Classical | **mean±std** | **0.8070±0.0655** | **0.6771±0.0355** | **0.6845±0.0324** | **0.4487±0.0131** | **0.4113±0.0201** | **0.3624±0.0231** |
| QGNN-v4 reference | 42 | 0.8682 | 0.6800 | 0.6898 | 0.3943 | 0.3793 | 0.3263 |
| QGNN-v4 reference | 43 | 0.6807 | 0.5950 | 0.5717 | 0.4496 | 0.5320 | 0.5541 |
| QGNN-v4 reference | 44 | 0.8197 | 0.6992 | 0.7005 | 0.4156 | 0.4248 | 0.3776 |
| QGNN-v4 reference | 45 | 0.7951 | 0.6156 | 0.6130 | 0.2900 | 0.3632 | 0.3111 |
| QGNN-v4 reference | 46 | 0.8924 | 0.6929 | 0.6947 | 0.4385 | 0.3941 | 0.3427 |
| QGNN-v4 reference | **mean±std** | **0.8112±0.0737** | **0.6565±0.0428** | **0.6539±0.0521** | **0.3976±0.0571** | **0.4187±0.0602** | **0.3824±0.0887** |

QGNN-v4's per-seed variance exceeds classical's on every metric shown,
most sharply on severity MCC (standard deviation 0.089 versus 0.023) —
driven substantially, though not exclusively, by seed 45's own severity
reversal under this architecture, discussed in Section 12. With only
five seeds, this table supports a directional statement about relative
variance and should not be read as a formally tested significance
result; no paired significance test was computed for any comparison in
this project.

## 10. Fresh-Onset Analysis

The primary split's test window contains zero fresh-onset positive
examples — it falls entirely inside the active window of the same
severity-5 event used to construct the severity split, during which
every positive example represents an already-ongoing disruption rather
than a new one — so fresh-onset PR-AUC is undefined on the primary split
for every configuration evaluated in this project, and is reported as
such rather than as zero. The severity split's test window contains 84
fresh-onset positive examples, identical across every seed and
configuration, since the underlying labels do not vary with model
configuration. Across every encoding configuration evaluated in the
final quantum-encoding investigation, fresh-onset PR-AUC on the severity
split ranged from 0.005 to 0.012 — a small positive value, not literally
zero, but statistically indistinguishable from a random ranking of
fresh-onset positives against the split's negatives. This same near-zero
pattern was observed for every model architecture evaluated throughout
this project, including the full classical GraphSAGE-Full baseline,
which has complete access to the graph structure and message-passing
information the frozen QGNN embedding does not directly propagate at
inference time. Because the full classical model shows the identical
floor, this project's evidence points to the absence of a genuine
leading indicator in the benchmark's available dynamic features at the
prediction horizon used, rather than to a limitation specific to any one
architecture, quantum or classical.

## 11. Calibration Analysis

Classical GraphSAGE-Full is the best-calibrated model in every direct
comparison performed in this project, on both splits, by a substantial
and consistent margin (primary expected calibration error 0.049 versus
QGNN-v4's 0.182; severity 0.090 versus 0.241). Within the space of
QGNN-v4 configurations investigated, output-side `LayerNorm` produced
the clearest calibration improvement over the un-normalized baseline,
adopted for this reason among others. Several later interventions — most
clearly, near-zero-start quantum-parameter initializations and certain
encoding variants — produced a calibration/ranking decoupling: numerically
better calibration (lower Brier score and expected calibration error)
accompanied by numerically worse ranking quality (lower PR-AUC) on the
same split. This pattern recurred often enough across independent phases
of this investigation (most clearly Phase 4 Stage 4b's initialization
scale sweep and Phase 4 Stage 5's ansatz comparison) that it is treated
as a real, reproducible property of this system rather than a one-off
coincidence, though the quantum-encoding investigation's own pilot did
not show this decoupling — calibration and ranking moved together for
every encoding configuration tested there — and this exception is stated
explicitly rather than smoothed over.

## 12. Discussion

The evidence gathered across this project's nine investigated factors is
most consistent with the following account. The frozen GraphSAGE
representation already encodes substantial, genuinely task-relevant
structure — confirmed independently through diagnostic classifiers,
principal-component structure, and direct target correlation — and this
representation is not the primary bottleneck limiting severity
performance, since a representation diagnostic on the identical
embedding reaches a higher severity PR-AUC than either full downstream
head. The severity condition's difficulty is better explained by a
genuine distribution-shift generalization problem than by an information
deficiency, supported directly by the finding that the same severity-5
periods are predicted comparably well when they are in-distribution
relative to training. Within this account, no single architectural
lever investigated in this project — the classical-to-quantum
projection, the quantum circuit's initialization or its scale, its
ansatz, or its input encoding — was shown to be the dominant factor
governing either the QGNN's seed sensitivity or its severity performance
gap relative to classical; each measurably affects some property of the
system (stability, calibration, representation redundancy) without
closing the performance gap. One specific, well-evidenced example of
seed-dependent behavior was traced to a concrete mechanism: seed 45's
baseline circuit concentrated an unusually large share of its
discriminative signal into a single output channel, and the introduction
of `LayerNorm` — which removes the optimization incentive for a circuit
to rely on one dominant channel — caused that seed's severity ranking to
regress sharply while every other seed's representation improved or held
flat under the identical change. This is treated as the strongest
available explanation for that specific seed's behavior, established
correlationally and mechanistically but not confirmed through a
controlled causal intervention (such as retraining under a modified
objective that explicitly discourages channel concentration), and it is
not claimed to be the sole or complete explanation for seed sensitivity
in this system more broadly. The one open, unresolved anomaly in this
project's evidence is the PCA-informed-projection result: a representation
diagnostic on a four-component PCA reduction of the frozen embedding
reaches strong primary performance, yet the identical PCA representation
feeding the quantum head performs substantially worse than the
unreduced 128-dimensional input does — a result this project's own
diagnostics cannot fully explain and that is flagged for a dedicated
follow-up rather than resolved here.

## 13. Limitations

The single most consequential limitation of this study is that the
severity/out-of-distribution test condition is drawn from a benchmark
containing only six disruption events in total, of which only one
reaches severity level 5 — the entire severity test population is drawn
from this one event's active window. Every severity finding in this
project, including the distribution-shift interpretation in Sections 5
and 12, should be read as evidence about this dataset's specific
severity-5 pattern, not as an established claim about generalization
across a diverse population of independent severe disruptions, which
this dataset cannot speak to. The primary split's test window contains
no fresh-onset examples, so fresh-onset anticipation could only be
evaluated on the severity split. The benchmark itself is a synthetic,
generator-produced dataset, and no claim in this project extends to
real-world supply-chain data, which was not evaluated. Several
investigations in this project — the projection, initialization-scale,
ansatz-adjacent, and encoding pilots — were conducted with two seeds
rather than the project's five-seed standing convention, and results
from these pilots are explicitly distinguished throughout this chapter
and the accompanying CSVs (`seed_type = pilot` versus `full`); the one
case where a two-seed pilot's apparent finding was subsequently tested
at five seeds (identity-like initialization) shows that a two-seed
result should not be treated as equivalent to a confirmed finding. The
qubit-count and ansatz comparisons in this project both carry a
parameter-count confound not fully separated from the architectural
factor under test. Finally, every result in this chapter comes from an
ideal, noise-free quantum simulator with no physical quantum hardware
involved at any stage, and no claim here should be read as a claim about
physical quantum-hardware behavior. A complete, dedicated treatment of
these and several additional limitations is provided in
`QGNN_V4_LIMITATIONS.md`.

## 14. Summary of Findings

Within the scope of this synthetic benchmark, this frozen GraphSAGE
encoder, this family of 6-qubit/2-layer QGNN-v4 architectures, and the
primary and severity splits evaluated, the systematic investigation
conducted across nine controlled experimental phases did not identify a
modification — to projection architecture, quantum initialization or its
scale, circuit ansatz, or quantum encoding — that produced a
reproducible, seed-robust improvement over the standing QGNN-v4
reference. Among end-to-end, fully-trained models, classical
GraphSAGE-Full remained the stronger model on the severity/out-of-
distribution condition throughout this investigation, while a
representation diagnostic on the same frozen embedding — not a competing
end-to-end model — reached a higher severity PR-AUC than either full
model, indicating the practical ceiling on this representation exceeds
what either downstream head currently extracts from it. This chapter's
contribution is not a higher benchmark score but a systematic,
evidence-scoped account of which architectural and training factors do
and do not govern this hybrid model's observed behavior on this
benchmark.
