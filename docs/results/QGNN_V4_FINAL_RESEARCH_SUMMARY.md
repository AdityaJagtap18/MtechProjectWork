# QGNN-v4 Final Research Summary

Consolidates every completed phase of the QGNN-v4 hybrid quantum-classical
architecture investigation into a single, evidence-scoped research
narrative. Companion documents: `QGNN_V4_MASTER_RESULTS.csv` (60-row
per-configuration/per-split metric table), `QGNN_V4_ABLATION_TABLE.csv`
(12-factor ruled-in/out table), `QGNN_V4_REFERENCE_COMPARISON.csv`
(Classical-vs-QGNN gap quantification), `QGNN_V4_LIMITATIONS.md` (full
limitations detail). This document does not repeat every number those
files carry — it synthesizes what they mean, in order.

Git commit at time of writing: `fc0f804b624e3953f9a9c301d90ac9d0dffe3d3e`,
branch `qgnn-package-restructure`.

---

## 1. Executive Summary

Across nine controlled experimental phases and ~230 individual training
runs, the hybrid QGNN-v4 architecture (a 6-qubit, 2-layer variational
quantum circuit on top of a frozen classical GraphSAGE embedding) was
systematically investigated for training instability, seed sensitivity,
and its performance relative to a full classical GraphSAGE-Full baseline
on a synthetic supply-chain disruption-prediction benchmark. **No
architectural, initialization, encoding, or ansatz modification tested
produced a reproducible, seed-robust improvement over the standing QGNN-v4
reference. Among the end-to-end, fully-trained models compared throughout
this project — classical GraphSAGE-Full vs. every QGNN-v4 configuration —
the classical GraphSAGE-Full baseline remained stronger on the
Severity/OOD generalization condition throughout.** This is an
end-to-end-model finding specifically, and it requires one important
qualification: a plain logistic regression trained only as a
*representation diagnostic* (Stage 1) — not a competing end-to-end
model, and not trained with the same protocol as either full model — on
the same frozen embedding reaches severity PR-AUC 0.465±0.005, which is
numerically **above both** the classical end-to-end reference (0.449) and
every QGNN-v4 configuration tested. This does not mean "the logistic
regression is the best model" in this project's own end-to-end sense; it
means the frozen embedding already carries more severity-relevant signal
than either full downstream head (classical or quantum) currently
extracts from it — the bottleneck is not (or not only) the quantum head's
expressiveness, and is not fully explained by the classical head's own
architecture either. A combination of the severity split's genuine
distribution shift (Stage 1) and the specific downstream training/
generalization behavior of every *full* head tested (quantum and
classical alike) better explains the observed gap than any single
architectural choice investigated.

## 2. Research Objective

To understand whether, and under what conditions, a hybrid
quantum-classical head (a variational quantum circuit reading a frozen,
classically-learned supply-chain graph embedding) can match or exceed a
classical graph neural network on supplier-disruption risk prediction —
and, more importantly, to systematically identify **which architectural
and training factors actually govern the hybrid model's behavior**,
rather than searching for the single highest test score.

## 3. Research Questions

1. Does the frozen GraphSAGE representation contain enough signal for
   this task, and is that signal the bottleneck for the quantum head?
2. Under what conditions is the QGNN's training stable, and what explains
   its seed-to-seed variance?
3. Does the QGNN outperform a classical GNN on temporal (primary)
   generalization? On severity/OOD generalization?
4. Which architectural factors (projection, initialization, ansatz,
   encoding, qubit count, depth) materially affect QGNN behavior, and
   which do not?
5. Can the QGNN anticipate genuinely new ("fresh-onset") disruptions, or
   only rank already-visible ones?

## 4. Experimental Setup

### 4.1 Dataset
`scm_v1_black_swan_seed43` — a synthetic, generator-produced supply-chain
benchmark (`SCM_DATASET_GENERATION_PLAN.md`). Horizon: 104 periods.
Target: `supplier_disrupted` over a 4-period-ahead prediction window.

### 4.2 Graph
2,670 nodes (300 suppliers, 100 materials, 50 plants, 200 products, 20
regions, 2,000 procurement-order nodes), 7,675 directed edges across 9
relation types. Degree distribution mean 5.75 (median 3.0, max 136) —
sparse (density 0.00215), long-tailed/hub-concentrated, one connected
component, average shortest path length 3.82. Supplier degree correlates
essentially not at all with disruption rate (Pearson r = 0.054) — labels
are not a trivial function of graph structure. Only 6 disruption events
total; only 1 reaches severity 5 (§ Limitations §1).

### 4.3 Classical GNN
`HeteroGraphSAGE`: type-specific input projection → 2 layers of
mean-aggregation `SAGEConv` (via PyG's `HeteroConv`) → supplier-node
readout → `MLP(128→64→1)` classifier. `hidden_dim=128`. Trained
end-to-end, then frozen for every downstream QGNN/matched-classical head.

### 4.4 QGNN
`Linear(128,6) → π·tanh → AngleEmbedding(RY) → StronglyEntanglingLayers
(6 qubits, 2 layers) → PauliZ(6) → LayerNorm(6, no affine) →
Linear(6,1)`. `qml.qnn.TorchLayer`, `default.qubit`,
`diff_method="backprop"` (exact, noise-free simulation) throughout every
phase.

### 4.5 Training
Adam, lr=0.001, weight_decay=0.0001, max_epochs=100, early-stopping
patience=10 (validated against patience=25 in Phase 2 — no material
difference), balanced class weighting (`pos_weight` from train-split
counts), threshold fixed at 0.5 throughout.

### 4.6 Evaluation
PR-AUC (primary metric — justified below, §9), ROC-AUC, F1, precision,
recall, specificity, balanced accuracy, MCC, Brier score, ECE, confusion
matrix, all at the fixed 0.5 threshold, computed identically for every
configuration via one shared `metrics.compute_classification_metrics`
implementation.

## 5. Classical Baseline

**Primary PR-AUC: 0.807 ± 0.066. Severity PR-AUC: 0.449 ± 0.013** (5
seeds, `QGNN_V4_CLASSICAL_BASELINE.md`). Full metric set in
`QGNN_V4_MASTER_RESULTS.csv` rows 1–2. Severity recall is exactly 0.4406
in every one of 5 independent seeds (the model catches the identical set
of already-ongoing positives every run) — a real, documented finding, not
a bug.

## 6. QGNN Baseline

**Primary PR-AUC: 0.811 ± 0.074. Severity PR-AUC: 0.398 ± 0.057**
(LayerNorm-no-affine, the standing reference adopted from Phase 2c
onward, 5 seeds). The *raw*, pre-LayerNorm baseline (Phase 1/2b) actually
had a nominally higher primary mean (0.833) but with double the variance
(std 0.140) and roughly 3× worse calibration (ECE 0.285 vs. classical's
0.049) — LayerNorm was adopted specifically because it traded a small
mean decline for substantially better stability and calibration, not
because it closed the gap to classical.

## 7. Phase-by-Phase Experimental Investigation

### 7.1 Initial Diagnostics (Phase 1)
Instrumented per-epoch train-PR-AUC and quantum-circuit gradient-norm
logging on the 20 existing benchmark runs. **Ruled out barren plateaus
and vanishing gradients** at this scale (6 qubits, 2 layers) — quantum
gradients stayed in the 0.1–0.3 range throughout, generally *growing*
during early-stopped runs. Found two distinct failure modes hiding under
"seed variance": severity's near-immediate early stopping (validation
saturates in 1–5 epochs for 4/5 seeds because the validation carve-out is
drawn from the same in-distribution severity 1–3 data), and a
non-monotonic train-PR-AUC trajectory on primary that, counter-intuitively,
*correlated with better* test generalization (r=0.93, n=5).

### 7.2 Patience (Phase 2)
Patience 10 → 25. **Ruled out early stopping as the cause**: a positive
control showed 18/20 runs select the identical checkpoint either way, and
every "unchanged" seed's validation PR-AUC genuinely never improved given
15–25 more epochs of search room. Recommended reverting to patience=10
(no benefit, 1.4–1.8× more compute).

### 7.3 Output Scaling (Phase 2b)
Four variants (trainable α, trainable α+β, fixed α=0.5, fixed α=2.0) —
mathematically redundant with the existing `Linear(6,1)`, an
optimization-dynamics test only. All four landed within noise of the raw
baseline on both splits. **Not supported as a major contributor.**

### 7.4 LayerNorm (Phase 2c)
The first genuinely non-redundant change tested. Cut severity recall/F1
standard deviation by ~10×, eliminated threshold-0.5 prediction collapse
(2 seeds → 0), and improved calibration substantially on both splits — at
the cost of primary PR-AUC mean (0.833→0.811) and one severity seed
(45) reversing sharply (0.385→0.290). Adopted as the standing output
stage specifically for stability/calibration, not for closing the
Classical gap (it did not).

### 7.5 Seed-45 Investigation (Phase 2d)
Diagnostic-only, no retraining beyond the existing three runs (baseline,
LN-no-affine, LN-affine). Reproducibility fully verified (identical
encoder checkpoint, identical test rows, zero numerical difference in a
from-scratch forward-pass reconstruction). **Mechanism found**: seed 45's
baseline circuit concentrated nearly all its discriminative signal into
one PauliZ channel (correlation 0.524 with the label — the highest of any
seed); LayerNorm removes the optimization incentive for this
concentration, and under LayerNorm that seed's best channel drops to
0.282 (roughly halved) while every other seed's improves or holds flat.
Gradient pathology, dataset differences, and general training-dynamics
explanations were all directly ruled out (§12). Classified: **seed-
dependent representation distortion, medium confidence, not proven
causal** — the correct, cautious framing given a diagnostic-only phase
that did not retrain under controlled interventions to confirm causality.

### 7.6 Qubit Count / Depth (Phase 3)
4q/2L reached the highest primary mean (0.847±0.102) of any configuration
in the whole project; 8q/2L was comparable to the 6q reference on both
metrics; 6q/1L had the tightest severity std observed anywhere (0.017)
but the worst primary mean (0.693); 6q/3L's severity run was killed
mid-batch (zero usable data). **Confounded with classical bottleneck
parameter count** (qubit count changes `Linear(128,n_qubits)` too) — not
a pure quantum-capacity ablation.

### 7.7 Representation Audit (Phase 4 Stage 1)
The single most consequential finding of the whole project. A plain
logistic regression on the identical frozen 128-D embedding reaches
severity PR-AUC 0.465±0.005 — **above both the classical head (0.449) and
every QGNN configuration tried (0.36–0.41)**, with far tighter variance.
PCA showed primary signal is low-rank (4 components: 0.781 PR-AUC, best
of any tested dimensionality — independently corroborating Phase 3's
4-qubit finding) while severity signal is distributed across all 128
dimensions (PCA-128 best: 0.480). Graph diagnostics confirmed labels are
not a trivial function of degree/centrality. Fresh-onset audit confirmed
zero fresh-onset examples in primary, 84 in severity, near-zero PR-AUC
for any head on severity's fresh-onset subset. A severity-level breakdown
showed the primary split's own test window sits entirely inside the
severity-5 event's *already-ongoing* period — the same severity-5 periods
score 0.81 PR-AUC there (in-distribution) vs. 0.38–0.44 under the true
OOD severity split, directly demonstrating the severity problem is a
**distribution-shift generalization failure**, not an information
deficiency.

### 7.8 Projection (Phase 4 Stage 2)
Nonlinear projection, pre-projection LayerNorm, and PCA-informed
projection (4/6/8 components) were all tested against the existing single
`Linear(128,6)` bottleneck. **Every alternative underperformed on both
splits** in a 2-seed pilot; none was expanded to 5 seeds. The PCA-informed
variant's underperformance — despite far fewer parameters — directly
contradicts Stage 1's own finding that a linear classifier on the same
PCA representation performs well, an unresolved anomaly flagged for
future work, not explained away.

### 7.9 Initialization (Phase 4 Stage 4 + 4b)
Default (uniform[0,2π]), small-Gaussian (std=0.01), and identity-like
(exact zero) were compared, then generalized to a 5-point Gaussian-std
sweep (0.001–0.050) plus a 5-seed expansion of identity-like. **Finding
A**: the primary-PR-AUC improvement from small-Gaussian/identity-like
initializations comes from being near-zero at all, not from a specific
std — primary PR-AUC is nearly identical across the entire 0.001–0.050
range for a given seed. **Finding B**: every near-zero initialization
consistently degrades severity, best explained by a ranking-specific
overfitting-to-validation failure (calibration metrics are actually
*better* than the reference despite far worse PR-AUC — decoupled, not a
general quality collapse). **Finding C**: identity-like's apparent
2-seed variance reduction (std 0.094→0.017) **did not survive 5-seed
expansion** (std rose to 0.059; severity std became worse than the
reference, 0.112 vs. 0.057) — the single clearest illustration in this
project of why 2-seed pilots cannot be read as confirmed findings.

### 7.10 Ansatz (Phase 4 Stage 5)
StronglyEntanglingLayers (reference, 36 quantum params), hardware-
efficient ring (24 params), and reduced-entanglement chain (12 params)
compared at matched 6 qubits/2 layers, 5 seeds each. All three land
within ~1% of each other on primary mean and ~4% on severity mean —
**ansatz topology is not the dominant factor behind the Classical gap.**
The reference ansatz has the tightest primary variance of the three; the
reduced-entanglement ansatz shows a reproducible severity specificity
collapse on 2 of 5 seeds and the most mutually-correlated (least
differentiated) quantum feature channels — a real representation property
without a corresponding score benefit or penalty on its own. Not
parameter-matched.

### 7.11 Quantum Encoding (Encoding Investigation)
Encoding scale (0.5π, 2π vs. the reference π), a bounded-linear
alternative to tanh (clip), and data re-uploading were tested, 2-seed
pilot. 2π scale is a clear, multiply-corroborated regression (measured:
widest angle spread, worst channel-target correlation, a distinct
both-seed severity convergence delay). Linear clip regresses with a
directly measured mechanism (46% of angles saturate exactly at the
boundary, more information loss than tanh's asymptotic approach). Neither
0.5π nor re-uploading showed a seed-consistent improvement. **No
configuration was expanded to 5 seeds — the investigation concluded that
encoding did not justify further budget.**

## 8. Cross-Phase Results

Full table: `QGNN_V4_MASTER_RESULTS.csv` (60 rows, every phase, every
configuration, both splits, with `n_seeds`/`seed_type` marking pilot vs.
full evaluations and a `decision` column per row).

## 9. Classical vs QGNN Comparison

**Why PR-AUC is the primary metric here, not accuracy**: the target
class is heavily imbalanced (overall supplier-period disruption rate
3.33%, Stage 1). Accuracy on a 3%-positive-rate target is dominated by
the majority class and is uninformative (a constant-negative predictor
already scores ~97%) — this project never reports "accuracy" as a
result for this reason. PR-AUC (unlike ROC-AUC) is specifically sensitive
to performance on the minority (positive/disruption) class, which is the
class that matters operationally, and — unlike F1/precision/recall — it
is threshold-independent, avoiding the fixed-0.5-threshold calibration
artifact documented directly in this project (`QGNN_V4_BENCHMARK.md` §7:
a poorly-calibrated head's threshold-based metrics can look dramatically
better or worse purely from where its probability distribution happens
to sit relative to 0.5, independent of true ranking quality).

**Primary/Temporal**: classical and QGNN are close on PR-AUC (0.807 vs.
0.811 — within each other's noise), but classical has meaningfully higher
recall (0.968 vs. 0.898) and is **substantially better calibrated** (ECE
0.049 vs. 0.182 — roughly 3.7× worse for QGNN).

**Severity/OOD**: classical leads on PR-AUC (0.449 vs. 0.398), ROC-AUC
(0.728 vs. 0.657), and calibration (ECE 0.090 vs. 0.241) — with tight,
directionally consistent per-seed variance on both sides. QGNN reaches
essential parity on F1 (0.419 vs. 0.411) and a marginally higher MCC
(0.382 vs. 0.362) — the one metric family where QGNN is not behind.

**Required qualification — end-to-end comparison vs. representation
diagnostic, not to be conflated.** Everything above compares two
end-to-end, fully-trained models under this project's shared protocol.
Stage 1 separately ran a plain logistic regression on the *same frozen
128-D embedding* purely as a representation diagnostic — a different
kind of measurement, not a third competing end-to-end model, and not
trained/evaluated under the identical protocol as the two full models
(different classifier family, different regularization, no early
stopping). That diagnostic reached severity PR-AUC 0.465±0.005 — **above
both** the classical end-to-end reference (0.449) and the QGNN reference
(0.398). This does **not** mean "logistic regression is the strongest
model" in the sense §5–§8 above use "strongest"/"stronger" — it means the
frozen embedding already contains more severity-relevant signal than
either full downstream head (classical or quantum) currently extracts
from it. Wherever this document says classical GraphSAGE-Full is
"stronger" or "the stronger model" on severity, that statement is scoped
to the end-to-end comparison specifically, and this diagnostic finding
should be read alongside it, not treated as contradicting it.

**Pattern, not a ranking**: ranking quality (PR-AUC/ROC-AUC) and
calibration both favor classical, more so on severity than primary;
classification-at-a-fixed-threshold metrics (F1, MCC) are closer to
parity, especially on severity; variance is comparable on primary but
QGNN's calibration variance is consistently worse throughout.

## 10. Primary / Temporal Generalization

Both models perform reasonably well and comparably on primary (both
above 0.8 PR-AUC), with the practical difference concentrated in
calibration, not ranking. Multiple QGNN configurations (4-qubit,
identity-init 2-seed pilot, small-Gaussian) nominally exceed 0.81–0.85
primary PR-AUC in various phases, but every such case either carries
substantially elevated variance (4-qubit std 0.102 vs. reference's 0.074;
small-Gaussian's paired-seed gain came with 1.5× the reference's variance)
or is a 2-seed pilot not confirmed at full scale.

## 11. Severity / OOD Generalization

The harder condition throughout — every model, every configuration,
scores substantially lower on severity than primary (roughly halved).
Stage 1's finding is the clearest evidence available on the mechanism:
the same severity-5 periods score 0.81 PR-AUC when in-distribution (under
the primary split) vs. 0.38–0.45 when genuinely out-of-distribution
(severity split) — **this strongly supports a predominantly
distribution-shift explanation, rather than an information-content
deficiency, for this dataset's severity gap.** No architectural
intervention tested (projection, initialization, ansatz, encoding) closed
this gap; several (small-Gaussian init, PCA projection, most encoding
alternatives) made it distinctly worse. Among end-to-end models,
classical GraphSAGE-Full remained the stronger model on this condition
throughout the entire project — though, as §9's correction notes, a
representation diagnostic on the same frozen embedding reaches a higher
severity PR-AUC than classical's own end-to-end head, so the ceiling is
not classical's architecture either. **This distribution-shift
interpretation is itself scoped by a real limitation**: the severity
test condition is dominated by a single underlying severity-5 event
(`QGNN_V4_LIMITATIONS.md` §1), so it should be read as strong evidence
for *this dataset's* specific severity-5 pattern, not as an established
general claim about generalization across many independent severe
events — that broader claim remains untested.

## 12. Fresh-Onset Detection

Primary split: **0 fresh-onset positive examples** in every configuration
tested (the test window falls entirely inside the standing severity-5
event's already-ongoing period) — fresh-onset PR-AUC is undefined there,
not zero, and never fabricated as such. Severity split: 84 fresh-onset
positives, identical across every seed and configuration (the underlying
labels don't change). **Fresh-onset PR-AUC sits at 0.004–0.012 across
every model tested in this entire project — including the full classical
GraphSAGE-Full baseline, which has complete graph and message-passing
access.** This is not "zero performance" in the literal sense (the metric
is a small positive number, not exactly 0), but it is statistically
indistinguishable from a random ranking. Because the *full* classical
model shows the identical floor, this is established as a property of
the benchmark's input features at the prediction horizon used (no
model tested captures a genuine leading indicator of a disruption that
hasn't started yet) — not a limitation specific to any architecture,
quantum or classical.

## 13. Seed Sensitivity

The dominant, recurring theme across every phase. The raw QGNN baseline's
primary std (0.140) was roughly double the classical control's (0.065)
from the very first benchmark. LayerNorm reduced but did not eliminate
this. Two specific, well-evidenced mechanisms were identified: (a) seed
45's representation-concentration/LayerNorm interaction (§7.5), and (b)
identity-like initialization's apparent stability gain not surviving
5-seed expansion (§7.9) — the clearest demonstration that seed
sensitivity in this system is itself seed-count-sensitive to measure
correctly. No single factor tested was shown to be the primary driver of
overall seed variance; several (LayerNorm, initialization scale, ansatz)
each measurably affect it in different, sometimes opposite directions on
primary vs. severity.

## 14. Quantum Representation Analysis

The PauliZ output channels were probed directly (not estimated) in three
separate investigations (Phase 2d, Stage 5, Encoding Investigation).
Consistent findings: (a) individual channels correlate strongly with the
target when the ansatz/encoding is unchanged (up to r≈0.87 raw embedding
dims per Stage 1; up to r≈0.72 PauliZ channels per Stage 5) — the circuit
clearly learns task-relevant structure; (b) channels are frequently
highly mutually correlated (cross-channel correlation 0.68–0.93 depending
on configuration) — the nominal 6-channel output often carries closer to
the information content of fewer effective channels; (c) both encoding
scale and ansatz choice measurably shift these properties (e.g., 2π
encoding scale degrades channel-target correlation; reduced-entanglement
ansatz maximizes cross-channel redundancy) — **the quantum representation
is demonstrably not inert to these choices, even where the resulting
downstream metrics don't move accordingly.**

## 15. Calibration

Classical GraphSAGE-Full is the best-calibrated model in every comparison
run in this project, on both splits, by a substantial and consistent
margin (§9). Within QGNN configurations, LayerNorm is a clear calibration
win over the raw baseline; near-zero-start initializations and several
encoding variants show a **calibration/ranking decoupling** — better
Brier/ECE alongside *worse* PR-AUC (most clearly: Stage 4b's small-
Gaussian severity result, Stage 5's hardware-ring ansatz) — a real,
repeatedly-observed pattern in this project, not a one-off. The Quantum
Encoding Investigation's own pilot did *not* show this decoupling
(calibration and ranking moved together for every encoding tested there)
— stated explicitly since it's an exception to the pattern seen
elsewhere, not confirmation that the pattern is universal.

## 16. Computational / Quantum Resource Analysis

| | Value |
|---|---|
| Qubits (standing reference) | 6 |
| Variational layers | 2 |
| Quantum trainable parameters | 36 (StronglyEntangling) / 24 (hardware-ring) / 12 (reduced-chain) |
| Total trainable head parameters | 817 / 805 / 793 respectively |
| Frozen encoder parameters | ≈1.2M (GraphSAGE-Full, `hidden_dim=128`, 2 layers) |
| Simulator | PennyLane `default.qubit` |
| Differentiation method | `backprop` (exact, analytic) |
| Shots | None (analytic expectation values) |
| Hardware | None — classical simulation throughout |
| Noise model | None (ideal simulation) |

No claim of quantum computational advantage is made or implied anywhere
in this project — every result is from an ideal, noise-free classical
simulation of a small (6-qubit) circuit. See `QGNN_V4_LIMITATIONS.md` §6
and §9 for the full computational-limitations statement, including that
no systematic runtime/speedup comparison was performed.

## 17. What Has Been Ruled Out

Full table: `QGNN_V4_ABLATION_TABLE.csv`. Summary: training patience and
output-scale reparameterization are the two factors most clearly "not
supported as a major contributor" — both showed no effect distinguishable
from noise. Projection architecture, quantum initialization (as tested),
ansatz topology, and quantum encoding are each "not supported as a major
contributor to closing the Classical gap," while each still measurably
affects secondary properties (stability, calibration, representation
redundancy) in ways worth carrying forward as context, not as a settled
"this doesn't matter at all" conclusion.

## 18. What Remains Uncertain

- **The PCA-informed-projection anomaly** (Stage 2): why a linear
  classifier on PCA(4) reaches 0.78 primary PR-AUC while the same PCA(4)
  representation feeding the quantum head reaches only 0.39 — flagged,
  not resolved.
- **Whether an intermediate encoding scale (~0.05π–0.10π range) offers a
  genuine primary/severity trade-off improvement** — the encoding
  investigation's std=0.050 result hinted at this but was driven by a
  single pilot seed.
- **The exact causal mechanism behind seed 45's representation
  distortion** — a well-evidenced correlational finding (Phase 2d),
  explicitly not confirmed via a controlled causal intervention.
- **Whether the severity distribution-shift problem is fundamentally
  solvable by any architecture on this frozen representation**, or
  requires new input features (e.g. explicit historical-disruption
  features, identified as absent by Stage 1's temporal-feature audit but
  never implemented or tested in this project).

## 19. Limitations

See `QGNN_V4_LIMITATIONS.md` for the full, dedicated treatment. Headline
items: the severity condition is dominated by a single underlying event;
fresh-onset cannot be evaluated on primary at all; the dataset is
synthetic; several investigations used 2-seed pilots not equivalent to
the 5-seed standing references; qubit-count and ansatz comparisons carry
parameter-count confounds; every result is from an ideal, noiseless
simulator with no physical quantum hardware involved.

## 20. Research Findings

1. The frozen GraphSAGE representation contains substantial, genuine
   predictive signal — confirmed independently by diagnostic classifiers,
   PCA structure, and direct target-correlation analysis (Stage 1).
2. The representation is not the exclusive bottleneck for severity
   performance: a simple linear model on it already matches/exceeds
   every quantum configuration tried, **and also exceeds the classical
   end-to-end head's own severity PR-AUC** (0.465 vs. 0.449) — as a
   representation diagnostic, not a competing end-to-end model (§9).
3. Severity's difficulty is predominantly a genuine train/test
   distribution-shift problem — strongly supported by the
   primary-split-contains-severity-5-in-distribution comparison (Stage
   1) — rather than an information deficiency, though this is evidence
   from one dataset with a single underlying severity-5 event, not a
   general claim (`QGNN_V4_LIMITATIONS.md` §1).
4. No single architectural factor investigated (projection, ansatz,
   initialization, encoding, qubit count, depth) produced a reproducible,
   seed-robust improvement over the standing QGNN-v4 reference.
5. Calibration and ranking quality repeatedly decouple in this system —
   several interventions improve one while worsening the other.
6. Seed sensitivity is real, multi-causal, and itself sensitive to how
   many seeds are used to measure it (the identity-init 2-seed→5-seed
   result is the clearest demonstration in the project).
7. Fresh-onset anticipation is not demonstrated by any model tested,
   quantum or classical — a property of the benchmark's available
   features, not of any architecture.
8. Among end-to-end, fully-trained models, classical GraphSAGE-Full
   remains the stronger model on the Severity/OOD condition throughout
   every phase of this investigation — with the representation-diagnostic
   qualification in Finding 2 above kept explicit, not implied away.

## 21. Final Conclusion

Within this synthetic SCM dataset, this frozen GraphSAGE-Full
representation, this ideal PennyLane `default.qubit` simulation, this
6-qubit/2-layer QGNN-v4 architecture family, and the primary/severity
splits evaluated, **the investigated modifications (projection,
initialization, initialization scale, ansatz, and quantum encoding) did
not produce a reproducible, seed-robust improvement over the standing
QGNN-v4 reference. Among end-to-end, fully-trained models, the classical
GraphSAGE-Full baseline remained stronger on the Severity/OOD
generalization condition throughout** — a representation diagnostic
(not an end-to-end model) on the same frozen embedding numerically
exceeds classical's own severity PR-AUC, which is itself evidence that
the ceiling is not attributable to the classical architecture either
(§9, Finding 2). This conclusion is scoped exactly to what was tested —
it is not a claim that quantum computing is inferior to classical GNNs
in general, that quantum hardware would behave identically, that a
different dataset would reproduce this result, that this architecture is
globally optimal, that every possible encoding has been exhausted, or
that a quantum advantage in this problem class is impossible. Within the
boundaries actually investigated, the evidence is consistent and
reproducible: it does not support adopting any of the tested
modifications as a replacement for the standing reference, and it
supports treating classical GraphSAGE-Full as the stronger current
end-to-end option for this specific severity/OOD condition — while the
representation diagnostic indicates the practical ceiling on this
frozen embedding is higher than either full model currently reaches.

### Explicitly demonstrated

- QGNN can learn useful predictive representations (channel-target
  correlations up to r≈0.87, comparable primary PR-AUC to classical).
- QGNN achieves strong Primary PR-AUC in several configurations.
- QGNN is measurably sensitive to initialization/seed/architecture
  interactions, in specific, evidenced (not merely asserted) ways.
- Quantum encoding measurably affects the learned representation (angle
  distribution, channel correlations), independent of whether it affects
  final metrics.
- Severity/OOD performance is substantially harder than primary for every
  model tested.
- Among end-to-end models, classical GraphSAGE-Full is stronger on the
  established Severity reference across ranking, calibration, and
  (mostly) classification metrics — while a representation-diagnostic
  logistic regression on the same frozen embedding (not an end-to-end
  model) numerically exceeds classical's own severity PR-AUC (§9).
- Fresh-onset detection remains near the noise floor for every model
  tested, including the full classical baseline.

### Explicitly not demonstrated

- That QGNNs are inherently inferior to classical GNNs as a model class.
- That real quantum hardware would reproduce this project's simulator
  results.
- That a different dataset (real-world or otherwise) would show the same
  pattern.
- That the specific 6-qubit/2-layer/StronglyEntangling architecture used
  as the reference is globally optimal among all possible QGNN designs.
- That all possible quantum encodings, ansätze, or initializations have
  been exhausted — only the specific, documented set tested here.
- That a quantum advantage on this or related problems is impossible.

## 22. Future Work

Per the evidence generated (not as a default "more experiments" list):

1. **Investigate causal, leakage-safe temporal features** (Stage 1's own
   proposal: `periods_since_last_disruption`, rolling disruption
   frequency, graph-propagated neighbor disruption history) — the one
   concrete, evidence-backed lever identified that targets the
   fresh-onset and severity-generalization weaknesses directly, rather
   than another architectural variant on the same frozen representation.
2. **Resolve the PCA-projection anomaly** (§18) before any further
   projection-architecture work — it is the one unexplained result that
   directly contradicts an otherwise-clean finding (Stage 1's PCA
   diagnostic) and deserves isolation before being dismissed or pursued
   further.
3. **If encoder architecture is investigated**, it should be preceded by
   confirming (or refuting) whether a different graph encoder (GCN, GAT)
   changes the diagnostic-classifier severity ceiling found in Stage 1 —
   otherwise a new encoder risks repeating this project's central
   finding (the head architecture is not the primary bottleneck) under a
   different downstream head.
4. **A real statistical significance test** (e.g. a paired bootstrap over
   matched seeds) for the handful of comparisons where 5-seed evidence
   exists and the direction is consistent (e.g. classical vs. QGNN on
   severity) would strengthen any claim beyond "directional evidence."

Per this consolidation's own instruction, none of the above is
recommended to be started automatically — each requires its own
scoped, controlled design before execution.

---

## Appendix A — Thesis-Ready Tables

### Table 1 — Technology Stack

| Component | Value |
|---|---|
| Language | Python 3.12.3 |
| Deep learning framework | PyTorch 2.14.0+cu130 |
| Graph learning | PyTorch Geometric (`HeteroConv`, `SAGEConv`) |
| Quantum framework | PennyLane 0.45.1 (`qml.qnn.TorchLayer`) |
| Quantum simulator | `default.qubit` (ideal, noise-free) |
| Differentiation method | `backprop` (exact/analytic) |
| CUDA | Available on the development machine, not used (all computation CPU, for consistency with the noise-free-simulator, no-hardware scope) |
| Classical ML diagnostics | scikit-learn (`LogisticRegression`, `MLPClassifier`, `RandomForestClassifier`, `LinearSVC`) |
| Version control | git, branch `qgnn-package-restructure` |

### Table 2 — Dataset Characteristics

| Property | Value |
|---|---|
| Dataset ID | `scm_v1_black_swan_seed43` |
| Horizon | 104 periods |
| Prediction target | `supplier_disrupted`, 4-period-ahead window |
| Overall supplier-period disruption rate | 3.33% |
| Total disruption events | 6 (3 logistics, 1 supplier_failure, 1 geopolitical, 1 cyberattack) |
| Event severity distribution | severity 1 ×2, severity 2 ×3, severity 5 ×1 |
| Period-level severity distribution | severity 0: 39 periods, 1: 2, 2: 26, 5: 37 |
| Primary split | temporal 70/15/15 train/validation/test |
| Severity split | train severity ≤3, test severity 4–5 |
| Fresh-onset positives | Primary: 0. Severity: 84 (identical across seeds/configs) |

### Table 3 — Graph Characteristics

| Property | Value |
|---|---|
| Total nodes | 2,670 |
| Node types | supplier (300), material (100), plant (50), product (200), region (20), procurement (2,000) |
| Total edges | 7,675, across 9 relation types |
| Degree distribution | mean 5.75, median 3.0, min 1, max 136 |
| Graph density | 0.00215 |
| Connected components | 1 (all 2,670 nodes) |
| Average shortest path length | 3.82 |
| Supplier degree vs. disruption-rate correlation | r = 0.054 (not a trivial structural predictor) |

### Table 4 — Classical vs. QGNN Reference Performance

Full 7-metric × 2-split gap table: `QGNN_V4_REFERENCE_COMPARISON.csv`
(absolute/relative gaps computed as Classical − QGNN). The wide-format
view below adds the Stage 1 logistic-regression **representation
diagnostic** explicitly, per this validation pass's own requirement not
to mix it with the end-to-end comparison without a clear label — it is
**not** an end-to-end model and was not trained under either full
model's protocol; it is included here specifically because it is
numerically above both end-to-end models on severity PR-AUC, a fact
that must not be obscured by omission.

| Model | Type | Primary PR-AUC | Severity PR-AUC | Primary ROC-AUC | Severity ROC-AUC | Primary MCC | Severity MCC |
|---|---|---:|---:|---:|---:|---:|---:|
| Classical GraphSAGE-Full | end-to-end (5 seeds) | 0.8070 | 0.4487 | 0.9871 | 0.7275 | 0.6845 | 0.3624 |
| QGNN-v4 standing reference | end-to-end (5 seeds) | 0.8112 | 0.3976 | 0.9775 | 0.6566 | 0.6539 | 0.3824 |
| Logistic regression (frozen embedding) | **representation diagnostic** (5 seeds, not an end-to-end model) | 0.6706 | **0.4646** | 0.9662 | 0.7311 | 0.5532 | 0.4336 |

**Gap statement (exact metric and split named, per this validation's own
requirement — never "X% worse overall")**: Classical − QGNN-v4 Primary
PR-AUC = −0.0042 absolute (−0.52% relative; QGNN nominally ahead, within
noise). Classical − QGNN-v4 Severity PR-AUC = +0.0511 absolute (+11.39%
relative; Classical ahead). Classical − QGNN-v4 Severity ECE = −0.1501
absolute (Classical better calibrated by this amount; lower is better
for ECE, so the negative sign here means QGNN's ECE is larger). Every
other metric/split pair is in `QGNN_V4_REFERENCE_COMPARISON.csv`,
individually, not aggregated into a single percentage.

### Table 5 — QGNN Ablation Experiments

See `QGNN_V4_MASTER_RESULTS.csv` (60 rows) and `QGNN_V4_ABLATION_TABLE.csv`
(12-factor summary with evidence and interpretation per factor).

### Table 6 — Quantum Resource Comparison

| Ansatz | Qubits | Layers | Quantum params | Total head params | Entanglement |
|---|---:|---:|---:|---:|---|
| StronglyEntanglingLayers (reference) | 6 | 2 | 36 | 817 | PennyLane built-in range pattern |
| Hardware-efficient ring | 6 | 2 | 24 | 805 | CNOT ring (6 edges) |
| Reduced entanglement | 6 | 2 | 12 | 793 | CNOT chain (5 edges) |
| 4-qubit variant (Phase 3) | 4 | 2 | 24 | 545 | StronglyEntangling range pattern |
| 8-qubit variant (Phase 3) | 8 | 2 | 48 | 1,089 | StronglyEntangling range pattern |

### Table 7 — Seed Stability (Primary PR-AUC std, 5-seed configurations)

| Configuration | Primary PR-AUC std |
|---|---:|
| Classical GraphSAGE-Full | 0.066 |
| QGNN raw baseline (no LayerNorm) | 0.140 |
| QGNN LayerNorm-no-affine (reference) | 0.074 |
| QGNN LayerNorm-affine | 0.073 |
| Ansatz: hardware-efficient ring | 0.147 |
| Ansatz: reduced entanglement | 0.105 |
| Init: identity-like (5-seed expansion) | 0.059 |

### Table 8 — OOD / Severity Analysis

| Condition | PR-AUC | Note |
|---|---:|---|
| Severity-5 periods, in-distribution (under primary split) | 0.807–0.811 | Same periods, model trained on data including this range |
| Severity-5 periods, out-of-distribution (severity split test) | 0.384–0.449 | Same periods, model never trained on severity ≥4 |

Directly demonstrates the severity gap is predominantly a distribution-
shift effect, not an information deficiency (Stage 1, §11 above).

### Table 9 — Fresh-Onset Analysis

| Split | n Fresh-Onset | Fresh-Onset PR-AUC range (all models tested) |
|---|---:|---|
| Primary | 0 | Undefined (no fresh-onset examples exist) |
| Severity | 84 | 0.004–0.012 (every architecture, quantum and classical) |

### Table 10 — Experimental Limitations

See `QGNN_V4_LIMITATIONS.md` for the full 10-point treatment (severity
event-diversity limitation, fresh-onset scope, synthetic-dataset scope,
seed/pilot-vs-full distinctions, parameter-count confounds, ideal-
simulator-only scope, frozen-encoder scope, fixed-threshold scope,
computational/runtime scope, and the bounded meaning of every "ruled
out" classification).

## Appendix B — Final Decision Matrix

| Area | Finding | Confidence | Status |
|---|---|---|---|
| Primary temporal performance | QGNN and Classical are close (0.811 vs. 0.807); no configuration shows a robust, seed-consistent edge over the reference | Medium | Established at current evidence level |
| Severity/OOD performance | Classical leads consistently (0.449 vs. 0.398); gap is predominantly distribution-shift-driven, not information-deficiency-driven | High | Established (5-seed reference + independent in-distribution/OOD comparison) |
| Seed stability | Real, multi-causal, and itself sensitive to seed count (2-seed pilots have repeatedly overstated stability gains) | High | Established as a general caution; specific mechanisms only partially explained |
| Representation quality | Frozen embedding contains strong signal; not the primary severity bottleneck | High | Established (diagnostic classifiers + PCA + severity-level breakdown, all 5-seed) |
| Initialization | Near-zero start trades primary gain for severity cost; apparent stability gains did not survive 5-seed expansion | Medium-High | Established for the specific configurations tested |
| Projection | No alternative to the existing linear bottleneck improved either split in a 2-seed pilot | Low-Medium | Pilot-level evidence only, not expanded |
| Ansatz | Topology is not the dominant factor behind the Classical gap; measurably affects stability and representation redundancy | Medium | Established (5-seed, but not parameter-matched) |
| Encoding | No scale/type/re-uploading variant showed a seed-consistent improvement | Low-Medium | Pilot-level evidence only, not expanded |
| Fresh-onset detection | Near noise floor for every model tested, including full classical | High | Established (multiple independent phases, consistent finding) |
| Calibration | Classical consistently better calibrated; QGNN shows a recurring calibration/ranking decoupling under several interventions | High | Established (consistent across nearly every phase) |

## Appendix C — Figures

`final_figures/` (built by `scripts/build_final_consolidation_figures.py`
directly from `QGNN_V4_MASTER_RESULTS.csv`, no numbers re-derived or
adjusted):

1. `fig1_classical_vs_qgnn_primary_severity.png` — Classical vs. QGNN
   reference, both splits, with std error bars.
2. `fig2_ablation_primary_pr_auc.png` — every tested configuration's
   Primary PR-AUC, full-vs-pilot color-coded.
3. `fig3_ablation_severity_pr_auc.png` — same, Severity PR-AUC. Visually
   the most important figure in this project: the three representation
   diagnostics (Stage 1) sit clearly above both the QGNN and Classical
   references at the top of the chart.
4. `fig4_primary_vs_severity_scatter.png` — every configuration plotted
   by (Primary, Severity) PR-AUC jointly; the classical reference sits
   near the top of the severity axis relative to the whole cloud.
5. `fig5_seedwise_pr_auc.png` — per-seed PR-AUC, classical vs. QGNN
   reference, both splits (seed 45's severity reversal visible directly).
6. `fig6_calibration_ece.png` — ECE across every configuration with a
   recorded value.

No plot in this set implies a statistical significance test was
performed; error bars are ± one standard deviation across the stated
seed count only.

## Appendix D — Master Experiment Inventory

One row per experiment (not per split). `Outcome` uses this project's
final validation-stage vocabulary — **KEEP** (adopted going forward),
**INVESTIGATE** (real, unresolved signal), **DROP** (not supported as a
contributor), **DIAGNOSTIC** (not an architectural change — measures the
representation or a mechanism, not a candidate replacement), **COMPLETE**
(the investigation itself is finished, independent of whether the result
was positive) — not the same vocabulary as `QGNN_V4_ABLATION_TABLE.csv`'s
per-factor `interpretation` column, which classifies evidence strength
rather than a forward-looking status. Full per-split metrics for every
row: `QGNN_V4_MASTER_RESULTS.csv`.

| Phase | Experiment | Main Variable | Seeds | Primary PR-AUC | Severity PR-AUC | Outcome |
|---|---|---|---:|---|---|---|
| Reference | Classical GraphSAGE-Full | (end-to-end reference) | 5 | 0.807±0.066 | 0.449±0.013 | COMPLETE |
| Reference | QGNN-v4 raw baseline (pre-LayerNorm) | (architecture reference) | 5 | 0.833±0.140 | 0.392±0.024 | COMPLETE |
| Phase 2 | Training patience | patience 10 vs. 25 | 5 | unchanged (±0.003, noise) | unchanged (±0.003, noise) | DROP |
| Phase 2b | Output scale (4 variants) | trainable/fixed α, +β | 5 each | 0.820–0.842 | 0.388–0.404 | DROP |
| Phase 2c | LayerNorm (no-affine) | output normalization | 5 | 0.811±0.074 | 0.398±0.057 | KEEP (standing reference) |
| Phase 2c | LayerNorm (affine) | output normalization + affine | 5 | 0.800±0.073 | 0.408±0.071 | INVESTIGATE |
| Phase 2d | Seed-45 root-cause investigation | (diagnostic only, no architecture change) | 5 (existing runs) | n/a | n/a | DIAGNOSTIC |
| Phase 3 | Qubit count (4, 8) | n_qubits | 5 each | 0.847 / 0.810 | 0.396 / 0.364 | INVESTIGATE |
| Phase 3 | Circuit depth (1, 3 layers) | n_layers | 5 / 0 (killed) | 0.693 / 0.804 | 0.386 / N/A | INVESTIGATE (1L) / COMPLETE-INCOMPLETE (3L severity, no data) |
| Phase 4 Stage 1 | Representation audit (diagnostics, PCA, graph, fresh-onset) | (diagnostic only, no architecture change) | 5 | 0.671–0.781 (diagnostics) | 0.463–0.480 (diagnostics) | DIAGNOSTIC |
| Phase 4 Stage 2 | Projection/bottleneck (nonlinear, pre-norm, PCA) | projection architecture | 2 (pilot) | 0.391–0.729 | 0.263–0.389 | DROP |
| Phase 4 Stage 4 | Initialization (small-Gaussian, identity-like) | quantum_init | 2 (pilot) | 0.726–0.828 | 0.265–0.380 | INVESTIGATE |
| Phase 4 Stage 4b | Initialization scale sweep (5 points) | gaussian_std | 2 (pilot) | 0.812–0.828 | 0.263–0.386 | DROP (redundant cluster) / INVESTIGATE (std=0.050 only) |
| Phase 4 Stage 4b | Identity-like, 5-seed expansion | quantum_init=identity_like | 5 | 0.777±0.059 | 0.386±0.112 | DROP (stability claim did not replicate) |
| Phase 4 Stage 5 | Ansatz comparison (ring, reduced-chain) | entanglement topology | 5 each | 0.807 / 0.814 | 0.354 / 0.360 | INVESTIGATE |
| Encoding Investigation | Encoding scale/type/re-uploading | encoding_scale, encoding_type, data_reuploading | 2 (pilot) | 0.581–0.803 | 0.263–0.432 | DROP |

---

## Abstract

This work investigates whether a hybrid quantum-classical graph neural
network (QGNN) — a small variational quantum circuit reading a frozen,
classically-trained GraphSAGE supplier-embedding — can match or exceed a
classical GraphSAGE-Full baseline for supply-chain disruption risk
prediction on a synthetic benchmark, and, more centrally, which
architectural and training factors govern the hybrid model's behavior.
Across nine controlled phases (training patience, output scaling,
LayerNorm normalization, a seed-specific representation-distortion
investigation, qubit-count/depth ablation, a frozen-representation
diagnostic audit, projection-architecture, quantum-parameter
initialization and its scale, quantum-circuit ansatz, and quantum
encoding), and approximately 230 training runs evaluated on both a
temporal (primary) and a severity/out-of-distribution generalization
split, no tested modification produced a reproducible, seed-robust
improvement over the standing QGNN-v4 reference, and, among end-to-end
fully-trained models, the classical GraphSAGE-Full baseline remained
stronger on the severity/out-of-distribution condition throughout. The
most consequential finding was representational rather than
architectural: a simple linear classifier trained only as a
representation diagnostic on the same frozen embedding — not a competing
end-to-end model — already matches or exceeds every quantum
configuration's severity performance and numerically exceeds the
classical end-to-end head's own severity PR-AUC as well, indicating the
frozen embedding carries more severity-relevant signal than either full
downstream head currently extracts. The severity split's difficulty is
shown to be predominantly consistent with a genuine distribution-shift
problem rather than an information deficiency, since the identical
severity-5 periods are predicted well when in-distribution — though this
evidence comes from a dataset whose severe condition is dominated by a
single underlying severity-5 event, so it is stated as evidence within
this benchmark, not a general claim about severe-event generalization.
The principal limitations — evaluation on a single, event-limited
synthetic dataset, an ideal noise-free quantum simulator with no physical
hardware in the loop, and several investigations conducted as 2-seed
pilots not equivalent to the project's 5-seed standing references — are
documented explicitly and scope every conclusion drawn. The contribution
of this work is not a higher benchmark score but a systematic,
evidence-based map of which
factors do and do not govern this hybrid architecture's behavior, and a
demonstrated methodology (frozen-representation diagnostics, seed-level
mechanism tracing, quantum-feature probing, and disciplined pilot-then-
expand experimental staging) for investigating quantum-classical model
behavior rigorously rather than through score-chasing.
