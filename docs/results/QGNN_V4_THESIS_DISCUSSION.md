# Discussion

This chapter interprets the results reported in
`QGNN_V4_THESIS_RESULTS_CHAPTER.md`. Throughout, each claim is labeled as
an **Observed Result** (a directly measured number or a fact re-verified
against saved artifacts), an **Interpretation** (a reading of what one or
more observed results most plausibly means, stated with its own
confidence level), or a **Hypothesis for Future Work** (an untested idea
this project's evidence motivates but does not confirm). Collapsing these
three categories into one another is exactly the kind of overreach this
final validation pass was tasked with removing from this project's
record, so the distinction is kept explicit rather than implied.

## 1. Why Primary and Severity Behave Differently

**Observed Result.** Every model and configuration evaluated in this
project scores substantially lower on the severity split than on the
primary split — roughly halved for both the classical and QGNN-v4
references (0.807→0.449 for classical; 0.811→0.398 for QGNN-v4). The
identical severity-5 periods used as the severity split's test set score
0.81 PR-AUC when they instead fall inside the primary split's
in-distribution test window.

**Interpretation (high confidence, within this dataset).** The severity
condition's difficulty is predominantly a distribution-shift
generalization problem — the model has never seen a severity-4-or-5
pattern during training under the severity split's design — rather than
an inherent information deficiency in the frozen representation or the
graph structure. This interpretation is well-supported by the
in-distribution/out-of-distribution comparison above, which isolates the
effect of *training exposure* from the effect of *what the severity-5
periods look like*, using the identical periods in both cases.

**Interpretation (necessarily lower confidence, generalization scope).**
Whether this same distribution-shift account would hold for a benchmark
containing many independent, diverse severe events — rather than the
single underlying severity-5 event this benchmark's severity condition
is built from — is not established by this evidence. The in-distribution
comparison controls for training exposure to *this* event's pattern; it
says nothing about whether a model trained without exposure to *any*
severe event would generalize to a *novel* one, which is arguably the
more practically important question and remains untested here.

## 2. Classical vs. Quantum Head Behavior

**Observed Result.** On primary, classical and QGNN-v4 reach comparable
PR-AUC (0.807 vs. 0.811) but classical shows meaningfully higher recall
and substantially better calibration. On severity, classical leads on
PR-AUC, ROC-AUC, and calibration; QGNN-v4 reaches parity or a small edge
on F1 and MCC specifically. QGNN-v4's cross-seed variance exceeds
classical's on every metric compared in Section 9 of the results
chapter.

**Interpretation (medium-high confidence).** The quantum head is not
categorically unable to learn useful structure from this representation
— its primary-split performance and several individual channels'
correlation with the target (up to r≈0.87 in related raw-embedding
analysis) demonstrate real learning. Its comparative weakness relative
to classical shows up most consistently in *calibration* and in
*cross-seed reliability*, not in raw ranking capability on primary. This
suggests the gap has more to do with how the quantum head's output scale
and optimization trajectory behave across different random
initializations than with a fundamental capacity limitation at 6
qubits/2 layers — though "fundamental capacity limitation" was not ruled
out by a capacity-matched comparison in every case (the qubit-count and
ansatz ablations both carry parameter-count confounds, Results Chapter
§13).

**Hypothesis for Future Work.** A capacity-matched comparison — e.g., a
classical bottleneck head with a number of trainable parameters equal to
the quantum circuit's own 36, rather than the existing matched-*width*
(not matched-parameter-count) classical control — was not run in this
project and could sharpen this interpretation further.

## 3. Representation Findings

**Observed Result.** A logistic regression trained as a representation
diagnostic on the frozen embedding reaches severity PR-AUC 0.465±0.005,
above both the classical end-to-end reference (0.449) and every QGNN-v4
configuration tested (0.36–0.41), with markedly tighter variance than
either. PCA shows primary-split signal concentrated in four components
(90.9% of variance, best downstream diagnostic score at that
dimensionality); severity-split signal is more evenly distributed across
all 128 dimensions.

**Interpretation (high confidence).** The frozen representation is not
the primary bottleneck for severity performance — there is more usable
signal in it than either full downstream head (classical or quantum)
currently extracts. This reframes the project's central question: the
gap between QGNN-v4 and classical, and between both of them and the
diagnostic ceiling, is at least partly a *downstream training/decision-
function* question, not purely a *representation* or *architecture*
question.

**Interpretation (medium confidence, not further tested).** The
asymmetry between primary's low-rank signal and severity's distributed
signal plausibly explains why a smaller quantum bottleneck (4 qubits)
performed comparably to or better than a wider one (8 qubits) on primary
specifically, in the earlier qubit-count ablation — a smaller bottleneck
loses little of a genuinely low-rank signal. This is offered as a
plausible, independently-motivated explanation connecting two separate
experiments' findings, not as a claim verified by a joint experiment
that manipulated both factors together.

**Hypothesis for Future Work.** Why the quantum head, when fed a
PCA-reduced version of the same embedding that a linear classifier
handles well, performs *worse* than when fed the full 128-dimensional
input (the Stage 2 PCA-projection anomaly) is not explained by any
evidence gathered in this project. A dedicated, narrowly-scoped follow-up
isolating this specific composition (PCA reduction → quantum head)
would be needed before drawing any conclusion about it.

## 4. Seed Sensitivity

**Observed Result.** QGNN-v4's cross-seed standard deviation exceeds
classical's on every metric examined (Results Chapter, Table 1), most
sharply on severity MCC (0.089 vs. 0.023). Seed 45's severity ranking
reverses sharply under `LayerNorm` while every other seed's improves or
holds flat, traced to that seed's baseline circuit concentrating signal
into one dominant output channel that `LayerNorm` specifically removes
the incentive for. A separate, later finding showed a pilot-observed
reduction in seed variance (identity-like initialization, 2-seed std
0.017) did not persist at the full 5-seed evaluation (std 0.059,
severity std actually higher than the reference).

**Interpretation (medium confidence for the seed-45 mechanism
specifically).** Seed sensitivity in this system is real, and at least
one specific instance of it has a concrete, quantitatively distinctive,
and mechanistically sensible explanation (channel concentration
interacting with output normalization) — this is stronger evidence than
"ordinary stochastic variation," but it was established correlationally
in a diagnostic-only phase that did not retrain under a controlled
intervention to confirm the mechanism causally, and is not claimed to be
the complete explanation for seed sensitivity across every configuration
in this project.

**Interpretation (high confidence, methodological).** Two-seed pilot
estimates of variance in this system are not reliable enough to treat as
confirmed findings — the identity-like initialization case is direct,
project-generated evidence for this, not an assumption. Any future
two-seed result in this line of work should be treated provisionally
until checked at the full seed count, exactly as this project's own
staged pilot-then-expand protocol was designed to enforce.

## 5. Quantum-Specific Observations

**Observed Result.** Ansatz choice and encoding configuration both
measurably change properties of the quantum representation itself —
cross-channel correlation among the six Pauli-Z outputs ranged from 0.68
to 0.93 depending on ansatz and encoding; channel-target correlation and
encoded-angle spread both shifted systematically with encoding scale (a
doubled scale produced the widest angle spread and the weakest
channel-target correlation observed in that investigation).

**Interpretation (medium confidence).** The quantum circuit is not inert
to these architectural choices — it demonstrably learns different
representations under different ansätze and encodings — but these
representational differences did not translate into a corresponding,
seed-robust difference in final task metrics in any configuration tested.
This suggests the representational sensitivity of this small (6-qubit,
2-layer) circuit is real but is not currently the limiting factor for
this task's performance ceiling, which the representation-diagnostic
evidence (Section 3 above) locates more plausibly downstream, in how the
extracted signal is used rather than in whether it is extracted at all.

**Hypothesis for Future Work.** Whether a larger or differently
structured quantum circuit would exhibit the same "representation
changes without a metrics benefit" pattern, or whether this is specific
to the modest circuit scale (6 qubits, 2 layers) investigated throughout
this project, is not addressed by any experiment here.

## 6. Limitations

Restated briefly from the results chapter and detailed fully in
`QGNN_V4_LIMITATIONS.md`: the severity condition is dominated by a
single underlying event; the primary split cannot evaluate fresh-onset
detection at all; the dataset is synthetic; several investigations used
2-seed pilots not equivalent to the project's 5-seed standing references;
the qubit-count and ansatz ablations carry parameter-count confounds; and
every result comes from an ideal, noise-free simulator with no physical
quantum hardware involved. Each of these bounds the scope of every
interpretation offered in this chapter, and none is treated as
incidental.

## 7. Implications

**Interpretation (scoped to this project's own evidence).** For this
specific benchmark, encoder, and QGNN-v4 architecture family, the
practical path to closing the severity gap is not well-supported by
further architectural search *within the space already explored*
(projection, initialization, ansatz, encoding) — nine factors were
investigated and none produced a robust improvement. The more
evidence-supported directions implied by this project's own findings
point toward the input side (richer, leakage-safe temporal features,
motivated directly by the absent-historical-feature finding in Section 7
of the results chapter) and toward the downstream decision function
(motivated by the representation-diagnostic ceiling exceeding what
either full head currently reaches) rather than toward further quantum-
circuit-specific tuning.

## 8. Future Work

Presented as hypotheses this project's evidence motivates, not as
findings or as work already begun:

1. **Leakage-safe historical features** (time since previous disruption,
   rolling disruption frequency, graph-propagated neighbor disruption
   history) — directly motivated by the confirmed absence of any such
   feature in the current pipeline (results chapter §7) and by the
   observed fresh-onset weakness. Not implemented or tested in this
   project.
2. **Isolating the PCA-projection anomaly** (§3 above) before any further
   projection-architecture work, since it is the one result in this
   project's evidence base that appears internally inconsistent and is
   not explained by anything else found.
3. **A capacity-matched classical control** (§2 above) to separate "the
   quantum circuit's own expressiveness" from "the bottleneck width" more
   cleanly than the existing matched-*width* control does.
4. **A dataset with multiple independent severe events**, to test whether
   the distribution-shift interpretation (§1 above) extends beyond this
   benchmark's single-event severity condition — a different, larger
   experimental undertaking than anything performed in this project.
5. **Real-world supply-chain data**, to test whether any pattern observed
   here persists outside a synthetic, generator-produced benchmark.
6. **Physical hardware or realistic noise-model evaluation**, to test
   whether the ideal-simulator behavior documented throughout this
   project persists once shot noise and hardware noise are introduced —
   entirely unaddressed by any experiment in this project.
7. **Alternative graph encoders** (GCN, GAT), only if a future
   representation audit — repeating Stage 1's diagnostic approach under a
   different encoder — indicates the encoder itself, rather than the
   downstream head, is the limiting factor; this project's own evidence
   (the diagnostic ceiling exceeding both downstream heads) argues against
   prioritizing this direction first.

None of the above is recommended to begin automatically; each requires
its own scoped, controlled experimental design, following the same
disciplined, single-variable-at-a-time protocol used throughout this
investigation.
