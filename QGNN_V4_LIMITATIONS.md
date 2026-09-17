# QGNN-v4 — Experimental Limitations

Consolidated from every phase's own individually-documented limitations
(Phase 1 through the Quantum Encoding Investigation). Read alongside
`QGNN_V4_FINAL_RESEARCH_SUMMARY.md` — every conclusion in that document
is scoped by what's documented here.

## 1. Severity-5 event limitation (the single most important caveat)

The dataset (`scm_v1_black_swan_seed43`) contains only **6 events total**,
of which only **1 reaches severity 5**. The entire severity/OOD test
condition is drawn from this one event's active window (which happens to
span 37 of 104 periods, so it is not thin in *row count* — 9,900 of the
severity test split's ~11,100 rows are severity-5 — but it is drawn from
one underlying disruption, not a diverse population of severe events).

**Consequence**: every "Severity PR-AUC" figure in this project answers
*"how well does this model generalize to this one severity-5 event's
pattern, having trained only on severity 1–3"* — not *"how well does this
model generalize across many independent severe disruptions."* Any claim
about QGNN-vs-classical OOD generalization should be read at this scope,
not extrapolated to severe-event diversity in general. (Established in
Phase 4 Stage 1's graph/event diagnostic audit.)

## 2. Fresh-onset limitation

The primary/temporal split's test window has **zero** fresh-onset
positive examples (it falls entirely inside the severity-5 event's
already-ongoing period, confirmed in Stage 1). The severity split has 84.
**Primary cannot be used to evaluate fresh-onset anticipation at all** —
not because performance is bad, but because there is nothing to measure.
Every fresh-onset PR-AUC number in this project (0.004–0.012, every
configuration, every phase, including the full classical GraphSAGE
baseline) comes from the severity split only.

## 3. Synthetic dataset

`scm_v1_black_swan_seed43` is a synthetic, generator-produced supply-chain
benchmark (`SCM_DATASET_GENERATION_PLAN.md`), not real-world data. Every
finding in this project is a finding about model behavior **on this
specific synthetic benchmark's graph structure, feature distributions,
and label-generation mechanism** — it should be read as a controlled
experimental result, not as evidence about real-world supply chains.

## 4. Seed and sample-size limitations

- **The project's standing full-evaluation convention is 5 seeds**
  (42–46) — not a large-N statistical study. No formal significance test
  (e.g. a paired t-test or bootstrap CI) was run for any comparison in
  this project; every "X vs. Y" comparison should be read as directional
  evidence from a small sample, not a statistically validated claim.
  Overlapping or non-overlapping standard deviations across 5 points are
  descriptive, not a substitute for a significance test.
- **Several investigations (Phase 4 Stage 2, Stage 4, Stage 4b's scale
  sweep, the Quantum Encoding Investigation) used 2-seed pilots** (seeds
  42, 43 only), per each investigation's own staged-expansion protocol
  ("run a small pilot, only expand to 5 seeds if it shows a credible
  reason"). **2-seed means/stds are explicitly NOT equivalent to the
  5-seed references** and are labeled `pilot` (not `full`) throughout
  `QGNN_V4_MASTER_RESULTS.csv`. The one case where a 2-seed pilot *was*
  expanded to 5 (Stage 4b's identity-like initialization) is the clearest
  illustration of why this distinction matters: the pilot's apparent
  variance reduction (std 0.017) did not survive the expansion (std rose
  to 0.059, and severity std became *worse* than the reference).
- **Phase 3's 6-qubit/3-layer severity run was killed mid-batch** (at the
  user's explicit request, to report interim results) with zero seeds
  completed — that cell is `N/A` throughout, not estimated or
  interpolated.

## 5. Parameter-count confounds

- **Qubit-count ablation (Phase 3)**: changing qubit count necessarily
  changes the classical `Linear(128, n_qubits)` reduction layer's size
  too (measured: going 4→8 qubits changes total head parameters by 544,
  of which only 24 is quantum). Any qubit-count effect is a mix of "more/
  fewer qubits" and "a bigger/smaller classical bottleneck," not isolated.
- **Ansatz comparison (Phase 4 Stage 5)**: the three ansätze tested have
  36 / 24 / 12 quantum trainable parameters respectively — explicitly
  **not** a parameter-matched comparison. A0's largest parameter budget
  coincides with its tightest primary variance; this pattern is
  consistent with, but does not prove, a capacity effect.

## 6. Ideal simulator, no hardware

**Every single result in this entire project** — Phase 1 through the
Quantum Encoding Investigation, ~200+ individual training runs — was
produced on `default.qubit`, PennyLane's classical statevector simulator,
using `diff_method="backprop"` (exact analytic gradients). There is no
hardware noise, no decoherence, no finite-shot sampling error, and no
physical quantum device anywhere in this project's pipeline. No claim in
this project's findings is a claim about how a physical quantum computer
would behave, and none should be read as one.

## 7. Frozen encoder, not co-trained

The GraphSAGE-Full encoder is trained once (per seed, per split) as a
classical model and then frozen (`requires_grad_(False)`, verified by
test in every phase) before any QGNN or classical-bottleneck head is
trained on its output embeddings. No experiment in this project trains
the graph encoder and the quantum head jointly, and no experiment
modifies the encoder architecture (GCN/GAT/Graph Transformer alternatives
were explicitly out of scope for every stage, deferred to a dedicated
"encoder architecture" investigation that was not run).

## 8. Threshold policy held fixed throughout

`threshold.policy: fixed, value: 0.5` was used for every classification
metric (F1, precision, recall, specificity, balanced accuracy, MCC,
confusion matrix) in every phase — never tuned per-configuration, never
selected using test data. This was flagged as early as
`QGNN_V4_BENCHMARK.md` (§7) as a real source of cross-seed instability
for a poorly-calibrated head (the quantum head's output-probability
distribution collapsing near-uniformly above/below 0.5 in some seeds
inflated or deflated every threshold-based metric simultaneously,
independent of true ranking quality) — PR-AUC (threshold-free) was
adopted as the primary metric specifically to avoid this artifact,
consistently from Phase 1 onward.

## 9. Computational / runtime limitations

- Total wall-clock time was measured for some phases but not
  systematically tracked project-wide. Documented figures: the original
  20-run QGNN-v4 benchmark (`QGNN_V4_BENCHMARK.md`) completed in ≈2.4
  minutes total; Phase 2c's 20 runs completed in ≈5.5 minutes. No
  systematic CPU-vs-GPU or classical-vs-quantum-simulator runtime
  comparison was performed — the CUDA-capable GPU present on the
  development machine was never used for the quantum simulator
  (`default.qubit` runs on CPU throughout); classical PyTorch operations
  also ran on CPU tensors throughout, for consistency across every phase.
  **No speedup or slowdown claim should be inferred** — this was not
  measured as a research question in this project.
- Phase 3's original 14-command, 70-run experimental matrix was
  interrupted (killed) partway through at the user's request after 7
  commands (35 runs) completed, to report interim findings rather than
  wait for the full batch — the remaining 6 configurations (depth-4 both
  splits, both new ansätze both splits at the time) were never
  subsequently completed as originally scoped; the ansatz comparison was
  instead completed later, as its own dedicated stage (Stage 5), with a
  smaller, more tightly controlled 3-ansatz/5-seed design.

## 10. Scope of the "ruled out" claims

Every factor in `QGNN_V4_ABLATION_TABLE.csv` classified as "not supported
as major contributor" means exactly that: **the specific configurations
tested for that factor did not produce a reproducible, seed-robust
improvement over the standing reference** — it does not mean the entire
space of possible values for that factor has been exhausted. For example:
only 3 ansätze (of many possible entanglement topologies) were tested;
only 4 discrete encoding scales plus one clip variant were tested (not a
continuous sweep); only 2 initialization families (Gaussian, identity)
were tested. No factor is claimed to be "mathematically proven" incapable
of helping — the evidence is empirical and bounded to what was actually run.
