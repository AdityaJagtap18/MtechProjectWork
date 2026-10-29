"use strict";
const H = require("./helpers");
const { AlignmentType } = require("docx");

function obsPara(text) {
  return new (require("docx").Paragraph)({
    spacing: { after: 160, line: 276 },
    alignment: AlignmentType.JUSTIFIED,
    children: [
      new (require("docx").TextRun)({ text: "Observed Result. ", bold: true }),
      new (require("docx").TextRun)({ text }),
    ],
  });
}
function interpPara(confidence, text) {
  return new (require("docx").Paragraph)({
    spacing: { after: 160, line: 276 },
    alignment: AlignmentType.JUSTIFIED,
    children: [
      new (require("docx").TextRun)({
        text: `Interpretation${confidence ? " (" + confidence + ")" : ""}. `,
        bold: true,
        italics: true,
      }),
      new (require("docx").TextRun)({ text }),
    ],
  });
}
function hypoPara(text) {
  return new (require("docx").Paragraph)({
    spacing: { after: 220, line: 276 },
    alignment: AlignmentType.JUSTIFIED,
    children: [
      new (require("docx").TextRun)({ text: "Hypothesis for Future Work. ", bold: true, italics: true, color: H.COLOR_MUTED }),
      new (require("docx").TextRun)({ text, color: H.COLOR_MUTED }),
    ],
  });
}

function build() {
  const c = [];

  // ---------------------------------------------------------------
  // 14. Discussion
  // ---------------------------------------------------------------
  c.push(H.h1("14. Discussion"));
  c.push(
    H.p(
      "This section interprets the results reported in Section 13. Every claim below is explicitly labeled as an Observed Result (a directly measured number, or a fact re-verified against saved artifacts), an Interpretation (a reading of what one or more observed results most plausibly means, stated with its own confidence level), or a Hypothesis for Future Work (an untested idea this project's evidence motivates but does not confirm). Collapsing these three categories into one another is exactly the kind of overreach this report's own validation pass was tasked with removing, so the distinction is kept explicit rather than implied throughout."
    )
  );

  c.push(H.h2("14.1 Why Primary and Severity Behave Differently"));
  c.push(
    obsPara(
      "Every model and configuration evaluated in this project scores substantially lower on the Severity split than on the Primary split — roughly halved for both references (0.807→0.449 Classical; 0.811→0.398 QGNN-v4). The identical severity-5 periods used as the Severity split's test set score 0.81 PR-AUC when they instead fall inside the Primary split's in-distribution test window."
    )
  );
  c.push(
    interpPara(
      "high confidence, within this dataset",
      "The Severity condition's difficulty is predominantly a distribution-shift generalization problem — the model has never seen a severity-4-or-5 pattern during training under the Severity split's design — rather than an inherent information deficiency in the frozen representation or the graph structure. This is well-supported by the in-distribution/out-of-distribution comparison above, which isolates the effect of training exposure from the effect of what the severity-5 periods look like, using the identical periods in both cases."
    )
  );
  c.push(
    interpPara(
      "necessarily lower confidence, generalization scope",
      "Whether this same distribution-shift account would hold for a benchmark containing many independent, diverse severe events — rather than the single underlying severity-5 event this benchmark's Severity condition is built from — is not established by this evidence. The in-distribution comparison controls for training exposure to this event's pattern; it says nothing about whether a model trained without exposure to any severe event would generalize to a novel one, arguably the more practically important question, and it remains untested here."
    )
  );

  c.push(H.h2("14.2 Classical vs. Quantum Head Behavior"));
  c.push(
    obsPara(
      "On Primary, Classical and QGNN-v4 reach comparable PR-AUC (0.807 vs. 0.811) but Classical shows meaningfully higher recall and substantially better calibration. On Severity, Classical leads on PR-AUC, ROC-AUC, and calibration; QGNN-v4 reaches parity or a small edge on F1 and MCC specifically. QGNN-v4's cross-seed variance exceeds Classical's on every metric compared in Section 13.9."
    )
  );
  c.push(
    interpPara(
      "medium-high confidence",
      "The quantum head is not categorically unable to learn useful structure from this representation — its Primary-split performance and several individual channels' correlation with the target (up to r ≈ 0.87 in related raw-embedding analysis) demonstrate real learning. Its comparative weakness relative to Classical shows up most consistently in calibration and in cross-seed reliability, not in raw ranking capability on Primary. This suggests the gap has more to do with how the quantum head's output scale and optimization trajectory behave across different random initializations than with a fundamental capacity limitation at 6 qubits / 2 layers — though “fundamental capacity limitation” was not ruled out by a capacity-matched comparison in every case, since the qubit-count and ansatz ablations both carry parameter-count confounds (Section 15.5)."
    )
  );
  c.push(
    hypoPara(
      "A capacity-matched comparison — for example, a classical bottleneck head with a number of trainable parameters equal to the quantum circuit's own 36, rather than the existing matched-width (not matched-parameter-count) classical control — was not run in this project and could sharpen this interpretation further."
    )
  );

  c.push(H.h2("14.3 Representation Findings"));
  c.push(
    obsPara(
      "A logistic regression trained as a representation diagnostic on the frozen embedding reaches Severity PR-AUC 0.465 ± 0.005, above both the classical end-to-end reference (0.449) and every QGNN-v4 configuration tested (0.36–0.41), with markedly tighter variance than either. PCA shows Primary-split signal concentrated in four components (90.9% of variance, best downstream diagnostic score at that dimensionality); Severity-split signal is more evenly distributed across all 128 dimensions."
    )
  );
  c.push(
    interpPara(
      "high confidence",
      "The frozen representation is not the primary bottleneck for Severity performance — there is more usable signal in it than either full downstream head, classical or quantum, currently extracts. This reframes the project's central question: the gap between QGNN-v4 and Classical, and between both of them and the diagnostic ceiling, is at least partly a downstream training/decision-function question, not purely a representation or architecture question."
    )
  );
  c.push(
    interpPara(
      "medium confidence, not further tested",
      "The asymmetry between Primary's low-rank signal and Severity's distributed signal plausibly explains why a smaller quantum bottleneck (4 qubits) performed comparably to or better than a wider one (8 qubits) on Primary specifically, in the earlier qubit-count ablation — a smaller bottleneck loses little of a genuinely low-rank signal. This is offered as a plausible, independently-motivated explanation connecting two separate experiments' findings, not as a claim verified by a joint experiment that manipulated both factors together."
    )
  );
  c.push(
    hypoPara(
      "Why the quantum head, when fed a PCA-reduced version of the same embedding that a linear classifier handles well, performs worse than when fed the full 128-dimensional input (the Stage 2 PCA-projection anomaly) is not explained by any evidence gathered in this project. A dedicated, narrowly-scoped follow-up isolating this specific composition (PCA reduction → quantum head) would be needed before drawing any conclusion about it."
    )
  );

  c.push(H.h2("14.4 Seed Sensitivity"));
  c.push(
    obsPara(
      "QGNN-v4's cross-seed standard deviation exceeds Classical's on every metric examined (Section 13.9), most sharply on Severity MCC (0.089 vs. 0.023). Seed 45's severity ranking reverses sharply under LayerNorm while every other seed's improves or holds flat, traced to that seed's baseline circuit concentrating signal into one dominant output channel that LayerNorm specifically removes the incentive for. A separate, later finding showed a pilot-observed reduction in seed variance (identity-like initialization, 2-seed std 0.017) did not persist at the full 5-seed evaluation (std 0.059, Severity std actually higher than the reference)."
    )
  );
  c.push(
    interpPara(
      "medium confidence for the seed-45 mechanism specifically",
      "Seed sensitivity in this system is real, and at least one specific instance of it has a concrete, quantitatively distinctive, and mechanistically sensible explanation — channel concentration interacting with output normalization. This is stronger evidence than “ordinary stochastic variation,” but it was established correlationally in a diagnostic-only phase that did not retrain under a controlled intervention to confirm the mechanism causally, and is not claimed to be the complete explanation for seed sensitivity across every configuration in this project."
    )
  );
  c.push(
    interpPara(
      "high confidence, methodological",
      "Two-seed pilot estimates of variance in this system are not reliable enough to treat as confirmed findings — the identity-like initialization case is direct, project-generated evidence for this, not an assumption. Any future two-seed result in this line of work should be treated provisionally until checked at the full seed count, exactly as this project's own staged pilot-then-expand protocol was designed to enforce."
    )
  );

  c.push(H.h2("14.5 Quantum-Specific Observations"));
  c.push(
    obsPara(
      "Ansatz choice and encoding configuration both measurably change properties of the quantum representation itself — cross-channel correlation among the six Pauli-Z outputs ranged from 0.68 to 0.93 depending on ansatz and encoding; channel-target correlation and encoded-angle spread both shifted systematically with encoding scale (a doubled scale produced the widest angle spread and the weakest channel-target correlation observed in that investigation)."
    )
  );
  c.push(
    interpPara(
      "medium confidence",
      "The quantum circuit is not inert to these architectural choices — it demonstrably learns different representations under different ansätze and encodings — but these representational differences did not translate into a corresponding, seed-robust difference in final task metrics in any configuration tested. This suggests the representational sensitivity of this small (6-qubit, 2-layer) circuit is real but is not currently the limiting factor for this task's performance ceiling, which the representation-diagnostic evidence (Section 14.3) locates more plausibly downstream, in how the extracted signal is used rather than in whether it is extracted at all."
    )
  );
  c.push(
    hypoPara(
      "Whether a larger or differently structured quantum circuit would exhibit the same “representation changes without a metrics benefit” pattern, or whether this is specific to the modest circuit scale (6 qubits, 2 layers) investigated throughout this project, is not addressed by any experiment here."
    )
  );

  c.push(H.h2("14.6 Implications"));
  c.push(
    interpPara(
      "scoped to this project's own evidence",
      "For this specific benchmark, encoder, and QGNN-v4 architecture family, the practical path to closing the Severity gap is not well-supported by further architectural search within the space already explored (projection, initialization, ansatz, encoding) — nine factors were investigated and none produced a robust improvement. The more evidence-supported directions implied by this project's own findings point toward the input side (richer, leakage-safe temporal features, motivated directly by the absent-historical-feature finding in Section 13.7) and toward the downstream decision function (motivated by the representation-diagnostic ceiling exceeding what either full head currently reaches) rather than toward further quantum-circuit-specific tuning."
    )
  );
  c.push(H.pageBreak());

  // ---------------------------------------------------------------
  // 15. Limitations
  // ---------------------------------------------------------------
  c.push(H.h1("15. Limitations"));
  c.push(
    H.h2("15.1 Severity-5 Event Limitation (the single most important caveat)")
  );
  c.push(
    H.p(
      "The dataset (scm_v1_black_swan_seed43) contains only six disruption events in total, of which only one reaches severity 5. The entire Severity/OOD test condition is drawn from this one event's active window (spanning 37 of 104 periods, so it is not thin in row count — 9,900 of the Severity test split's ≈ 11,100 rows are severity-5 — but it is drawn from one underlying disruption, not a diverse population of severe events). Every “Severity PR-AUC” figure in this project answers “how well does this model generalize to this one severity-5 event's pattern, having trained only on severity 1–3” — not “how well does this model generalize across many independent severe disruptions.” Any claim about QGNN-vs-classical OOD generalization should be read at this scope, not extrapolated to severe-event diversity in general."
    )
  );

  c.push(H.h2("15.2 Fresh-Onset Limitation"));
  c.push(
    H.p(
      "The Primary split's test window has zero fresh-onset positive examples (it falls entirely inside the severity-5 event's already-ongoing period). The Severity split has 84. Primary cannot be used to evaluate fresh-onset anticipation at all — not because performance is bad, but because there is nothing to measure. Every fresh-onset PR-AUC number in this project (0.004–0.012, every configuration, every phase, including the full classical GraphSAGE baseline) comes from the Severity split only."
    )
  );

  c.push(H.h2("15.3 Synthetic Dataset"));
  c.push(
    H.p(
      "scm_v1_black_swan_seed43 is a synthetic, generator-produced supply-chain benchmark, not real-world data. Every finding in this project is a finding about model behavior on this specific synthetic benchmark's graph structure, feature distributions, and label-generation mechanism — it should be read as a controlled experimental result, not as evidence about real-world supply chains."
    )
  );

  c.push(H.h2("15.4 Seed and Sample-Size Limitations"));
  c.push(
    ...H.bullets([
      "The project's standing full-evaluation convention is 5 seeds (42–46) — not a large-N statistical study. No formal significance test (e.g. a paired t-test or bootstrap CI) was run for any comparison in this project; every “X vs. Y” comparison should be read as directional evidence from a small sample, not a statistically validated claim.",
      "Several investigations (Phase 4 Stage 2, Stage 4, Stage 4b's scale sweep, the Quantum Encoding Investigation) used 2-seed pilots (seeds 42, 43 only), per each investigation's own staged-expansion protocol. 2-seed means/stds are explicitly not equivalent to the 5-seed references and are labeled pilot (not full) throughout QGNN_V4_MASTER_RESULTS.csv. The one case where a 2-seed pilot was expanded to 5 (Stage 4b's identity-like initialization) illustrates why this distinction matters: the pilot's apparent variance reduction (std 0.017) did not survive expansion (std rose to 0.059, and Severity std became worse than the reference).",
      "Phase 3's 6-qubit/3-layer Severity run was killed mid-batch (at the user's explicit request, to report interim results) with zero seeds completed — that cell is N/A throughout, not estimated or interpolated.",
    ])
  );

  c.push(H.h2("15.5 Parameter-Count Confounds"));
  c.push(
    ...H.bullets([
      "Qubit-count ablation (Phase 3): changing qubit count necessarily changes the classical Linear(128, n_qubits) reduction layer's size too (measured: going 4→8 qubits changes total head parameters by 544, of which only 24 is quantum). Any qubit-count effect is a mix of “more/fewer qubits” and “a bigger/smaller classical bottleneck,” not isolated.",
      "Ansatz comparison (Phase 4 Stage 5): the three ansätze tested have 36 / 24 / 12 quantum trainable parameters respectively — explicitly not a parameter-matched comparison. The reference ansatz's largest parameter budget coincides with its tightest Primary variance; this pattern is consistent with, but does not prove, a capacity effect.",
    ])
  );

  c.push(H.h2("15.6 Ideal Simulator, No Hardware"));
  c.push(
    H.p(
      "Every single result in this entire project — across roughly 230 individual training runs — was produced on default.qubit, PennyLane's classical statevector simulator, using diff_method=\"backprop\" (exact analytic gradients). There is no hardware noise, no decoherence, no finite-shot sampling error, and no physical quantum device anywhere in this project's pipeline. No claim in this project's findings is a claim about how a physical quantum computer would behave, and none should be read as one."
    )
  );

  c.push(H.h2("15.7 Frozen Encoder, Not Co-Trained"));
  c.push(
    H.p(
      "The GraphSAGE-Full encoder is trained once (per seed, per split) as a classical model and then frozen (requires_grad set to False, verified by an explicit assertion in every phase) before any QGNN or classical-bottleneck head is trained on its output embeddings. No experiment in this project trains the graph encoder and the quantum head jointly, and no experiment modifies the encoder architecture."
    )
  );

  c.push(H.h2("15.8 Threshold Policy Held Fixed Throughout"));
  c.push(
    H.p(
      "threshold.policy: fixed, value: 0.5 was used for every classification metric in every phase — never tuned per-configuration, never selected using test data. This was flagged early as a real source of cross-seed instability for a poorly-calibrated head; PR-AUC (threshold-free) was adopted as the primary metric specifically to avoid this artifact, consistently from the earliest phase onward."
    )
  );

  c.push(H.h2("15.9 Computational / Runtime Limitations"));
  c.push(
    H.p(
      "Total wall-clock time was measured for some phases but not systematically tracked project-wide; no systematic CPU-vs-GPU or classical-vs-quantum-simulator runtime comparison was performed. No speedup or slowdown claim should be inferred — this was not measured as a research question in this project. Phase 3's original 14-command, 70-run experimental matrix was interrupted partway through at the user's request after 7 commands (35 runs) completed, to report interim findings rather than wait for the full batch; the ansatz comparison was instead completed later as its own dedicated, more tightly controlled stage (Stage 5)."
    )
  );

  c.push(H.h2("15.10 Scope of the “Ruled Out” Claims"));
  c.push(
    H.p(
      "Every factor classified as “not supported as major contributor” in this report means exactly that: the specific configurations tested for that factor did not produce a reproducible, seed-robust improvement over the standing reference — it does not mean the entire space of possible values for that factor has been exhausted. For example: only three ansätze (of many possible entanglement topologies) were tested; only four discrete encoding scales plus one clip variant were tested (not a continuous sweep); only two initialization families (Gaussian, identity) were tested. No factor is claimed to be mathematically proven incapable of helping — the evidence is empirical and bounded to what was actually run."
    )
  );
  c.push(H.pageBreak());

  // ---------------------------------------------------------------
  // 16. Conclusion
  // ---------------------------------------------------------------
  c.push(H.h1("16. Conclusion"));
  c.push(
    H.p(
      "Within this synthetic SCM dataset, this frozen GraphSAGE-Full representation, this ideal PennyLane default.qubit simulation, this 6-qubit/2-layer QGNN-v4 architecture family, and the Primary/Severity splits evaluated, the investigated modifications — projection, initialization, initialization scale, ansatz, and quantum encoding — did not produce a reproducible, seed-robust improvement over the standing QGNN-v4 reference. Among end-to-end, fully-trained models, the classical GraphSAGE-Full baseline remained stronger on the Severity/OOD generalization condition throughout this investigation, while a representation diagnostic — not an end-to-end model — on the same frozen embedding numerically exceeds Classical's own Severity PR-AUC, which is itself evidence that the ceiling is not attributable to the classical architecture either."
    )
  );
  c.push(
    H.p(
      "This conclusion is scoped exactly to what was tested. It is not a claim that quantum computing is inferior to classical GNNs in general, that quantum hardware would behave identically, that a different dataset would reproduce this result, that this architecture is globally optimal, that every possible encoding has been exhausted, or that a quantum advantage in this problem class is impossible. Within the boundaries actually investigated, the evidence is consistent and reproducible: it does not support adopting any of the tested modifications as a replacement for the standing reference, and it supports treating classical GraphSAGE-Full as the stronger current end-to-end option for this specific severity/OOD condition — while the representation diagnostic indicates the practical ceiling on this frozen embedding is higher than either full model currently reaches."
    )
  );

  c.push(H.h2("16.1 Explicitly Demonstrated"));
  c.push(
    ...H.bullets([
      "QGNN can learn useful predictive representations (channel-target correlations up to r ≈ 0.87, comparable Primary PR-AUC to Classical).",
      "QGNN achieves strong Primary PR-AUC in several configurations.",
      "QGNN is measurably sensitive to initialization/seed/architecture interactions, in specific, evidenced ways.",
      "Quantum encoding measurably affects the learned representation (angle distribution, channel correlations), independent of whether it affects final metrics.",
      "Severity/OOD performance is substantially harder than Primary for every model tested.",
      "Among end-to-end models, classical GraphSAGE-Full is stronger on the established Severity reference across ranking, calibration, and mostly classification metrics — while a representation-diagnostic logistic regression on the same frozen embedding numerically exceeds Classical's own Severity PR-AUC (Section 13.7).",
      "Fresh-onset detection remains near the noise floor for every model tested, including the full classical baseline.",
    ])
  );

  c.push(H.h2("16.2 Explicitly Not Demonstrated"));
  c.push(
    ...H.bullets([
      "That QGNNs are inherently inferior to classical GNNs as a model class.",
      "That real quantum hardware would reproduce this project's simulator results.",
      "That a different dataset (real-world or otherwise) would show the same pattern.",
      "That the specific 6-qubit/2-layer/StronglyEntangling architecture used as the reference is globally optimal among all possible QGNN designs.",
      "That all possible quantum encodings, ansätze, or initializations have been exhausted — only the specific, documented set tested here.",
      "That a quantum advantage on this or related problems is impossible.",
    ])
  );
  c.push(H.pageBreak());

  // ---------------------------------------------------------------
  // 17. Future Work
  // ---------------------------------------------------------------
  c.push(H.h1("17. Future Work"));
  c.push(
    H.p(
      "Presented as hypotheses this project's evidence motivates, not as findings or as work already begun. None of the following is recommended to begin automatically; each requires its own scoped, controlled experimental design, following the same disciplined, single-variable-at-a-time protocol used throughout this investigation."
    )
  );
  c.push(
    ...H.numbered([
      [
        { text: "Leakage-safe historical features. ", bold: true },
        "Time since previous disruption, rolling disruption frequency, graph-propagated neighbor disruption history — directly motivated by the confirmed absence of any such feature in the current pipeline (Section 13.7) and by the observed fresh-onset weakness (Section 13.10). Not implemented or tested in this project.",
      ],
      [
        { text: "Isolating the PCA-projection anomaly. ", bold: true },
        "Before any further projection-architecture work, since it is the one result in this project's evidence base that appears internally inconsistent and is not explained by anything else found (Section 14.3).",
      ],
      [
        { text: "A capacity-matched classical control. ", bold: true },
        "To separate “the quantum circuit's own expressiveness” from “the bottleneck width” more cleanly than the existing matched-width control does (Section 14.2).",
      ],
      [
        { text: "A dataset with multiple independent severe events. ", bold: true },
        "To test whether the distribution-shift interpretation (Section 14.1) extends beyond this benchmark's single-event severity condition — a different, larger experimental undertaking than anything performed in this project.",
      ],
      [
        { text: "Real-world supply-chain data. ", bold: true },
        "To test whether any pattern observed here persists outside a synthetic, generator-produced benchmark.",
      ],
      [
        { text: "Physical hardware or realistic noise-model evaluation. ", bold: true },
        "To test whether the ideal-simulator behavior documented throughout this project persists once shot noise and hardware noise are introduced — entirely unaddressed by any experiment in this project.",
      ],
      [
        { text: "Alternative graph encoders (GCN, GAT). ", bold: true },
        "Only if a future representation audit — repeating Stage 1's diagnostic approach under a different encoder — indicates the encoder itself, rather than the downstream head, is the limiting factor; this project's own evidence (the diagnostic ceiling exceeding both downstream heads) argues against prioritizing this direction first.",
      ],
      [
        { text: "A real statistical significance test. ", bold: true },
        "A paired bootstrap over matched seeds, for the handful of comparisons where 5-seed evidence exists and the direction is consistent (e.g. Classical vs. QGNN on Severity), would strengthen any claim beyond “directional evidence.”",
      ],
    ])
  );
  c.push(H.pageBreak());

  // ---------------------------------------------------------------
  // 18. Reproducibility
  // ---------------------------------------------------------------
  c.push(H.h1("18. Reproducibility"));
  c.push(
    H.p(
      "Each item below marked VERIFIED was directly re-checked against a live command or a saved artifact during this report's own validation pass — not assumed. PARTIALLY VERIFIED means some but not all of the claim was checked. NOT AVAILABLE means the item was not tracked and is stated as such, not estimated."
    )
  );
  c.push(H.tableCaption("Final reproducibility checklist."));
  c.push(
    H.dataTable(
      [
        { header: "Item", width: 0.22 },
        { header: "Status", width: 0.16, align: AlignmentType.CENTER },
        { header: "Detail", width: 0.62 },
      ],
      [
        ["Branch", "VERIFIED", "qgnn-package-restructure (git branch --show-current)"],
        ["Python version", "VERIFIED", "3.12.3 (python3 --version, run live in the project's .venv)"],
        ["PyTorch version", "VERIFIED", "2.14.0+cu130 (torch.__version__, run live)"],
        ["PennyLane version", "VERIFIED", "0.45.1 (qml.__version__, run live)"],
        [
          "CUDA status",
          "VERIFIED",
          "torch.cuda.is_available() returns True on the development machine, but not used — every quantum-circuit run uses default.qubit (CPU simulator); classical PyTorch tensors also ran on CPU throughout, for consistency across every phase. A deliberate scope choice, not an oversight.",
        ],
        [
          "Simulator / backend",
          "VERIFIED",
          "default.qubit, diff_method=\"backprop\", confirmed in every quantum_resource_summary.json produced by every training run in this project",
        ],
        [
          "Dataset",
          "VERIFIED",
          "scm_v1_black_swan_seed43, data/benchmark/scm_v1_black_swan_seed43/ on disk; graph/event statistics re-verified directly from the raw CSVs",
        ],
        [
          "Split definitions",
          "VERIFIED",
          "src/scm_dataset/benchmark/splits.py: temporal_split (Primary, 70/15/15 by period), severity_split (train severity ≤ 3, test severity 4–5)",
        ],
        [
          "Seeds",
          "VERIFIED",
          "42, 43, 44, 45, 46 for every full (seed_type=full) row in QGNN_V4_MASTER_RESULTS.csv; 42, 43 only for every row marked pilot",
        ],
        [
          "Configuration files",
          "VERIFIED",
          "configs/qgnn_v4.yaml, configs/qgnn_v4_severity.yaml, configs/graphsage.yaml / graphsage_severity.yaml — all present in the repository",
        ],
        [
          "Experiment commands",
          "VERIFIED",
          "Every phase report includes the exact scripts/run_qgnn_v4_experiment.py invocation used, with every non-default flag shown explicitly",
        ],
        [
          "Output locations",
          "VERIFIED",
          "experiments/qgnn_v4/, experiments/classical_gnn/ — 91 GraphSAGE-Full checkpoint directories confirmed present. Raw experiment outputs are not committed to git by design; the committed record is the reports, scripts, and consolidated CSVs.",
        ],
        [
          "Metric calculation",
          "VERIFIED",
          "Single shared implementation throughout: metrics.py::compute_classification_metrics, plus mcc_from_confusion / specificity_from_confusion and calibration.py::expected_calibration_error",
        ],
        [
          "Model checkpoints",
          "PARTIALLY VERIFIED",
          "Checkpoints exist on disk for every spot-checked run; not every one of roughly 230 individual QGNN run checkpoints was re-loaded and re-scored during this specific validation pass",
        ],
        [
          "Threshold policy",
          "VERIFIED",
          "fixed, value: 0.5 for every classification metric in every phase, confirmed by inspecting every config file",
        ],
        [
          "Frozen-encoder isolation",
          "VERIFIED",
          "assert all(not p.requires_grad for p in encoder.parameters()) is asserted in the run loop for every QGNN training run, not just claimed in prose",
        ],
        ["Test suite", "VERIFIED", "334/334 tests passing at this validation's starting commit (python3 -m pytest tests/ -q, run live)"],
      ],
      { zebra: true }
    )
  );

  c.push(H.h2("18.1 How to Reproduce the Standing References"));
  const codeLines = [
    "# Classical GraphSAGE-Full (already-trained checkpoints, not retrained):",
    "python scripts/extract_classical_gnn_baseline.py",
    "",
    "# QGNN-v4 standing reference (LayerNorm-no-affine), Primary, 5 seeds:",
    "python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4.yaml \\",
    "    --tag repro_primary --seeds 42,43,44,45,46 \\",
    "    --head-variant layernorm_noaffine --quantum-only --diagnostics",
    "",
    "# QGNN-v4 standing reference (LayerNorm-no-affine), Severity, 5 seeds:",
    "python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4_severity.yaml \\",
    "    --tag repro_severity --seeds 42,43,44,45,46 \\",
    "    --head-variant layernorm_noaffine --quantum-only --diagnostics",
  ];
  codeLines.forEach((line, idx) => {
    c.push(
      new (require("docx").Paragraph)({
        shading: { type: require("docx").ShadingType.CLEAR, fill: "F2F2F2", color: "auto" },
        spacing: { before: idx === 0 ? 120 : 0, after: idx === codeLines.length - 1 ? 240 : 0, line: 252 },
        keepNext: idx < codeLines.length - 1,
        children: [new (require("docx").TextRun)({ text: line === "" ? " " : line, font: "Consolas", size: 17 })],
      })
    );
  });
  c.push(
    H.p(
      "Every other configuration's exact command is recorded in its own phase report and in QGNN_V4_MASTER_RESULTS.csv's source column, which names the specific report each row's numbers were verified against."
    )
  );
  c.push(H.pageBreak());

  // ---------------------------------------------------------------
  // Appendix — Final Decision Matrix
  // ---------------------------------------------------------------
  c.push(H.h1("Appendix A — Final Decision Matrix"));
  c.push(H.tableCaption("Final decision matrix across every major area investigated in this project."));
  c.push(
    H.dataTable(
      [
        { header: "Area", width: 0.18 },
        { header: "Finding", width: 0.44 },
        { header: "Confidence", width: 0.13, align: AlignmentType.CENTER },
        { header: "Status", width: 0.25 },
      ],
      [
        [
          "Primary temporal performance",
          "QGNN and Classical are close (0.811 vs. 0.807); no configuration shows a robust, seed-consistent edge over the reference",
          "Medium",
          "Established at current evidence level",
        ],
        [
          "Severity/OOD performance",
          "Classical leads consistently (0.449 vs. 0.398); gap is predominantly distribution-shift-driven, not information-deficiency-driven",
          "High",
          "Established (5-seed reference + independent in-distribution/OOD comparison)",
        ],
        [
          "Seed stability",
          "Real, multi-causal, and itself sensitive to seed count (2-seed pilots have repeatedly overstated stability gains)",
          "High",
          "Established as a general caution; specific mechanisms only partially explained",
        ],
        [
          "Representation quality",
          "Frozen embedding contains strong signal; not the primary severity bottleneck",
          "High",
          "Established (diagnostic classifiers + PCA + severity-level breakdown, all 5-seed)",
        ],
        [
          "Initialization",
          "Near-zero start trades Primary gain for Severity cost; apparent stability gains did not survive 5-seed expansion",
          "Medium-High",
          "Established for the specific configurations tested",
        ],
        [
          "Projection",
          "No alternative to the existing linear bottleneck improved either split in a 2-seed pilot",
          "Low-Medium",
          "Pilot-level evidence only, not expanded",
        ],
        [
          "Ansatz",
          "Topology is not the dominant factor behind the Classical gap; measurably affects stability and representation redundancy",
          "Medium",
          "Established (5-seed, but not parameter-matched)",
        ],
        [
          "Encoding",
          "No scale/type/re-uploading variant showed a seed-consistent improvement",
          "Low-Medium",
          "Pilot-level evidence only, not expanded",
        ],
        [
          "Fresh-onset detection",
          "Near noise floor for every model tested, including full classical",
          "High",
          "Established (multiple independent phases, consistent finding)",
        ],
        [
          "Calibration",
          "Classical consistently better calibrated; QGNN shows a recurring calibration/ranking decoupling under several interventions",
          "High",
          "Established (consistent across nearly every phase)",
        ],
      ],
      { zebra: true }
    )
  );

  return c;
}

module.exports = { build };
