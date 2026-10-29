"use strict";
const H = require("./helpers");
const { AlignmentType } = require("docx");

function build() {
  const c = [];

  c.push(H.h1("13. Results"));
  c.push(
    H.p(
      "Each subsection below follows the same structure: the experiment being reported, the architecture or process under test, its results as both a table and a figure, an interpretation of what the results mean, and the decision the project reached. Every number is drawn from QGNN_V4_MASTER_RESULTS.csv, QGNN_V4_REFERENCE_COMPARISON.csv, QGNN_V4_ABLATION_TABLE.csv, or a directly-recomputed statistic over a saved per-example prediction file — re-verified during this report's own validation pass, not carried forward from an earlier summary."
    )
  );

  // ---------------- 13.1 Classical Baseline ----------------
  c.push(H.h2("13.1 Classical Baseline"));
  c.push(
    H.labelPara(
      "Experiment.",
      "Establish the classical, fully end-to-end reference this report compares every QGNN-v4 configuration against."
    )
  );
  c.push(
    H.labelPara(
      "Architecture / process.",
      "HeteroGraphSAGE (Section 5), trained end-to-end including its own two-layer MLP classifier head, five independent seeds (42–46), identical training and evaluation protocol as every QGNN configuration (Section 11.4)."
    )
  );
  c.push(
    H.labelPara(
      "Results.",
      "Primary PR-AUC 0.8070 ± 0.0655; Severity PR-AUC 0.4487 ± 0.0131 (Tables 6 and 7). Severity recall is exactly 0.4406 in every one of the five seeds — a genuine, previously documented property of the benchmark (the model consistently catches the same population of already-visibly-disrupted suppliers), not a measurement artifact."
    )
  );
  c.push(
    H.labelPara(
      "Interpretation.",
      "The classical model is a strong, stable, well-calibrated reference on both splits, and its behavior is the yardstick every subsequent QGNN-v4 result in this chapter is read against."
    )
  );
  c.push(H.labelPara("Decision.", "COMPLETE — standing classical reference for the remainder of this report."));

  // ---------------- 13.2 QGNN Baseline ----------------
  c.push(H.h2("13.2 QGNN-v4 Baseline"));
  c.push(
    H.labelPara(
      "Experiment.",
      "Establish the standing QGNN-v4 reference (Section 6) that every subsequent architectural modification in this chapter is compared against."
    )
  );
  c.push(
    H.labelPara(
      "Architecture / process.",
      "6-qubit, 2-layer StronglyEntanglingLayers circuit with output-side LayerNorm (no affine parameters), five independent seeds, identical protocol to the classical baseline."
    )
  );
  c.push(
    H.labelPara(
      "Results.",
      "Primary PR-AUC 0.8112 ± 0.0737; Severity PR-AUC 0.3976 ± 0.0571. The earlier, pre-LayerNorm raw baseline reached a nominally higher primary mean (0.833 ± 0.140) but with more than double the cross-seed variance and substantially worse calibration (expected calibration error 0.285 vs. classical's 0.049)."
    )
  );
  c.push(
    H.labelPara(
      "Interpretation.",
      "LayerNorm was adopted specifically for its stability and calibration benefit — it reduced severity recall/F1 standard deviation by roughly an order of magnitude and eliminated a threshold-0.5 prediction collapse observed in two seeds — not because it closed the performance gap to the classical baseline, which it did not."
    )
  );
  c.push(H.labelPara("Decision.", "KEEP — LayerNorm-no-affine adopted as the standing QGNN-v4 reference for every subsequent comparison."));
  c.push(H.pageBreak());

  // ---------------- 13.3 Extended Metrics: Classical vs QGNN ----------------
  c.push(H.h2("13.3 Classical vs. QGNN-v4: Extended Metrics"));
  c.push(
    H.labelPara(
      "Experiment.",
      "Compare the two standing, fully-trained end-to-end references directly, on every metric this project tracks, with Primary and Severity reported separately throughout — never combined into a single \"overall\" score."
    )
  );
  c.push(H.tableCaption("Extended metrics, Primary (temporal) split — Classical GraphSAGE-Full vs. QGNN-v4 reference."));
  c.push(
    H.dataTable(
      [
        { header: "Metric", width: 0.34 },
        { header: "Classical", width: 0.22, align: AlignmentType.CENTER },
        { header: "QGNN-v4", width: 0.22, align: AlignmentType.CENTER },
        { header: "Gap (C − Q)", width: 0.22, align: AlignmentType.CENTER },
      ],
      [
        ["PR-AUC", "0.8070", "0.8112", "−0.0042"],
        ["ROC-AUC", "0.9871", "0.9775", "+0.0096"],
        ["F1", "0.6771", "0.6565", "+0.0206"],
        ["Precision", "0.5220", "0.5191", "+0.0029"],
        ["Recall", "0.9678", "0.8975", "+0.0703"],
        ["Specificity", "0.9353", "0.9400", "−0.0047"],
        ["Balanced accuracy", "0.9515", "0.9188", "+0.0327"],
        ["MCC", "0.6845", "0.6539", "+0.0306"],
        ["Brier score (lower better)", "0.0418", "0.0655", "−0.0237"],
        ["ECE (lower better)", "0.0493", "0.1823", "−0.1330"],
      ],
      { zebra: true }
    )
  );
  c.push(H.tableCaption("Extended metrics, Severity (out-of-distribution) split — Classical GraphSAGE-Full vs. QGNN-v4 reference."));
  c.push(
    H.dataTable(
      [
        { header: "Metric", width: 0.34 },
        { header: "Classical", width: 0.22, align: AlignmentType.CENTER },
        { header: "QGNN-v4", width: 0.22, align: AlignmentType.CENTER },
        { header: "Gap (C − Q)", width: 0.22, align: AlignmentType.CENTER },
      ],
      [
        ["PR-AUC", "0.4487", "0.3976", "+0.0511"],
        ["ROC-AUC", "0.7275", "0.6566", "+0.0709"],
        ["F1", "0.4113", "0.4187", "−0.0074"],
        ["Precision", "0.3871", "0.4608", "−0.0737"],
        ["Recall", "0.4406", "0.4166", "+0.0240"],
        ["Specificity", "0.9438", "0.9513", "−0.0075"],
        ["Balanced accuracy", "0.6922", "0.6840", "+0.0082"],
        ["MCC", "0.3624", "0.3824", "−0.0200"],
        ["Brier score (lower better)", "0.0857", "0.1225", "−0.0368"],
        ["ECE (lower better)", "0.0904", "0.2405", "−0.1501"],
      ],
      { zebra: true }
    )
  );
  const fig1f = H.figure(
    "fig1_classical_vs_qgnn_primary_severity",
    H.FINAL_FIGS,
    "Classical GraphSAGE-Full vs. QGNN-v4 reference, PR-AUC on both splits, mean ± one standard deviation across 5 seeds. Error bars show cross-seed spread only — no significance test is implied.",
    { maxWidthIn: 5.4 }
  );
  c.push(...fig1f.paragraphs);
  c.push(
    H.p(
      "Introduce: the figure plots the same two PR-AUC numbers that head Tables 6 and 7 above, with error bars showing ± one standard deviation across the five seeds. Observe: on Primary the two bars are close enough that their error bars overlap substantially; on Severity, Classical's bar is visibly higher and the two error bars barely overlap. Explain: this visual gap matches the tabulated gap of +0.0511 PR-AUC (11.4% relative) in Classical's favor on Severity, while the Primary gap of −0.0042 is well within either model's own cross-seed noise and is not read as a directional finding for either model. Limitation, with an example: overlapping or non-overlapping error bars across only five seeds are descriptive, not a formal significance test — no paired test (for example a paired bootstrap over matched seeds) was computed for this or any other comparison in this project (Section 15.4)."
    )
  );
  c.push(
    H.labelPara(
      "Required qualification.",
      "Every number above compares two end-to-end, fully-trained models under this project's shared protocol. Section 13.7 separately reports a logistic-regression representation diagnostic on the same frozen embedding — not a competing end-to-end model — that reaches Severity PR-AUC 0.465, numerically above both rows in the table above. Wherever this report says Classical is “stronger” on Severity, that statement is scoped to the end-to-end comparison in this subsection specifically; it does not contradict Section 13.7's diagnostic finding, and the two should always be read together."
    )
  );
  c.push(H.labelPara("Decision.", "COMPLETE — this comparison is the project's standing end-to-end reference point for every subsequent ablation."));
  c.push(H.pageBreak());

  // ---------------- 13.4 Primary Evaluation ----------------
  c.push(H.h2("13.4 Primary (Temporal) Evaluation"));
  c.push(
    H.labelPara(
      "Experiment.",
      "Evaluate forward-in-time generalization: train on earlier periods, test on later ones under conditions similar to training."
    )
  );
  c.push(
    H.labelPara(
      "Results.",
      "Classical and QGNN-v4 reach comparable PR-AUC (0.807 vs. 0.811), a difference well within either model's own cross-seed variance. The two models diverge more clearly on secondary metrics: Classical achieves meaningfully higher recall at the fixed 0.5 threshold (0.968 vs. 0.898) and is substantially better calibrated (ECE roughly 3.7× smaller: 0.049 vs. 0.182)."
    )
  );
  c.push(
    H.labelPara(
      "Interpretation.",
      "Across every architectural and training modification investigated in this project — projection, initialization, ansatz, and encoding — no single change produced a Primary PR-AUC improvement that held consistently across all five seeds where a full evaluation was run. Several 2-seed pilots appeared to show a Primary gain (most notably small-Gaussian initialization), but these gains were consistently accompanied by materially higher cross-seed variance, and the one case subsequently expanded to five seeds (identity-like initialization) had its apparent gain fail to hold at the larger seed count."
    )
  );
  c.push(H.labelPara("Decision.", "COMPLETE — no Primary-split intervention promoted beyond the standing reference."));

  // ---------------- 13.5 Severity Evaluation ----------------
  c.push(H.h2("13.5 Severity (Out-of-Distribution) Evaluation"));
  c.push(
    H.labelPara(
      "Experiment.",
      "Evaluate genuine distribution-shift generalization: train only on periods reaching severity ≤ 3, test exclusively on periods reaching severity 4–5, which the model never observes during training."
    )
  );
  c.push(
    H.labelPara(
      "Results.",
      "Classical leads QGNN-v4 on Severity PR-AUC (0.449 vs. 0.398), ROC-AUC (0.728 vs. 0.657), and calibration (ECE 0.090 vs. 0.241). A dedicated diagnostic (Section 13.7) found that when the identical severity-5 periods used as the Severity split's test set are instead evaluated as part of the Primary split's in-distribution test window, the same frozen representation supports a PR-AUC of 0.81 — close to Primary's ordinary performance level, rather than the 0.38–0.45 range observed under the genuine out-of-distribution condition."
    )
  );
  c.push(
    H.labelPara(
      "Interpretation.",
      "This in-distribution/out-of-distribution comparison strongly supports the interpretation that the Severity gap is predominantly a distribution-shift generalization problem rather than an inherent information deficiency in the input representation. This should be read as evidence specific to this benchmark's severity-5 pattern — which is dominated by a single underlying event (Section 15.1) — not as an established general claim about generalization across many independent severe disruptions, a question this dataset cannot directly answer. No architectural modification tested (projection, initialization, ansatz, encoding) closed this gap in a way that held across seeds; several made it distinctly worse, most notably small-Gaussian-family initializations."
    )
  );
  c.push(H.labelPara("Decision.", "COMPLETE — Classical remains the stronger end-to-end model on Severity throughout this investigation; see Section 13.7's required qualification."));
  c.push(H.pageBreak());

  // ---------------- 13.6 Ablation Studies ----------------
  c.push(H.h2("13.6 Ablation Studies"));
  c.push(
    H.labelPara(
      "Experiment.",
      "Nine distinct architectural and training factors were investigated as controlled ablations against the standing QGNN-v4 reference, summarized in QGNN_V4_ABLATION_TABLE.csv and, per configuration, in QGNN_V4_MASTER_RESULTS.csv."
    )
  );
  c.push(H.tableCaption("Ablation summary: nine investigated factors, their evidence, and this project's interpretation."));
  c.push(
    H.dataTable(
      [
        { header: "Factor (phase)", width: 0.24 },
        { header: "Evidence", width: 0.5 },
        { header: "Interpretation", width: 0.26 },
      ],
      [
        [
          "Training patience (Phase 2)",
          "18/20 runs select the identical checkpoint at patience 25; a positive control confirms validation PR-AUC never improves beyond the patience-10 optimum.",
          "Not supported as a major contributor",
        ],
        [
          "Output scale (Phase 2b)",
          "All four variants land within [0.82, 0.84] Primary / [0.388, 0.404] Severity — indistinguishable from the raw baseline; mathematically redundant with the existing Linear(6,1).",
          "Not supported as a major contributor",
        ],
        [
          "LayerNorm (Phase 2c)",
          "Genuine, non-redundant change. Cuts severity recall/F1 std by ≈10×, eliminates threshold-0.5 collapse. Costs Primary PR-AUC mean and reverses one severity seed (45).",
          "Mixed — adopted for stability/calibration, not for closing the Classical gap",
        ],
        [
          "Seed-45 behavior (Phase 2d)",
          "Seed 45's baseline circuit concentrated signal into one dominant channel (r=0.524); LayerNorm removes the incentive, and that channel's correlation nearly halves while every other seed's improves or holds.",
          "Supported contributor to seed-dependent distortion (medium confidence, not proven causal)",
        ],
        [
          "Qubit count (Phase 3)",
          "4-qubit gives the highest Primary mean (0.847) with the tightest Severity std of the three; confounded with classical Linear(128,n) parameter count.",
          "Mixed — capacity-confounded",
        ],
        [
          "Circuit depth (Phase 3)",
          "1 layer: worst Primary (0.693) but tightest Severity std anywhere (0.017). 3-layer Severity run killed mid-batch, no data.",
          "Inconclusive",
        ],
        [
          "Representation quality (Stage 1)",
          "Logistic regression on the frozen embedding matches/exceeds both full heads on Severity PR-AUC (0.465 vs. 0.449/0.398) with far tighter std.",
          "Supported contributor — representation is not the ceiling either head is hitting",
        ],
        [
          "Projection architecture (Stage 2)",
          "Every alternative underperforms the existing Linear(128,6) on both splits in a 2-seed pilot; PCA-informed projection underperforms despite far fewer parameters.",
          "Not supported (2-seed pilot; PCA anomaly unresolved)",
        ],
        [
          "Quantum initialization (Stage 4/4b)",
          "Small-Gaussian/identity-like improve Primary/calibration but degrade Severity in a 2-seed pilot. Identity-like's variance reduction did not survive 5-seed expansion (std 0.017→0.059).",
          "Not supported — real trade-off, stability claim did not replicate",
        ],
        [
          "Ansatz / entanglement (Stage 5)",
          "All three ansätze land within ~1% Primary / ~4% Severity of each other (5 seeds each). Reference has tightest Primary variance; not parameter-matched (36/24/12).",
          "Not supported as major contributor to the Classical gap",
        ],
        [
          "Quantum encoding (Encoding Inv.)",
          "2π scale (E2) is a clear regression (widest angle spread, weakest correlation). Linear clip (E3) saturates 46% of angles at the boundary.",
          "Not supported (2-seed pilot; no config expanded to 5 seeds)",
        ],
        [
          "Fresh-onset detection (multiple)",
          "PR-AUC 0.004–0.012 across every QGNN configuration and the full classical baseline.",
          "Supported — property of the input features, not of any architecture",
        ],
      ],
      { zebra: true }
    )
  );

  const fig2f = H.figure(
    "fig2_ablation_primary_pr_auc",
    H.FINAL_FIGS,
    "Primary-split PR-AUC across every tested configuration in this project (green = 5-seed full evaluation, red = 2-seed pilot), with the QGNN-v4 and Classical references marked by dashed lines.",
    { maxWidthIn: 5.6, maxHeightIn: 8.4 }
  );
  c.push(...fig2f.paragraphs);
  const fig3f = H.figure(
    "fig3_ablation_severity_pr_auc",
    H.FINAL_FIGS,
    "Severity-split PR-AUC across every tested configuration in this project (green = 5-seed full evaluation, red = 2-seed pilot), with the QGNN-v4 and Classical references marked by dashed lines.",
    { maxWidthIn: 5.6, maxHeightIn: 8.4 }
  );
  c.push(...fig3f.paragraphs);
  c.push(
    H.p(
      "Introduce: these two figures plot every individually tested configuration's mean PR-AUC, Primary and Severity kept as two separate figures rather than combined into one “overall” score, color-coded by whether the configuration was evaluated at the project's full 5-seed convention (green) or as a 2-seed pilot (red). Observe: the top of the Severity figure is occupied by the three representation diagnostics from Stage 1 (logistic regression, MLP, and the PCA-4 diagnostic), sitting visibly above both the QGNN-v4 and Classical end-to-end references — the single most visually direct statement of this project's central representation finding (Section 13.7). Explain: because pilot (red) and full (green) results are never visually merged, a reader cannot mistake a 2-seed pilot's apparent ranking for a confirmed 5-seed finding — several red bars near the top of the Primary figure (for instance the Gaussian-initialization pilots) do not correspond to a confirmed improvement, exactly the caution Section 15.4 states in prose. Limitation, with an example: bar length alone does not convey variance — error bars are included, but a bar's ranking position should not be read as statistically distinguishable from its near neighbors without checking the accompanying standard deviation, as Section 13.3's discussion of overlapping error bars already cautions."
    )
  );
  c.push(H.labelPara("Decision.", "See Section 12.2's Master Experiment Inventory for the per-experiment KEEP / INVESTIGATE / DROP / DIAGNOSTIC outcome."));
  c.push(H.pageBreak());

  // ---------------- 13.7 Representation Analysis ----------------
  c.push(H.h2("13.7 Representation Analysis"));
  c.push(
    H.labelPara(
      "Experiment.",
      "A dedicated representation-audit phase (Phase 4 Stage 1) investigated the frozen GraphSAGE embedding directly, independent of any quantum or classical downstream head, using diagnostic classifiers, PCA, target-correlation analysis, and graph-level diagnostics."
    )
  );
  c.push(
    H.labelPara(
      "Architecture / process.",
      "Lightweight classifiers — logistic regression, a small MLP, a linear SVM, and a random forest — were trained directly on the frozen 128-dimensional embedding, using the identical train/validation/test partition every end-to-end model uses. These are diagnostic probes, not competing end-to-end models: they were not trained under the same protocol (optimizer, regularization, early-stopping regime) as either full model."
    )
  );
  c.push(
    H.labelPara(
      "Results.",
      "On Primary, the strongest diagnostic (a small MLP) reached PR-AUC 0.733, below both full models — expected, since the diagnostics received no architecture-specific tuning. On Severity, a plain logistic regression reached PR-AUC 0.465 ± 0.005 across five seeds — numerically above both the classical end-to-end reference (0.449) and every QGNN-v4 configuration evaluated in this project (0.36–0.41), with markedly tighter cross-seed variance than either. PCA showed Primary-split signal concentrated in four components (90.9% of variance, PR-AUC 0.781, the single highest observed across the entire dimensionality sweep), while Severity-split signal was distributed across the full 128 dimensions, with performance improving monotonically as more components were retained."
    )
  );
  c.push(
    H.labelPara(
      "Interpretation.",
      "This finding does not establish that logistic regression is the strongest model in this project's own end-to-end sense; it establishes that the frozen embedding already carries more severity-relevant signal than either full downstream head currently extracts from it, and that the Severity gap observed throughout this project is not solely attributable to insufficient representational capacity in either the classical or the quantum head. The Primary/Severity signal-concentration asymmetry independently explains why a 4-qubit quantum bottleneck performed comparably to or better than wider bottlenecks on Primary specifically (Section 13.6), without requiring any assumption about the quantum circuit's own behavior. A target-aware correlation analysis found individual raw embedding dimensions correlating with the disruption target at up to r ≈ 0.87, confirming the representation encodes strongly task-relevant structure. A graph-level audit found no meaningful correlation between a supplier's network degree and its disruption rate (r = 0.054), ruling out a trivial structural explanation for the labels."
    )
  );
  c.push(H.labelPara("Decision.", "DIAGNOSTIC — not an architecture change; the single most consequential finding of this project's representation-side investigation."));
  c.push(H.pageBreak());

  // ---------------- 13.8 Quantum Resource Analysis ----------------
  c.push(H.h2("13.8 Quantum Resource Analysis"));
  c.push(
    H.labelPara(
      "Experiment.",
      "State the exact computational resources every quantum-circuit result in this project draws on, and compare the parameter budgets of the three ansätze evaluated in the controlled ansatz comparison (Section 13.6)."
    )
  );
  c.push(H.tableCaption("Quantum resource comparison across every circuit configuration evaluated in this project."));
  c.push(
    H.dataTable(
      [
        { header: "Configuration", width: 0.32 },
        { header: "Qubits", width: 0.12, align: AlignmentType.CENTER },
        { header: "Layers", width: 0.12, align: AlignmentType.CENTER },
        { header: "Quantum params", width: 0.19, align: AlignmentType.CENTER },
        { header: "Total head params", width: 0.25, align: AlignmentType.CENTER },
      ],
      [
        ["StronglyEntanglingLayers (reference)", "6", "2", "36", "817"],
        ["Hardware-efficient ring", "6", "2", "24", "805"],
        ["Reduced entanglement", "6", "2", "12", "793"],
        ["4-qubit variant (Phase 3)", "4", "2", "24", "545"],
        ["8-qubit variant (Phase 3)", "8", "2", "48", "1,089"],
      ],
      { zebra: true }
    )
  );
  c.push(
    H.labelPara(
      "Results.",
      "Every circuit in this project was executed on PennyLane's default.qubit simulator using exact, analytic backpropagation (diff_method=\"backprop\"). There is no shot noise, no hardware noise model, and no physical quantum device anywhere in this investigation's results. The frozen GraphSAGE encoder contributes approximately 1.2 million parameters, roughly 1,500× the size of the entire trainable QGNN-v4 head."
    )
  );
  c.push(
    H.labelPara(
      "Interpretation.",
      "The two alternative ansätze evaluated in the controlled comparison used fewer quantum parameters at the same qubit count and depth — 24 for the ring topology, 12 for the reduced-entanglement topology — and reached comparable mean PR-AUC on both splits despite this reduced budget, though the comparison is not parameter-matched and this pattern should not be read as demonstrating that fewer quantum parameters are inherently sufficient."
    )
  );
  c.push(H.labelPara("Decision.", "DIAGNOSTIC — resource accounting, not a candidate architectural change."));
  c.push(H.pageBreak());

  // ---------------- 13.9 Seed Sensitivity ----------------
  c.push(H.h2("13.9 Seed Sensitivity"));
  c.push(
    H.labelPara(
      "Experiment.",
      "Characterize cross-seed variance for the two standing references, and trace one specific instance of seed-dependent behavior (seed 45) to a concrete mechanism."
    )
  );
  c.push(H.tableCaption("Per-seed results, standing references (test split). Re-verified directly against raw per-seed metric artifacts."));
  c.push(
    H.dataTable(
      [
        { header: "Model", width: 0.14 },
        { header: "Seed", width: 0.10, align: AlignmentType.CENTER },
        { header: "Primary PR-AUC", width: 0.16, align: AlignmentType.CENTER },
        { header: "Primary F1", width: 0.15, align: AlignmentType.CENTER },
        { header: "Severity PR-AUC", width: 0.16, align: AlignmentType.CENTER },
        { header: "Severity F1", width: 0.15, align: AlignmentType.CENTER },
        { header: "Severity MCC", width: 0.14, align: AlignmentType.CENTER },
      ],
      [
        ["Classical", "42", "0.7874", "0.7236", "0.4430", "0.3838", "0.3312"],
        ["Classical", "43", "0.6888", "0.6382", "0.4560", "0.4431", "0.3991"],
        ["Classical", "44", "0.8745", "0.7032", "0.4378", "0.4228", "0.3753"],
        ["Classical", "45", "0.8491", "0.6865", "0.4358", "0.4022", "0.3518"],
        ["Classical", "46", "0.8354", "0.6340", "0.4708", "0.4045", "0.3544"],
        ["Classical", "mean±std", "0.8070±0.0655", "0.6771±0.0355", "0.4487±0.0131", "0.4113±0.0201", "0.3624±0.0231"],
        ["QGNN-v4 ref.", "42", "0.8682", "0.6800", "0.3943", "0.3793", "0.3263"],
        ["QGNN-v4 ref.", "43", "0.6807", "0.5950", "0.4496", "0.5320", "0.5541"],
        ["QGNN-v4 ref.", "44", "0.8197", "0.6992", "0.4156", "0.4248", "0.3776"],
        ["QGNN-v4 ref.", "45", "0.7951", "0.6156", "0.2900", "0.3632", "0.3111"],
        ["QGNN-v4 ref.", "46", "0.8924", "0.6929", "0.4385", "0.3941", "0.3427"],
        ["QGNN-v4 ref.", "mean±std", "0.8112±0.0737", "0.6565±0.0428", "0.3976±0.0571", "0.4187±0.0602", "0.3824±0.0887"],
      ],
      { zebra: true, boldRows: [5, 11], fontSize: 16 }
    )
  );
  const fig5f = H.figure(
    "fig5_seedwise_pr_auc",
    H.FINAL_FIGS,
    "Per-seed PR-AUC, Classical vs. QGNN-v4 reference, Primary (left) and Severity (right). Seed 45's severity reversal is directly visible as the one point where the QGNN-v4 line drops sharply below the Classical line.",
    { maxWidthIn: 6.1 }
  );
  c.push(...fig5f.paragraphs);
  c.push(
    H.p(
      "Introduce: each panel plots one line per model across the five standing-reference seeds, Primary on the left and Severity on the right. Observe: on the Severity panel, seeds 42–44 and 46 show QGNN-v4 tracking reasonably close to Classical, but seed 45 drops to 0.290, the single largest gap between the two lines anywhere in the figure. Explain: Figure 14 below traces this specific drop to a measured mechanism — seed 45's baseline circuit concentrated an unusually large share of its discriminative signal into one Pauli-Z output channel (correlation 0.524, the highest of any seed), and LayerNorm removes the optimization incentive for this concentration, causing that seed's severity ranking to regress while every other seed's representation improves or holds flat under the identical change. Limitation, with an example: this is a correlational, diagnostic-only finding — no controlled causal intervention (such as retraining seed 45 under an objective that explicitly discourages channel concentration) was run to confirm the mechanism causally; it is the strongest available explanation for this specific seed's behavior, not a proven cause."
    )
  );
  const seed45 = H.figure(
    "chart_seed45_analysis",
    H.CHARTS,
    "Seed-45 visual analysis. Left: predicted-probability distribution on the Severity test split, baseline circuit vs. LayerNorm-no-affine. Right: maximum |Pauli-Z channel ↔ label| correlation per seed, baseline vs. LayerNorm, computed fresh from each seed's raw per-example output.",
    { maxWidthIn: 6.1 }
  );
  c.push(...seed45.paragraphs);
  c.push(
    H.p(
      "Introduce: the left panel overlays seed 45's predicted-probability histogram before and after LayerNorm; the right panel shows every seed's best single-channel correlation with the label, baseline vs. LayerNorm, independently recomputed for this report via scipy's point-biserial correlation on each seed's raw per-example CSV. Observe: seed 45 is the one seed where the LayerNorm bar is markedly lower than the baseline bar (0.524 → 0.282); every other seed's LayerNorm bar is equal to or higher than its baseline bar. Explain: this is why seed 45 alone reverses under LayerNorm while LayerNorm's aggregate benefit (chiefly stability and calibration) is preserved across the five-seed population — LayerNorm's benefit is a population-level effect across five seeds, not a uniform per-seed improvement, and this figure makes the one exception directly visible rather than hiding it inside a mean and a standard deviation. Limitation: the fresh recomputation shown here was cross-checked against Phase 2d's originally reported values and found to match exactly ({42: 0.270→0.314, 43: 0.360→0.401, 44: 0.387→0.411, 45: 0.524→0.282, 46: 0.415→0.409}) — stated explicitly because an independent numerical cross-check, not merely a repeated claim, is what establishes this figure's reliability."
    )
  );
  c.push(H.labelPara("Decision.", "DIAGNOSTIC — seed-dependent representation distortion, supported contributor to instability (medium confidence, not proven causal)."));
  c.push(H.pageBreak());

  // ---------------- 13.10 Fresh-Onset Analysis ----------------
  c.push(H.h2("13.10 Fresh-Onset Analysis"));
  c.push(
    H.labelPara(
      "Experiment.",
      "Measure whether any model tested can anticipate a genuinely new (“fresh-onset”) disruption, as opposed to ranking already-visible, ongoing ones."
    )
  );
  c.push(H.tableCaption("Fresh-onset analysis by split."));
  c.push(
    H.dataTable(
      [
        { header: "Split", width: 0.2, align: AlignmentType.CENTER },
        { header: "n Fresh-Onset", width: 0.25, align: AlignmentType.CENTER },
        { header: "Fresh-Onset PR-AUC (all models tested)", width: 0.55, align: AlignmentType.CENTER },
      ],
      [
        ["Primary", "0", "N/A — undefined (no fresh-onset examples exist; never plotted or reported as zero)"],
        ["Severity", "84", "0.005–0.012 (every architecture, quantum and classical)"],
      ]
    )
  );
  const chartFresh = H.figure(
    "chart_fresh_onset",
    H.CHARTS,
    "Fresh-onset PR-AUC, Severity split only. The Primary split has zero fresh-onset positive examples and is not plotted — the metric is undefined there, not zero.",
    { maxWidthIn: 5.4 }
  );
  c.push(...chartFresh.paragraphs);
  c.push(
    H.p(
      "Introduce: the chart compares the Classical and QGNN-v4 references' Severity-split fresh-onset PR-AUC against the range observed across the quantum-encoding investigation's five pilot configurations. Observe: all three bars sit within a narrow band close to zero (0.005–0.012), an order of magnitude below any of this project's main PR-AUC results. Explain: because the full Classical GraphSAGE-Full baseline — with complete graph structure and message-passing access — shows the identical near-zero floor as every QGNN-v4 configuration, this project's evidence points to the absence of a genuine leading indicator in the benchmark's available dynamic features at the four-period prediction horizon used, rather than to a limitation specific to any one architecture, quantum or classical. Limitation, with an example: the Primary split's test window falls entirely inside the same severity-5 event's already-ongoing period, so it contains zero fresh-onset positives by construction — this is a property of how the split was built, not evidence that Primary-split fresh-onset detection would also be near-zero; that comparison simply cannot be made with this dataset, and the report states this explicitly rather than substituting a zero."
    )
  );
  c.push(H.labelPara("Decision.", "Supported contributor to understanding this limitation — ruled out as architecture-specific; a property of the benchmark's input features at this horizon."));
  c.push(H.pageBreak());

  // ---------------- 13.11 Calibration Analysis ----------------
  c.push(H.h2("13.11 Calibration Analysis"));
  c.push(
    H.labelPara(
      "Experiment.",
      "Compare predicted-probability calibration — not just ranking quality — between Classical and QGNN-v4, and across the QGNN-v4 configurations investigated."
    )
  );
  const fig6f = H.figure(
    "fig6_calibration_ece",
    H.FINAL_FIGS,
    "Expected calibration error (lower is better) across every configuration this project recorded a value for, colored by 5-seed full evaluation (green) vs. 2-seed pilot (red).",
    { maxWidthIn: 5.6, maxHeightIn: 8.3 }
  );
  c.push(...fig6f.paragraphs);
  const chartCal = H.figure(
    "chart_calibration_reliability",
    H.CHARTS,
    "Reliability diagrams, Classical vs. QGNN-v4 reference, Primary (left) and Severity (right), computed fresh by pooling all 5 seeds' test-split predictions per split and binning into deciles.",
    { maxWidthIn: 6.1 }
  );
  c.push(...chartCal.paragraphs);
  c.push(
    H.p(
      "Introduce: the bar chart ranks every recorded configuration by expected calibration error; the reliability diagrams show, for the two standing references specifically, how closely predicted probability matches empirical positive frequency within each decile bin. Observe: Classical GraphSAGE-Full occupies the two lowest (best) bars in the ECE ranking on both splits, and its reliability curve stays visibly closer to the diagonal (perfect calibration) than QGNN-v4's on both panels. Explain: this ranking and the reliability-diagram shape are two independent measurements of the same underlying property — the bar chart is read directly from saved per-run ECE values, while the reliability diagram was computed fresh for this report from pooled per-example predictions.csv files, and the two are consistent with each other, which is itself a cross-check on both. Limitation, with an example: several interventions (most clearly near-zero-start quantum-parameter initializations and certain ansatz variants) produced a calibration/ranking decoupling — numerically better calibration accompanied by numerically worse PR-AUC on the same split — so a low position in this ECE chart should never be read as “the better model” without checking the corresponding PR-AUC figure (Section 13.6) alongside it; the quantum-encoding investigation's own pilot is a stated exception where calibration and ranking moved together for every configuration tested."
    )
  );
  c.push(H.labelPara("Decision.", "Established — Classical is the best-calibrated model in every direct comparison in this project, on both splits, by a substantial and consistent margin."));
  c.push(H.pageBreak());

  // ---------------- 13.12 Quantum Representation Analysis ----------------
  c.push(H.h2("13.12 Quantum Representation Analysis"));
  c.push(
    H.labelPara(
      "Experiment.",
      "Probe the six Pauli-Z output channels directly — not estimated — across the ansatz comparison (Stage 5) and the encoding investigation, to test whether ansatz and encoding choices change the quantum circuit's learned representation even where they do not change final task metrics."
    )
  );
  const chartQchan = H.figure(
    "chart_quantum_channel_correlation",
    H.CHARTS,
    "Maximum channel–target correlation vs. maximum cross-channel correlation, across the three ansätze (A0–A2) and five encoding configurations (E0–E4) directly probed in this project.",
    { maxWidthIn: 6.1 }
  );
  c.push(...chartQchan.paragraphs);
  const chartAngles = H.figure(
    "chart_encoding_angle_distributions",
    H.CHARTS,
    "Encoded-angle standard deviation per (seed, split), E0 (reference, π-scale tanh) through E4 (data re-uploading), showing E2's markedly wider spread (2π scale) and E3's boundary saturation (linear clip).",
    { maxWidthIn: 6.1 }
  );
  c.push(...chartAngles.paragraphs);
  c.push(
    H.p(
      "Introduce: the first chart plots each configuration's two most important representation properties against each other — how strongly individual channels track the label, and how redundant the six channels are with one another; the second chart shows the raw spread of encoded rotation angles feeding the circuit for each encoding variant. Observe: the reduced-entanglement ansatz (A2) shows the highest cross-channel correlation (≈ 0.93) among the three ansätze — its six outputs are the most mutually redundant — while E2 (2π encoding scale) shows the lowest channel-target correlation (≈ 0.29) and, in the second chart, visibly the widest angle spread of the five encoding variants. Explain: both patterns are directly measured mechanisms, not inferred from downstream metrics alone — E2's wide angle spread pushes rotation angles further around the periodic encoding, which independently explains its weak channel-target correlation and the training-convergence delay documented in Section 13.6; A2's channel redundancy is consistent with (though does not alone prove) its observed severity specificity collapse on two of five seeds. Limitation, with an example: neither of these representational differences translated into a corresponding, seed-robust difference in final task PR-AUC in any configuration tested — the quantum circuit is demonstrably not inert to these architectural choices, but this project's evidence locates the performance ceiling downstream of the representation (Section 13.7), not in whether a distinctive representation is extracted at all, so these figures should be read as characterizing representational sensitivity, not as evidence of a metrics benefit."
    )
  );
  c.push(H.labelPara("Decision.", "DIAGNOSTIC — the quantum representation is measurably sensitive to ansatz and encoding; this sensitivity did not translate into a metrics benefit in any configuration tested."));
  c.push(H.pageBreak());

  // ---------------- 13.13 Primary vs Severity Comparison ----------------
  c.push(H.h2("13.13 Primary vs. Severity Comparison"));
  c.push(
    H.labelPara(
      "Experiment.",
      "Plot every tested configuration jointly by its Primary and Severity PR-AUC, to check whether any configuration improves on one split without regressing the other."
    )
  );
  const fig4f = H.figure(
    "fig4_primary_vs_severity_scatter",
    H.FINAL_FIGS,
    "Every tested configuration plotted by (Primary, Severity) PR-AUC jointly. The Classical reference sits near the top of the Severity axis relative to the whole cloud of QGNN-v4 configurations.",
    { maxWidthIn: 5.2 }
  );
  c.push(...fig4f.paragraphs);
  c.push(
    H.p(
      "Introduce: each point is one tested configuration's mean (Primary PR-AUC, Severity PR-AUC), with the Classical and QGNN-v4 references marked separately as stars. Observe: the cloud of QGNN-v4 configurations spans roughly 0.40 to 0.85 on the Primary axis but is compressed into a narrower band, mostly below 0.40, on the Severity axis, while the Classical reference sits above nearly the entire cloud on Severity despite a middling Primary score relative to the QGNN-v4 cluster. Explain: this visual asymmetry is the scatter-plot restatement of Section 13.5's Severity gap — no point in the QGNN-v4 cloud simultaneously matches Classical's Severity height while also leading on Primary, which is exactly what “no configuration closed the gap on both splits at once” looks like in two dimensions rather than in a table. Limitation: this plot aggregates configurations across both 5-seed full evaluations and 2-seed pilots without visually distinguishing them (unlike Figures 11–12's explicit green/red coding) — a reader should cross-reference Section 12.2's inventory table before treating any individual point's position as a confirmed result rather than, in several cases, a pilot-stage estimate."
    )
  );
  c.push(H.labelPara("Decision.", "COMPLETE — confirms no tested configuration improved both splits simultaneously relative to the standing references."));
  c.push(H.pageBreak());

  return c;
}

module.exports = { build };
