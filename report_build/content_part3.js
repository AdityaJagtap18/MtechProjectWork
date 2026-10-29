"use strict";
const H = require("./helpers");
const { AlignmentType } = require("docx");

function build() {
  const c = [];

  // ---------------------------------------------------------------
  // 9. Training and Evaluation Workflow
  // ---------------------------------------------------------------
  c.push(H.h1("9. Training and Evaluation Workflow"));
  const fig7 = H.figure(
    "fig07_training_evaluation_workflow",
    H.DIAGRAMS,
    "Training and evaluation workflow used for every experiment in this report, from configuration through per-seed training, shared-metric evaluation, and cross-seed aggregation, ending in an explicit KEEP / INVESTIGATE / DROP decision."
  );
  c.push(...fig7.paragraphs);
  c.push(
    H.p(
      "Every experiment in this report follows the same ten-stage workflow, from configuration to a KEEP / INVESTIGATE / DROP decision. Both models were trained with Adam (learning rate 0.001, weight decay 0.0001), a maximum of 100 epochs with early stopping on validation PR-AUC (patience 10, empirically validated against patience 25 in Section 13.6 with no material difference), balanced class weighting, and a classification threshold fixed at 0.5 for every threshold-based metric reported anywhere in this project."
    )
  );
  c.push(
    H.p(
      "Introduce: the workflow runs once per seed and is then aggregated across seeds before any decision is made — no single seed's result is treated as a finding on its own. Observe: the decision step at the bottom of the diagram has exactly three possible outcomes, and every one of the sixteen experiments summarized in Section 12.2's inventory table was assigned one of them explicitly. Explain: this uniform decision vocabulary (KEEP: adopted going forward; INVESTIGATE: a real, unresolved signal; DROP: not supported as a contributor; plus DIAGNOSTIC for measurements that are not architecture changes, and COMPLETE for investigations that finished regardless of outcome) is what allows Section 12.2's inventory table to summarize nine experimental phases in one consistent format. Limitation: the workflow diagram shows the ordinary path from a completed run to a decision; it does not depict the pilot-then-expand protocol used for several phases (Section 15.4's two-seed-pilot caveat) — that staging decision is made before this workflow begins, not inside it, and is documented separately in Section 12.2 and Section 15.4."
    )
  );
  c.push(H.pageBreak());

  // ---------------------------------------------------------------
  // 10. Research Experiment Roadmap
  // ---------------------------------------------------------------
  c.push(H.h1("10. Research Experiment Roadmap"));
  const fig8 = H.figure(
    "fig08_research_roadmap",
    H.DIAGRAMS,
    "Research experiment roadmap: thirteen experimental stages in execution order, each with the specific research question it addressed."
  );
  c.push(...fig8.paragraphs);
  c.push(
    H.p(
      "Figure 9 lays out the full sequence of experimental stages in the order they were actually run, each annotated with the specific research question it was designed to answer. The roadmap is not a plan for future work — it is a record of the investigation as executed, from the initial benchmark through this report's own final consolidation."
    )
  );
  c.push(
    H.p(
      "Introduce: the roadmap begins with the initial QGNN benchmark and the raw-baseline/LayerNorm comparison, moves through the seed-45 diagnostic, the qubit-count and depth ablation, the representation audit, the projection and initialization investigations, the controlled ansatz comparison, the encoding investigation, and ends at this report's own final consolidation. Observe: several stages branch directly from a finding in the previous stage rather than following a fixed pre-registered plan — for instance, the seed-45 investigation (the sixth stage in the roadmap) exists specifically because the fifth stage's LayerNorm comparison surfaced one seed's anomalous reversal. Explain: this stage-by-stage, finding-driven structure is why Section 12.2's inventory table records nine distinct experimental phases rather than one large factorial sweep — each stage's scope was set by what the previous stage's evidence actually showed, consistent with this project's single-variable-at-a-time protocol. Limitation: the roadmap records what was run and in what order; it does not itself carry outcome labels (KEEP/INVESTIGATE/DROP) — those are attached per-experiment in Section 12.2, since a single roadmap stage sometimes produced more than one distinct decision (for example, the tenth stage's initialization investigation produced both a DROP outcome for the general finding and a separate INVESTIGATE flag for one specific configuration)."
    )
  );
  c.push(H.pageBreak());

  // ---------------------------------------------------------------
  // 11. Experimental Setup
  // ---------------------------------------------------------------
  c.push(H.h1("11. Experimental Setup"));
  c.push(
    H.p(
      "This section states, in tabular form, the exact technology stack, dataset, and graph characteristics underlying every result in this report. Every value below was re-verified against a live command or a saved artifact during this report's own validation pass, not retyped from an earlier summary."
    )
  );

  c.push(H.h2("11.1 Technology Stack"));
  c.push(
    H.tableCaption(
      "Technology stack used for every experiment in this report."
    )
  );
  c.push(
    H.dataTable(
      [
        { header: "Component", width: 0.38 },
        { header: "Value", width: 0.62 },
      ],
      [
        ["Language", "Python 3.12.3"],
        ["Deep learning framework", "PyTorch 2.14.0+cu130"],
        ["Graph learning", "PyTorch Geometric (HeteroConv, SAGEConv)"],
        ["Quantum framework", "PennyLane 0.45.1 (qml.qnn.TorchLayer)"],
        ["Quantum simulator", "default.qubit (ideal, noise-free)"],
        ["Differentiation method", "backprop (exact / analytic gradients)"],
        [
          "CUDA",
          "Available on the development machine, not used — all computation ran on CPU, for consistency with the noise-free-simulator, no-hardware scope",
        ],
        [
          "Classical ML diagnostics",
          "scikit-learn (LogisticRegression, MLPClassifier, RandomForestClassifier, LinearSVC)",
        ],
        ["Version control", "git, branch qgnn-package-restructure"],
      ]
    )
  );

  c.push(H.h2("11.2 Dataset Characteristics"));
  c.push(H.tableCaption("Benchmark dataset characteristics."));
  c.push(
    H.dataTable(
      [
        { header: "Property", width: 0.42 },
        { header: "Value", width: 0.58 },
      ],
      [
        ["Dataset ID", "scm_v1_black_swan_seed43"],
        ["Horizon", "104 periods"],
        ["Prediction target", "supplier_disrupted, 4-period-ahead window"],
        ["Overall supplier-period disruption rate", "3.33%"],
        [
          "Total disruption events",
          "6 (3 logistics, 1 supplier failure, 1 geopolitical, 1 cyberattack)",
        ],
        ["Event severity distribution", "severity 1 × 2, severity 2 × 3, severity 5 × 1"],
        [
          "Period-level severity distribution",
          "severity 0: 39 periods; 1: 2; 2: 26; 5: 37",
        ],
        ["Primary split", "temporal 70/15/15 train/validation/test"],
        ["Severity split", "train severity ≤ 3, test severity 4–5"],
        [
          "Fresh-onset positives",
          "Primary: 0. Severity: 84 (identical across seeds/configurations)",
        ],
      ]
    )
  );

  c.push(H.h2("11.3 Graph Characteristics"));
  c.push(H.tableCaption("Benchmark graph characteristics."));
  c.push(
    H.dataTable(
      [
        { header: "Property", width: 0.42 },
        { header: "Value", width: 0.58 },
      ],
      [
        ["Total nodes", "2,670"],
        [
          "Node types",
          "supplier (300), material (100), plant (50), product (200), region (20), procurement (2,000)",
        ],
        ["Total edges", "7,675, across 9 relation types"],
        ["Degree distribution", "mean 5.75, median 3.0, min 1, max 136"],
        ["Graph density", "0.00215"],
        ["Connected components", "1 (all 2,670 nodes)"],
        ["Average shortest path length", "3.82"],
        [
          "Supplier degree vs. disruption-rate correlation",
          "r = 0.054 (not a trivial structural predictor)",
        ],
      ]
    )
  );

  c.push(H.h2("11.4 Training and Evaluation Protocol"));
  c.push(
    ...H.bullets([
      "Optimizer: Adam, learning rate 0.001, weight decay 0.0001",
      "Maximum epochs: 100, early stopping on validation PR-AUC, patience 10 (validated against patience 25, Section 13.6)",
      "Class weighting: balanced (pos_weight from train-split counts)",
      "Classification threshold: fixed at 0.5 for every metric in every phase (never tuned per-configuration, never selected using test data)",
      "Metrics: PR-AUC (primary metric), ROC-AUC, F1, precision, recall, specificity, balanced accuracy, MCC, Brier score, expected calibration error — all computed by one shared metrics.compute_classification_metrics implementation",
      "Seeds: 42–46 (5 seeds) for every full evaluation; 42–43 (2 seeds) for pilot-stage investigations, explicitly labeled seed_type=pilot throughout the master results table",
    ])
  );
  c.push(
    H.p(
      "PR-AUC, rather than accuracy, is used as the primary metric throughout because the target class is heavily imbalanced (3.33% positive rate) — a constant-negative predictor already scores roughly 97% accuracy, making accuracy uninformative for this task. PR-AUC is specifically sensitive to performance on the minority (disruption) class and, unlike F1, precision, or recall, is threshold-independent, avoiding a fixed-0.5-threshold calibration artifact documented directly in this project's earliest benchmark: a poorly-calibrated head's threshold-based metrics can look dramatically better or worse purely from where its probability distribution happens to sit relative to 0.5, independent of its true ranking quality."
    )
  );
  c.push(H.pageBreak());

  // ---------------------------------------------------------------
  // 12. Experiment Control: What Changed, What Stayed Fixed
  // ---------------------------------------------------------------
  c.push(H.h1("12. Experiment Control: What Changed, What Stayed Fixed"));
  c.push(
    H.p(
      "This section states, for every experimental phase, exactly which factor was deliberately varied and which factors were held fixed as controls — the single-variable-at-a-time discipline this project's own decision rule required throughout — followed by a single inventory table summarizing every experiment's headline outcome."
    )
  );

  c.push(H.h2("12.1 What Changed in Each Experiment?"));
  c.push(
    H.tableCaption(
      "Experiment-control matrix: the one factor deliberately varied in each phase, against the factors held fixed as controls."
    )
  );
  c.push(
    H.dataTable(
      [
        { header: "Experiment", width: 0.26 },
        { header: "Changed", width: 0.4 },
        { header: "Fixed", width: 0.34 },
      ],
      [
        [
          "Phase 2 — Patience",
          "Early-stopping patience (10 vs. 25)",
          "Architecture, optimizer, seeds, every other setting",
        ],
        [
          "Phase 2b — Output scaling",
          "Output-side multiplicative scale / additive bias",
          "Patience, projection, circuit, LayerNorm (absent)",
        ],
        [
          "Phase 2c — LayerNorm",
          "Output normalization (none / no-affine / affine)",
          "Projection, circuit, patience, output scale",
        ],
        [
          "Phase 2d — Seed-45 diagnostic",
          "Nothing (diagnostic-only, no retraining)",
          "Every setting — reuses existing checkpoints",
        ],
        [
          "Phase 3 — Qubit count",
          "n_qubits (4, 6, 8)",
          "Ansatz, layers (2), patience, LayerNorm",
        ],
        [
          "Phase 3 — Circuit depth",
          "n_layers (1, 2, 3)",
          "Qubits (6), ansatz, patience, LayerNorm",
        ],
        [
          "Phase 4 Stage 1 — Representation audit",
          "Nothing architectural (diagnostic classifiers on frozen embedding)",
          "Encoder, embedding, split protocol",
        ],
        [
          "Phase 4 Stage 2 — Projection",
          "Projection architecture (linear / nonlinear / pre-norm / PCA)",
          "Circuit, qubits (6), ansatz, initialization",
        ],
        [
          "Phase 4 Stage 4 / 4b — Initialization",
          "quantum_init strategy and gaussian_std",
          "Projection, circuit structure, ansatz, encoding",
        ],
        [
          "Phase 4 Stage 5 — Ansatz",
          "Entanglement topology (StronglyEntangling / ring / reduced-chain)",
          "Qubits (6), layers (2), initialization, encoding",
        ],
        [
          "Encoding Investigation",
          "encoding_scale, encoding_type, data_reuploading",
          "Qubits, layers, ansatz, initialization",
        ],
      ],
      { zebra: true }
    )
  );

  c.push(H.h2("12.2 Master Experiment Inventory"));
  c.push(
    H.p(
      "One row per experiment (not per split). Outcome uses this project's final decision vocabulary: KEEP (adopted going forward), INVESTIGATE (a real, unresolved signal), DROP (not supported as a contributor), DIAGNOSTIC (measures the representation or a mechanism, not a candidate architectural replacement), COMPLETE (the investigation itself finished, independent of whether the result was positive). Full per-split metrics for every row are in QGNN_V4_MASTER_RESULTS.csv."
    )
  );
  c.push(H.tableCaption("Master experiment inventory across all nine experimental phases."));
  c.push(
    H.dataTable(
      [
        { header: "Phase", width: 0.13 },
        { header: "Experiment", width: 0.24 },
        { header: "Seeds", width: 0.10, align: AlignmentType.CENTER },
        { header: "Primary PR-AUC", width: 0.16, align: AlignmentType.CENTER },
        { header: "Severity PR-AUC", width: 0.16, align: AlignmentType.CENTER },
        { header: "Outcome", width: 0.21, align: AlignmentType.CENTER },
      ],
      [
        ["Reference", "Classical GraphSAGE-Full", "5", "0.807±0.066", "0.449±0.013", "COMPLETE"],
        ["Reference", "QGNN-v4 raw baseline (pre-LayerNorm)", "5", "0.833±0.140", "0.392±0.024", "COMPLETE"],
        ["Phase 2", "Training patience", "5", "unchanged", "unchanged", "DROP"],
        ["Phase 2b", "Output scale (4 variants)", "5 ea.", "0.820–0.842", "0.388–0.404", "DROP"],
        ["Phase 2c", "LayerNorm (no-affine)", "5", "0.811±0.074", "0.398±0.057", "KEEP (reference)"],
        ["Phase 2c", "LayerNorm (affine)", "5", "0.800±0.073", "0.408±0.071", "INVESTIGATE"],
        ["Phase 2d", "Seed-45 root-cause diagnostic", "5 (existing)", "n/a", "n/a", "DIAGNOSTIC"],
        ["Phase 3", "Qubit count (4, 8)", "5 ea.", "0.847 / 0.810", "0.396 / 0.364", "INVESTIGATE"],
        ["Phase 3", "Circuit depth (1, 3 layers)", "5 / 0", "0.693 / 0.804", "0.386 / N/A", "INVESTIGATE / INCOMPLETE"],
        ["Phase 4 St.1", "Representation audit", "5", "0.671–0.781", "0.463–0.480", "DIAGNOSTIC"],
        ["Phase 4 St.2", "Projection / bottleneck", "2 (pilot)", "0.391–0.729", "0.263–0.389", "DROP"],
        ["Phase 4 St.4", "Initialization (Gaussian, identity)", "2 (pilot)", "0.726–0.828", "0.265–0.380", "INVESTIGATE"],
        ["Phase 4 St.4b", "Init. scale sweep (5 points)", "2 (pilot)", "0.812–0.828", "0.263–0.386", "DROP"],
        ["Phase 4 St.4b", "Identity-like, 5-seed expansion", "5", "0.777±0.059", "0.386±0.112", "DROP"],
        ["Phase 4 St.5", "Ansatz (ring, reduced-chain)", "5 ea.", "0.807 / 0.814", "0.354 / 0.360", "INVESTIGATE"],
        ["Encoding Inv.", "Scale / type / re-uploading", "2 (pilot)", "0.581–0.803", "0.263–0.432", "DROP"],
      ],
      { zebra: true, fontSize: 16 }
    )
  );
  c.push(H.pageBreak());

  return c;
}

module.exports = { build };
