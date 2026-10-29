"use strict";
const H = require("./helpers");
const { AlignmentType } = require("docx");

function build() {
  const c = [];

  // ---------------------------------------------------------------
  // Abstract
  // ---------------------------------------------------------------
  c.push(H.h1("Abstract"));
  c.push(
    H.p(
      "This work investigates whether a hybrid quantum-classical graph neural network (QGNN) — a small variational quantum circuit reading a frozen, classically-trained GraphSAGE supplier embedding — can match or exceed a classical GraphSAGE-Full baseline for supply-chain disruption risk prediction on a synthetic benchmark, and, more centrally, which architectural and training factors govern the hybrid model's behavior. Across nine controlled experimental phases and approximately 230 training runs, evaluated on both a temporal (primary) and a severity/out-of-distribution generalization split, no tested modification produced a reproducible, seed-robust improvement over the standing QGNN-v4 reference, and, among end-to-end fully-trained models, the classical GraphSAGE-Full baseline remained stronger on the severity/out-of-distribution condition throughout."
    )
  );
  c.push(
    H.p(
      "The most consequential finding was representational rather than architectural: a simple linear classifier trained only as a representation diagnostic on the same frozen embedding — not a competing end-to-end model — already matches or exceeds every quantum configuration's severity performance, and numerically exceeds the classical end-to-end head's own severity PR-AUC as well. This indicates the frozen embedding carries more severity-relevant signal than either full downstream head currently extracts. The severity split's difficulty is shown to be predominantly consistent with a genuine distribution-shift problem rather than an information deficiency, since the identical severity-5 periods are predicted well when in-distribution — though this evidence comes from a dataset whose severe condition is dominated by a single underlying severity-5 event, so it is stated as evidence within this benchmark, not a general claim about severe-event generalization."
    )
  );
  c.push(
    H.p(
      "The principal limitations — evaluation on a single, event-limited synthetic dataset; an ideal, noise-free quantum simulator with no physical hardware in the loop; and several investigations conducted as 2-seed pilots not equivalent to the project's 5-seed standing references — are documented explicitly and scope every conclusion drawn. The contribution of this work is not a higher benchmark score but a systematic, evidence-based map of which factors do and do not govern this hybrid architecture's behavior, and a demonstrated methodology — frozen-representation diagnostics, seed-level mechanism tracing, quantum-feature probing, and disciplined pilot-then-expand experimental staging — for investigating quantum-classical model behavior rigorously rather than through score-chasing."
    )
  );
  c.push(H.pageBreak());

  // ---------------------------------------------------------------
  // 1. Introduction
  // ---------------------------------------------------------------
  c.push(H.h1("1. Introduction"));
  c.push(
    H.p(
      "Supply-chain risk prediction — anticipating which suppliers are likely to be disrupted, and how severely — is a graph-structured problem: suppliers, materials, plants, products, regions, and procurement orders interact through a heterogeneous network of dependencies, and a disruption at one node can propagate along that network. Graph neural networks are a natural architectural fit for this structure. This project asks a narrower, more specific question than \"can a graph neural network predict supply-chain risk\": given a classical graph neural network already capable of learning a useful supplier representation, does replacing (or augmenting) its downstream decision-making head with a small variational quantum circuit change what the system can do — and, if it changes nothing beneficial, can the investigation still produce a rigorous, reusable account of which architectural and training factors actually govern the hybrid system's behavior?"
    )
  );

  c.push(H.h2("1.1 Research Objective"));
  c.push(
    H.p(
      "To understand whether, and under what conditions, a hybrid quantum-classical head — a variational quantum circuit reading a frozen, classically-learned supply-chain graph embedding — can match or exceed a classical graph neural network on supplier-disruption risk prediction, and, more importantly, to systematically identify which architectural and training factors actually govern the hybrid model's behavior, rather than searching for the single highest test score."
    )
  );

  c.push(H.h2("1.2 Research Questions"));
  c.push(
    ...H.numbered([
      "Does the frozen GraphSAGE representation contain enough signal for this task, and is that signal the bottleneck for the quantum head?",
      "Under what conditions is the QGNN's training stable, and what explains its seed-to-seed variance?",
      "Does the QGNN outperform a classical GNN on temporal (primary) generalization? On severity/out-of-distribution generalization?",
      "Which architectural factors (projection, initialization, ansatz, encoding, qubit count, depth) materially affect QGNN behavior, and which do not?",
      "Can the QGNN anticipate genuinely new (“fresh-onset”) disruptions, or only rank already-visible ones?",
    ])
  );

  c.push(H.h2("1.3 How This Report Is Organized"));
  c.push(
    H.p(
      "Sections 2 through 10 describe the system as built: the end-to-end architecture, how data flows through it, the benchmark graph, the classical encoder, the quantum head and its circuit, what is trainable versus frozen, the training and evaluation workflow, and the roadmap of experiments in the order they were run. Sections 11 and 12 state the experimental setup and, for every experiment, exactly what was changed and what was held fixed. Section 13 presents the results in detail, organized so that every major result carries both a numerical table and a visual figure. Section 14 discusses what the results mean, explicitly separating observed results from interpretations and from hypotheses for future work. Sections 15 through 18 cover limitations, the conclusion, future work, and reproducibility. Appendix A carries the final decision matrix."
    )
  );
  c.push(
    H.p(
      "Every number in this report is drawn from saved experiment artifacts — per-seed metric CSVs, quantum-feature CSVs, and per-example prediction files — re-verified against those artifacts during a dedicated validation pass rather than carried forward from memory. Every figure is generated either directly from those artifacts or, for the architecture and workflow diagrams, drawn to accurately represent the system's actual code path; none is decorative or fabricated. Every quantum-circuit result in this report comes from PennyLane's default.qubit simulator using exact analytic backpropagation — there is no physical quantum hardware, shot-based sampling, or noise model anywhere in this project, and no claim in this report should be read as a claim about physical quantum-hardware behavior."
    )
  );
  c.push(H.pageBreak());

  // ---------------------------------------------------------------
  // 2. System Architecture Overview
  // ---------------------------------------------------------------
  c.push(H.h1("2. System Architecture Overview"));
  const fig1 = H.figure(
    "fig01_overall_system_architecture",
    H.DIAGRAMS,
    "End-to-end architecture of the proposed hybrid classical-quantum supply-chain risk prediction system. Both downstream heads consume the identical frozen 128-dimensional GraphSAGE supplier embedding; only the head differs."
  );
  c.push(...fig1.paragraphs);
  c.push(
    H.p(
      "Figure 1 shows the complete, end-to-end system: raw benchmark data on the left, through graph construction and the classical GraphSAGE encoder, branching into the two downstream heads compared throughout this report — the classical MLP head and the QGNN-v4 quantum head — and converging on a shared evaluation protocol. This is Architecture View A, the outside-in, system-level view; Section 6 presents Architecture View B, a complementary inside-out view of the QGNN head's own internal data flow, at a level of detail this system-level diagram intentionally omits."
    )
  );
  c.push(
    H.p(
      "Introduce: the diagram separates the pipeline into four stages — data and graph construction, the classical encoder, two parallel downstream heads, and a shared evaluation stage. Observe: both heads consume the identical frozen embedding; nothing about the input to the head differs between the classical and quantum conditions, which is what makes the downstream comparison in Section 13 a controlled one. Explain: because the encoder is frozen (Section 8), any performance difference measured between the two heads is attributable to the head and its training, not to a difference in what each head is allowed to see. Limitation, with an example: the diagram is schematic at the encoder-internals level — for instance, it shows “GraphSAGE Encoder” as one box rather than expanding its two SAGEConv layers and heterogeneous per-type projections, which Section 5 covers separately — so that this figure stays readable as a single-page system overview."
    )
  );
  c.push(H.pageBreak());

  // ---------------------------------------------------------------
  // 3. Data and Information Flow
  // ---------------------------------------------------------------
  c.push(H.h1("3. Data and Information Flow"));
  c.push(
    H.p(
      "This section traces exactly how a raw benchmark record becomes a risk prediction, as ten explicit, numbered steps. Figure 2 renders this pipeline visually; the enumeration below is the same pipeline in prose, with the specific artifact produced at each step named."
    )
  );
  c.push(
    ...H.numbered([
      [
        { text: "Raw benchmark generation. ", bold: true },
        "The synthetic dataset generator produces the scm_v1_black_swan_seed43 benchmark: entity tables (suppliers, materials, plants, products, regions, procurement orders), a period-indexed event log, and the supplier_disrupted target label, over a 104-period horizon.",
      ],
      [
        { text: "Graph construction. ", bold: true },
        "Entity tables and their relationships are assembled into a single heterogeneous graph: 2,670 nodes of six types connected by 7,675 directed edges across nine relation types (Section 4).",
      ],
      [
        { text: "Feature engineering. ", bold: true },
        "Per-node static and rolling dynamic features are computed (e.g. rolling procurement and delivery aggregates for supplier nodes) and attached to every node, type by type.",
      ],
      [
        { text: "Target and horizon definition. ", bold: true },
        "The prediction target, supplier_disrupted, is defined as a four-period-ahead binary indicator per supplier-period observation; the overall positive rate is 3.33%.",
      ],
      [
        { text: "Split construction. ", bold: true },
        "Two independent evaluation conditions are built from the same underlying graph and labels: a primary (temporal) split — train on earlier periods, test on later ones — and a severity (out-of-distribution) split — train only on periods whose most severe active disruption reaches severity 3 or below, test exclusively on periods reaching severity 4 or 5.",
      ],
      [
        { text: "Classical encoder training. ", bold: true },
        "A two-layer HeteroGraphSAGE model is trained end-to-end, per seed and per split, on the full heterogeneous graph, producing a 128-dimensional embedding for every supplier node (Section 5).",
      ],
      [
        { text: "Encoder freezing. ", bold: true },
        "Once trained, the encoder's parameters are frozen (requires_grad set to False and asserted in code) before any downstream head is trained on its output (Section 8).",
      ],
      [
        { text: "Downstream head training. ", bold: true },
        "Two downstream heads are trained independently on the identical frozen embedding: the classical MLP head, and the QGNN-v4 quantum head (Sections 5–6). Only the head's own parameters receive gradient updates.",
      ],
      [
        { text: "Inference and scoring. ", bold: true },
        "Each trained head produces a risk probability per supplier-period observation on the held-out test partition of its split, at a fixed classification threshold of 0.5.",
      ],
      [
        { text: "Evaluation and aggregation. ", bold: true },
        "Predictions are scored with a single shared metrics implementation (PR-AUC, ROC-AUC, F1, precision, recall, specificity, balanced accuracy, MCC, Brier score, expected calibration error), aggregated across five independent seeds, and written to the per-run and consolidated result artifacts this report draws from (Sections 13 onward).",
      ],
    ])
  );
  const fig2 = H.figure(
    "fig02_data_processing_pipeline",
    H.DIAGRAMS,
    "Data processing pipeline from raw benchmark generation through split construction into the Primary (temporal) and Severity (out-of-distribution) evaluation conditions."
  );
  c.push(...fig2.paragraphs);
  c.push(
    H.p(
      "Introduce: this figure renders steps 1 through 5 above, ending in the branch point where the same graph and labels produce two structurally different train/test partitions. Observe: the primary and severity splits are not two different datasets — they are two different partitions of one dataset, sharing the same nodes, edges, and features. Explain: this shared-graph design is what makes the in-distribution/out-of-distribution comparison in Section 13.5 meaningful — the identical severity-5 periods can be scored under both partitions, isolating the effect of training exposure from the effect of what those periods look like. Limitation: the diagram does not depict the specific rolling-window feature computations (step 3); it names the step but leaves the computation itself to the project's feature-engineering code, since enumerating every engineered feature would not aid a reader's understanding of the overall flow."
    )
  );
  c.push(H.pageBreak());

  // ---------------------------------------------------------------
  // 4. Graph Construction
  // ---------------------------------------------------------------
  c.push(H.h1("4. Graph Construction"));
  c.push(
    H.p(
      "The benchmark graph contains 2,670 nodes of six types (300 suppliers, 100 materials, 50 plants, 200 products, 20 regions, and 2,000 procurement-order nodes) connected by 7,675 directed edges across nine relation types, forming a single connected component with mean node degree 5.75 (median 3.0, maximum 136) and average shortest path length 3.82. Graph density is 0.00215 — sparse and long-tailed, with a small number of hub nodes carrying disproportionately many edges. A supplier's network degree correlates only weakly with its empirical disruption rate (Pearson r = 0.054), ruling out a trivial structural explanation for the disruption labels: the graph's structure alone does not determine which suppliers are disrupted."
    )
  );
  const fig3 = H.figure(
    "fig03_graph_construction_schematic",
    H.DIAGRAMS,
    "Schematic node-and-relation-type structure of the benchmark heterogeneous graph. This is a small illustrative subgraph by node type and relation type — not a rendering of all 2,670 nodes and 7,675 edges, which are listed numerically instead."
  );
  c.push(...fig3.paragraphs);
  c.push(
    H.p(
      "Introduce: the figure shows the six node types and the nine relation types that connect them, using a handful of representative nodes per type rather than the full population. Observe: procurement-order nodes are the most numerous type (2,000 of 2,670 nodes) and sit at the center of the transaction relations connecting suppliers, materials, plants, and products; region nodes are comparatively few (20) and connect to suppliers geographically. Explain: this heterogeneity is exactly why the encoder (Section 5) uses type-specific input projections before any message passing — a supplier node's raw features are not directly comparable to a procurement-order node's, so each type is first projected into a shared representation space. Limitation, with an example: drawing all 2,670 nodes would produce an unreadable diagram dominated by procurement-order nodes; the schematic instead fixes a small, fixed count per type (for example, 4 supplier nodes and 3 material nodes shown, not 300 and 100) purely for legibility, and the true node and edge counts are stated numerically in the caption and in Section 11's dataset and graph tables rather than implied by the drawing."
    )
  );
  c.push(H.pageBreak());

  return c;
}

module.exports = { build };
