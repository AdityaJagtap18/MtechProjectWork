"use strict";
const H = require("./helpers");

function build() {
  const c = [];

  // ---------------------------------------------------------------
  // 5. Classical GraphSAGE Processing
  // ---------------------------------------------------------------
  c.push(H.h1("5. Classical GraphSAGE Processing"));
  const fig4 = H.figure(
    "fig04_classical_graphsage_flow",
    H.DIAGRAMS,
    "Classical GraphSAGE-Full processing flow: type-specific input projection, two layers of heterogeneous mean-aggregation SAGEConv, supplier-node readout, and the branch into the classical MLP head versus the frozen 128-dimensional embedding consumed by the QGNN-v4 head."
  );
  c.push(...fig4.paragraphs);
  c.push(
    H.p(
      "The classical encoder, HeteroGraphSAGE, is a two-layer heterogeneous graph neural network built from PyTorch Geometric's HeteroConv and mean-aggregation SAGEConv operators, with hidden_dim=128 throughout. It is trained end-to-end, once per seed and per split, and its own final classifier head (a two-layer MLP) is what produces the classical baseline's predictions reported throughout this report. The identical trained encoder — before its own classifier head — is reused as the frozen embedding source for every QGNN-v4 experiment: the encoder is trained exactly once per seed/split and never retrained for a downstream-head experiment."
    )
  );
  c.push(
    H.p(
      "Introduce: each node type's raw features are first projected into a shared 128-dimensional space, then two rounds of heterogeneous message passing update every node's representation using its type-specific neighbors. Observe: only the supplier-node representations are read out after the second layer — material, plant, product, region, and procurement-order nodes participate in message passing (they carry and propagate information) but are not themselves classified. Explain: this is why the frozen embedding handed to the QGNN head is exactly 128-dimensional per supplier, regardless of how many other node types or edges exist in the graph — the readout step is a fixed contraction point shared by both downstream heads. Limitation: the diagram shows two message-passing layers as two boxes; it does not expand the internal mean-aggregation formula for SAGEConv itself, since that formula is standard and unmodified from the original GraphSAGE operator."
    )
  );
  c.push(H.pageBreak());

  // ---------------------------------------------------------------
  // 6. QGNN-v4 Detailed Architecture (Architecture View B)
  // ---------------------------------------------------------------
  c.push(H.h1("6. QGNN-v4 Detailed Architecture"));
  const fig5 = H.figure(
    "fig05_qgnn_detailed_architecture",
    H.DIAGRAMS,
    "Detailed internal architecture of the QGNN-v4 head (Architecture View B), from the frozen 128-dimensional embedding through the classical projection, angle encoding, the two-layer variational quantum circuit, Pauli-Z measurement, LayerNorm, and the final linear classifier."
  );
  c.push(...fig5.paragraphs);
  c.push(
    H.p(
      "This is Architecture View B — the internal, inside-out view of the QGNN-v4 head specifically, complementing Figure 1's system-level Architecture View A. The QGNN-v4 head reduces the frozen 128-dimensional supplier embedding through a trainable Linear(128, 6) layer, bounds the result with π·tanh(·) so every value lies in (−π, π), encodes it into six qubits via RY angle embedding, applies a two-layer StronglyEntanglingLayers variational circuit (36 trainable quantum parameters), measures the Pauli-Z expectation value of each of the six qubits, passes the six resulting values through a LayerNorm with no learnable affine parameters, and finally applies a Linear(6, 1) layer to produce a single risk logit. The complete head has 817 trainable parameters: 774 in the input projection (128×6 weights plus 6 biases), 36 in the quantum circuit, and 7 in the output layer (6 weights plus 1 bias)."
    )
  );
  c.push(
    H.p(
      "Introduce: the nested box in the center of the figure expands the two-layer variational circuit specifically, distinguishing it from the classical projection stage before it and the classical output stage after it. Observe: exactly one nonlinearity (π·tanh) sits between the frozen embedding and the quantum circuit, and exactly one normalization layer (LayerNorm, no affine) sits between the quantum circuit and the final classifier — every other transformation shown is linear or a fixed quantum operation. Explain: the tanh bound exists because angle-embedding gates are periodic in their input angle, so an unbounded projection output could wrap around unpredictably across the 2π period; bounding to (−π, π) keeps the encoding one-to-one over its full useful range. Limitation, with an example: the figure depicts the StronglyEntanglingLayers block as a single labeled box rather than unrolling all twelve individual single-qubit rotation gates and six entangling operations per layer; the exact gate-level circuit is shown separately in Figure 6 specifically so this figure can stay focused on data flow rather than gate-level detail."
    )
  );
  c.push(H.pageBreak());

  // ---------------------------------------------------------------
  // 7. Quantum Circuit Diagram
  // ---------------------------------------------------------------
  c.push(H.h1("7. Quantum Circuit Diagram"));
  c.push(
    H.p(
      "Figure 6 renders the QGNN-v4 quantum circuit in the conventional logical-circuit style: one horizontal wire per qubit, gates applied left to right. Six qubits (q0–q5) are each initialized with an RY(θ) angle-embedding gate carrying one of the six projected-and-bounded classical features, followed by two repeated variational layers. Each variational layer applies one general single-qubit rotation, Rot(φ, θ, ω), to every qubit, followed by a ring of controlled-NOT entangling gates connecting each qubit to a neighbor. After both layers, every qubit is measured in the Pauli-Z basis, producing six real-valued expectation values ⟨Z⟩ that become the classical feature vector passed to LayerNorm."
    )
  );
  const fig6 = H.figure(
    "fig06_quantum_circuit_diagram",
    H.DIAGRAMS,
    "Logical circuit diagram of the QGNN-v4 variational quantum circuit: RY angle embedding on six qubits, two variational layers of general single-qubit rotation plus ring entanglement, and Pauli-Z measurement. The entangling pattern shown is the connectivity StronglyEntanglingLayers actually applies at this qubit count; PennyLane's own template logic sets each layer's exact range parameter, so this is the logical circuit architecture, not a hardware-transpiled or fabricated connectivity diagram."
  );
  c.push(...fig6.paragraphs);
  c.push(
    H.p(
      "Introduce: reading left to right, the circuit is encode once, then entangle-and-rotate twice, then measure once — there is no mid-circuit measurement and no classical feedback within a single forward pass. Observe: the entangling gates connect each qubit to its neighbor in a ring, so information originating on any one qubit can reach every other qubit within the two layers shown. Explain: this connectivity is what allows the six Pauli-Z outputs to become mutually correlated — a property measured directly in Section 13.12's quantum-representation analysis, where cross-channel correlation is shown to depend on exactly this entangling structure and on the ansatz variant used. Limitation, stated directly in the figure's own caption as required: the per-layer entangling range is determined internally by PennyLane's StronglyEntanglingLayers template rather than fixed by this project's own code, so the ring pattern drawn here is the representative, logically-equivalent connectivity for six qubits — not a claim that every layer's range parameter has been individually hand-verified gate by gate."
    )
  );
  c.push(H.pageBreak());

  // ---------------------------------------------------------------
  // 8. What Is Trainable? Frozen vs. Trainable Components
  // ---------------------------------------------------------------
  c.push(H.h1("8. What Is Trainable? Frozen vs. Trainable Components"));
  c.push(
    H.p(
      "A single design decision governs every experiment in this report: the GraphSAGE encoder is frozen during the QGNN experiments. It is trained exactly once per seed and per split, as a classical model, and then its parameters are frozen (requires_grad set to False, and this condition is asserted directly in the training script's own run loop for every QGNN run, not merely claimed in prose) before any quantum or classical downstream head is trained on its output embeddings. No experiment in this report trains the graph encoder and a downstream head jointly, and no experiment modifies the encoder's own architecture."
    )
  );
  const fig9 = H.figure(
    "fig09_trainable_vs_frozen",
    H.DIAGRAMS,
    "What is trainable in the QGNN-v4 system. Frozen: the GraphSAGE encoder (≈ 1.2 million parameters), trained once per seed/split and never updated during any downstream-head experiment. Trainable: the Linear(128,6) projection, the 36 quantum-circuit parameters, and the Linear(6,1) output layer — 817 parameters total; LayerNorm (no-affine) contributes no trainable parameters of its own."
  );
  c.push(...fig9.paragraphs);
  c.push(
    H.p(
      "Introduce: the figure separates the system into a single frozen block and a small trainable block, then breaks the trainable block into its three parameter-bearing components. Observe: the frozen encoder (≈ 1.2 million parameters, HeteroGraphSAGE, hidden_dim=128, two layers) is roughly 1,500 times larger than the entire trainable QGNN head (817 parameters) — the quantum circuit itself contributes only 36 of those 817. Explain: because the encoder is frozen and shared identically between the classical and quantum conditions, every comparison in this report is a comparison between two small downstream heads reading the same fixed input, not a comparison between two differently-sized end-to-end models; this is what makes the QGNN-vs-classical-head comparison in Section 13 a matched-input (though not parameter-matched) comparison. Limitation, with an example: this parameter accounting is head-only — it does not include the classical baseline's own MLP(128→64→1) classifier head, whose parameter count (8,449) is deliberately larger than the QGNN head's 817; the two heads are matched in the width of their read of the shared embedding, not in total trainable parameter count, a distinction discussed further as an open question in Section 14.2."
    )
  );
  c.push(H.pageBreak());

  return c;
}

module.exports = { build };
