# QGNN-v2 Architecture Analysis

**Analysis-only.** No source code, datasets, labels, preprocessing,
configs, or existing experiment artifacts were modified. Nothing in this
document has been implemented or run. All claims below are either (a)
read directly from the current repository state, cited with file:line,
or (b) explicitly marked as a recommendation/hypothesis rather than a
measured fact.

---

## 1. Executive Recommendation

Build QGNN-v2 as a **frozen shared graph encoder** feeding two matched
heads:

```
Existing, unmodified GraphSAGE-Full checkpoint (per seed)
    -> HeteroGraphSAGE.encode(...) -- ALREADY a separate method, no code change needed
    -> supplier embedding, [n_suppliers, 128], frozen (no gradients ever flow back into it)
    -> PCA (fit on train-split embeddings only, reusing reduction.py's existing pattern)
    -> 8-dimensional bottleneck, identical for both heads
         /                                      \
  classical MLP head                    quantum head (existing qgnn.py, unchanged)
         |                                       |
       risk                                    risk
```

This is Option C from §3 (frozen encoder, two heads trained on top) —
the only one of the four options that gives both heads byte-for-byte
identical input at every timestep, which is the whole point of this
exercise (§4/§5). It requires **no new graph-encoder training at all**:
reuse the existing, already-frozen `GraphSAGE-Full` checkpoints in
`experiments/classical_gnn/`, which are already leakage-audited.
`HeteroGraphSAGE.encode()` already exists as a method separate from
`forward()` (`src/scm_dataset/modeling/graphsage.py:61-66`) specifically
because it returns pre-classifier embeddings — this was not built for
QGNN-v2, but it happens to be exactly what QGNN-v2 needs.

The existing `qgnn.py` quantum model does not need to change at all —
`QGNN`/`QuantumCircuitLayer`/`train_qgnn`/`evaluate_qgnn` already consume
a plain `[batch, n_qubits]` tensor and have no idea whether it came from
raw supplier features (v1) or a graph embedding (v2).

---

## 2. Existing GraphSAGE Architecture (as implemented, not as documented)

| Aspect | Detail | Source |
|---|---|---|
| Graph container | `SupplyChainGraph` — typed node dict + edge list, no networkx | `src/scm_dataset/schema/graph.py:42` |
| Node types (6) | `SUPPLIER, PROCUREMENT, MATERIAL, PLANT, PRODUCT, REGION` | `src/scm_dataset/schema/nodes.py:23-29` |
| Edge types (9 base) | `SUPPLIER_MATERIAL, SUPPLIER_PROCUREMENT, PROCUREMENT_MATERIAL, PROCUREMENT_PLANT, MATERIAL_PLANT, PLANT_PRODUCT, PRODUCT_REGION, SUPPLIER_REGION, PLANT_REGION` | `src/scm_dataset/schema/edges.py:19-27` |
| Relations at message-passing time | 18 = 9 base + 9 `rev_<type>` (reverse), added only in the PyG representation, never on disk | `src/scm_dataset/modeling/hetero_graph.py:84-85` (`build_edge_index_dict`) |
| Topology | Static, built once per dataset, reused for every timestep snapshot — only node feature tensors vary by `t` | `hetero_graph.py:93-98` (`build_topology`), `100-` (`HeteroGraphSnapshotBuilder`) |
| Node id → index | Deterministic natural sort by trailing integer | `hetero_graph.py:46-49` (`_natural_sort_key`) |
| Node feature tensors | `input_proj[node_type]`: per-type `Linear(in_dim, hidden_dim)`, in_dim = 60 (supplier, full mode) / 48 (material) / 38 (plant) / 45 (product) / 12 (region) / 16 (procurement) | `graphsage.py:43` |
| Message passing | `HeteroConv({rel: SAGEConv(hidden_dim, hidden_dim, aggr) for rel in edge_types}, aggr="mean")` × `num_layers` (default 2), ReLU+Dropout between layers | `graphsage.py:45-51, 61-66` |
| Hidden dim | 128 (`ModelConfig.hidden_dim`, `config.py:55`) |
| Supplier representation (tap point) | `h_dict["supplier"]` returned by `encode()`, **before** the classifier | `graphsage.py:61-66` |
| Prediction head | `classifier = Linear(128,64) → ReLU → Dropout → Linear(64,1)`, applied only to `h_dict["supplier"]`, no sigmoid (raw logits) | `graphsage.py:54-59, 68-72` |
| Training loop | One epoch = loop over shuffled distinct train-split prediction times `t`; each `t` = **one full forward pass over the entire graph** (all ~2,670 nodes at once, not a sampled minibatch of suppliers) | `train.py:124-139` |
| Temporal handling | Static topology + time-indexed feature tensors per snapshot (`HeteroGraphSnapshotBuilder.build(t)`); no recurrence/temporal attention — each `t` is an independent forward pass | `hetero_graph.py:100-` |
| "Batching" | None in the minibatch sense — full-graph, full-batch per timestep; `snapshots = {t: builder.build(t) for t in ...}` precomputed once before the epoch loop | `train.py:98` (original), same pattern in `qgnn.py:_flatten_split` (v1's post-hoc flat-batching, §see below) |
| Loss | `BCEWithLogitsLoss`, class-weighted (`class_weighting: balanced`) | `losses.py:compute_pos_weight`, `build_loss` |
| Class weighting | Computed from **train-split** targets only | `train.py:110-112` |
| Early stopping | Watches **validation PR-AUC only**, patience configurable (default 10); test split never touched during training | `train.py:114-118, 175-185` |
| Validation | Full forward pass over `val_times`, `model.eval()` + `torch.no_grad()` | `train.py:141-159` |
| Test evaluation | `evaluate_experiment`/`generate_predictions`, threshold selected from validation split only, applied to test | `evaluate.py:45-65, 281-288` |

## 3. Existing QGNN Architecture (v1)

```
reduced supplier tensor [batch, d]  (already standardized)
    -> angle = tanh(x) * pi                     (deterministic, label-free, bounded (-pi, pi))
    -> qml.AngleEmbedding(angles, rotation="Y")  (d qubits)
    -> per layer: RY(trainable weight) per qubit, then linear CNOT chain (wire i -> i+1)
    -> [qml.expval(PauliZ(i)) for i in range(d)]
    -> Linear(d, 8) -> ReLU -> Linear(8, 1)      (no sigmoid, BCEWithLogitsLoss)
```

`src/scm_dataset/modeling/qgnn.py`: `QuantumCircuitLayer` (56-89),
`QGNN` (92-119), `train_qgnn` (mirrors `train_graphsage`'s optimizer/
loss/early-stopping exactly, but trains over **flat mini-batches of
(supplier, time) rows** rather than per-timestep snapshots — QGNN has no
graph dependency, so this was restructured for a ~140x speed win, see
`qgnn.py:194-` docstring). Backend: `default.qubit` (CPU, `backprop`) by
default — measured faster than `lightning.gpu`/adjoint at this qubit
count; GPU backend remains implemented and verified working.

Its input in v1 comes from `reduction.py`'s `PCASupplierReducer` /
`DomainSelectedReducer`, applied to the **raw** supplier feature frame
(before `FeaturePreprocessor`, before any GraphSAGE encoder) —
`src/scm_dataset/modeling/reduction.py:1-30` explicitly documents this:
"Operates on the SUPPLIER node's raw, pre-`FeaturePreprocessor` combined
static+dynamic frame." **This is the fairness gap QGNN-v2 exists to
close.**

## 4. Current Fairness Limitation

| | GraphSAGE-Reduced (v1) | QGNN-Reduced (v1) |
|---|---|---|
| Input | 8D of the supplier's own raw features | Same 8D of the supplier's own raw features |
| Graph access | **Yes** — full heterogeneous message passing, every other node type at full width | **No** — reads only the 8D supplier tensor, no edges, no other node type |

`ReducedGraphSnapshotBuilder` (`reduction.py:169-192`) swaps in the
reduced supplier tensor while leaving every other node type and every
`edge_index` untouched — so GraphSAGE-Reduced still benefits from
2-hop-reachable information from materials, plants, products, regions,
and procurement orders. QGNN-Reduced never had access to any of that.
`GRAPHSAGE_PHASE_A5_ROOT_CAUSE_FINDINGS.md` and the QGNN-v1 benchmark's
own resource analysis both note this asymmetry; it was not corrected in
v1 because v1's purpose was isolating dimensionality-reduction effects
from quantum-processing effects, not graph-access effects. QGNN-v2's
purpose is the reverse: hold graph access **constant** and isolate the
quantum-vs-classical question at the processing-head level.

## 5. Candidate Hybrid Architectures — Where to Tap the Graph Representation

**Tap point: `h_dict["supplier"]`, the output of `HeteroGraphSAGE.encode()`, before the classifier.**

| Property | Value | Basis |
|---|---|---|
| Shape | `[n_suppliers, hidden_dim]` = `[300, 128]` (primary config) | `graphsage.py:43,55` (`hidden_dim=128` default), confirmed structurally by `encode()`'s return |
| Contains neighbor info | Yes — after `num_layers=2` SAGEConv rounds over 18 relations (9 base + 9 reverse), incorporates up to 2-hop neighborhood information from every other node type reachable within 2 hops | `graphsage.py:61-66`, `hetero_graph.py:84-85` |
| Target-dependent | The embedding is computed by a **pure forward pass** (no label used in its computation) — but the **weights** that produced it were fit end-to-end with the classification loss, so it is task-optimized, not a task-agnostic representation. This matters for how it's shared (§below), not for leakage. | `train.py:124-139` |
| Trainable (as reused in v2) | **No, by recommendation** — treated as a frozen activation; see Option C |
| Leakage risk | None beyond what GraphSAGE-Full already carries, **provided** the checkpoint being reused was itself trained via the existing `train_graphsage` (train-only loss, validation-only early stopping, test never touched) — true of every existing GraphSAGE-Full checkpoint. Extracting an embedding for a test-period timestep uses only that timestep's own (contemporaneous) feature values across the graph — identical to how GraphSAGE-Full/Reduced already score test examples today; not a new risk. |

An alternative, **rejected** tap point: the post-classifier logit (1-dim,
already collapses all information into a single scalar — not useful as a
d-qubit input) or an intermediate SAGEConv-layer-1-only embedding (skips
half the message passing GraphSAGE-Full actually uses, making the
comparison to GraphSAGE-Full less apples-to-apples). `h_dict["supplier"]`
after the full 2-layer encoder is the only tap point that matches what
GraphSAGE-Full's own classifier actually sees.

## 6. Recommended Architecture

See §1 diagram. In prose: reuse an **existing, unmodified GraphSAGE-Full
checkpoint** as a frozen encoder; extract supplier embeddings for every
needed `(supplier, time)`; fit a PCA reducer (reusing the existing,
tested `PCASupplierReducer` pattern) on the train-split embeddings only;
apply it identically to both a classical MLP head and the existing
quantum head.

## 7. Graph Encoder Strategy

Evaluated against the plan's own four options:

| | A — Separate encoders per model | B — Same architecture, independently trained | C — One frozen encoder, two heads | D — Jointly trained shared encoder |
|---|---|---|---|---|
| Fairness | Each head shapes its own encoder end-to-end — confounds "head type" with "what the encoder happened to learn" | Same confound as A, marginally more disciplined (architecture held fixed) | **Cleanest** — both heads see byte-for-byte identical input | Encoder is shaped by both heads' gradients simultaneously — most confounded |
| Leakage risk | Baseline (same as existing `train_graphsage`) | Baseline | **Lowest** — reuses an already-audited frozen checkpoint, introduces zero new encoder-training surface | Baseline, but fragile (two very different gradient sources into one shared object) |
| Parameter-count implications | Two full encoder+head budgets | Two full encoder+head budgets | Only the small heads (§11) count as "trainable" per model — cleanest for a parameter-fair comparison | One shared large budget, hard to attribute |
| Optimization implications | Two independent full end-to-end runs; quantum head's noisier gradients could reshape its own encoder differently than the classical run's encoder | Same | Each head is a small, fast, stable optimization problem — no backprop through the 1.2M-param encoder at all once embeddings are cached | Gradient-magnitude imbalance between a classical and quantum head sharing one encoder is a known, hard multi-task-learning problem |
| Isolates quantum contribution | No — the primary confound this exercise exists to avoid | No, same reason | **Yes** — this is the entire point | No — worst of the four |
| Unfair advantage | Neither model favored, but the comparison itself is muddied | Same | None — symmetric by construction | Risk of one head's optimization dominating the shared encoder |
| Reproducibility | Fine (deterministic given seeds) | Fine | **Best** — one deterministic frozen artifact + deterministic embedding extraction + two small deterministic head-training runs | Hardest — joint optimization dynamics are the least reproducible of the four |

**Recommendation: Option C.** It is the only option that actually
achieves the fairness diagram in the task prompt (§5 of the request:
"SAME... Same 8D input... Classical head / Quantum head"), it introduces
the least new leakage surface (reuses an already-audited artifact rather
than training anything new), and it matches this project's established
philosophy from QGNN-v1 (GraphSAGE-Reduced vs QGNN-Reduced was already a
matched-input comparison — v2 just moves the matching point later in the
pipeline, from raw features to a graph embedding).

## 8. Dimensionality Reduction Strategy

| Method | Applicable to a 128-dim *learned* embedding? | Recommendation |
|---|---|---|
| A. PCA | Yes — directly reuses the existing `PCASupplierReducer` (`reduction.py`), which already fits on train-split rows only via `supplier_fit_mask` and is completely agnostic to whether its input columns are raw features or embedding dimensions | **Primary recommendation** |
| B. Learned classical projection (`Linear(128,8)`) | Yes, but if trained end-to-end per head it reintroduces exactly the Option-A/B confound from §7 at a different layer. To stay fair it would need to be trained **once**, frozen, and shared — extra implementation surface (a new, separately-leakage-audited training step) for an unproven benefit | Not recommended for the primary experiment; reasonable **future** ablation, out of scope now |
| C. Domain-selected | **Does not meaningfully apply** — the 128 embedding dimensions are opaque, rotated combinations with no individual semantic meaning (unlike the raw supplier fields v1 used); "selecting" 8 of 128 arbitrary learned dimensions has no principled basis | Not recommended |
| D. Trainable bottleneck | Same as B | Same as B |
| E. Other | — | — |

`Graph embedding → PCA → 8D`, fit on train-split embeddings only, frozen,
applied identically to train/val/test/cross-dataset-target — the same
`source train → fit → freeze → target` contract `reduction.py` and
`pipeline.py`'s `prepare_reduced_for_cross_dataset_eval` already
implement and test. No new leakage surface if this pattern is followed
exactly (it should be, since the code to do so already exists).

## 9. Quantum Circuit Strategy

The existing architecture (§3) should be **retained**, not redesigned —
it is implemented, tested (forward/backward/gradient correctness), and
already produced an informative depth-ablation finding in v1.

| Choice | Recommendation | Basis |
|---|---|---|
| Qubits | **8** | Already validated by v1's dimension sweep (monotonic improvement 4→6→8, selected on validation PR-AUC); re-sweeping is not necessary (§12) |
| Layers (primary) | **2** | v1's own depth ablation found 2 layers a real, mostly-consistent improvement over 1 (paired diff +0.104, 4/5 seeds), not just a hypothesis — start from the already-better config, don't re-litigate |
| Layers (ablation) | Optionally 3, **but see below** | The task says "do not tune layer counts beyond 1 and 2" for the *original* ablation; for v2 specifically I'd substitute a different, single, well-justified ablation (below) rather than pushing further on depth alone |
| Entanglement | Linear CNOT chain (unchanged) | Already simple, already tested; changing this is architecture search, not warranted by any current evidence |
| Measurement | PauliZ expectation per qubit (unchanged) | Same reasoning |
| **One ablation, if any**: data re-uploading | Re-apply `AngleEmbedding` between variational layers, not only at the start | Directly answers "was 1 layer too shallow because of insufficient *entanglement depth* or because the input is only presented once" — a well-established, low-parameter-cost QML technique, and a single well-scoped addition rather than an open-ended search |

## 10. Model Comparison — Is the Proposed Hierarchy Valid?

The task's suggested hierarchy is:

```
MLP -> GraphSAGE-Hybrid-Control -> Hybrid-QGNN -> GraphSAGE-Full
```

**This needs one correction before it is well-defined.** If "MLP" means
"a classical MLP on the *same* 8D graph-derived embedding," it is not a
separate model — that IS GraphSAGE-Hybrid-Control's own head. For "MLP"
to be a meaningful, distinct rung (and to actually answer RQ1), it must
mean a classical MLP with **no graph access at all**.

There is a free way to build exactly that control with **zero new code**:
`HeteroGraphSAGE(num_layers=0)` — verified directly
(`graphsage.py:45-51,61-66`): with `num_layers=0`, `self.convs` is empty,
`encode()`'s loop over `self.convs` never executes, so `h_dict` is
literally just `{nt: input_proj[nt](x)}` — a per-node-type linear
projection with **no cross-node-type mixing whatsoever**. Calling the
existing `build_model(..., num_layers=0)` on the existing 8D reduced
supplier input (reusing v1's `prepare_reduced`) already produces a valid,
zero-graph-access "MLP" baseline via a **config value change only**.

With that correction, the model set that actually disentangles the three
research questions is not a single line but three overlapping
comparisons:

| Model | Graph access | Input source | Status |
|---|---|---|---|
| GraphSAGE-NoGraph (`num_layers=0`) | None | 8D raw supplier features | New, but zero new code — config-only |
| GraphSAGE-Reduced (v1) | Full message passing | 8D raw supplier features (narrowed *before* message passing) | Existing, unchanged |
| GraphSAGE-Hybrid-Control (v2) | Full message passing | 8D PCA of the 128D *post-message-passing* embedding (narrowed *after*) | New |
| Hybrid-QGNN (v2) | Full message passing (via the frozen encoder only) | Same 8D as GraphSAGE-Hybrid-Control | New |
| GraphSAGE-Full | Full message passing | Full 128D embedding, no bottleneck | Existing, unchanged |

Note GraphSAGE-Reduced and GraphSAGE-Hybrid-Control are **not the same
model** despite both being "8D + graph": one narrows the input before
message passing (the graph mixes an already-narrow signal), the other
narrows a full-richness embedding after message passing (PCA discards
information the graph already mixed in). Which of the two performs
better is an open, measurable question, not something to assume.

**The strict linear ordering is a hypothesis about what a *successful*
hybrid design might show, not a scientific guarantee** — e.g., there is
no a priori reason GraphSAGE-Hybrid-Control must beat GraphSAGE-Reduced
(PCA is unsupervised/variance-based and is not guaranteed to preserve the
directions of an already-task-optimized embedding that matter most for
classification). Treat the ordering as something to measure, not assume.

- **RQ1** (does graph structure help?): GraphSAGE-NoGraph vs {GraphSAGE-Reduced, GraphSAGE-Hybrid-Control, GraphSAGE-Full}.
- **RQ2** (does quantum processing add value given matched graph-derived input?): GraphSAGE-Hybrid-Control vs Hybrid-QGNN — **the primary comparison**, exactly as the task specifies.
- **RQ3** (does the hybrid approach close in on the ceiling?): Hybrid-QGNN vs GraphSAGE-Full.

## 11. Parameter-Count Analysis

| Component | Parameters | Trainable in v2? |
|---|---:|---|
| Graph encoder (`input_proj` × 6 node types + 2 × 18-relation `HeteroConv`/`SAGEConv`, hidden_dim=128) | ≈1.2M (estimated: input_proj ≈29K + 2 layers × 18 relations × ≈33K/relation-layer ≈1.19M) | **No — frozen**, reused from an existing checkpoint |
| PCA bottleneck (128→8) | Not a gradient-trained parameter (fitted statistics, like a scaler) | No — fit once, frozen |
| Classical head (`Linear(8,8)→ReLU→Linear(8,1)`, matched in scale to the quantum head's own MLP) | ≈81 | Yes |
| Quantum head: circuit (8 qubits × 2 layers) | 16 | Yes |
| Quantum head: MLP (`Linear(8,8)→ReLU→Linear(8,1)`) | ≈81 | Yes |
| **Quantum head total** | **≈97** | Yes |

Both heads have a **comparably tiny** trainable-parameter budget (81 vs
97) against a shared, frozen 1.2M-parameter encoder neither can modify —
this directly avoids the "quantum model has fewer parameters, therefore
any win is meaningless / any loss is unsurprising" confound the task
explicitly warns against: neither model has a meaningfully larger
trainable budget than the other.

**Qubits**: 8. **Circuit depth**: 2 layers (primary). **Circuit
evaluations**: one per training step, same convention as v1 — but likely
**faster** than v1 overall, because embeddings need only be extracted
**once** (a handful of frozen forward passes through the encoder, no
backward pass needed) and cached; head training then never touches the
encoder or the graph again. **Memory**: cached embeddings are
`[n_examples, 128]` or `[n_examples, 8]` float32 — trivial. **Training
cost**: expected lower than v1's already-fast (~2 min/model) runs, since
the expensive graph-forward-pass step is amortized to a one-time
extraction rather than repeated every epoch.

## 12. Leakage Analysis

| Control | Status |
|---|---|
| Graph construction | Unchanged, already audited |
| Temporal features | Unchanged, already audited |
| Labels | Never touch the frozen encoder (it was trained earlier, via the already-audited `train_graphsage`, on train-split-only loss with validation-only early stopping) |
| PCA/projection fitting | Must fit on train-split embeddings only — reuses `reduction.py`'s existing `supplier_fit_mask`-based pattern, applied to embeddings instead of raw features; no new leakage-prone code path, same tested contract |
| Normalization | Same train-only philosophy throughout |
| Class weighting | Unchanged — train-split only |
| Threshold selection | Unchanged — validation-split only |
| Graph encoder training | **None needed** for the recommended design — this is the safest possible choice specifically because no new encoder training happens; the existing checkpoint's own safety was already established |
| Validation/test separation | Unchanged |

**Specifically on "could a graph encoder trained using future-period
information leak test information"**: not a risk in the recommended
design, because the encoder is not retrained at all. Embeddings are
extracted using each timestep's own (contemporaneous, not future)
feature values across the graph — identical to how GraphSAGE-Full/Reduced
already score test-period examples today, an already-accepted pattern,
not a new one. If a *fresh* encoder were trained specifically for v2
instead (not recommended), it would need to follow the exact same
existing safe procedure — no different from GraphSAGE-Full/Reduced.

## 13. Training Protocol

The task's own draft procedure assumes a **new** graph encoder is trained
as part of this experiment. Based on the actual implementation and the
Option-C recommendation, the correct procedure is different in one key
respect — no new encoder training:

1. **(Reused, not retrained)** An existing, already-frozen GraphSAGE-Full checkpoint (per model seed, from `experiments/classical_gnn/`).
2. Extract `h_dict["supplier"]` via `model.eval()` + `model.encode(...)` for every `(supplier, time)` row needed across train/validation/test.
3. Fit a PCA reducer on the **train-split subset** of those embeddings only (same `supplier_fit_mask` boundary already used elsewhere).
4. Transform all rows (train/validation/test) with the frozen, fit-once PCA.
5. Train the classical head on the 8D train-split embeddings (Adam, class weighting from train split, early stopping on validation PR-AUC) — structurally identical to `train_qgnn`'s existing flat-batch loop, since neither head needs the graph anymore once embeddings are cached.
6. Separately train the quantum head the same way, same split boundaries.
7. Evaluate both, frozen, on the test split via the existing, unmodified `_finalize_evaluation`.

**Preserved**: temporal split, 4-period horizon, seeds 42-46, train-only
preprocessing (both the original `FeaturePreprocessor` that produced the
frozen encoder's own inputs, and the new PCA-on-embeddings step),
train-only class weighting, validation-PR-AUC early stopping, identical
evaluation code, existing artifact conventions.

## 14. Cross-Dataset Protocol

Retain all four directions: `43→43`, `44→44`, `43→44`, `44→43` —
consistent with v1 and with `GRAPHSAGE_PHASE_A5_ROOT_CAUSE_FINDINGS.md`'s
established finding that this asymmetry is worth re-checking under any
new architecture variant.

**Critical rule, directly answering the task's own concern**: for a
cross-dataset direction (e.g., seed43→seed44), use the **seed43-trained,
frozen encoder** to extract embeddings for **seed44's own graph** —
i.e., run the source-trained checkpoint's `encode()` on the target
dataset's own topology and features, never train or fine-tune anything
on seed44. This is exactly the same "fit on source, apply unchanged to
target, never refit" contract `prepare_for_cross_dataset_eval` and
`prepare_reduced_for_cross_dataset_eval` already implement and test for
v1 — extended one level earlier in the pipeline (the encoder itself,
rather than only the reducer/preprocessor). The PCA reducer must
similarly be fit only on the source's train-split embeddings and applied
to the target's embeddings unchanged. No target-world information
influences the source-trained representation at any stage, by
construction — no new mechanism needs to be invented, only reused.

## 15. Proposed Experiment Matrix

Minimum scientifically defensible set:

| Model | Purpose | Seeds |
|---|---|---|
| GraphSAGE-Full | Existing, ceiling reference — **not rerun** | (existing) |
| GraphSAGE-NoGraph (`num_layers=0`) | RQ1 control — cheap (config-only), recommended to include | 42-46 |
| GraphSAGE-Reduced | Existing, secondary reference — **not rerun** | (existing) |
| GraphSAGE-Hybrid-Control | RQ2 classical control — **new, primary** | 42-46 |
| Hybrid-QGNN (2 layers) | RQ2 quantum model — **new, primary** | 42-46 |
| Hybrid-QGNN (2 layers + data re-uploading) | The one justified ablation | 42-46 |

Plus cross-dataset (both directions) for GraphSAGE-Hybrid-Control and
Hybrid-QGNN (primary configuration only, not the ablation, to avoid
experiment explosion).

**Is a fresh 4/6/8-qubit sweep necessary?** No. v1 already established a
monotonic, validation-selected preference for 8 qubits with the same
general circuit family, and there is no specific reason to expect the
qubit-count relationship to invert merely because the input source
changed from raw features to a PCA'd graph embedding. Re-sweeping would
be exactly the "unnecessary experiment explosion" the task warns against.
Flagged as a low-priority optional sanity check for later, not a
requirement now.

## 16. Success/Failure Interpretation

**Outcome A — Strong result**: Hybrid-QGNN ≈ GraphSAGE-Hybrid-Control
(e.g., within roughly one pooled standard deviation), possibly
approaching GraphSAGE-Full. Scientific meaning: once given an
information-matched, graph-derived representation, this quantum
architecture can match classical processing of the same input — a
genuinely interesting result for QGNN feasibility on this task, though
still not evidence of quantum *advantage* (matching, not beating; and
simulator-based).

**Outcome B — Interesting result**: Hybrid-QGNN clearly improves over
v1's feature-only QGNN-Reduced (0.284-0.388 PR-AUC range) but remains
clearly below GraphSAGE-Hybrid-Control. Scientific meaning: lack of graph
access *was* a real handicap in v1 and fixing it helps — but the
remaining classical-vs-quantum gap is not primarily an information-access
problem; it narrows the explanation toward the quantum processing/head
capacity itself.

**Outcome C — Negative result**: Hybrid-QGNN remains substantially below
GraphSAGE-Hybrid-Control, similar in magnitude to v1's gap. Scientific
meaning: even with matched, graph-derived information, this specific
shallow quantum architecture adds no value — this *strengthens* (not
weakens) v1's conclusion by ruling out "no graph access" as the
explanation, and is exactly as valid and reportable a result as A or B,
consistent with this project's standing negative-results policy.

## 17. Exact Source Files That Would Need Modification (future implementation, not now)

None of the following would be **modified** — all new functionality
belongs in new, additive files, mirroring how every prior phase of this
project (D2 cross-dataset, reduction.py, qgnn.py) added new modules
rather than editing existing ones:

- **New**: something like `src/scm_dataset/modeling/graph_embedding_reduction.py` (embedding extraction + PCA-on-embeddings glue — the fitting logic reuses `reduction.py`'s existing `supplier_fit_mask` pattern, applied to a different input).
- **New**: `scripts/run_qgnn_v2_experiment.py`, `scripts/run_qgnn_v2_crossdataset_experiment.py` (mirroring the existing v1 scripts' structure).
- **New**: `configs/qgnn_v2.yaml`.
- **Reused, unmodified**: `src/scm_dataset/modeling/qgnn.py` (`QGNN`, `QuantumCircuitLayer`, `train_qgnn`, `evaluate_qgnn` — all already input-agnostic).
- **Reused, unmodified**: `src/scm_dataset/modeling/graphsage.py` (`HeteroGraphSAGE.encode()` already does exactly what's needed; `num_layers=0` already gives the RQ1 control for free).

## 18. Files That MUST NOT Be Modified

- `src/scm_dataset/modeling/graphsage.py`, `train.py`, `evaluate.py`, `pipeline.py`, `features.py`, `preprocessing.py`, `hetero_graph.py`, `config.py`, `experiment.py` — the entire existing GraphSAGE/QGNN-v1 shared infrastructure.
- `src/scm_dataset/modeling/qgnn.py`, `reduction.py` — reused as-is.
- `configs/graphsage.yaml`, `configs/graphsage_severity.yaml`, `configs/graphsage_scenario.yaml`, `configs/qgnn.yaml` — existing configs.
- Anything under `data/`, `experiments/classical_gnn/`, `experiments/qgnn/` — existing artifacts and datasets.
- `GRAPHSAGE_WORK_SUMMARY.md`, `GRAPHSAGE_PHASE_A_GENERALIZATION_FINDINGS.md`, `GRAPHSAGE_PHASE_A5_ROOT_CAUSE_FINDINGS.md`, `QGNN_IMPLEMENTATION_AND_BENCHMARK.md`, `QGNN_FINAL_BENCHMARK.md` — frozen research records.

## 19. Implementation Risks

- **PCA-of-a-learned-embedding risk**: PCA is unsupervised/variance-based; a supervised embedding's directions of highest *variance* are not guaranteed to align with directions of highest *classification relevance*. GraphSAGE-Hybrid-Control could plausibly underperform expectations for this reason alone, independent of anything about the quantum head — worth watching for when results come in, not assuming away.
- The v2 experiment's ceiling is tied to whichever existing GraphSAGE-Full checkpoint (per seed) is reused — any quirk in an existing checkpoint's training carries over unchanged.
- Angle-encoding a PCA-of-embedding input may need different effective scaling than the raw-feature case; the existing `tanh(x)*π` mapping is robustly bounded regardless of input scale, but should be sanity-checked (a cheap smoke test) once implemented, not assumed identical to v1's behavior.
- Data re-uploading (the recommended single ablation) is a genuine code change to the circuit construction — small and well-scoped, but real, and deferred entirely to implementation time.
- Loading old `model.pt` checkpoints assumes the `HeteroGraphSAGE` architecture (`in_dims`, `edge_types`, `hidden_dim`) is reconstructed identically before `load_state_dict` — needs the same config that produced the checkpoint, not verified here (analysis-only), should be a first smoke-test step at implementation time.

## 20. Recommended Next Step

Do not implement yet (per this task's own instruction). When implementation
is authorized: build the embedding-extraction + PCA glue as a small,
isolated, additively-tested module first (mirroring `reduction.py`'s own
test-first pattern), smoke-test loading one existing checkpoint and
extracting embeddings for a handful of timesteps, verify shapes and
leakage boundaries with unit tests before touching any multi-seed
training loop — the same incremental discipline every prior phase of
this project has used.

---

## IMPLEMENTATION DECISION

- **Should QGNN-v2 be implemented?** Yes — it closes a real, previously-identified fairness gap (§4) with a design that reuses existing, already-audited infrastructure almost entirely.
- **Exact architecture**: frozen GraphSAGE-Full encoder → `h_dict["supplier"]` (128D) → PCA (train-fit, frozen) → 8D → {classical MLP head | quantum head}.
- **Exact input dimension**: 8.
- **Exact qubit count**: 8.
- **Exact circuit depth**: 2 variational layers (primary); +data re-uploading as the one ablation.
- **Graph encoder**: **frozen** — reuse an existing GraphSAGE-Full checkpoint per seed; do not retrain.
- **PCA or learned projection**: **PCA**, fit on train-split embeddings only, frozen, shared identically by both heads.
- **Exact classical control**: GraphSAGE-Hybrid-Control = frozen-encoder embedding → PCA(8) → `Linear(8,8)→ReLU→Linear(8,1)`.
- **Exact QGNN control**: Hybrid-QGNN = same frozen-encoder embedding → PCA(8) → existing `qgnn.py` circuit (8 qubits, 2 layers) → same MLP head shape.
- **Exact experiments to run**: GraphSAGE-NoGraph (`num_layers=0`, cheap RQ1 control), GraphSAGE-Hybrid-Control, Hybrid-QGNN (2-layer primary), Hybrid-QGNN (2-layer + data re-uploading, the one ablation) — 5 seeds each, plus both-direction cross-dataset evaluation for the two primary new models. No 4/6/8-qubit re-sweep.
- **What remains unchanged**: GraphSAGE-Full, GraphSAGE-Reduced, and every existing artifact/config/test — none rerun, none modified.
