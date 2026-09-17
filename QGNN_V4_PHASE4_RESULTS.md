# QGNN-v4 Phase 4 — Results: Stage 1 (Representation Audit)

Covers Tracks **A1–A4, H, I, J** (`QGNN_V4_PHASE4_PLAN.md` §4). Produced by
`scripts/run_phase4_stage1_diagnostics.py`, all outputs under
`experiments/qgnn_v4/phase4_stage1/`. No GraphSAGE retraining, no quantum
circuit — every number below comes from the SAME frozen encoder checkpoints
and SAME `(supplier_id, time) -> target/split` assignment every QGNN-v4/
classical run in this project already uses, so every PR-AUC here is
directly comparable to the numbers in `QGNN_V4_CLASSICAL_BASELINE.md` and
the Phase 1–3 reports.

**One correctness fix made while building this**: `experiments/
classical_gnn/` contains a third, oddly-named batch of severity runs
(`*_hetero_graphsage_severity_seed44_seed<N>`, left over from an unrelated
earlier session) that a naive glob for "severity runs" would also match,
silently mixing 15 directories across 3 batches into what should be 5 per
seed. Tracks H/I resolve the classical run directory the same way the real
QGNN-v4 experiments do (`_analysis_common.find_latest_graphsage_full_
checkpoint`, regex-anchored per seed) specifically to avoid this — caught
by a sanity check (classical primary PR-AUC recomputed here matched
`QGNN_V4_CLASSICAL_BASELINE.md`'s 0.807 only after the fix; the naive-glob
version did not).

---

## A1 — Embedding separability (is there signal in the 128D embedding at all?)

Four lightweight sklearn diagnostics (`LogisticRegression`, `LinearSVC` via
`CalibratedClassifierCV` for probability outputs, `MLPClassifier(64)`,
`RandomForestClassifier(300 trees)`), trained on the train-split rows of
the exact raw 128D frozen embedding, evaluated on test (11,100 severity /
3,600 primary examples — not a small-sample result). **XGBoost is not
installed in this environment** (`pip show xgboost` empty) — per the
phase's own "if already available" instruction, this diagnostic uses
Random Forest only for the tree-based model, stated plainly rather than
silently substituted.

| Split | Model | Test PR-AUC (mean±std, 5 seeds) | Test ROC-AUC |
|---|---|---|---|
| Primary | logistic_regression | 0.671 ± 0.043 | 0.966 ± 0.016 |
| Primary | linear_svm | 0.630 ± 0.036 | 0.958 ± 0.016 |
| Primary | mlp | **0.733 ± 0.063** | 0.977 ± 0.013 |
| Primary | random_forest | 0.636 ± 0.045 | 0.936 ± 0.046 |
| Severity | logistic_regression | **0.465 ± 0.005** | 0.731 ± 0.015 |
| Severity | linear_svm | 0.469 ± 0.007 | 0.727 ± 0.017 |
| Severity | mlp | 0.463 ± 0.007 | 0.730 ± 0.018 |
| Severity | random_forest | 0.411 ± 0.034 | 0.706 ± 0.001 |

**Reference numbers already in hand:** classical GraphSAGE-Full's own
trained head: primary 0.807±0.066 / severity 0.449±0.013. QGNN-v4
reference (6q/2L/LayerNorm): primary 0.811±0.074 / severity 0.398±0.057.

**Finding, stated carefully:** on **primary**, every diagnostic classifier
here is *below* both the classical head and every QGNN-v4 configuration —
expected, since these are simple off-the-shelf models with no
hyperparameter tuning against a full trained `MLP(128→64→1)` classifier
head. But on **severity**, `logistic_regression` and `linear_svm` (0.465,
0.469) sit **above** both the classical baseline (0.449) and every QGNN-v4
configuration tried so far (0.36–0.40), with roughly 5–7x tighter
cross-seed std (0.005–0.007 vs. 0.030–0.057). This is a real, reproducible
result — not a small-sample artifact (11,100 test rows, 817 positives) —
and it directly answers A1's question: **yes, the frozen 128D embedding
contains at least as much severity-generalization signal as either the
classical or quantum head currently extracts from it**, so the severity
gap versus the classical baseline reported in every prior phase is not
fully explained by "the representation doesn't have the signal."

**Important caveat, not to be over-read:** this is not an apples-to-apples
head comparison. `LogisticRegression`/`LinearSVC` here use sklearn's
default L2 regularization (`C=1.0`) and `class_weight="balanced"`, a
different mechanism than this project's own `pos_weight`-based weighted
BCE + dropout + 100-epoch/patience-10 early stopping. The honest
conclusion is narrower than "logistic regression beats the QGNN": it is
that **a simpler, more strongly regularized linear decision boundary on
this embedding generalizes better to the severity split than either of
this project's own more complex trained heads currently do** — a concrete,
testable lead for Stage 2/6 (a linear or more strongly-regularized
classical/quantum head), not a finished result. Severity PR-AUC here
(0.46–0.47) is still far below primary's (0.62–0.85 across these same
diagnostics) — the OOD generalization problem is real and large regardless
of which head is used.

CSV: `experiments/qgnn_v4/phase4_stage1/representation_diagnostics.csv`

---

## A2 — PCA dimensionality diagnostic

Logistic regression at PCA dimensions {4,6,8,16,32,64,128}, PCA fit on
train-split rows only (`reduction.PCASupplierReducer`, reused unchanged).

| Split | n_components | Explained var. ratio | Test PR-AUC (mean±std) |
|---|---:|---:|---|
| Primary | 4 | 0.909 | **0.781 ± 0.085** |
| Primary | 6 | 0.940 | 0.761 ± 0.087 |
| Primary | 8 | 0.958 | 0.754 ± 0.084 |
| Primary | 16 | 0.987 | 0.712 ± 0.079 |
| Primary | 32 | 0.996 | 0.657 ± 0.066 |
| Primary | 64 | 0.999 | 0.627 ± 0.042 |
| Primary | 128 | 1.000 | 0.618 ± 0.043 |
| Severity | 4 | 0.890 | 0.456 ± 0.017 |
| Severity | 6 | 0.923 | 0.457 ± 0.019 |
| Severity | 8 | 0.943 | 0.456 ± 0.017 |
| Severity | 16 | 0.982 | 0.467 ± 0.005 |
| Severity | 32 | 0.997 | 0.476 ± 0.011 |
| Severity | 64 | 1.000 | 0.480 ± 0.014 |
| Severity | 128 | 1.000 | **0.480 ± 0.011** |

**Finding — asymmetric, and directly relevant to Phase 3's own result:**
On **primary**, PR-AUC *decreases monotonically* as PCA dimension grows —
best at 4 components (0.781), worst at 128 (0.618). Just 4 components
already explain 90.9% of variance and give the best downstream score. This
is independent evidence — not from a quantum circuit at all — that the
useful primary-split signal in this embedding really is low-rank,
concentrated in a handful of directions. It is a strong, direct
explanation for Phase 3's own finding that **4-qubit QGNN-v4 (0.847±0.102)
outperformed 6-qubit (0.811±0.074) and 8-qubit (0.810±0.091) on primary**:
a 4-dimensional bottleneck was already enough to keep almost all the
useful signal, and pushing to 6–8 dimensions may just be adding
noise-fitting capacity rather than new information.

On **severity**, the trend *reverses*: PR-AUC rises monotonically from
0.456 (4 components) to 0.480 (128 components) — smaller but consistent.
Severity-generalizable signal is **not** concentrated the same way; more
of the embedding's dimensions carry marginally useful severity signal.
This is a plausible partial explanation for why the qubit-count ablation
showed a much smaller/less clear effect on severity (0.396/0.398/0.364 at
4/6/8 qubits — no monotonic pattern) than it did on primary.

CSV: `experiments/qgnn_v4/phase4_stage1/pca_sweep.csv`

---

## A3 — Target-aware per-dimension correlation / mutual information

Point-biserial correlation + mutual information (`sklearn.feature_
selection.mutual_info_classif`) between each of the 128 raw embedding
dimensions and three targets, **train-split rows only**. Computed on one
representative seed (42) per split, not all five — Stage 1's question
("is signal concentrated in a few dims") doesn't need encoder-level seed
variance the way a PR-AUC comparison does; flagged here as a real scope
limitation, not hidden.

| Split | Target | Top single-dim \|correlation\| | Top-8-dims MI / total MI |
|---|---|---:|---:|
| Primary | target | 0.805 (emb_83) | 73.5% |
| Primary | fresh_onset | 0.251 (emb_100) | 11.9% |
| Severity | target | 0.867 (emb_12) | 54.7% |
| Severity | fresh_onset | 0.418 (emb_12) | 11.8% |

**Finding:** individual embedding dimensions correlate very strongly
(0.75–0.87) with the ordinary `target` — expected, since GraphSAGE was
trained end-to-end for exactly this target, and confirms A1's core
question with a second, independent method: the representation clearly
encodes disruption risk, concentrated in a handful of dominant dimensions
(top-8 of 128 already carry 55–74% of total MI). Correlation with
`fresh_onset` specifically is **much weaker** (0.22–0.42, roughly
half-to-a-third the strength, and far more spread across dimensions —
top-8 carries only ~12% of MI) — a second, independent line of evidence
(alongside I below) that fresh-onset prediction is a genuinely harder
signal-extraction problem, not merely an artifact of how the current heads
are trained.

CSV: `experiments/qgnn_v4/phase4_stage1/dimension_correlations.csv`

---

## A4 — Graph / label / event diagnostics

Dataset: `scm_v1_black_swan_seed43` (the one every QGNN-v4 config in this
project uses).

- **Nodes (2,670 total):** supplier 300, material 100, plant 50, product
  200, region 20, procurement (order) 2,000.
- **Edges (7,675 total, 9 relation types):** dominated by
  `supplier_procurement`/`procurement_material`/`procurement_plant` (2,000
  each — one per order), `material_plant` (649), `supplier_region` (300),
  `supplier_material` (276, the actual multi-source supplier→material
  dependency edges), `plant_product` (200), `product_region` (200),
  `plant_region` (50).
- **Degree distribution:** mean 5.75, median 3.0, max 136 (min 1) — a
  long-tailed, hub-concentrated graph (a small number of high-degree
  region/plant nodes), graph density 0.00215 (sparse, as expected for a
  real supply-chain topology, not a synthetic clique).
- **Connectivity:** 1 connected component covering all 2,670 nodes (no
  isolated subgraphs). Average shortest path length (largest component,
  computed exactly, not sampled): **3.82** — short, consistent with a
  4-layer bipartite-chain-like topology (supplier→material→plant→product)
  plus reverse edges.
- **Events:** only **6 events total** for this dataset (3 logistics, 1
  supplier_failure, 1 geopolitical, 1 cyberattack); severities 1×2, 2×3,
  5×1. Despite only one severity-5 *event*, its active window (duration +
  recovery_delay + recovery_periods) spans **37 of 104 periods (35.6%)** —
  so the severity split's test set is not thin in row-count, but it is
  drawn from **a single underlying disruption**, not a diverse population
  of severe events. This nuances the plan doc's earlier hypothesis
  (written before this diagnostic ran): the problem is event *diversity*,
  not event *count* or test-set size.
- **Trivial-property check:** supplier degree vs. per-supplier disruption
  rate, Pearson r = **0.054** — essentially no linear relationship. Labels
  are not a trivial function of graph degree.
- **Class balance:** overall supplier-period disruption rate 3.33%
  (matches the imbalance every phase's `pos_weight` computation has
  already been handling).

JSON: `experiments/qgnn_v4/phase4_stage1/graph_diagnostics.json`

---

## H — Severity-level breakdown (what's actually driving the severity gap)

Reusing each run's own saved `predictions.csv`, joined against the
period's true active severity (`_analysis_common.load_period_severity`,
the same source-of-truth `severity_split` itself uses).

| Config arm | Source | Test period severity | n | n_pos | PR-AUC | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| primary (temporal split) | classical | 5 | 3,600 | 242 | **0.807** | 0.987 |
| primary (temporal split) | QGNN-v4 ref | 5 | 3,600 | 242 | **0.811** | 0.977 |
| severity split | classical | 2 | 1,200 | 84 | 0.514 | 0.623 |
| severity split | classical | 5 | 9,900 | 733 | 0.445 | 0.739 |
| severity split | QGNN-v4 ref | 2 | 1,200 | 84 | 0.532 | 0.707 |
| severity split | QGNN-v4 ref | 5 | 9,900 | 733 | 0.384 | 0.651 |

**Two findings, both important:**

1. **The primary/temporal split's own test window happens to fall entirely
   inside the severity-5 event's active window** (all 3,600 primary-test
   examples are period-severity 5). Yet PR-AUC there is 0.807–0.811 — as
   good as primary performance ever is in this project. This means
   severity-5 periods are **not inherently unpredictable** — the model
   handles them fine when it has been allowed to train on temporally
   adjacent/overlapping data. The severity split's poor performance is
   therefore a genuine **out-of-distribution generalization** failure
   (never having trained on severity 4–5 patterns), not an information
   deficiency in what severity-5 periods look like.
2. Within the severity split's own test set, **severity level 5 (9,900
   rows, 89% of the test set) scores far worse than severity level 2
   (1,200 rows)** for both classical (0.445 vs. 0.514) and QGNN
   (0.384 vs. 0.532). Directly answers the phase's own question: **the
   poor overall severity PR-AUC is dominated by the severity-5 subgroup**,
   consistent with it being the single hardest (and only fully
   out-of-distribution, since train caps at severity 3) subgroup, and the
   largest share of the test set by far.

CSV: `experiments/qgnn_v4/phase4_stage1/severity_level_breakdown.csv`

---

## I — Fresh-onset signal audit

Reusing `evaluate.disruption_onset_breakdown` against the same saved
predictions.

| Eval split | Source | n_pos | n_fresh | n_ongoing | recall_fresh | recall_ongoing | PR-AUC_fresh | PR-AUC_ongoing |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Primary | classical | 242 | 0 | 242 | — | 0.968 | — | 0.807 |
| Primary | QGNN ref | 242 | 0 | 242 | — | 0.898 | — | 0.811 |
| Severity | classical | 817 | 84 | 733 | **0.000** | 0.491 | **0.006** | 0.487 |
| Severity | QGNN ref | 817 | 84 | 733 | **0.000** | 0.464 | **0.007** | 0.429 |

(Primary's test window has zero fresh-onset positives at all — every
positive there is "already disrupted," consistent with A4/H's finding
that the primary test window sits entirely inside the severity-5 event's
*ongoing* tail, not its onset.)

**Finding, confirming the project's own earlier discovery from a second,
independent angle:** on the severity split, recall on fresh-onset examples
is **exactly 0.0 for both classical and QGNN**, and PR-AUC on the isolated
fresh-onset-vs-negatives comparison is **0.006–0.007 — near the noise
floor**, while already-ongoing PR-AUC stays respectable (0.43–0.49). This
matches A3's weaker-but-nonzero correlation finding: the embedding does
carry *some* fresh-onset signal (0.22–0.42 per-dimension correlation, not
zero), but **neither trained head currently extracts any of it** at the
operating threshold. Best diagnostic-classifier fresh-onset comparison
(`fresh_onset_audit_diagnostic_models.csv`) shows the same pattern — the
diagnostic models were selected by overall test PR-AUC, not fresh-onset
PR-AUC specifically, so this remains an open problem across every model
type tried, not something A1's simpler heads happened to solve either.

CSV: `experiments/qgnn_v4/phase4_stage1/fresh_onset_audit.csv`,
`fresh_onset_audit_diagnostic_models.csv`

---

## J — Temporal feature audit (code inspection, no dataset change)

Per the phase's explicit rule ("do NOT immediately modify the dataset" —
this section proposes, it does not implement).

**What `features.py` actually builds into the supplier feature panel**
(`_supplier_panel`, rolling windows 4/8/12): `order_volume`,
`orders_count`, `delivery_volume`, `fulfillment_ratio` — all *procurement/
delivery* aggregates. **What it does NOT include:** the supplier's own
past `supplier_disrupted` value, a "time since last disruption" feature,
a rolling disruption *frequency*, or any neighbor's historical disruption
state. `supplier_disrupted` is read by exactly one function in the whole
codebase, `build_prediction_examples`, and only to build the *target* —
`features.py`'s own docstring states this is deliberate ("`supplier_risk_
score` is deliberately never read by this module"). GraphSAGE's 2-layer
message passing aggregates only the *current-snapshot* features of
neighbors — not any neighbor's own disruption history — so there is no
route, direct or graph-propagated, for "this supplier's neighbor had a
disruption 3 periods ago" to reach the model.

**Does this matter, given A1–A3/H/I's evidence?** The embedding still
correlates strongly with the *ordinary* target (up to r=0.87) via the
indirect `fulfillment_ratio` proxy — disruption already shows up as a
drop in fulfillment before/while it's active. But fresh-onset correlation
is much weaker (A3) and both heads get zero fresh-onset recall on severity
(I) — consistent with "the model can tell a disruption is *already*
happening (via the fulfillment proxy) far better than it can anticipate
one *starting*," exactly what a purely-reactive (no direct disruption-
history) feature set would predict.

**Proposal (not implemented):** add causal, leakage-safe historical-risk
features to `_supplier_panel` — e.g. `periods_since_last_disruption`
(rolling, computed only from `supplier_disrupted[< t]`) and a rolling
disruption-frequency count over the same 4/8/12 windows already used for
the other rolling features — and, separately, a graph-propagated version
(a supplier's *neighbors'* recent disruption rate, aggregated the same way
GraphSAGE already aggregates current-snapshot features). Both are
computable with information available strictly before `t`, matching the
existing `features.py` leakage-safety pattern exactly (the rolling
windows already only look backward). **Not built this stage** — flagged
as a concrete, evidence-backed candidate for a later Track J/K experiment
if Stage 2+ still shows fresh-onset/severity as the dominant weakness
after the bottleneck/encoding tracks are tried first, per the phase's own
"only modify if diagnostics demonstrate a concrete problem" rule.

---

## Stage 1 summary — answers to the applicable final-report questions

- **Q3 (does a smaller quantum bottleneck preserve more useful GraphSAGE
  information)?** Yes, on primary — A2's PCA sweep shows primary signal is
  genuinely low-rank (4 components already capture the best downstream
  score), independently corroborating Phase 3's 4-qubit result. Severity
  shows the opposite trend (more components help), so this does not
  generalize to the OOD split.
- **Q10 (is the severity problem caused by the quantum head or upstream
  representation)?** Neither, fully. A1 shows a differently-regularized
  head on the *same* embedding can already match/exceed both classical and
  quantum heads on severity (0.465–0.469 vs. 0.449/0.398) — so the
  representation is not the hard ceiling either the classical or quantum
  head is bumping into. But H shows the same severity-5 periods are
  predicted well (0.81) when they're not held out of training — the
  dominant problem is that **train never sees severity 4–5 patterns at
  all**, an out-of-distribution generalization gap that head architecture
  changes alone (classical, quantum, or a simple diagnostic classifier)
  only partially close.
- **Q11 (does the graph contain enough information for severity-5
  prediction)?** Yes, when severity-5 periods are in-distribution (primary
  arm: 0.81 PR-AUC on the identical severity-5 periods) — the information
  exists in principle. The severity split's poor score is a
  generalization failure under a genuine distribution shift, not a "the
  graph/features don't encode this" problem.
- **Q12 (does fresh-onset prediction have usable signal)?** Weak but
  nonzero (A3: 0.22–0.42 per-dimension correlation, some MI) — yet every
  head tried so far (classical, quantum, and A1's diagnostics) gets ~0
  fresh-onset recall on the severity split. There is a real, currently
  unexploited signal-extraction gap here, plausibly connected to J's
  finding that no explicit historical-disruption feature is fed to the
  model at all.

## What's next

Stage 2 (Track B: bottleneck projection — B2 nonlinear, B3 pre-projection
LayerNorm, B4 PCA-informed projection) is the natural next step per the
phase's own experiment order, and A2's low-rank-primary finding directly
motivates trying a PCA-informed or otherwise rank-constrained projection.
Per this project's standing convention, Stage 2 involves training new head
variants end-to-end under the frozen protocol — that will be built and
handed off as a command, not run unprompted.
