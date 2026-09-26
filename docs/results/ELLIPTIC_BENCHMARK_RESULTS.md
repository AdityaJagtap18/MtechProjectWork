# Elliptic++ Benchmark: Validation Study Results

This is a validation study, run to check whether this project's classical
and quantum models behave sanely on a known, independently-published, real
graph benchmark — separate from and not a replacement for the SCM
supplier-disruption benchmark, which remains this project's primary
contribution. Full rationale, data provenance and implementation notes are
in `scripts/run_elliptic_benchmark.py`'s module docstring.

**Headline (v2, current):** classical GraphSAGE-Full test PR-AUC 0.446 ±
0.038; hybrid QGNN 0.405 ± 0.072 — a 9% relative gap, down from 26% in the
first pass (v1). Two diagnosed, documented representation fixes (not blind
tuning — see "v1 → v2" below) did almost all of that work. Full per-seed
numbers: `data/generated/elliptic_benchmark/results_per_seed.csv`
(gitignored, regenerate with the commands at the bottom of this file).

**Data:** Elliptic++ transactions (Elmougy & Liu, KDD 2023) — the same
203,769 Bitcoin transactions, and identical illicit(4,545)/licit(42,019)/
unknown(157,205) label counts, as the original Weber et al. (2019) Elliptic
dataset, with a richer 182-feature set (vs. the original 165). 965/203,769
rows (0.47%) had missing values in 17 Elliptic++-added on-chain-lookup
columns, imputed with 0. Edges symmetrized (reverse edges added).

**Split:** chronological, `train` = time steps 1–29 (26,381 labeled rows),
`val` = 30–34 (3,513 rows), `test` = 35–49 (16,670 rows) — the field-standard
Weber/EvolveGCN train/test boundary (step 35), with a validation slice
carved from the tail of train per this project's own "never tune the
threshold on test" rule. Threshold selected by F1-optimal policy on `val`
only, applied unchanged to `test`.

**Models**, all trained for 5 seeds (42–46):

| Model | Architecture | Trainable params |
|---|---|---|
| Classical GraphSAGE-Full | `HeteroGraphSAGE`, 2-layer mean-aggregation SAGEConv, hidden_dim=128, full-batch, end-to-end | full model |
| Standalone QGNN | raw 182 features → PCA(6) (train-fit only, 36.5% variance retained) → 6-qubit, 1-layer hand-rolled RY+CNOT-chain circuit → MLP(8) — **no graph structure used at all** | 71 |
| Hybrid QGNN | **frozen** GraphSAGE-Full embedding (128-dim) → `Linear(128,6)` → 6-qubit, 2-layer `StronglyEntanglingLayers` → `LayerNorm(6, no affine)` → `Linear(6,1)` — this project's own "QGNN-v4 standing reference" architecture | 817 (36 quantum) |

## Results (test split, mean ± std over 5 seeds)

### v1: first pass

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **Classical GraphSAGE-Full** | **0.388 ± 0.084** | **0.815 ± 0.055** | **0.398 ± 0.096** | 0.338 ± 0.080 | 0.486 ± 0.123 |
| Hybrid QGNN | 0.288 ± 0.085 | 0.765 ± 0.034 | 0.345 ± 0.086 | 0.295 ± 0.136 | 0.465 ± 0.060 |
| Standalone QGNN | 0.083 ± 0.029 | 0.596 ± 0.107 | 0.151 ± 0.030 | 0.090 ± 0.024 | 0.628 ± 0.277 |

**Classical GraphSAGE-Full: a seed-variance finding.** Per-seed test PR-AUC:
seed42=0.314, **seed43=0.493**, **seed44=0.429**, seed45=0.292,
**seed46=0.410**. Two seeds (42, 45) early-stopped at epoch 4–5 (val PR-AUC
≈ 0.62–0.64) while the other three trained to ~50 epochs (val PR-AUC ≈
0.83–0.87) before early-stopping. `patience=10` was apparently too tight for
this graph/split combination — it could lock in a poor local optimum within
the first several epochs on an unlucky seed.

### v2: after three diagnosed fixes

Rather than tune parameters blindly, each model's actual representation was
inspected empirically first, and each fix targets a specific, measured
failure mode (all three changes and their diagnostics are in
`scripts/run_elliptic_benchmark.py`'s inline comments, not just this doc):

1. **Classical GraphSAGE-Full: `patience` 10→20, epoch cap 60→90.** Directly
   addresses the v1 seed-variance finding above — an unfair, under-converged
   baseline makes any quantum-vs-classical comparison meaningless regardless
   of which direction it points.
2. **Standalone QGNN: `PCA(whiten=True)`.** Measured directly: raw PCA
   scores (182 raw features → 6 components) are badly variance-skewed (PC0
   std≈4.6 vs. PC5 std≈2.5). Angle-encoding via `tanh(z)·π` saturates
   (`|tanh(z)|>0.95`) on 25–78% of each component's values — most of the
   circuit's rotation range was collapsing to a near-binary ±π signal
   instead of a continuous one. Whitening (unit variance per component)
   drops saturation to 2–8.5% across all six components.
3. **Hybrid QGNN: `pre_projection_norm=True`** (a `LayerNorm(128)` already
   built into `HybridQuantumHeadLayerNorm`, just not previously enabled).
   Measured directly: the frozen GraphSAGE embedding's per-dimension std
   ranges from 0.002 to 9.45 on the train split — a ~4,700× spread — before
   its `Linear(128,6)` reduction. Without normalizing first, that single
   linear layer has to learn to correct for the scale imbalance on top of
   learning a useful projection; LayerNorm does the scale-correction
   directly so the reduction layer's job is just "find a good subspace."

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **Classical GraphSAGE-Full** | **0.446 ± 0.038** | **0.851 ± 0.007** | **0.485 ± 0.038** | 0.447 ± 0.081 | 0.545 ± 0.046 |
| Hybrid QGNN | 0.405 ± 0.072 | 0.838 ± 0.011 | 0.473 ± 0.034 | 0.474 ± 0.044 | 0.476 ± 0.044 |
| Standalone QGNN | 0.121 ± 0.034 | 0.714 ± 0.072 | 0.189 ± 0.055 | 0.132 ± 0.035 | 0.549 ± 0.368 |

### v1 → v2, side by side

| Model | Metric | v1 | v2 | Change |
|---|---|---|---|---|
| Classical | PR-AUC | 0.388 ± 0.084 | 0.446 ± 0.038 | +15% mean, std **less than half** |
| Classical | F1 | 0.398 ± 0.096 | 0.485 ± 0.038 | +22% mean, std less than half |
| Hybrid QGNN | PR-AUC | 0.288 ± 0.085 | 0.405 ± 0.072 | **+41%** |
| Hybrid QGNN | F1 | 0.345 ± 0.086 | 0.473 ± 0.034 | **+37%**, std less than half |
| Hybrid QGNN | ROC-AUC | 0.765 ± 0.034 | 0.838 ± 0.011 | +7 points, std one-third |
| Standalone QGNN | PR-AUC | 0.083 ± 0.029 | 0.121 ± 0.034 | +46% |
| Standalone QGNN | ROC-AUC | 0.596 ± 0.107 | 0.714 ± 0.072 | barely-above-random → solidly discriminative |
| **Classical vs. Hybrid gap (PR-AUC)** | | 0.388 vs 0.288 (**26% relative gap**) | 0.446 vs 0.405 (**9% relative gap**) | gap cut by roughly two-thirds |

The classical-vs-hybrid gap did not close because classical got worse — it
closed mainly because hybrid QGNN improved faster (+41% vs. classical's
+15%) once its input scale problem was fixed. Not every seed improved
uniformly for standalone QGNN specifically (seed43 dropped slightly,
0.129→0.111, while seed42/44/45 improved substantially) — reported here
rather than smoothed over, consistent with how v1's results were reported.

## Comparison with published numbers

| Source | Model | F1 | Notes |
|---|---|---|---|
| Weber et al. 2019 | GCN | 0.70 | Original 165-feature set; widely cited |
| Weber et al. 2019 / EvolveGCN | EvolveGCN | 0.77 | Later reported figure |
| Elmougy & Liu 2023 (KDD, Elliptic++) | Logistic Regression | ≈0.45 | Same underlying transactions as this study |
| Elmougy & Liu 2023 | RF+MLP+XGB ensemble | 0.826 | Precision 0.962, recall 0.723 |
| 2026 strict-temporal re-evaluation (arXiv 2604.19514) | GCN | 0.26 | Same chronological-split philosophy as this study |
| 2026 strict-temporal re-evaluation | GraphSAGE | 0.28 | Directly comparable architecture family to our classical model |
| 2026 strict-temporal re-evaluation | GAT | 0.33 | |
| 2026 strict-temporal re-evaluation | RF / XGBoost | 0.72 / 0.71 | Non-graph tree models beat GNNs under strict eval |
| **This study (v2)** | **Classical GraphSAGE-Full** | **0.485 ± 0.038** | Full-batch, symmetrized edges, Elliptic++ 182-feature set |
| **This study (v2)** | **Hybrid QGNN** | **0.473 ± 0.034** | |
| **This study (v2)** | **Standalone QGNN** | **0.189 ± 0.055** | No graph signal at all |

## Interpretation

1. **Classical GraphSAGE-Full still edges out both quantum variants, but
   the gap to hybrid QGNN is now small** (PR-AUC 0.446 vs. 0.405, a 9%
   relative gap; F1 0.485 vs. 0.473, essentially tied). Standalone QGNN
   remains far behind on every metric except raw recall, which is an
   artifact of poor precision rather than good separation (one seed still
   predicts positive on nearly every test example). This still matches this
   project's own SCM-benchmark finding — classical ≥ quantum head on a
   frozen embedding — but the margin, at least on this benchmark, turned out
   to depend heavily on whether the quantum head's input was properly
   scaled first, not on quantum vs. classical processing being
   fundamentally different in capability here.

2. **Hybrid QGNN closing most of the gap to classical is the headline
   finding of the v2 pass.** It consumes the exact same graph-derived
   embedding classical does — the only architectural difference is a
   quantum circuit instead of an MLP head — so once both sides see a
   properly-scaled input, the comparison is much closer to apples-to-apples.
   This nuances (does not overturn) this project's SCM-benchmark finding:
   there, the same "quantum head vs. classical head on an identical frozen
   embedding" comparison also favored classical, but this result shows that
   gap is at least partly attributable to unaddressed input-scale mismatch,
   which is worth checking for on the SCM benchmark too as a follow-up
   (`HybridQuantumHeadLayerNorm.pre_projection_norm` is already available
   there, just never enabled in the QGNN-v4 investigation's own phases).

3. **Standalone QGNN improved substantially (PR-AUC 0.083→0.121, ROC-AUC
   0.596→0.714) but remains far below both graph-aware models** — it is
   architecturally an ablation that discards the graph entirely and
   compresses 182 features into 6 PCA dimensions retaining only 36.5% of
   variance. Its still-low PR-AUC mainly demonstrates that **graph structure
   carries real information on this benchmark**: fixing the encoding
   narrowed the "quantum vs. classical" gap a little, but removing the graph
   entirely still costs far more than that.

4. **Classical F1 (0.485) is still below the widely-cited original
   Weber/EvolveGCN numbers (0.70/0.77), but now sits closer to them, and the
   more rigorous comparison point tells the same story as before, more
   favorably.** The v1 write-up's caveats about feature-set and architecture
   differences still apply. The **2026 strict-temporal re-evaluation
   paper is still the more appropriate comparison** (same chronological-
   split philosophy as this study, vs. the original paper's transductive
   evaluation that paper argues leaks information): its own GraphSAGE
   scores F1=0.28 under that protocol — **our v2 classical GraphSAGE-Full
   at F1=0.485 is now clearly higher, by a wider margin than in v1** — and
   Elmougy & Liu's plain Logistic Regression on the same underlying data
   (F1≈0.45) is now below our mean rather than roughly tied with it. The
   "non-graph or simpler methods hold up better than expected under strict
   evaluation" pattern from three independent papers on this exact dataset
   family is still worth citing as context, but the v2 classical result
   makes a stronger case that this implementation itself was the limiting
   factor in v1, not some more fundamental property of the benchmark.

## Reproducing

```bash
python scripts/run_elliptic_benchmark.py --stage cache       # ~9s, needs data/raw/elliptic_pp/*.csv (see .gitignore)
python scripts/run_elliptic_benchmark.py --stage classical --seeds 42 43 44 45 46 --epochs 90  # patience=20 (default)
python scripts/run_elliptic_benchmark.py --stage qgnn       --seeds 42 43 44 45 46 --epochs 40  # whiten=True (default)
python scripts/run_elliptic_benchmark.py --stage hybrid     --seeds 42 43 44 45 46 --epochs 40  # pre_projection_norm=True (default); needs classical's embeddings
python scripts/run_elliptic_benchmark.py --stage report     --seeds 42 43 44 45 46
```

`--epochs` passed on the command line only sets the cap; `patience`,
`whiten` and `pre_projection_norm` are current code defaults (see the v1→v2
diffs in `scripts/run_elliptic_benchmark.py`'s git history on this branch
for the v1 values, if reproducing the original, unfixed numbers is ever
needed).
