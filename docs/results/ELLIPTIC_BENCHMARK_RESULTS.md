# Elliptic++ Benchmark: Validation Study Results

This is a validation study, run to check whether this project's classical
and quantum models behave sanely on a known, independently-published, real
graph benchmark — separate from and not a replacement for the SCM
supplier-disruption benchmark, which remains this project's primary
contribution. Full rationale, data provenance and implementation notes are
in `scripts/run_elliptic_benchmark.py`'s module docstring.

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

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **Classical GraphSAGE-Full** | **0.388 ± 0.084** | **0.815 ± 0.055** | **0.398 ± 0.096** | 0.338 ± 0.080 | 0.486 ± 0.123 |
| Hybrid QGNN | 0.288 ± 0.085 | 0.765 ± 0.034 | 0.345 ± 0.086 | 0.295 ± 0.136 | 0.465 ± 0.060 |
| Standalone QGNN | 0.083 ± 0.029 | 0.596 ± 0.107 | 0.151 ± 0.030 | 0.090 ± 0.024 | 0.628 ± 0.277 |

Per-seed values: `data/generated/elliptic_benchmark/results_per_seed.csv`
(gitignored, regenerate with `python scripts/run_elliptic_benchmark.py
--stage report --seeds 42 43 44 45 46` after re-running the three training
stages).

### Classical GraphSAGE-Full: a seed-variance finding

Per-seed test PR-AUC: seed42=0.314, **seed43=0.493**, **seed44=0.429**,
seed45=0.292, **seed46=0.410**. Two seeds (42, 45) early-stopped at epoch
4–5 (val PR-AUC ≈ 0.62–0.64) while the other three trained to ~50 epochs
(val PR-AUC ≈ 0.83–0.87) before early-stopping. `patience=10` on validation
PR-AUC is apparently too tight for this graph/split combination — it can
lock in a poor local optimum within the first several epochs on an unlucky
seed. This is reported as-is rather than cherry-picking favorable seeds; a
follow-up run with higher patience (e.g. 20–25) would likely raise the mean
and shrink the std, and is a natural next step, not done here to keep this
validation study bounded in scope.

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
| **This study** | **Classical GraphSAGE-Full** | **0.40 ± 0.10** | Full-batch, symmetrized edges, Elliptic++ 182-feature set |
| **This study** | **Hybrid QGNN** | **0.34 ± 0.09** | |
| **This study** | **Standalone QGNN** | **0.15 ± 0.03** | No graph signal at all |

## Interpretation

1. **Classical GraphSAGE-Full beats both quantum variants on every metric
   except raw recall** (where standalone QGNN's 0.63 is an artifact of poor
   precision, not good separation — one seed predicted positive on literally
   every test example). This matches this project's own SCM-benchmark
   finding: classical ≥ quantum head on a frozen embedding, and both ≥ a
   model with the graph signal removed entirely.

2. **Hybrid QGNN tracks classical more closely than standalone QGNN does**
   (PR-AUC 0.288 vs. 0.388, vs. standalone's 0.083), because it consumes the
   same graph-derived embedding classical does — the only difference is a
   quantum circuit instead of an MLP head. The consistent gap in classical's
   favor (visible seed-by-seed, not just in the mean) is the same pattern
   this project's SCM QGNN-v4 investigation already documented: swapping a
   frozen embedding's head from classical to quantum has not, in either
   dataset tested here, produced a net improvement.

3. **Standalone QGNN performing far worse is expected, not a quantum
   failure** — it is architecturally an ablation that discards the graph
   entirely and compresses 182 features into 6 PCA dimensions retaining only
   36.5% of variance. Its very low PR-AUC (0.083, barely above the ≈0.065
   base rate) mainly demonstrates that **graph structure carries real
   information on this benchmark** — removing it costs far more than
   swapping classical for quantum does.

4. **Our classical F1 (0.40) is well below the widely-cited original
   Weber/EvolveGCN numbers (0.70/0.77), but this is not a straightforward
   apples-to-apples comparison, and the more rigorous comparison point tells
   a different story.** Three concrete, checkable reasons for the gap:
   - Two of five seeds under-converged due to early-stopping sensitivity
     (see above) — restricting to the three well-converged seeds
     (43/44/46) alone gives mean F1 ≈ 0.47, already closer to the published
     figures.
   - Different feature set: Elliptic++'s 182 columns (post-imputation)
     vs. the original paper's 165.
   - Full-batch, symmetrized-edge classical GraphSAGE, not the exact
     Weber/EvolveGCN architecture or training recipe.

   The **2026 strict-temporal re-evaluation paper is the more appropriate
   comparison**, because it uses the same chronological-split philosophy
   this study does (rather than the original paper's transductive
   evaluation, which that paper argues leaks information). Under that
   protocol, its own GraphSAGE scores F1=0.28 — **our classical
   GraphSAGE-Full at F1=0.40 is actually higher**, and Elmougy & Liu's
   plain Logistic Regression on the same underlying data (F1≈0.45) is
   roughly comparable to our mean and above our under-converged seeds. This
   is the same "non-graph or simpler methods hold up better than expected
   under strict evaluation" pattern independently reported across three
   different papers on this exact dataset family — a legitimate, citable
   phenomenon, not evidence this implementation is broken.

## Reproducing

```bash
python scripts/run_elliptic_benchmark.py --stage cache       # ~9s, needs data/raw/elliptic_pp/*.csv (see .gitignore)
python scripts/run_elliptic_benchmark.py --stage classical --seeds 42 43 44 45 46 --epochs 60
python scripts/run_elliptic_benchmark.py --stage qgnn       --seeds 42 43 44 45 46 --epochs 40
python scripts/run_elliptic_benchmark.py --stage hybrid     --seeds 42 43 44 45 46 --epochs 40   # needs classical's embeddings
python scripts/run_elliptic_benchmark.py --stage report     --seeds 42 43 44 45 46
```
