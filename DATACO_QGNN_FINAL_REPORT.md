# DataCo QGNN: Final Report

Consolidated final results for the DataCo real-world classical-vs-QGNN track
(a second, independent benchmark alongside the synthetic `scm_dataset`
project). This is a new document, separate from
`QGNN_DATACO_ABLATION_BENCHMARK.md` (the 42-run ablation sweep write-up) --
this report adds a properly-powered 10-seed deep dive on the ablation's
best-looking config, the full metric set, a threshold-tuning test, and the
final verdict.

## Setup recap

- **Dataset**: DataCo Smart Supply Chain (Mendeley, CC-BY-4.0), 180,519
  order-item rows, chronological 70/15/15 split, `Late_delivery_risk` target
  predicted from leakage-audited pre-shipment features only.
- **Graph**: `order`/`customer`/`product`/`region`/`order_country` nodes,
  enriched with train-only historical late-rates and a causal 7-day
  regional-congestion feature.
- **QGNN**: frozen GraphSAGE order-embedding -> PCA -> variational quantum
  circuit (`scm_dataset.modeling.qgnn`, unmodified).
- Full methodology: `notebooks/dataco_exploration.ipynb`; literature
  references: `DATACO_REFERENCES.md`.

## The mode-collapse fix (recap)

A pervasive training-instability bug was found and fixed: a randomly
initialized final-layer bias could start predictions saturated toward one
class, where BCE gradients vanish before the model corrects course. Fix:
initialize that bias to `logit(train positive rate)`, plus gradient clipping
(`max_norm=1.0`) -- applied only in the DataCo-specific training wrappers
(`scripts/qgnn_ablation.py`, `scripts/final_comparison.py`), never in
`graphsage.py`/`qgnn.py` themselves, since those are frozen and shared with
the synthetic benchmark's own QGNN v1-v4 work. Verified: GraphSAGE collapse
rate 40% -> 0%; QGNN collapse rate up to 50% -> 0-17% depending on
experiment. All results below are post-fix.

## Full comparison, complete metric set

5 seeds each for LR/RF (LR/RF are deterministic enough that seed variance
wasn't the concern here -- their `random_state=42` single run is reported),
5 seeds for GraphSAGE (both graphs), **10 seeds** for QGNN 8-qubit/1-layer
(the ablation's best-looking config, now properly powered instead of the
original 3-seed estimate). 0/10 QGNN seeds collapsed.

| model | F1 | ROC-AUC | PR-AUC | Balanced acc. | Brier (&darr; better) | Precision | Recall |
|---|---|---|---|---|---|---|---|
| Logistic regression | 0.661 | 0.743 | 0.813 | 0.704 | 0.199 | 0.825 | 0.552 |
| Random forest | 0.671 | 0.733 | 0.805 | 0.680 | 0.208 | 0.751 | 0.607 |
| GraphSAGE, base graph | 0.658 &plusmn; .006 | **0.745** &plusmn; .001 | **0.817** &plusmn; .0003 | **0.708** &plusmn; .001 | **0.196** &plusmn; .0005 | **0.844** &plusmn; .006 | 0.539 &plusmn; .011 |
| GraphSAGE, enriched graph | **0.685** &plusmn; .001 | 0.712 &plusmn; .001 | 0.756 &plusmn; .002 | 0.684 &plusmn; .014 | 0.220 &plusmn; .001 | 0.745 &plusmn; .028 | 0.636 &plusmn; .022 |
| **QGNN, 8-qubit/1-layer (10 seeds)** | 0.679 &plusmn; .012 | 0.675 &plusmn; .014 | 0.702 &plusmn; .019 | 0.646 &plusmn; .029 | 0.238 &plusmn; .006 | 0.688 &plusmn; .037 | **0.676** &plusmn; .063 |

**Bold** marks the best value per column. QGNN wins exactly one column
(recall) and is worst on five of the other six. Its F1 (0.679) sits between
Random Forest and GraphSAGE-enriched -- competitive, not exceptional.

This 10-seed mean (0.679) is lower than the ablation's original 3-seed
estimate for this same config (0.693, reported in
`QGNN_DATACO_ABLATION_BENCHMARK.md`) -- a reminder that 3 seeds wasn't
enough to trust, exactly the kind of small-sample overconfidence this whole
investigation kept running into.

## Best-of-42 vs. honest mean

The ablation sweep's single best run anywhere (F1=0.701, ROC-AUC=0.703) does
edge out every other model's F1. But that's the best of 42 attempts, not a
result you'd get by running QGNN once -- the honest comparison point is the
10-seed mean above, and it does not beat classical models except on recall.
Reporting the best-of-N as "QGNN's result" without disclosing N is the
single easiest way to make this look like a bigger win than it is.

## Threshold tuning: tried, doesn't help

Hypothesis: QGNN's low precision/high recall at a fixed 0.5 threshold might
reflect a fixed-threshold artifact rather than genuinely worse probability
estimates -- if so, F1-optimal threshold selection on validation
(`metrics.select_threshold`, already in the codebase) should rebalance it.

Tested on 3 seeds (`scripts/test_qgnn_threshold.py`):

| seed | optimal threshold | F1 @ 0.5 | F1 @ optimal | precision @ optimal | recall @ optimal |
|---|---|---|---|---|---|
| 0 | 0.070 | 0.690 | 0.710 | 0.550 | 0.9999 |
| 1 | 0.465 | 0.701 | 0.709 | 0.551 | 0.9950 |
| 2 | 0.247 | 0.707 | 0.707 | 0.559 | 0.9617 |

**Result: this doesn't rescue QGNN -- it just re-finds the trivial
classifier.** For seeds 0 and 2, the "optimal" threshold is so low that
recall approaches 1.0 and precision collapses to the ~0.55 base rate -- the
same degenerate "predict positive for almost everyone" pattern from the
mode-collapse investigation, just reached through thresholding instead of
bad training. ROC-AUC is unchanged (expected -- it's threshold-independent),
confirming the underlying ranking ability didn't improve. This is consistent
with QGNN having the worst Brier score (calibration) of every model tested:
its probability outputs don't carry a clean separating boundary for any
threshold search to find.

## Answering "did we get better results"

- **On F1, best case**: yes, marginally -- best-of-42 (0.701) beats
  GraphSAGE-enriched's honest mean (0.685), a genuine +2.3% relative gain
  once the mode-collapse bug was fixed (it was a wash pre-fix).
- **On F1, honest mean**: no -- 0.679 vs. GraphSAGE-enriched's 0.685.
- **On every other metric (ROC-AUC, PR-AUC, balanced accuracy, Brier,
  precision)**: no, not once, in any framing tried (best run, mean, or
  re-thresholded).

**QGNN does not show a real advantage on this dataset with the architecture
and encoding tried here.** This is a legitimate, well-supported negative
result: the qubit/layer sweep (`QGNN_DATACO_ABLATION_BENCHMARK.md`) ruled
out "undersized architecture," the data-efficiency sweep ruled out
"data-starved," the mode-collapse fix ruled out "broken optimizer," and
threshold tuning ruled out "fixed-threshold artifact." What's left
unexplored, and could genuinely change this conclusion: a fundamentally
different encoding or entanglement strategy (the existing project's own
QGNN-v3 entanglement-topology study, on the synthetic benchmark, is the
natural template to port here), or accepting that a 6-8 qubit
angle-encoding circuit on a PCA-compressed classical embedding simply
doesn't carry more signal than the classical embedding already had --
which, given PCA explained ~98% of the embedding's variance in 6-8
components, is a plausible and unglamorous explanation on its own.

## Literature, for context

See `DATACO_REFERENCES.md` for full citations and caveats. Headline
numbers: EAGLE (graph-based, peer-reviewed) F1=0.876; SOM+ANN (peer-reviewed
but an outlier next to everything else tried on this dataset) F1=0.962; the
"Integrated ML" preprint's F1=0.854 is not trustworthy (not peer-reviewed,
cites features absent from the real DataCo schema).

## Reproducibility

| Artifact | Purpose |
|---|---|
| `notebooks/dataco_exploration.ipynb` | Full pipeline: EDA, leakage audit, classical baselines, graph construction/enrichment, GraphSAGE, single QGNN run |
| `scripts/qgnn_ablation.py` | 42-run ablation: qubit/layer sweep, data efficiency, noise robustness |
| `scripts/test_mode_collapse_fix.py`, `_v2.py` | Diagnosis and verification of the mode-collapse fix |
| `scripts/final_comparison.py` | 10-seed 8-qubit deep dive with the full metric set (this report's main table) |
| `scripts/test_qgnn_threshold.py` | The threshold-tuning test above |
| `data/processed/qgnn_ablation_results.csv`, `final_comparison_full_metrics.csv` | Raw per-run results (gitignored -- regenerate via the scripts above) |
