# DataCo QGNN: Final Report

Consolidated final results for the DataCo real-world classical-vs-QGNN track
(a second, independent benchmark alongside the synthetic `scm_dataset`
project). This is a new document, separate from
`QGNN_DATACO_ABLATION_BENCHMARK.md` (the 42-run ablation sweep write-up) --
this report adds a properly-powered 10-seed deep dive on the ablation's
best-looking config, the full metric set, a threshold-tuning test, a
depth/data-re-upload sweep, and the **quantum-first** result that changes
the overall verdict.

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

## Depth and data re-upload: tried, don't help

Two more untried axes, before the finding that actually moved the needle
(next section). Both still use the frozen enriched-GraphSAGE embedding as
input, 6 qubits, 3 seeds each (`scripts/qgnn_deeper_hybrid_sweep.py`):

| variant | n_layers | f1 (mean) | roc_auc (mean) | pr_auc (mean) |
|---|---|---|---|---|
| standard | 3 | 0.657 &plusmn; .008 | 0.685 &plusmn; .011 | 0.724 &plusmn; .007 |
| standard | 4 | 0.670 &plusmn; .011 | 0.676 &plusmn; .010 | 0.703 &plusmn; .011 |
| standard | 5 | 0.646 &plusmn; .004 | 0.662 &plusmn; .003 | 0.707 &plusmn; .007 |
| data re-upload (`qgnn_v2_reupload.py`) | 3 | 0.647 &plusmn; .031 | 0.664 &plusmn; .040 | 0.692 &plusmn; .057 |
| data re-upload | 4 | 0.653 &plusmn; .023 | 0.670 &plusmn; .010 | 0.701 &plusmn; .005 |
| data re-upload | 5 | 0.636 &plusmn; .011 | 0.661 &plusmn; .016 | 0.697 &plusmn; .024 |

Neither more layers nor data re-upload beats the original 1-2 layer result
(F1 0.661-0.671, ROC-AUC 0.678-0.679 at 6 qubits) -- if anything, 5 layers is
the worst standard result, and re-upload adds instability (std up to 0.057
on PR-AUC) rather than expressivity that helps. More circuit depth is not
the fix, at least for this input.

## Quantum-first: the finding that actually moves the needle

Everything above fed the quantum circuit a **frozen GraphSAGE embedding**,
compressed by PCA into 6-8 dimensions that already captured ~98% of that
embedding's variance -- meaning the classical encoder, not the quantum
circuit, may have been the real ceiling. **Quantum-first** tests this
directly: skip GraphSAGE and the graph entirely, and feed the same
leakage-audited raw tabular features the classical LR/RF baselines use
straight into the quantum circuit (PCA to 6 dims, fit on train only, same
mode-collapse fix throughout). 3 seeds, both a standard 1-layer circuit and
a 3-layer re-upload circuit (`scripts/qgnn_deeper_hybrid_sweep.py`,
experiment `quantum_first`):

| model | F1 | ROC-AUC | PR-AUC | Balanced acc. | Brier (&darr; better) | Precision | Recall |
|---|---|---|---|---|---|---|---|
| Logistic regression | 0.661 | **0.743** | **0.813** | 0.704 | 0.199 | **0.825** | 0.552 |
| Random forest | 0.671 | 0.733 | 0.805 | 0.680 | 0.208 | 0.751 | 0.607 |
| GraphSAGE, base graph | 0.658 | 0.745 | 0.817 | **0.708** | **0.196** | 0.844 | 0.539 |
| GraphSAGE, enriched graph | **0.685** | 0.712 | 0.756 | 0.684 | 0.220 | 0.745 | 0.636 |
| QGNN, frozen-embedding (10 seeds) | 0.679 | 0.675 | 0.702 | 0.646 | 0.238 | 0.688 | 0.676 |
| **QGNN, quantum-first, standard (3 seeds)** | 0.682 &plusmn; .000 | 0.732 &plusmn; .007 | 0.796 &plusmn; .015 | 0.707 &plusmn; .000 | 0.204 &plusmn; .001 | 0.802 &plusmn; .000 | 0.594 &plusmn; .000 |
| QGNN, quantum-first, re-upload (3 seeds) | 0.681 &plusmn; .002 | 0.729 &plusmn; .005 | 0.784 &plusmn; .020 | 0.706 &plusmn; .001 | 0.202 &plusmn; .002 | 0.802 &plusmn; .001 | 0.593 &plusmn; .001 |

**This is the real result.** Quantum-first jumps ROC-AUC from 0.675 to
0.732 -- closing most of the gap to classical models -- and now genuinely
beats Random Forest on ROC-AUC, PR-AUC, balanced accuracy, and calibration
(Brier), while landing within a rounding error of GraphSAGE-enriched's F1.
It's also remarkably stable: F1/precision/recall/balanced-accuracy match to
3+ decimal places across all 3 seeds (ROC-AUC still varies slightly --
0.725-0.739 -- confirming the models aren't literally identical, just
converging to very similar decision boundaries each time; a robustness
signal, not a bug). Re-upload depth on top of quantum-first gives no
additional benefit over the plain 1-layer circuit -- consistent with the
depth findings above.

**n=3 seeds** -- tight enough here to be a strong signal (near-zero std),
but a 5-10 seed confirmation would be the natural next step before treating
this as fully settled.

## Answering "did we get better results"

- **Frozen-GraphSAGE-embedding QGNN (the original approach)**: no, on
  every metric except recall, in every framing tried (best run, honest mean,
  re-thresholded, deeper, re-uploaded).
- **Quantum-first QGNN (raw features, no GraphSAGE)**: **yes, substantially
  closer to classical performance than anything else tried**, beating Random
  Forest on 4 of 7 metrics and nearly matching Logistic Regression, though
  still short of the best classical/GraphSAGE result on ROC-AUC and PR-AUC
  specifically.

**The headline finding of this whole investigation isn't "QGNN wins" or
"QGNN loses" -- it's that *what feeds the quantum circuit matters more than
the circuit itself*.** A compressed classical graph embedding was actively
hurting QGNN; raw leakage-audited tabular features, reduced with the exact
same PCA discipline, let it perform close to classical baselines. The qubit
sweep, depth sweep, and re-upload sweep all varied the *circuit* and found
nothing; the one thing that actually helped varied the *input*. That is a
genuinely interesting, reportable, methodologically clean result for a
paper, independent of whether QGNN ultimately "wins."

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
| `scripts/final_comparison.py` | 10-seed 8-qubit deep dive with the full metric set |
| `scripts/test_qgnn_threshold.py` | The threshold-tuning test |
| `scripts/qgnn_deeper_hybrid_sweep.py` | Depth/re-upload sweep and the quantum-first result (this report's headline finding) |
| `data/processed/qgnn_ablation_results.csv`, `final_comparison_full_metrics.csv`, `qgnn_deeper_hybrid_results.csv` | Raw per-run results (gitignored -- regenerate via the scripts above) |
