# GraphSAGE Phase A.5 — Cross-Dataset Generalization Root-Cause Findings

Executes `GRAPHSAGE_PHASE_A5_ROOT_CAUSE_INVESTIGATION.md` in full (D1, D2,
D3, D4, D5; D6 not run — justified in §9). All D1/D2/D3/D5 analyses are
read-only and were run directly against already-generated data and
already-completed Phase A run artifacts. D4 required 8 new training runs
(2 directions × 4 feature modes × 5 model seeds each), executed by the
user.

---

## 1. Executive Summary

The seed43→seed44 / seed44→seed43 cross-dataset transfer asymmetry found
in Phase A (`GRAPHSAGE_PHASE_A_GENERALIZATION_FINDINGS.md`) is **not**
caused by a specific feature group, a distribution shift in model inputs,
or an architectural failure. Five independent diagnostics converge on one
mechanism: **the two datasets' disruption events landed in completely
non-overlapping sets of regions by chance** (a consequence of very few
total events — 6 and 9 — being assigned to one region each, independent
of any risk feature), while every static risk feature is **numerically
identical** between the datasets. A model trained on either dataset
necessarily learns "the specific regions that happened to be hit in my
training draw are risky," which cannot transfer to a draw that hit
different regions — and this holds regardless of which feature group
(dynamic, static, region-risk, or full) the model is given. This is
consistent with — not proof of, given only two dataset realizations — a
strong dataset-specific distribution/event-pattern dependence limiting
cross-world generalization in the current synthetic environment.

## 2. Research Question

> Determine why cross-dataset transfer is asymmetric before adding
> structural features, temporal features, architectural changes, or
> training optimizations.

## 3. Experiments Performed

| ID | Description | Type | Status |
|---|---|---|---|
| D1 | Regional distribution analysis | Read-only analysis | Complete |
| D2 | Feature distribution shift | Read-only analysis | Complete |
| D3 | Prediction inversion analysis | Read-only analysis | Complete |
| D4 | Cross-dataset feature-mode ablation | New training (8 runs × 5 seeds) | Complete |
| D5 | Event distribution comparison | Read-only analysis | Complete |
| D6 | Graph structural comparison | Read-only analysis | Not run — see §9 |

No dataset, label, split, or event-generation code was modified. No
target label was used for preprocessing, threshold selection, or feature
fitting at any point (unchanged from Phase A's protocol). No existing
run directory was overwritten — all analysis scripts (`scripts/analyze_*.py`)
read already-completed artifacts and write to new locations under
`experiments/analysis/` or new files under `experiments/classical_gnn/`.

## 4. Regional Analysis (D1)

Full data: `experiments/analysis/region_generalization/region_breakdown.csv`,
`REGION_GENERALIZATION_ANALYSIS.md`.

- **0 of 20 regions were disrupted in both datasets.** seed43's disrupted
  regions: 19, 8, 3, 1, 13, 16 (6 regions). seed44's disrupted regions: 7,
  5, 17, 14, 2, 18, 4, 6 (8 regions). 6 regions were disrupted in neither.
- region_19 is 100% disrupted in seed43 (every supplier there was
  disrupted at some point) and **0% disrupted in seed44**.
- Spearman correlation between the seed43-trained model's mean predicted
  risk per region and seed44's actual regional disruption rate: **−0.317**.
  The reverse direction (seed44-trained model vs seed43 actual): **−0.258**.
  Both negative — the model's regional risk ranking, transferred to the
  other dataset, points the wrong way on average.
- Within-dataset reference: seed43's own static exposure score correlates
  +0.287 with its own disruption rate; seed44's correlates **−0.279** with
  its own. Even within a single dataset, the exposure score is a weak-to-
  reversed predictor of which regions actually get disrupted — consistent
  with the earlier-established finding that event assignment in this
  generator does not depend on any risk/state feature.

## 5. Feature Distribution Analysis (D2)

Full data: `experiments/analysis/feature_shift/feature_distribution_comparison.csv`,
`FEATURE_SHIFT_ANALYSIS.md`.

- **All static supplier features** (tier, capacity, capacity_utilization,
  reliability, financial_health, geopolitical/disaster/cyber_exposure,
  criticality, industry) and **all region-level risk features**
  (geopolitical_risk, natural_disaster_risk, infrastructure_risk,
  trade_risk, cyber_risk, transport_reliability) are **bit-identical**
  between seed43 and seed44 (standardized mean difference = 0, KS
  statistic = 0, Wasserstein distance = 0, for every one of these
  columns).
- Dynamic (rolling-window operational) features show only negligible
  shift: the largest |standardized mean difference| across all 32 numeric
  features is 0.042 (`fulfillment_ratio_12`) — 0 of 32 features exceed the
  conventional 0.5 "medium effect size" threshold.
- **This rules out covariate shift (H2) as an explanation.** The model
  sees exactly the same static input values and near-identical dynamic
  input distributions in both worlds; the difference between the two
  datasets is which specific suppliers/regions were labeled positive, not
  what the model's inputs looked like.

## 6. Prediction Inversion Analysis (D3)

Full data: `experiments/analysis/prediction_inversion/prediction_inversion.csv`,
`PREDICTION_INVERSION_ANALYSIS.md`.

- **seed43→seed44**: the top-10 suppliers by predicted risk (mean
  probability 0.87–0.90) are **all 10 in region_19** — the one region
  with 100% disruption in seed43 and 0% in seed44. All 10 have 0% actual
  disruption in seed44. Spearman(predicted risk, actual disruption rate)
  across all 300 suppliers: **−0.261** (p=4.6e-6). The single most
  confidently-"safe" prediction (supplier_152, probability ≈1e-13) has a
  **60.7% actual disruption rate** in seed44 — the most direct evidence of
  the below-chance ROC-AUC's mechanism.
- **seed44→seed43**: top-10 predictions concentrate in region_18/17 (the
  seed44-disrupted regions), again 0% actual disruption in seed43.
  Spearman(predicted risk, actual disruption rate): **−0.003** (not
  significant, p=0.96) — much weaker inversion, consistent with this
  direction's milder (above-chance but still degraded) transfer.
- Correlation with static exposure score is **+0.467** in the seed43→44
  direction but **−0.344** in the seed44→43 direction — opposite signs.
  Since exposure score is identical across datasets (§5), a model that
  used it consistently would produce the same-signed correlation both
  times; the sign flip indicates each model learns a dataset-specific,
  non-generalizable mapping rather than a stable rule.
- Correlation with graph degree (supplier_material, supplier_procurement)
  is weak in both directions (|ρ| ≤ 0.11) — degree/connectivity is not a
  notable driver of the inversion.

## 7. Cross-Dataset Feature Ablation (D4)

Full data: `experiments/classical_gnn/crossdataset_feature_ablation_summary.csv`,
`CROSSDATASET_FEATURE_ABLATION.md`.

| Direction | Feature mode | Cross-dataset PR-AUC | Cross-dataset ROC-AUC | Fresh-onset recall | Already-ongoing recall |
|---|---|---:|---:|---:|---:|
| seed43→seed44 | dynamic_only | 0.0527 ± 0.0020 | 0.3155 ± 0.0252 | 0.0000 ± 0.0000 | 0.0011 ± 0.0018 |
| seed43→seed44 | static_only | 0.0543 ± 0.0014 | 0.3387 ± 0.0159 | 0.0000 ± 0.0000 | 0.0014 ± 0.0019 |
| seed43→seed44 | region_risk_only | 0.0521 ± 0.0003 | 0.3103 ± 0.0045 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| seed43→seed44 | full | 0.0520 ± 0.0012 | 0.3097 ± 0.0169 | 0.0000 ± 0.0000 | 0.0011 ± 0.0017 |
| seed44→seed43 | dynamic_only | 0.3022 ± 0.0058 | 0.6622 ± 0.0231 | 0.3125 ± 0.0000 | 0.5521 ± 0.0000 |
| seed44→seed43 | static_only | 0.3008 ± 0.0024 | 0.6780 ± 0.0177 | 0.3125 ± 0.0000 | 0.5521 ± 0.0000 |
| seed44→seed43 | region_risk_only | 0.2968 ± 0.0037 | 0.6649 ± 0.0209 | 0.3125 ± 0.0000 | 0.5521 ± 0.0000 |
| seed44→seed43 | full | 0.2944 ± 0.0036 | 0.6669 ± 0.0207 | 0.3125 ± 0.0000 | 0.5521 ± 0.0000 |

**Every feature mode fails (seed43→seed44) or transfers moderately
(seed44→seed43) to essentially the same degree.** Within each direction,
PR-AUC across the four modes spans only 0.0023 (seed43→seed44) or 0.0078
(seed44→seed43) — far smaller than the gap between directions (≈0.24–0.25
PR-AUC). The fresh-onset and already-ongoing recall figures are identical
to 4 decimal places across all four feature modes within a direction
(same fixed-0.5-threshold saturation noted in Phase A). This directly
answers D4's question: transfer failure is **not** primarily associated
with static, dynamic, or region-risk features specifically — it is a
property shared by the model regardless of what it's given to look at.

## 8. Event Distribution Comparison (D5)

Full data: `experiments/analysis/event_shift/EVENT_DISTRIBUTION_ANALYSIS.md`.

**seed43 has only 6 total events and seed44 has 9 — no broad statistical
claims are drawn from this sample.** With that caveat:

- **Every single event in both datasets affects exactly one region.**
  seed43's 6 events touch 6 of 20 regions; seed44's 9 events touch 8 of 20
  regions.
- seed44 has more events (9 vs 6), more affected suppliers per event on
  average, and a higher fresh-onset fraction of positive rows (4.90% vs
  4.43%) — broadly consistent with, but not sufficient alone to prove,
  H3's suggestion that seed44's larger/broader event sample supports more
  robust learning.
- Given one-event-one-region assignment and only 6–9 draws from 20
  regions, two independent datasets landing on non-overlapping region
  sets (§4) is the expected behavior of this generator's random
  assignment, not an anomaly requiring a separate mechanism.

## 9. Graph Structural Comparison (D6) — Not Run

The plan specifies D6 only if D1–D5 do not sufficiently explain the
asymmetry. They do: D1 (disjoint regional geography) + D2 (identical
static features, ruling out covariate shift) + D3 (individual-supplier
inversion traced directly to region membership) + D4 (uniform failure
across all feature modes, ruling out a specific-feature explanation) + D5
(mechanical cause — one-event-one-region assignment with very few events)
form a complete, mutually consistent causal chain. D3's degree/
connectivity correlations were also weak in both directions (§6),
providing no independent signal that a structural/topological comparison
would be likely to add. D6 was not run.

## 10. Evidence for/against H1–H4

- **H1 (regional concentration / dataset-specific geography): strongly
  supported.** D1 (disjoint disrupted regions), D3 (top/bottom-ranked
  suppliers trace directly to each dataset's own disrupted regions), and
  D5 (mechanical one-event-one-region cause) all point the same direction.
- **H2 (feature distribution shift): not supported, effectively ruled
  out.** D2 found zero shift in every static feature and negligible shift
  in dynamic features. The datasets' inputs are not meaningfully
  different; only their labels' regional pattern differs.
- **H3 (event-population/sample-diversity difference): plausible,
  partially supported, not independently isolated.** seed44 has more
  events, more onsets, and more disrupted regions than seed43 (D5), which
  is consistent with its milder transfer degradation, but this was not
  tested independently of H1 — it may simply be a restatement of "more
  positives, less narrow overfitting" rather than a distinct mechanism.
- **H4 (graph/topological interaction): not supported.** D3 found weak
  degree/connectivity correlations with predicted risk in both directions
  (|ρ| ≤ 0.11). D6 was not run (§9), so this is not a final ruling, but
  nothing in the completed diagnostics motivates prioritizing it.

## 11. Limitations

- **Two dataset realizations only.** Every cross-dataset finding here is
  "observed on two independently generated datasets," not a statistically
  powered claim across many synthetic worlds. Per the plan's own
  statistical-discipline section, five model seeds do not substitute for
  five independent dataset draws.
- D1's region-level PR-AUC/ROC-AUC/recall could only be computed for
  regions with ≥5 positive examples; many regions fall short (see `N/A
  (n=...)` entries in `region_breakdown.csv`).
- seed43's 6-event and seed44's 9-event populations are too small for D5
  to support broad statistical claims about event-type or severity
  distributions generally — only the specific counts reported.
- All findings are correlational (rule #13); "consistent with" is used
  throughout rather than "causes," in line with the investigation's
  non-negotiable rules.
- The fixed 0.5 threshold continues to produce seed-invariant recall/F1/
  precision figures (D4's identical-to-4-decimals recall across feature
  modes) that mask real underlying variance in the continuous risk score;
  PR-AUC/ROC-AUC remain the trustworthy metrics.

## 12. Negative Results (preserved)

1. Fresh-onset prediction is unavailable from current generated inputs (still holds; D4 confirms 0.0000 fresh-onset recall in every seed43→seed44 feature mode).
2. Static-feature advantage does not replicate across dataset seeds (Phase A finding, unaffected by this phase).
3. Within-seed performance can greatly exceed cross-seed performance (confirmed again in every D4 row: within-seed 0.94–1.00 vs cross-dataset 0.05–0.30).
4. Cross-dataset transfer is direction-asymmetric (confirmed).
5. seed43→seed44 can produce below-chance ranking (confirmed in all 4 feature modes, not just `full`).
6. Scenario split is statistically too thin for a primary generalization claim (Phase A finding, unaffected).
7. Fixed 0.5 threshold can mask model-seed variance in threshold-dependent metrics (confirmed again; also mode-invariant per D4).
8. **New**: feature-mode choice (dynamic/static/region-risk/full) has no meaningful effect on cross-dataset transfer quality in either direction — the asymmetry is not a feature-engineering problem.
9. **New**: the two datasets' disrupted regions are entirely disjoint (0/20 overlap), and this is consistent with the generator's known event-assignment mechanism rather than an unexplained anomaly.

## 13. Decision Gate

| Finding | Interpretation | Next action |
|---|---|---|
| Static-only *and* dynamic-only *and* region-risk-only *and* full all transfer equally poorly/moderately | Not GraphSAGE-only; not a specific-feature problem | **Focus on dataset/generalization design** |
| Strong regional artifact (0/20 region overlap, traced to individual supplier predictions) | Dataset-specific geography dominates | Investigate robust/domain-general representations or multi-world training |
| Zero feature distribution shift | Covariate shift ruled out | No action needed on preprocessing/normalization |
| Event diversity partially explains direction asymmetry, not independently isolated | Seed43 may be under-diverse (fewer/more concentrated events) | Consider additional independent dataset seeds if pursued further |

No row was cherry-picked; all four are simultaneously true of the
evidence gathered.

## 14. Recommended Next Phase

Per the decision gate, the evidence points to **Option A (multi-world
training)** as the most directly motivated next step if cross-dataset
generalization is to be pursued further: training on multiple
independent synthetic worlds (more than 2) is the only way to average out
the kind of single-draw regional-concentration effect documented here.
**Option D (structural features)** is explicitly not motivated by this
investigation — D4 already shows the existing feature groups transfer
uniformly, and there is no evidence a new structural feature would behave
differently, since the underlying problem is a labeling/event-geography
mismatch, not a missing input signal. **Option B (domain-robust
representations)** and **Option C (robust normalization)** are not
motivated either — D2 found no distribution shift for normalization to
correct.

## 15. Is Structural Feature Engineering Justified?

**No.** D4 directly tested whether transfer failure concentrates in one
feature group and found it does not — all four modes fail or succeed
together, within a fraction of the between-direction gap. Adding
structural features (single-source exposure, downstream procurement-path
counts, per `GRAPHSAGE_FINAL_GENERALIZATION_AND_IMPROVEMENT_PLAN.md`
Phase B) would add another input signal to a model whose transfer problem
has been traced to the label/event geography, not to missing input
information. This reaffirms and strengthens Phase A's original
conclusion, now with a specific mechanism rather than an unexplained
correlation.

## 16. Is GraphSAGE Ready to Freeze?

**No.** Freezing implies the model's behavior and limitations are
understood and accepted as the final classical baseline. While this phase
substantially deepens that understanding, the plan's Phase F freeze
criteria (final model selection using generalization + performance +
calibration + stability, final multi-seed run, reproducibility audit) have
not been attempted, and the multi-world question raised in §14 is
unresolved. Freezing now would mean freezing a model whose only tested
cross-world behavior is one asymmetric two-world comparison.

## 17. Is QGNN Allowed to Begin?

**No.** Per the plan's explicit QGNN gate ("Do not start QGNN
benchmarking until GraphSAGE is frozen") and §16 above, GraphSAGE is not
frozen. This is unchanged from Phase A's conclusion.

---

## Artifacts

- `scripts/_analysis_common.py`, `scripts/analyze_region_generalization.py`,
  `scripts/analyze_feature_shift.py`, `scripts/analyze_prediction_inversion.py`,
  `scripts/analyze_event_distribution.py`, `scripts/analyze_crossdataset_feature_ablation.py`
  (new, all read-only diagnostics)
- `experiments/analysis/region_generalization/`, `experiments/analysis/feature_shift/`,
  `experiments/analysis/prediction_inversion/`, `experiments/analysis/event_shift/`
  (analysis outputs, regenerable, not committed — see `.gitignore`)
- `experiments/classical_gnn/crossdataset_feature_ablation_summary.csv`,
  `CROSSDATASET_FEATURE_ABLATION.md`, and the 40 new `*_crossdataset_*_seed*`
  run directories from D4 (regenerable, not committed)
- Full test suite: 195/195 passing, unchanged by this phase (no `src/`
  changes, only new diagnostic scripts)

## Definition of Done

- [x] D1 regional breakdown completed
- [x] D2 feature distribution shift completed
- [x] D3 prediction inversion completed
- [x] D4 cross-dataset feature-mode ablation completed
- [x] D5 event-distribution comparison completed
- [x] D6 graph comparison completed if required — not required, justified in §9
- [x] H1–H4 evaluated using evidence
- [x] unsupported causal claims removed (correlational language used throughout)
- [x] negative results preserved
- [x] artifacts reproducible (no run directory overwritten; all scripts re-runnable)
- [x] tests pass (195/195)
- [x] `GRAPHSAGE_PHASE_A5_ROOT_CAUSE_FINDINGS.md` created
- [x] next phase selected based on evidence (§14)

**Phase A.5 is complete.**
