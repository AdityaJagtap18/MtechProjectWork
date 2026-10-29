# GraphSAGE Phase A — Generalization Findings

Executes Phase A of `GRAPHSAGE_FINAL_GENERALIZATION_AND_IMPROVEMENT_PLAN.md`:
D0 (dataset integrity), D1 (within-seed44 diagnostic), D2 (true cross-seed
generalization). All training runs were executed by the user; this
document records what was built, what was run, and what the numbers say.
Companion to `GRAPHSAGE_IMPROVEMENT_WORK_SUMMARY.md` (prior phase, seed43
only).

All figures below are `mean ± std` over 5 model seeds (42–46), read
directly from each run's `metrics.json`/`calibration.json` on disk unless
noted otherwise.

---

## 1. D0 — Dataset Integrity (seed43 vs seed44)

`scripts/compare_datasets.py` confirms `scm_v1_black_swan_seed43` and
`scm_v1_black_swan_seed44` are **structurally compatible**: identical node
counts (300 suppliers, 2000 procurement orders, 100 materials, 50 plants,
200 products, 20 regions), identical edge counts across all 9 edge types,
identical horizon (104 periods), identical feature/label schemas, all
required files present in both. `compatible_for_cross_dataset_evaluation:
true`.

They differ, as expected from independent random draws, in event
frequency:

| | seed43 | seed44 |
|---|---:|---:|
| `supplier_disrupted` prevalence | 3.33% (1039/31200) | 6.08% (1897/31200) |
| Onset count (0→1 transitions) | 46 | 93 |

seed44 has roughly twice the disruption events of seed43 — relevant
context for the asymmetry found in D2 below.

---

## 2. D1 — Within-Seed44 Diagnostic

**Correction made mid-run**: the first attempt used the default
`configs/graphsage.yaml` (temporal split) and produced
`recall_already_ongoing = 1.0` uniformly with zero fresh-onset examples —
the same "wrong split" signature that misled the original seed43 ablation
(GRAPH_SAGE_IMPROVEMENT_PLAN.md phase). Reran on
`configs/graphsage_severity.yaml`, matching the methodology that produced
the original decisive seed43 finding.

### Ablations table (severity split, test)

**seed43** (original finding, from prior phase):

| Variant | PR-AUC | ROC-AUC | F1 | Recall | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Dynamic only | 0.3644 ± 0.0157 | 0.6646 ± 0.0153 | 0.3485 ± 0.0250 | 0.4411 ± 0.0010 | 0.1072 ± 0.0085 | 0.1170 ± 0.0086 |
| Static only | **0.4580 ± 0.0107** | **0.7441 ± 0.0177** | 0.4051 ± 0.0106 | 0.4406 ± 0.0000 | 0.0877 ± 0.0054 | 0.0926 ± 0.0047 |
| Region risk only | 0.3736 ± 0.0532 | 0.6844 ± 0.0469 | 0.3673 ± 0.0340 | 0.4406 ± 0.0000 | 0.1037 ± 0.0150 | 0.1089 ± 0.0153 |
| Full | 0.4487 ± 0.0131 | 0.7275 ± 0.0156 | 0.4113 ± 0.0201 | 0.4406 ± 0.0000 | 0.0857 ± 0.0079 | 0.0904 ± 0.0074 |

**seed44** (this phase):

| Variant | PR-AUC | ROC-AUC | F1 | Recall | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Dynamic only | 0.1436 ± 0.0134 | 0.6945 ± 0.0248 | 0.1546 ± 0.0077 | 0.2126 ± 0.0041 | 0.1807 ± 0.0103 | 0.1914 ± 0.0097 |
| Static only | 0.1501 ± 0.0224 | 0.6971 ± 0.0362 | 0.1794 ± 0.0160 | 0.2169 ± 0.0226 | 0.1541 ± 0.0146 | 0.1598 ± 0.0155 |
| Region risk only | **0.1578 ± 0.0115** | **0.7323 ± 0.0242** | 0.1579 ± 0.0046 | 0.2100 ± 0.0047 | 0.1738 ± 0.0046 | 0.1825 ± 0.0071 |
| Full | 0.1642 ± 0.0084 | 0.7298 ± 0.0252 | 0.1953 ± 0.0295 | 0.2413 ± 0.0515 | 0.1556 ± 0.0013 | 0.1593 ± 0.0039 |

Fresh-onset recall: **0.0000** in every mode on both dataset seeds except
seed44's `static_only` (0.0111 ± 0.0222, i.e. ~1 case out of ~144, still
statistically indistinguishable from zero) and seed44 `full`
(0.0125 ± 0.0250) — consistent with the standing "fresh-onset unavailable
from current generated inputs" finding, replicated on a second dataset.

### Interpretation

On seed43, `static_only` (0.458) clearly beats `dynamic_only` (0.364) —
a 0.094 PR-AUC gap, roughly 6–9× either mode's own standard deviation, a
real effect. **On seed44 this gap disappears**: `static_only` (0.150) and
`dynamic_only` (0.144) differ by 0.006, well inside one standard
deviation of either. `region_risk_only` edges out both by a similarly
small, non-significant margin. The specific ordering that motivated this
whole investigation does not replicate.

Separately, absolute predictive power collapsed across **every** feature
mode from seed43 to seed44 (roughly −60% PR-AUC uniformly), and this is
not a base-rate artifact — seed44's severity-split test prevalence
(8.77%) is actually *higher* than seed43's (7.36%), which should make
PR-AUC easier, not harder, for an equally-skilled model. ROC-AUC, by
contrast, barely moved (0.727 → 0.730 for `full`). Something about
ranking quality at the high-confidence end of the distribution
specifically degraded on seed44, which is consistent with — but does not
by itself prove — the regional-concentration explanation below.

---

## 3. D2 — True Cross-Seed Generalization (the decisive experiment)

Trained on one dataset's own train/validation split (temporal, full
features — the primary configuration), evaluated both within-seed (same
dataset's test split) and cross-dataset (the *entire* other dataset,
using the source's fitted preprocessor and validation-selected threshold,
per `evaluate_on_target_dataset`). No target label ever touched fitting,
preprocessing, or thresholding.

### Generalization table

| Train | Test | Model | PR-AUC | ROC-AUC | F1 | Recall | Brier | ECE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| seed43 | seed43 | GraphSAGE | 0.8070 ± 0.0655 | 0.9871 ± 0.0083 | 0.6771 ± 0.0355 | 0.9678 ± 0.0164 | 0.0418 ± 0.0119 | 0.0493 ± 0.0103 |
| seed44 | seed44 | GraphSAGE | 0.9751 ± 0.0256 | 0.9992 ± 0.0008 | 0.2821 ± 0.0251 | 1.0000 ± 0.0000 | 0.1603 ± 0.0195 | 0.1994 ± 0.0266 |
| **seed43** | **seed44** | **GraphSAGE** | **0.0520 ± 0.0012** | **0.3097 ± 0.0169** | 0.0009 ± 0.0013 | 0.0010 ± 0.0014 | 0.1301 ± 0.0159 | 0.1414 ± 0.0133 |
| **seed44** | **seed43** | **GraphSAGE** | **0.2944 ± 0.0036** | **0.6669 ± 0.0207** | 0.1622 ± 0.0281 | 0.5239 ± 0.0000 | 0.1751 ± 0.0343 | 0.2079 ± 0.0381 |

Baselines, fit on source only, same protocol:

| Train→Test | Majority PR-AUC | Logistic Regression PR-AUC |
|---|---:|---:|
| seed43 → seed43 (within) | 0.067 | 0.304 |
| seed43 → seed44 (cross) | **0.079** | 0.057 |
| seed44 → seed44 (within) | 0.043 | 0.046 |
| seed44 → seed43 (cross) | 0.041 | 0.026 |

### The finding

Within-seed, GraphSAGE looks excellent in both directions (PR-AUC 0.81
and 0.98, ROC-AUC 0.99 in both — inflated by the temporal split's
persistence-detection ceiling, per the standing onset-breakdown finding).
Cross-dataset, the two directions diverge completely:

- **seed43 → seed44: catastrophic negative transfer.** PR-AUC 0.052,
  *worse than the majority-class baseline's 0.079* — a model trained on
  seed43 loses to a model that ignores the input entirely and predicts
  the training base rate. ROC-AUC 0.310, **below chance** (0.5): the
  model's risk ordering is anti-correlated with the truth on seed44, not
  merely uninformative. Fresh-onset recall 0.000 (0/352); already-ongoing
  recall ≈0.001 (fails to flag disruptions it would catch with recall
  0.968 within-seed).
- **seed44 → seed43: real, substantial transfer.** PR-AUC 0.294, roughly
  **7× the majority baseline** (0.041) and **11× logistic regression**
  (0.026). ROC-AUC 0.667, clearly above chance. Fresh-onset recall
  0.3125 (40/128) — the only time in this entire investigation (across
  both improvement phases) that fresh-onset recall has been meaningfully
  above zero.

**This asymmetry, not a single verdict, is the headline result.**
GraphSAGE's ability to generalize across dataset seeds is real but highly
sensitive to which dataset it is trained on — it is not simply "does not
generalize" nor "generalizes fine."

### Hypothesis (not confirmed)

seed44 has roughly twice seed43's onset count (93 vs 46) and a less
concentrated disruption pattern (seed43's 46 onsets are disproportionately
drawn from 4 of its 20 regions, per the prior phase's diagnosis). A model
trained on seed43's thinner, more concentrated sample plausibly memorizes
"these specific regions/suppliers are risky" rather than the general
region-risk relationship, which then actively misfires on seed44's
different disruption geography. A model trained on seed44's broader
sample has more evidence to learn the general relationship from, and that
more general pattern holds up reasonably — though not perfectly — on
seed43. GraphSAGE has no per-entity parameters (only per-node-type and
per-relation weights), so this would have to operate through the
input feature distributions and graph structure, not literal ID
memorization; confirming this would require a dedicated feature-attribution
or per-region breakdown, which was not run.

### Calibration caveat worth flagging now

The onset-breakdown recall for seed44→seed43 came out **bit-identical to
four decimal places across all 5 independently-trained model seeds**
(0.5239 exactly, std 0.0000) — the same pattern seen in both directions'
within-seed persistence recall. `configs/graphsage.yaml` uses a **fixed**
threshold policy (`value: 0.5`), not one selected per-model from
validation data, and training `val_loss` climbed as high as 8–10 nats in
several runs (severe sigmoid saturation/overconfidence). The PR-AUC/
ROC-AUC numbers above are threshold-independent and do vary properly by
seed, so this doesn't undermine the headline finding — but it means the
precision/recall/F1 columns in this report understate genuine seed-to-seed
variance and should be revisited if Phase E (calibration) is reached.

---

## 4. Scenario Split

Ran on seed43 via a new `configs/graphsage_scenario.yaml` (identical to
`graphsage_severity.yaml` except `split.strategy: scenario`).

**This dataset cannot statistically support the question the scenario
split is meant to answer.** Total events in seed43's full 104-period
horizon: 6 (3 logistics, 1 supplier failure, 1 geopolitical, 1
cyberattack). The held-out test period contains exactly **1 cyberattack
and 1 geopolitical event** (plus 2 logistics). Training positives:
**61 out of 8,100 examples** (0.75%) — far thinner than any other split
used in this project. A single held-out event can swing recall from 0 to
1; this is at most an anecdote, not a generalization measurement.

| Variant | PR-AUC | ROC-AUC | F1 | Recall | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Full (scenario split, seed43) | 0.0498 ± 0.0098 | 0.4002 ± 0.0695 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1014 ± 0.0347 | 0.1566 ± 0.1014 |

Recall is exactly 0.0000 in all 5 seeds (the fixed 0.5 threshold never
classifies anything positive here). PR-AUC (0.050) sits below the test
set's own base rate (0.059) and ROC-AUC (0.400) is below chance — both
consistent in direction with the seed43→seed44 D2 finding (a model
trained on seed43's thin, concentrated positive class fails to transfer
outside the exact conditions it trained on), but given the n=2 held-out
events, this is **weak, anecdotal support at best**, not independent
confirmation. Recorded as a negative result per the plan's own guidance,
not as a new finding to build on.

Per the plan's Definition of Done, this satisfies "scenario split
evaluated **or shown statistically insufficient**" — evaluated, and shown
statistically insufficient to serve as a primary generalization result.

---

## 5. Phase A Decision

Per the plan's own gate ("decide whether the next phase is scientifically
justified before proceeding — do not blindly execute every optimization
if an earlier diagnostic invalidates its premise"):

**Phase B (structural features) is not yet justified.** Two independent
diagnostics point the same direction:

1. D1: the specific static-beats-dynamic ordering that motivated interest
   in static/structural signal does not replicate on an unseen dataset
   seed.
2. D2: cross-dataset transfer of the full model itself is unstable enough
   to actively invert (below-chance ROC-AUC) depending on training seed.

Building more structural features on top of a foundation whose transfer
behavior is this direction-dependent risks optimizing for the same kind
of dataset-specific artifact D1 and D2 just caught. The priority is
understanding *why* seed43-trained models generalize so poorly relative
to seed44-trained ones before adding anything on top.

---

## 6. Negative Results (preserved per plan's requirement)

- Fresh-onset prediction is unavailable from current generated inputs —
  replicated on seed44 (recall ≈0 across all feature modes and both
  within-seed and cross-dataset conditions, except the one 31%
  cross-dataset exception in §3).
- **Static feature advantage is dataset-specific** (the plan's own
  anticipated candidate negative result — now confirmed, not merely
  suspected): the seed43 static-only PR-AUC advantage over dynamic-only
  does not hold on seed44.
- Cross-dataset generalization is direction-asymmetric and can be
  actively harmful (below-chance ROC-AUC), not merely weaker than
  within-seed performance.
- The scenario split (seed43, held-out cyberattack + geopolitical
  events) shows the same below-chance direction, though the sample is far
  too thin (2 held-out events, 61 training positives) to count as
  independent confirmation.
- The temporal split's fixed 0.5 threshold produces seed-invariant
  recall/F1/precision figures that mask real underlying model variance;
  PR-AUC/ROC-AUC remain the trustworthy metrics until a data-driven
  threshold policy is evaluated (Phase E, not yet started).

None of these are being treated as failures to hide — they directly
answer the plan's own research questions and gate its next phase.

---

## 7. Phase A Definition-of-Done Status

- [x] seed44 integrity verified (D0)
- [x] D1 complete
- [x] true cross-seed train→test evaluation complete (D2, both directions)
- [x] scenario split evaluated or shown statistically insufficient —
      evaluated on seed43; shown statistically insufficient (§4)
- [x] seed44 full-model robustness evaluated (obtained as the within-seed
      byproduct of the seed44→seed43 D2 run: PR-AUC 0.975 ± 0.026, ROC-AUC
      0.999 ± 0.001, temporal split, full features)
- [ ] justified structural/temporal/architecture/training experiments — **on hold, Phase B not justified per §5**

**Phase A is complete.**

---

## 8. Artifacts

- `dataset_comparison_seed43_seed44.json` (D0 output, regenerable via
  `scripts/compare_datasets.py`, not committed — see `.gitignore`)
- `configs/graphsage_scenario.yaml` (new, scenario-split config)
- `experiments/classical_gnn/*_severity_*_seed44_seed*` (D1 runs)
- `experiments/classical_gnn/*_crossdataset_seed43_to_seed44_seed*`,
  `*_crossdataset_seed44_to_seed43_seed*`, and their `_baselines` siblings
  (D2 runs)
- `experiments/classical_gnn/*_scenario_seed*` (scenario-split runs, §4)
- `experiments/classical_gnn/multiseed_summary_*` (aggregate summaries)
- All runs use the primary architecture/hyperparameters unchanged from
  `GRAPHSAGE_WORK_SUMMARY.md`'s frozen classical baseline — no leakage,
  no manufactured signal, nothing overwritten.

## 9. Open Items

- The regional-concentration hypothesis in §3 is untested; a per-region
  or feature-attribution breakdown would be needed to confirm it before
  treating it as more than a plausible explanation.
- QGNN work remains gated: GraphSAGE is not frozen, and per §5, still has
  an open scientific question (the D2 asymmetry) rather than a justified
  path into Phase B.
