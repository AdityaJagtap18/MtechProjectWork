# GraphSAGE Work Summary

Everything done implementing `CLASSICAL_GRAPHSAGE_IMPLEMENTATION_PLAN.md` end
to end, on top of the already-complete SCM benchmark documented in
`WORK_SUMMARY.md`. Companion to `GRAPHSAGE_PROGRESS.md` (the living status
tracker) — this file is the one-time, detailed account of what was built,
what went wrong along the way, and what the results actually are.

---

## 1. Session narrative

1. Read `WORK_SUMMARY.md`, `CLASSICAL_GRAPHSAGE_IMPLEMENTATION_PLAN.md`
   (93 sections), and the actual generated benchmark files/schema before
   writing any code, per the plan's own §1 instruction — confirmed real
   column names, node/edge types, split file formats, and (critically)
   `data/benchmark/feature_audit.csv`'s existing allow/disallow rules.
2. Resolved the plan's open question about `supplier_risk_score` directly
   from that audit file: it already marks all of `labels/supplier_labels.csv`
   `allowed=False`, and `labels/risk_labels.py` confirms it's a realized
   fulfillment shortfall computed from the same simulation outcomes as the
   target — excluded from the start, no ambiguity.
3. Chose `scm_v1_black_swan_seed43` as the primary/frozen dataset (has all
   3 split types, a severity-5 event, and is the dataset `WORK_SUMMARY.md`
   already traced end to end).
4. Installed `torch` (CPU wheel), `torch_geometric`, `scikit-learn` into
   `.venv`; added them to `pyproject.toml`'s `[modeling]` extra and
   `requirements.txt`.
5. Built the full modeling package (`src/scm_dataset/modeling/`, 15
   modules — full list in §3) implementing the pipeline in plan §90's
   order: data loading → leakage-safe temporal features → train-only
   preprocessing → heterogeneous PyG graph → 2-layer GraphSAGE → training
   → evaluation → experiment artifacts.
6. Smoke-tested each stage by hand against the real benchmark as it was
   built (not just at the end) — caught and fixed a `feature_audit.csv`
   parsing bug this way (§4.1) before it could hide behind passing tests.
7. Wrote the full test suite (71 new tests across 8 files + a shared
   fixture) — all passing alongside the original 90.
8. Ran the first real experiment (100-epoch budget, seed 42) on the actual
   benchmark; then a 5-seed sweep; then Majority/Logistic Regression
   baselines. Wrote `experiment_report.md` inside the run directory.
9. While explaining the severity-generalization config to the user, found
   and fixed a real bug: two config fields (`severity_train_max`,
   `scenario_test_event_types`) looked configurable but were silently
   never read anywhere (§4.2).
10. Ran the severity generalization experiment (5 seeds). Its recall being
    bit-identical across all 5 independently-seeded models led to the
    single most important finding of this phase: the model's high recall
    is persistence detection, not early warning (§5). Built this into the
    codebase permanently (`disruption_onset_breakdown`) rather than
    leaving it as a one-off observation, and corrected both experiment
    reports and `GRAPHSAGE_PROGRESS.md` to lead with it.
11. Found and fixed a multi-seed summary filename collision bug (§4.3) —
    the severity sweep had silently overwritten the temporal sweep's
    summary file on disk, because both used the same filename.
12. User independently reran both the primary and severity experiments
    from a fresh terminal session. Every number matched the earlier runs
    to full floating-point precision (§6) — the strongest reproducibility
    evidence available short of a second machine.

---

## 2. What the task was

Predict, from information available at week `t`, whether a supplier will
be disrupted at any point in weeks `t+1..t+4`, using a 2-layer
heterogeneous GraphSAGE over the benchmark's 6 node types / 9 edge types,
trained and evaluated with full leakage discipline (train-only
preprocessing, validation-only threshold/early-stopping selection, test
touched exactly once), and compared against Majority and Logistic
Regression baselines on the identical target/split/features. This is the
"classical baseline" that a later QGNN stage will be compared against
under the same protocol.

---

## 3. What was built

**Package** — `src/scm_dataset/modeling/`:

| Module | Responsibility |
|---|---|
| `config.py` | `GraphSAGEConfig` dataclasses + YAML loader |
| `data.py` | Loads one benchmark dataset directory, validates schema |
| `features.py` | Static + rolling-window dynamic features, target construction, the automated leakage audit (`audit_feature_sources`) |
| `preprocessing.py` | Train-only scaling/imputation/categorical encoding, JSON save/load |
| `hetero_graph.py` | Node id mapping, edge validation, `HeteroData` snapshot construction (+ reverse edges) |
| `graphsage.py` | The 2-layer heterogeneous GraphSAGE model |
| `losses.py` | `pos_weight` computation + weighted `BCEWithLogitsLoss` |
| `metrics.py` | Precision/recall/F1/PR-AUC/ROC-AUC/balanced accuracy, threshold selection |
| `calibration.py` | Brier score, Expected Calibration Error, calibration curve data |
| `pipeline.py` | Shared setup (`prepare`/`prepare_from_benchmark`) so train/evaluate/baselines/tests all run the identical pipeline |
| `train.py` | The training loop, early stopping on validation PR-AUC |
| `evaluate.py` | Predictions, metrics, risk ranking, early warning, **`disruption_onset_breakdown`**, plots |
| `baselines.py` | Majority + Logistic Regression, same pipeline |
| `experiment.py` | Run directory management, reproducibility metadata |

**Scripts** — `scripts/run_graphsage_experiment.py` (main entry point:
prepare once, train+evaluate per seed, optional `--baselines`, auto
multi-seed summary), `train_graphsage.py`, `evaluate_graphsage.py`
(re-evaluate a saved run without retraining, reloading its own saved
preprocessing rather than refitting).

**Configs** — `configs/graphsage.yaml` (primary, temporal split),
`configs/graphsage_severity.yaml` (severity generalization).

**Tests** — `tests/conftest.py` (shared fixture: a small in-memory
benchmark built through the real generator/simulation/labels/splits code,
seed=3, chosen for non-degenerate positive rates in all 3 splits) plus
`test_model_data.py`, `test_temporal_features.py`, `test_preprocessing.py`,
`test_hetero_graph.py`, `test_graphsage.py`, `test_training.py`,
`test_evaluation.py`, `test_integration.py` — **71 new tests**, all
exercising real production code paths, not reimplementations.

---

## 4. Issues found and fixed

### 4.1 `feature_audit.csv` comma-joined column parsing
The audit file's `operations/procurement.csv` row lists its allowed
columns as one string, `"quantity_ordered, quantity_fulfilled"`, not two
separate rows. The first `prepare()` smoke test against the real benchmark
failed loudly (`audit_feature_sources` refused to proceed) because the
naive lookup checked each column name against that literal joined string.
Fixed by splitting on `,` before building the lookup. Caught in minutes
because the leakage audit fails closed by design — exactly what it's for.

### 4.2 Dead scenario/severity config fields
`SplitConfig.severity_train_max` and `scenario_test_event_types` looked
like they configured the severity/scenario experiments. They didn't —
`pipeline.py` reads the benchmark's own pre-generated
`splits/{scenario,severity}_split.csv` as-is (correctly, per the plan's
"reuse the existing split files" instruction), so these fields were never
actually consulted. Found while explaining to the user how to run the
severity experiment — would have silently misled anyone who edited
`severity_train_max` expecting a behavior change. Removed the fields and
documented the real (fixed) composition instead: severity holds out
severity>3, scenario holds out `{cyberattack, geopolitical}`.

### 4.3 Multi-seed summary filename collision
`multiseed_summary_seeds_<...>.json`'s name didn't include the split
strategy. Running the severity sweep with the same 5 seeds silently
overwrote the temporal sweep's summary file on disk. Fixed by tagging the
filename with the split strategy (reusing the tag already used for run
directory names); recovered the lost temporal summary by recomputing it
from the 5 already-saved per-seed `metrics.json` files (no retraining
needed) and confirmed it matched the original console output exactly.

### 4.4 Sigmoid overflow warning (cosmetic)
`train.py`'s validation loop computed probabilities as a manual
`1/(1+exp(-x))`, which triggers a harmless `RuntimeWarning: overflow
encountered in exp` for large-magnitude logits (result is still correctly
`0.0`, just noisy). Replaced with `torch.sigmoid`. No effect on any
reported number.

---

## 5. The central finding: persistence detection, not early warning

This is the most important result of this phase, and it wasn't something
the plan anticipated — it emerged from the severity-generalization run's
behavior looking too clean to be an accident.

**What was observed:** running the severity-generalization experiment
across 5 independently-initialized-and-trained models produced *bit-
identical* test recall (`0.4406000...`) in every single one.

**Why that's suspicious:** five separately trained models agreeing to
machine precision on a continuous metric doesn't happen from normal
model variance. It means something structural, not the models
themselves, is determining the outcome.

**What investigating it found:** all 5 models flag the *exact same* 360
(supplier, week) examples as positive, out of 817 actual positives in the
test split. Splitting those 817 by whether the supplier was **already**
visibly disrupted at the prediction time (`supplier_disrupted[t]==1`, read
from the raw label table only for this diagnostic, never as a model
input) versus not yet disrupted (a genuine fresh onset still ahead):

| | count | recall |
|---|---:|---:|
| Already disrupted at `t` | 733 | 49.1% |
| Fresh onset (nothing wrong yet at `t`) | 84 | **0.0%** |

Every model gets roughly half of the "is this still going to be broken"
cases right, and **none** of the "is something about to break" cases —
and it's not a threshold issue: fresh-onset positives get a mean predicted
probability of 0.0000 (max 0.0001), statistically indistinguishable from
an ordinary negative example. Isolating fresh-onset positives against
negatives and computing ROC-AUC on that pair alone gives 0.41–0.53 — at or
below chance.

**The same pattern was confirmed in the primary (temporal-split) run,
too** — its validation split (which has 40 fresh onsets, unlike its test
split, which happens to have zero) also shows exactly 0% recall on them.
Cross-checking the "early warning" analysis (`early_warning.csv`) that had
been reported as "22 of 46 disruptions warned about 4 weeks early"
revealed all 22 were **training-split** onsets warned about by
**training-split** predictions — i.e. the model recognizing patterns it
was directly fit to, not a validated generalization result. Zero of the
10 validation-split onsets were ever warned about in advance.

**Why, mechanistically:** `events/base.py`'s `event_intensity` function
applies an event's capacity/lead-time effects as a hard step function —
exactly `0.0` (no effect at all) for every period before the event's
`start_time`, with no ramp-up. So the simulator's operational data
(orders, deliveries, inventory, production) contains, by construction, no
leading indicator of an upcoming disruption in the weeks before it starts.
Two concrete examples from the real data illustrate this directly:

- `supplier_279`, predicted at week 73: `fulfillment_ratio` was healthy
  (~0.80) through week 65, crashed to 0.24 the moment the severity-5 event
  hit at week 67, and was still depressed (~0.58) at week 73 — six weeks
  of visible, legitimate (non-leaked) evidence. Model: risk=0.845, correct.
- `supplier_36`, predicted at week 82: `fulfillment_ratio` was a flat,
  perfect 1.0 every week from 78 through 86. Its disruption doesn't start
  until week 86 — 4 weeks after the prediction, the very edge of the
  window. At week 82 there is nothing anywhere to distinguish this
  supplier from a permanently healthy one. Model: risk≈0.0000000049, wrong.

**What this means for interpreting the headline numbers:** the reported
96.3% test recall on the primary run is real, but it measures "will this
visible disruption still be visible in 4 weeks," not "will a new one
start" — a materially easier and different claim than the plan's "early
prediction of future supplier disruptions" framing. This appears to be a
property of the benchmark's event-engine design (no model, classical or
quantum, could learn a genuine forward precursor from dynamic operational
features alone, since none exists in the data-generating process) rather
than a GraphSAGE-specific shortfall.

**What was done about it, permanently:** `evaluate.py::
disruption_onset_breakdown` now runs on every experiment automatically,
saving `onset_breakdown.json` alongside every other artifact, so this
distinction is checked by default going forward — including whichever
future QGNN run needs to be compared fairly against this baseline.

---

## 6. Results

### 6.1 Primary experiment — temporal split
Dataset: `scm_v1_black_swan_seed43`. Target: `supplier_disrupted`.
Horizon: 4 weeks. Split: train=[0,73), validation=[73,88), test=[88,104).
26,700 examples (300 suppliers × 89 usable weeks).

| split | n | positive | negative | positive rate |
|---|---:|---:|---:|---:|
| train | 18,600 | 481 | 18,119 | 2.59% |
| validation | 4,500 | 365 | 4,135 | 8.11% |
| test | 3,600 | 242 | 3,358 | 6.72% |

Test-split results (threshold = 0.5):

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Majority | 0.067 | 0.500 | 0.000 | 0.000 | 0.000 | 0.064 |
| Logistic Regression | 0.304 | 0.821 | 0.181 | 0.632 | 0.282 | 0.145 |
| **GraphSAGE (mean ± std, 5 seeds)** | **0.807 ± 0.066** | **0.987 ± 0.008** | **0.522 ± 0.042** | **0.968 ± 0.016** | **0.677 ± 0.035** | **0.042 ± 0.012** |

Per-seed test PR-AUC: 0.787 (42), 0.689 (43), 0.874 (44), 0.849 (45), 0.835 (46).
Best epoch varied 11–19 across seeds; early stopping (patience 10) fired
in all 5 runs.

**Read with §5's caveat**: the test split's 242 positives are entirely
"already-ongoing" cases (the one severity-5 event dominates that window),
so this recall is persistence detection, not early warning.

### 6.2 Severity generalization — train severity 1-3, test severity 4-5
Same dataset/target/horizon/architecture; only the split differs
(`splits/severity_split.csv`, joined via each example's target window).
26,700 examples, redistributed: 13,200 train / 2,400 validation / 11,100
test (the one severity-5 event's long recovery tail dominates the back
half of the timeline, making "test" here much larger than "validation").

| split | n | positive | negative | positive rate |
|---|---:|---:|---:|---:|
| train | 13,200 | 191 | 13,009 | 1.45% |
| validation | 2,400 | 80 | 2,320 | 3.33% |
| test | 11,100 | 817 | 10,283 | 7.36% |

Test-split results (threshold = 0.5, mean ± std over 5 seeds):

| Metric | Value |
|---|---:|
| PR-AUC | 0.449 ± 0.013 |
| ROC-AUC | 0.727 ± 0.016 |
| F1 | 0.411 ± 0.020 |
| Precision | 0.387 ± 0.036 |
| Recall | 0.4406 ± 0.0000 |
| Balanced accuracy | 0.692 ± 0.004 |
| Brier | 0.086 ± 0.008 |

Substantially weaker than the temporal split, as expected for a genuine
severity-holdout generalization test — and, per §5, the recall here is
exclusively on already-ongoing cases (49.1%), with 0% on the 84 fresh
onsets in every one of the 5 seeds.

### 6.3 Reproducibility
Both experiments have now been run **three independent times** and match
to full floating-point precision every time:
1. This session's original run (temporal seed 42: `pr_auc=0.787372691381994`).
2. This session's 5-seed sweep, same seed (`pr_auc=0.787372691381994`).
3. **The user's own independent rerun from a fresh terminal**, both the
   temporal sweep and the severity sweep — every single per-seed and
   aggregate number matches this session's runs exactly (e.g. temporal
   seed 42: `pr_auc=0.787372691381994`; severity multi-seed recall:
   `0.4406 +/- 0.0000` both times). No CUDA involved in any run.

This is strong, direct evidence the pipeline is genuinely deterministic
given a fixed seed/config, on CPU, across process invocations and machine
sessions.

---

## 7. Test suite

161 tests passing (90 original dataset-framework tests, unchanged and
still green, + 71 new). New tests cover: benchmark loading and schema
validation, the leakage audit (including a regression test that injects a
future record and asserts a past feature value is unaffected), target/
temporal alignment, preprocessing (train-only fit, missing-value handling,
categorical encoding, save/load), heterogeneous graph construction (node
id determinism, edge validation), the GraphSAGE model (forward pass, no
internal sigmoid — checked architecturally, not by numeric luck),
training (`pos_weight` computed from train only, early stopping restores
the best-val-PR-AUC checkpoint not the last epoch, never builds a
test-split graph snapshot), evaluation/metrics/calibration (hand-computed
formulas, threshold selection observed to touch only validation rows,
safe single-class handling), `disruption_onset_breakdown`, and two
integration tests (an in-process run, and a subprocess invocation of the
actual CLI script against a tiny on-disk benchmark).

---

## 8. Definition of Done (plan §91) — status

Every item is satisfied: benchmark used without modification, schema and
feature audit inspected, target/horizon documented and configurable,
leakage checks implemented and tested, train-only preprocessing, all 6
node types and 9 edge types handled, 2-layer GraphSAGE with mean
aggregation and type-specific projections, weighted `BCEWithLogitsLoss`,
validation-PR-AUC early stopping, full metrics/calibration/plots/risk
ranking/run metadata saved, multiple seeds run, Majority and Logistic
Regression baselines run, new tests added and all existing tests still
pass, one (in fact several) complete experiments succeed, no leakage
detected, and reproducibility empirically confirmed three times over.

**Not required by the plan but completed anyway:** the severity
generalization experiment (plan sequences this as optional, "after" a
stable baseline) and the onset-breakdown analysis that came out of it.

**Genuinely optional and not started** (plan explicitly sequences these
last, §70/§77/§80): scenario generalization experiment, Ablations B–E
(Ablation A is covered by the Logistic Regression baseline), a robustness
rerun on `scm_v1_black_swan_seed44`, and formally declaring the
configuration frozen before the QGNN stage begins.

---

## 9. Where everything lives

- Detailed per-experiment reports (full methodology, all numbers,
  limitations): `experiments/classical_gnn/<run>/experiment_report.md`
  (primary temporal run and severity run each have one). This directory
  is gitignored (regenerable — see §6.3) but present locally.
- Living status / how-to-resume: `GRAPHSAGE_PROGRESS.md` (tracked in git).
- This file: a one-time detailed account, not meant to be kept updated —
  see `GRAPHSAGE_PROGRESS.md` for current status instead.
- Reproduce everything: `.venv/bin/python scripts/run_graphsage_experiment.py
  --config configs/graphsage.yaml --seeds 42,43,44,45,46 --baselines` (primary)
  and the same with `configs/graphsage_severity.yaml` (severity, no `--baselines`).
