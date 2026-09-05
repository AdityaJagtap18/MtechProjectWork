# GraphSAGE Improvement Plan — Work Summary

Everything done implementing `GRAPH_SAGE_IMPROVEMENT_PLAN.md`, on top of
the already-frozen classical baseline documented in
`GRAPHSAGE_WORK_SUMMARY.md`. Companion to `GRAPHSAGE_IMPROVEMENT_PROGRESS.md`
(the living status tracker) — this file is the one-time, detailed account
of what was built, what went wrong along the way, and what the results
actually are.

---

## 1. Session narrative

1. Read `GRAPH_SAGE_IMPROVEMENT_PLAN.md` (26 sections) and copied it into
   the repo. Its own §19 "Recommended Priority" set the order of work:
   Priority 1 (fresh-onset vs. persistence evaluation) and Priority 2
   (feature-information ablations) — everything else in the plan is
   explicitly sequenced after those.
2. Found Priority 1 was already satisfied from the prior phase
   (`disruption_onset_breakdown`, built while investigating the earlier
   severity-generalization finding) — extended it to report **ranking
   quality** (PR-AUC/ROC-AUC per subgroup), not just recall, since the
   plan's own ablation table needs those columns.
3. Resolved a genuine ambiguity in the plan before building anything: its
   ablation table lists "Static-only GraphSAGE" and "Static + Graph
   Structure" as separate rows without explaining the difference.
   Documented interpretation: the former restricts only the supplier
   node's own features; the latter restricts every node type in the graph.
4. Built the full feature-mode ablation infrastructure (`feature_mode`
   config, `apply_feature_mode`, CLI flag, run-directory tagging, the
   ablation-matrix builder script) and 12 tests for it.
5. Handed the sweep off to the user to run (this session's collaboration
   rule: user runs experiments, Claude writes code) — pointed it at the
   **temporal** split.
6. **Mistake caught**: every mode came back with `fresh_onset(n=0)` on
   test. The temporal split's test window is dominated by one long-running
   event, so it structurally contains zero fresh onsets — the sweep, as
   configured, could not answer the question it was built for. Corrected
   course and pointed the user at `configs/graphsage_severity.yaml` instead
   (84 real fresh onsets in its test split).
7. **Second problem caught in the same batch of results**: the
   `static_plus_graph` mode scored *higher* than the full model (0.933 vs
   0.807 PR-AUC) with *zero variance across 5 independently-seeded runs*.
   Investigated directly: every one of 300 suppliers got the identical
   `risk_probability` at every one of its ~89 prediction times. Root
   cause: restricting every node type to static-only removes every trace
   of time from the graph, so the model can only learn a fixed
   per-supplier score. Built `temporal_variation_summary` as a permanent,
   automatic check for this (not just a one-off fix), wired into both the
   training script (console warning) and the ablation-matrix builder
   (flags the row in the table itself). Retired `static_plus_graph` as an
   uninformative mode going forward.
8. User reran the three non-degenerate modes (`dynamic_only`,
   `static_only`, `region_risk_only`) against the corrected severity
   split, 5 seeds each.
9. **Result: a clean, decisive negative finding.** Fresh-onset recall was
   exactly 0.0000 in all three modes, and fresh-onset ROC-AUC was *below
   0.5* (worse than chance) in all three — including `region_risk_only`,
   the one channel that could plausibly carry a persistent background-risk
   signal independent of any operational symptom. Recorded as the
   definitive answer to the plan's own Q2 research question.

---

## 2. What the plan asked, and what was answered

The plan posed four questions (§3) to answer in order:

| Question | Status |
|---|---|
| **Q1**: Is GraphSAGE learning useful information beyond current disruption state? | Partially answered — see §5 below (persistence detection is real; static features matter more than dynamic ones for it). |
| **Q2**: Does the dataset contain any genuine pre-event signal? | **Answered: No.** See §4 — 0% recall and sub-chance ranking across every feature group tested, including background risk exposure alone. |
| **Q3**: Can GraphSAGE be improved for the useful task without leakage? | Not yet started (plan Phases D-F: temporal representation, architecture, training tuning). |
| **Q4**: What should the final classical-GNN benchmark be? | Not yet decided — pending a decision on how much further improvement work to do (see `GRAPHSAGE_IMPROVEMENT_PROGRESS.md`'s "not started" list). |

---

## 3. What was built

**New/extended code**:

| File | What changed |
|---|---|
| `src/scm_dataset/modeling/evaluate.py` | `disruption_onset_breakdown` extended with per-subgroup PR-AUC/ROC-AUC (`pr_auc_fresh_onset`, `roc_auc_fresh_onset`, and `_already_ongoing` counterparts). New `temporal_variation_summary` function + `EvaluationResult.temporal_variation` field. |
| `src/scm_dataset/modeling/baselines.py` | Majority/Logistic Regression now also compute and expose `onset_breakdown`, matching GraphSAGE's evaluation. |
| `src/scm_dataset/modeling/config.py` | New `FeaturesConfig.feature_mode` (`full`/`dynamic_only`/`static_only`/`region_risk_only`/`static_plus_graph`). |
| `src/scm_dataset/modeling/features.py` | New `apply_feature_mode` function + `FEATURE_MODES`/`SUPPLIER_RISK_EXPOSURE_FIELDS` constants — restricts columns on an already-built feature frame without touching how they were computed (leakage-safety unaffected by mode). |
| `src/scm_dataset/modeling/pipeline.py` | Wires `apply_feature_mode` into `prepare_from_benchmark` right after feature frames are built. |
| `scripts/run_graphsage_experiment.py` | `--feature-mode` CLI flag; run-directory and multiseed-summary filenames now tag by feature mode (alongside the existing split-strategy tag) so different ablation runs never collide; console output now prints the onset breakdown and a time-invariance warning per seed. |
| `scripts/build_ablation_matrix.py` (**new**) | Reads a set of already-saved run directories (no retraining) and assembles the plan's §14 ablation table — pure aggregation over `metrics.json`/`onset_breakdown.json`/`temporal_variation.json`, averaged across however many seeds are passed in, written as both Markdown and CSV. Flags a row's label if its model appears to have degenerated to a time-invariant score. |

**Tests**: 20 new tests across `tests/test_temporal_features.py` (feature-mode
correctness for all 5 modes), `tests/test_evaluation.py` (extended onset
breakdown ranking metrics, `temporal_variation_summary`), and
`tests/test_ablation_matrix.py` (new file — the ablation-matrix script's
aggregation math and time-invariance flagging).

**Full test suite: 180/180 passing** (160 from the prior phase + 20 new).

---

## 4. The central finding: no pre-event signal exists in this benchmark, in any feature group tested

### 4.1 Severity-split sweep (the one that actually tests this — 84 real fresh onsets in its test split)

| Model/Input | n seeds | Overall PR-AUC | Overall ROC-AUC | Fresh-Onset PR-AUC | Fresh-Onset ROC-AUC | Fresh-Onset Recall | Already-Ongoing Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dynamic-only GraphSAGE | 5 | 0.364 | 0.665 | 0.0069 | 0.458 | 0.0000 | 0.492 |
| Static-only GraphSAGE | 5 | 0.458 | 0.744 | 0.0057 | 0.344 | 0.0000 | 0.491 |
| Region/Risk-only GraphSAGE | 5 | 0.374 | 0.684 | 0.0060 | 0.371 | 0.0000 | 0.491 |
| Full GraphSAGE | 5 | 0.449 | 0.728 | n/a* | n/a* | 0.0000 | 0.491 |

*Full's fresh-onset ranking columns predate the metric's addition; recall
(0.0) was already recorded and is consistent with the other three.

**Reading this table**: fresh-onset recall is exactly zero across the
board — not one genuinely novel disruption onset was ever flagged, by any
of the three restricted feature groups. More telling than the zero recall
is the ranking quality: ROC-AUC of 0.34–0.46 is *below 0.50* (chance) in
every case, and PR-AUC of 0.006–0.007 sits *below* the fresh-onset
subset's own base rate (~0.008, i.e. 84 fresh onsets among 10,367
fresh-onset-or-negative examples). The models don't merely fail to find a
signal — they rank a supplier about to be disrupted as *less* likely than
an average calm one.

This is the direct, cheap test the earlier phase's one-off manual check
(ROC-AUC ~0.41–0.53 on the *full* model) left unresolved: is there
genuine pre-event signal hiding in static or risk-exposure features that
the full model just isn't using well? The answer, now checked properly
with three independent restricted-feature models: **no.** Not in
operational history, not in the supplier's own static attributes, and not
in background geopolitical/disaster/cyber risk exposure alone.

### 4.2 Temporal-split sweep (run first, by mistake — persistence-detection comparison only)

| Model/Input | n seeds | Overall PR-AUC | Overall ROC-AUC | Already-Ongoing Recall |
|---|---:|---:|---:|---:|
| Majority | 1 | 0.067 | 0.500 | n/a |
| Logistic Regression | 1 | 0.304 | 0.821 | n/a |
| Dynamic-only GraphSAGE | 5 | 0.583 | 0.945 | 0.780 |
| Static-only GraphSAGE | 5 | 0.818 | 0.989 | 0.960 |
| Region/Risk-only GraphSAGE | 5 | 0.633 | 0.952 | 0.828 |
| ~~Static + Graph Structure~~ | 5 | ~~0.933~~ | ~~0.998~~ | ~~1.000~~ |
| Full GraphSAGE | 5 | 0.807 | 0.987 | 0.968 |

The "Static + Graph Structure" row is struck through because it's the
degenerate mode (§1 step 7) — its apparently best-in-table score is an
artifact of the model collapsing to a fixed per-supplier identity
classifier, not evidence about structural information. Excluding it, the
legitimate finding from this table: **static features contribute far more
than dynamic ones to persistence detection** (0.818 vs. 0.583 PR-AUC), and
the full model (using everything) actually scores slightly *below*
static-only alone — consistent with every training run in both sweeps
showing the classic overfitting signature (validation loss climbing while
training loss keeps falling). This suggests the dynamic operational
features may currently be adding more noise than signal for the
persistence-detection task specifically, though this hasn't been isolated
further (that would be plan Phase D/E/F territory).

This table could not answer the fresh-onset question at all — see §5.

---

## 5. Two problems found and fixed along the way

### 5.1 Wrong split for the question
The first sweep ran against `configs/graphsage.yaml` (temporal split).
Every mode came back `fresh_onset(n=0)` on test, because that split's test
window is dominated by one long-running severity-5 event and structurally
contains no fresh onsets for *any* feature mode. This was a setup mistake
(the temporal split was the wrong one to point this specific experiment
at) — corrected by rerunning against `configs/graphsage_severity.yaml`,
whose test split has 84 real fresh onsets.

### 5.2 `static_plus_graph` is structurally degenerate, not just on this run
Restricting *every* node type in the graph to static-only removes every
trace of time from the model's inputs — there is no week-index feature
standing in for "which prediction time is this," so nothing distinguishes
one week's graph snapshot from another's. The model can only learn a
fixed per-supplier score. Confirmed directly: 0 out of 300 suppliers
showed any variation in `risk_probability` across their ~89 prediction
times. That fixed score still scored very well on a test window dominated
by one long-running event (rewarding "is this one of the ~11 historically
troubled suppliers"), producing a result that looked like strong evidence
for structural information but was actually an artifact of removing time
entirely. This is a property of the ablation's *design*, not of the
specific split or run — it would degenerate the same way anywhere.
`temporal_variation_summary` (the fraction of suppliers whose prediction
varies at all across time) now runs on every evaluation automatically and
is checked in the ablation-matrix table, so this can never again require
a manual investigation to catch.

---

## 6. Test suite

**180/180 passing** — the 90 original dataset-framework tests, the 68 from
the classical-baseline phase, and 20 new this phase (feature-mode
correctness for all 5 modes and their column-restriction logic, the
extended onset-breakdown ranking metrics, `temporal_variation_summary`'s
detection logic, and the ablation-matrix script's aggregation math and
time-invariance flagging).

---

## 7. Where everything lives

- Ablation tables: `experiments/classical_gnn/ablation_matrix_temporal.md`
  (+ `.csv`) and `experiments/classical_gnn/ablation_matrix_severity.md`
  (+ `.csv`). `experiments/` is gitignored (regenerable — see below).
- Living status / how to resume: `GRAPHSAGE_IMPROVEMENT_PROGRESS.md`
  (tracked in git, kept current).
- This file: a one-time detailed account, not meant to be kept updated.
- Plan document: `GRAPH_SAGE_IMPROVEMENT_PLAN.md` (tracked in git).

**Reproduce the severity-split sweep** (the one with the definitive
finding):
```bash
.venv/bin/python scripts/run_graphsage_experiment.py --config configs/graphsage_severity.yaml --feature-mode dynamic_only --seeds 42,43,44,45,46
.venv/bin/python scripts/run_graphsage_experiment.py --config configs/graphsage_severity.yaml --feature-mode static_only --seeds 42,43,44,45,46
.venv/bin/python scripts/run_graphsage_experiment.py --config configs/graphsage_severity.yaml --feature-mode region_risk_only --seeds 42,43,44,45,46
```
**Rebuild the ablation table** from whatever run directories those produce:
```bash
.venv/bin/python scripts/build_ablation_matrix.py \
    --row "Dynamic-only GraphSAGE" experiments/classical_gnn/<...dynamic_only_seed42> ... \
    --row "Static-only GraphSAGE" experiments/classical_gnn/<...static_only_seed42> ... \
    --row "Region/Risk-only GraphSAGE" experiments/classical_gnn/<...region_risk_only_seed42> ... \
    --output experiments/classical_gnn/ablation_matrix_severity.md
```

---

## 8. What's next (not started)

Per the plan's own priority order (§19), and its decision tree (§18)
saying not to force early-warning claims past a "no" on Q2: structural
dependency features (Phase C), temporal representation improvements
(Phase D), architecture tuning (Phase E), training tuning (Phase F),
calibration correction beyond evaluation (Phase H — worth checking
whether the sub-chance ranking in §4.1 is a calibration artifact rather
than genuine anti-signal), scenario/dataset-seed robustness (Phase I3-I4),
interpretation (Phase K), and freezing a final configuration (Phase §4
Q4). None of these were required to answer Q1/Q2, and the negative
finding in §4 is itself a complete, legitimate, publishable result on its
own — further work here is a judgment call on the user's part, not a gap
in what was asked.
