# GraphSAGE Improvement Plan — Status

Companion to `GRAPH_SAGE_IMPROVEMENT_PLAN.md`. That plan's own §19
"Recommended Priority" governs the order of work here: Priority 1 (fresh-
onset vs persistence) and Priority 2 (feature-information ablations) are
covered below; everything after is not started.

## Priority 1 — Fresh-onset vs persistence evaluation: already done

This was built in the prior phase (`disruption_onset_breakdown` in
`evaluate.py`, `onset_breakdown.json` on every run) before this plan
existed. Extended this session to also report **ranking quality**
(`pr_auc_fresh_onset`, `roc_auc_fresh_onset`, and the `_already_ongoing`
counterparts), not just recall — needed for the plan's §14 ablation table
columns. Also wired into the Majority/Logistic Regression baselines
(`baselines.py`), which didn't have it before.

## Priority 2 — Feature-information ablations: infrastructure built, not yet run

**A genuine ambiguity in the plan, resolved and documented**: §14's
ablation table lists both "Static-only GraphSAGE" and "Static + Graph
Structure" as separate rows, but §6 doesn't spell out the difference.
Interpretation used (see `features.py::apply_feature_mode`'s docstring for
the full reasoning):
- `dynamic_only` / `static_only` / `region_risk_only`: restrict only the
  **supplier** node's own features to that group; every other node type
  keeps its full feature set. Isolates what the supplier's own information
  contributes, while its neighbors can still be fully informative via
  message passing.
- `static_plus_graph`: restricts **every** node type to static-only —
  zero dynamic information anywhere in the graph. A stricter test of
  whether structure over purely static information carries signal.
- `full`: unchanged, the existing frozen benchmark.

`region_risk_only` uses `{geopolitical_exposure, disaster_exposure,
cyber_exposure}` as the supplier's own "risk exposure" fields (plan's own
example list under B3), not `criticality` (an importance measure, not an
exposure variable).

### What was built
- `config.features.feature_mode` (`full` | `dynamic_only` | `static_only` |
  `region_risk_only` | `static_plus_graph`), read by `pipeline.py` right
  after `build_feature_frames`.
- `scripts/run_graphsage_experiment.py --feature-mode <mode>` — no need
  for 5 separate config files.
- Run directories and multi-seed summaries now tag by feature mode too
  (alongside the existing split-strategy tag), so ablation runs don't
  collide with each other or with the primary runs.
- Per-seed console output now prints the onset breakdown (fresh-onset vs
  already-ongoing recall/PR-AUC/ROC-AUC) directly, not just overall
  metrics — this is the number that actually matters per this plan.
- `scripts/build_ablation_matrix.py`: reads a set of already-saved run
  directories (no retraining) and assembles the exact §14 table (one row
  per feature mode/baseline, averaged across however many seeds you pass
  in), writing both markdown and CSV. Now also reads `temporal_variation.
  json` and flags a row's label if the model appears to have degenerated
  to a fixed per-supplier score (see below).
- `evaluate.py::temporal_variation_summary` (`temporal_variation.json` on
  every run): the fraction of suppliers whose `risk_probability` varies at
  all across their own prediction times. Added after the finding below.
- 20 new tests total across this phase. **Full suite: 180/180 passing.**

### The sweep was run — and it revealed two real problems, not just results

The user ran all 4 feature modes (5 seeds each) against `configs/
graphsage.yaml` (the **temporal** split). Two things went wrong with that
choice, both now fixed/documented, not swept under the rug:

**Problem 1 — wrong split for the question being asked.** The temporal
split's test window is dominated by one long-running severity-5 event, so
it structurally contains **zero fresh-onset examples** for every feature
mode. The whole point of this ablation (does static/region-risk info
predict a disruption *before* it starts?) cannot be answered on a test set
that has no such examples in it. Every mode's `Fresh-Onset *` columns come
back `n/a`. **This was a mistake in how the sweep was set up (my error,
correcting it here)** — the severity split (`configs/graphsage_severity.
yaml`) has 84 fresh onsets in its test split and is the one that can
actually test this.

**Problem 2 — the `static_plus_graph` mode is structurally degenerate.**
It scored 0.933 PR-AUC (beating the full model's 0.807) with **exactly
zero variance across 5 independently-seeded runs** — precisely the kind
of too-clean result that turned out to be real trouble the last time it
showed up (recall §5's severity finding). Checked directly: every one of
300 suppliers gets the *identical* `risk_probability` at *every one* of
its ~89 prediction times. Once every node type in the graph is
static-only, nothing anywhere in the model's input varies from one week
to the next (no week-index feature stands in for time), so the model can
only learn a fixed per-supplier score — which then scores very well on a
test window dominated by one long-running event (rewarding "is this one
of the ~11 historically troubled suppliers") without doing anything
resembling temporal prediction. `temporal_variation_summary` now catches
this automatically and `build_ablation_matrix.py` flags it in the table.
**`static_plus_graph` should be treated as retired/uninformative** — it's
kept in the code for transparency (it's a real, run, plan-specified
variant) but `static_only` (restrict only the supplier; neighbors keep
their time-varying features, which still reach the supplier via message
passing) is the mode that actually answers "does graph structure over
static information carry signal," without this degeneracy.

**What the run DID legitimately show** (persistence-detection comparison
only, since fresh-onset is unanswerable on this split): static features
contribute much more than dynamic ones to persistence detection
(static-only 0.818 PR-AUC vs dynamic-only 0.583), region-risk alone gives
a middling result (0.633), and the full model (0.807) scores slightly
*below* static-only — suggestive that the dynamic operational features may
currently add more noise than signal for this specific task, consistent
with every training run in the sweep showing validation loss rising while
training loss keeps falling (classic overfitting). This is a real,
usable finding, just not the fresh-onset finding the sweep set out to get.

### Actual next step
Rerun the three non-degenerate modes against the **severity** split,
which has fresh onsets to actually measure against:

```bash
.venv/bin/python scripts/run_graphsage_experiment.py --config configs/graphsage_severity.yaml --feature-mode dynamic_only --seeds 42,43,44,45,46
.venv/bin/python scripts/run_graphsage_experiment.py --config configs/graphsage_severity.yaml --feature-mode static_only --seeds 42,43,44,45,46
.venv/bin/python scripts/run_graphsage_experiment.py --config configs/graphsage_severity.yaml --feature-mode region_risk_only --seeds 42,43,44,45,46
# "full" (severity) already exists from the prior phase -- no need to rerun.
# static_plus_graph is not worth rerunning -- see "Problem 2" above.
```

Watch the console output for the `WARNING: ... likely time-invariant`
line — if any of these three unexpectedly trip it too, stop and
investigate before trusting the numbers, the same way `static_plus_graph`
was caught.

**What this will actually answer** (plan §2 Q1/Q2): whether
`region_risk_only` recovers ANY fresh-onset ranking signal above chance —
the direct, cheap test of whether background risk exposure alone can hint
at an upcoming disruption before any operational symptom appears, which
the prior phase's one-off manual check (ROC-AUC ~0.41–0.53 on the full
model) left unresolved as to whether the signal exists but is unused, or
genuinely doesn't exist.

## Not started (plan §19 priorities 3-8)

Structural dependency features (Phase C), temporal representation
improvements (Phase D), architecture tuning (Phase E), training tuning
(Phase F), threshold selection beyond what already exists (Phase G,
partially done — `select_threshold` already supports `f1_optimal`/
`recall_constrained`/`precision_constrained`), calibration correction
(Phase H, evaluation already exists — Platt/isotonic correction does not),
scenario/dataset-seed robustness (Phase I2-I4; temporal and severity are
already done from the prior phase), interpretation (Phase K), and optional
explainability (§16).
