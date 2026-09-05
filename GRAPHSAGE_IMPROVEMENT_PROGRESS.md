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
  in), writing both markdown and CSV.
- 12 new tests (`apply_feature_mode` correctness for all 5 modes, the
  extended onset breakdown's ranking metrics, the ablation matrix script's
  aggregation math). **Full suite: 173/173 passing.**

### What's NOT done — this is where to run next
The infrastructure is ready; the actual sweep hasn't been run. To produce
the real §14 table:

```bash
# One command per feature mode, 5 seeds each (~5 min per mode on CPU):
.venv/bin/python scripts/run_graphsage_experiment.py --config configs/graphsage.yaml --feature-mode dynamic_only --seeds 42,43,44,45,46
.venv/bin/python scripts/run_graphsage_experiment.py --config configs/graphsage.yaml --feature-mode static_only --seeds 42,43,44,45,46
.venv/bin/python scripts/run_graphsage_experiment.py --config configs/graphsage.yaml --feature-mode region_risk_only --seeds 42,43,44,45,46
.venv/bin/python scripts/run_graphsage_experiment.py --config configs/graphsage.yaml --feature-mode static_plus_graph --seeds 42,43,44,45,46
# "full" already exists from the prior phase's runs -- no need to rerun.
```

Then build the table (substitute the actual timestamped directories each
sweep prints):

```bash
.venv/bin/python scripts/build_ablation_matrix.py \
    --row "Majority" experiments/classical_gnn/<majority_run> \
    --row "Logistic Regression" experiments/classical_gnn/<logreg_run> \
    --row "Dynamic-only GraphSAGE" experiments/classical_gnn/*_hetero_graphsage_dynamic_only_seed* \
    --row "Static-only GraphSAGE" experiments/classical_gnn/*_hetero_graphsage_static_only_seed* \
    --row "Region/Risk-only GraphSAGE" experiments/classical_gnn/*_hetero_graphsage_region_risk_only_seed* \
    --row "Static + Graph Structure" experiments/classical_gnn/*_hetero_graphsage_static_plus_graph_seed* \
    --row "Full GraphSAGE" experiments/classical_gnn/*_hetero_graphsage_seed* \
    --output experiments/classical_gnn/ablation_matrix.md
```

(Shell globs expand before Python sees them, so the `--row` glob patterns
above need to actually match only the intended run directories — check
with `ls` first if `full` and `dynamic_only` directory names could
otherwise collide; they won't here since `_strategy_tag` inserts the mode
name into the directory name itself.)

**What this will actually answer** (plan §2 Q1/Q2): whether the model's
persistence-detection ability (already established as strong) survives
when the supplier's own dynamic features are removed (`static_only`) —
if yes, that points to graph-neighbor information carrying the signal
instead of the supplier's own history. And whether `region_risk_only`
recovers ANY fresh-onset ranking signal above chance — this is the
direct, cheap test of whether background risk exposure alone can hint at
an upcoming disruption before any operational symptom appears, which the
prior phase's manual check (ROC-AUC ~0.41–0.53 in the full model) left
unresolved as to whether the signal exists but is unused, or doesn't
exist at all.

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
