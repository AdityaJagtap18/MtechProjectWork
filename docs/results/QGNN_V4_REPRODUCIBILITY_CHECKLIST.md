# QGNN-v4 — Final Reproducibility Checklist

Each item marked **VERIFIED** was directly re-checked against a live
command or a saved artifact during this validation pass — not assumed.
**PARTIALLY VERIFIED** means some but not all of the claim was checked.
**NOT AVAILABLE** means the item was not tracked and is stated as such,
not estimated.

| Item | Status | Detail |
|---|---|---|
| Repository commit | **VERIFIED** | `211286b939dcfdcb0f56fad5d937937e46837bcd` (re-checked live via `git rev-parse HEAD` during this validation pass; will advance with this validation's own commit) |
| Branch | **VERIFIED** | `qgnn-package-restructure` (`git branch --show-current`) |
| Python version | **VERIFIED** | 3.12.3 (`python3 --version`, run live in the project's `.venv`) |
| PyTorch version | **VERIFIED** | 2.14.0+cu130 (`torch.__version__`, run live) |
| PennyLane version | **VERIFIED** | 0.45.1 (`qml.__version__`, run live) |
| CUDA status | **VERIFIED** | `torch.cuda.is_available()` returns `True` on the development machine, but **not used** — every quantum-circuit run in this project uses `default.qubit` (CPU simulator), and classical PyTorch tensors also ran on CPU throughout, for consistency across every phase. This is a deliberate scope choice, not an oversight. |
| Simulator / backend | **VERIFIED** | `default.qubit`, `diff_method="backprop"`, confirmed in every `quantum_resource_summary.json` produced by every training run in this project (`"backend": "default.qubit"`) |
| Dataset | **VERIFIED** | `scm_v1_black_swan_seed43`, `data/benchmark/scm_v1_black_swan_seed43/` on disk; graph/event statistics re-verified directly from the raw CSVs during Phase 4 Stage 1 (2,670 nodes, 7,675 edges, 6 events, confirmed again during this validation pass by reading the same source files) |
| Split definitions | **VERIFIED** | `src/scm_dataset/benchmark/splits.py`: `temporal_split` (primary, 70/15/15 by period), `severity_split` (train severity ≤3, test severity 4–5, `train_max_severity=3` default) — read directly from source during this validation |
| Seeds | **VERIFIED** | 42, 43, 44, 45, 46 for every "full" (`seed_type=full`) row in `QGNN_V4_MASTER_RESULTS.csv`; 42, 43 only for every row marked `pilot` — the seed_type column itself is the audit trail, not a separate claim |
| Configuration files | **VERIFIED** | `configs/qgnn_v4.yaml` (primary), `configs/qgnn_v4_severity.yaml` (severity), `configs/graphsage.yaml`/`graphsage_severity.yaml` (classical encoder) — all present in the repository, confirmed via `ls configs/` during this validation |
| Experiment commands | **VERIFIED** | Every phase report (`docs/phases/`, `docs/results/`) includes the exact `scripts/run_qgnn_v4_experiment.py` invocation used, with every non-default flag shown explicitly (e.g. `--head-variant layernorm_noaffine --ansatz hardware_efficient_ring --output-subdir phase4_stage5/hardware_ring`) |
| Output locations | **VERIFIED** | `experiments/qgnn_v4/` (QGNN runs, gitignored), `experiments/classical_gnn/` (encoder + classical baseline, gitignored) — 91 GraphSAGE-Full checkpoint directories confirmed present via `find` during this validation. Raw experiment outputs are not committed to git (by design, per this project's `.gitignore`); the committed record is the reports, scripts, and consolidated CSVs. |
| Metric calculation | **VERIFIED** | Single shared implementation throughout: `src/scm_dataset/modeling/metrics.py::compute_classification_metrics` (PR-AUC, ROC-AUC, F1, precision, recall, balanced accuracy, Brier, confusion matrix) plus `_analysis_common.py::mcc_from_confusion`/`specificity_from_confusion` and `calibration.py::expected_calibration_error` — every phase's numbers trace to these same functions, not a per-phase reimplementation |
| Model checkpoints | **PARTIALLY VERIFIED** | Checkpoints exist on disk for every run this validation spot-checked (91 GraphSAGE-Full checkpoints, plus every QGNN run's own `model.pt`) — not every one of the ~230 individual QGNN run checkpoints was re-loaded and re-scored during this specific validation pass (that level of exhaustive re-verification was performed for the Phase 4 Stage 4b/5/Encoding pilots' "reuse valid existing artifacts" checks specifically, not for every historical run) |
| Threshold policy | **VERIFIED** | `fixed, value: 0.5` for every classification metric in every phase — confirmed by inspecting `configs/qgnn_v4*.yaml`/`configs/graphsage*.yaml` (`threshold.policy: fixed`) and cross-checked against every report's own stated protocol |
| Frozen-encoder isolation | **VERIFIED** | `assert all(not p.requires_grad for p in encoder.parameters())` is asserted in `scripts/run_qgnn_v4_experiment.py`'s own run loop for every QGNN training run, not just claimed in prose |
| Test suite | **VERIFIED** | 334/334 tests passing at this validation's starting commit (`python3 -m pytest tests/ -q`, run live) |

## How to reproduce the standing references

```bash
# Classical GraphSAGE-Full (already-trained checkpoints, not retrained):
python scripts/extract_classical_gnn_baseline.py

# QGNN-v4 standing reference (LayerNorm-no-affine), primary + severity, 5 seeds:
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4.yaml          --tag repro_primary  --seeds 42,43,44,45,46 --head-variant layernorm_noaffine --quantum-only --diagnostics
python scripts/run_qgnn_v4_experiment.py --config configs/qgnn_v4_severity.yaml --tag repro_severity --seeds 42,43,44,45,46 --head-variant layernorm_noaffine --quantum-only --diagnostics
```

Every other configuration's exact command is recorded in its own phase
report (`docs/phases/`, `docs/results/`) and in `QGNN_V4_MASTER_RESULTS.csv`'s
`source` column, which names the specific report each row's numbers were
verified against.
