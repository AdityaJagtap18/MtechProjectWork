"""Integration tests (plan §67/§87): benchmark -> features -> HeteroData ->
GraphSAGE -> prediction -> metrics, end to end.

Two levels:
- `test_prepare_train_evaluate_in_process`: fast, in-process, using the
  `tiny_prepared_data` fixture (real pipeline code, no disk I/O).
- `test_run_graphsage_experiment_script_end_to_end`: writes a tiny
  benchmark to disk and invokes `scripts/run_graphsage_experiment.py` as a
  subprocess exactly as a user would, mirroring
  `tests/test_benchmark.py::test_build_benchmark_script_end_to_end`'s
  existing pattern in this repo -- the only test that exercises the CLI/
  config-file/experiment-directory wiring rather than calling library
  functions directly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scm_dataset.export.csv import export_graph
from scm_dataset.modeling.baselines import run_logistic_regression_baseline, run_majority_baseline
from scm_dataset.modeling.evaluate import evaluate_experiment
from scm_dataset.modeling.train import train_graphsage

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_prepare_train_evaluate_in_process(tiny_prepared_data):
    tiny_prepared_data.config.training.epochs = 3
    train_result = train_graphsage(tiny_prepared_data, seed=42, verbose=False)
    eval_result = evaluate_experiment(train_result.model, tiny_prepared_data, tiny_prepared_data.config.threshold)

    predictions = eval_result.predictions
    # plan §66 Checks 9/10 and §92's "after coding" verification, run as assertions
    assert not predictions[["risk_probability"]].isna().any().any()
    assert np.isfinite(predictions["risk_probability"]).all()
    assert predictions["risk_probability"].between(0.0, 1.0).all()
    assert set(predictions["actual_disruption"].unique()) <= {0, 1}
    assert set(predictions["predicted_disruption"].unique()) <= {0, 1}

    for split in ("train", "validation", "test"):
        assert split in eval_result.metrics_by_split
        m = eval_result.metrics_by_split[split]
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
        assert 0.0 <= m["f1"] <= 1.0

    majority = run_majority_baseline(tiny_prepared_data, tiny_prepared_data.config.threshold)
    logreg = run_logistic_regression_baseline(tiny_prepared_data, tiny_prepared_data.config.threshold, seed=42)
    assert majority.predictions["risk_probability"].between(0.0, 1.0).all()
    assert logreg.predictions["risk_probability"].between(0.0, 1.0).all()


def _write_benchmark_to_disk(benchmark, benchmark_root: Path) -> None:
    dataset_dir = benchmark_root / benchmark.dataset_id
    export_graph(benchmark.graph, str(dataset_dir))

    ops_dir = dataset_dir / "operations"
    ops_dir.mkdir(parents=True, exist_ok=True)
    for filename, df in benchmark.operations.items():
        df.to_csv(ops_dir / filename, index=False)

    labels_dir = dataset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_filenames = {"supplier": "supplier_labels.csv", "material": "material_labels.csv", "plant": "plant_labels.csv", "product": "product_labels.csv"}
    for key, df in benchmark.labels.items():
        df.to_csv(labels_dir / label_filenames[key], index=False)

    splits_dir = dataset_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_filenames = {"temporal": "temporal_split.csv", "scenario": "scenario_split.csv", "severity": "severity_split.csv"}
    for key, df in benchmark.splits.items():
        df.to_csv(splits_dir / split_filenames[key], index=False)

    benchmark.feature_audit.to_csv(benchmark_root / "feature_audit.csv", index=False)


def test_run_graphsage_experiment_script_end_to_end(tiny_benchmark, tmp_path):
    benchmark_root = tmp_path / "benchmark"
    _write_benchmark_to_disk(tiny_benchmark, benchmark_root)

    output_dir = tmp_path / "experiments"
    config = {
        "dataset": {"benchmark_path": str(benchmark_root), "dataset_id": tiny_benchmark.dataset_id},
        "prediction": {"target": "supplier_disrupted", "horizon_periods": 4},
        "features": {"rolling_windows": [4, 8], "min_history_periods": 12},
        "split": {"strategy": "temporal", "train_frac": 0.7, "val_frac": 0.15},
        "model": {"name": "hetero_graphsage", "hidden_dim": 8, "num_layers": 2, "dropout": 0.0, "aggregation": "mean"},
        "training": {"epochs": 2, "learning_rate": 0.01, "weight_decay": 0.0001, "early_stopping_patience": 5, "class_weighting": "balanced"},
        "threshold": {"policy": "fixed", "value": 0.5},
        "experiment": {"output_dir": str(output_dir), "seeds": [42]},
        "seed": 42,
    }
    config_path = tmp_path / "graphsage.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    proc = subprocess.run(
        [sys.executable, "scripts/run_graphsage_experiment.py", "--config", str(config_path), "--seeds", "42", "--baselines"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # "*hetero_graphsage*seed42" (not an exact suffix match): run directory
    # names get an extra tag for a non-primary dataset id
    # (scripts/run_graphsage_experiment.py::_strategy_tag), and this
    # fixture's tiny benchmark isn't the primary scm_v1_black_swan_seed43.
    run_dirs = sorted(output_dir.glob("*hetero_graphsage*seed42"))
    assert len(run_dirs) == 1, f"expected exactly one run dir, found {run_dirs}"
    run_dir = run_dirs[0]

    for filename in (
        "config.yaml", "model.pt", "predictions.csv", "supplier_risk_ranking.csv", "metrics.json",
        "classification_report.json", "calibration.json", "training_history.csv", "run_metadata.json",
    ):
        assert (run_dir / filename).exists(), filename
    for plot in ("pr_curve.png", "roc_curve.png", "confusion_matrix.png", "calibration_curve.png", "training_curve.png"):
        assert (run_dir / "plots" / plot).exists(), plot

    predictions = pd.read_csv(run_dir / "predictions.csv")
    assert set(predictions.columns) >= {"supplier_id", "time", "risk_probability", "predicted_disruption", "actual_disruption", "split"}
    assert predictions["risk_probability"].between(0.0, 1.0).all()

    with open(run_dir / "metrics.json") as f:
        metrics = json.load(f)
    assert "test" in metrics["by_split"]
    assert metrics["by_split"]["test"]["prediction_horizon"] == 4

    with open(run_dir / "run_metadata.json") as f:
        metadata = json.load(f)
    assert metadata["dataset_id"] == tiny_benchmark.dataset_id
    assert metadata["seed"] == 42
    assert metadata["prediction_horizon"] == 4

    baseline_dirs = list(output_dir.glob("*majority_baseline*")) + list(output_dir.glob("*logistic_regression_baseline*"))
    assert len(baseline_dirs) == 2
    for baseline_dir in baseline_dirs:
        assert (baseline_dir / "predictions.csv").exists()
        assert (baseline_dir / "metrics.json").exists()


def test_run_cross_dataset_experiment_script_end_to_end(tiny_benchmark, tiny_benchmark_b, tmp_path):
    """D2 (GRAPHSAGE_FINAL_GENERALIZATION_AND_IMPROVEMENT_PLAN.md): trains
    on one tiny benchmark, evaluates cross-dataset on a genuinely
    different one, via the actual CLI script -- mirrors
    test_run_graphsage_experiment_script_end_to_end's pattern for the
    primary script."""
    benchmark_root = tmp_path / "benchmark"
    _write_benchmark_to_disk(tiny_benchmark, benchmark_root)
    _write_benchmark_to_disk(tiny_benchmark_b, benchmark_root)

    output_dir = tmp_path / "experiments"
    config = {
        "dataset": {"benchmark_path": str(benchmark_root), "dataset_id": tiny_benchmark.dataset_id},
        "prediction": {"target": "supplier_disrupted", "horizon_periods": 4},
        "features": {"rolling_windows": [4, 8], "min_history_periods": 12},
        "split": {"strategy": "temporal", "train_frac": 0.7, "val_frac": 0.15},
        "model": {"name": "hetero_graphsage", "hidden_dim": 8, "num_layers": 2, "dropout": 0.0, "aggregation": "mean"},
        "training": {"epochs": 2, "learning_rate": 0.01, "weight_decay": 0.0001, "early_stopping_patience": 5, "class_weighting": "balanced"},
        "threshold": {"policy": "fixed", "value": 0.5},
        "experiment": {"output_dir": str(output_dir), "seeds": [42]},
        "seed": 42,
    }
    config_path = tmp_path / "graphsage.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    proc = subprocess.run(
        [
            sys.executable, "scripts/run_cross_dataset_experiment.py",
            "--config", str(config_path),
            "--source-dataset-id", tiny_benchmark.dataset_id,
            "--target-dataset-id", tiny_benchmark_b.dataset_id,
            "--seeds", "42",
        ],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    run_dirs = list(output_dir.glob("*crossdataset*seed42"))
    assert len(run_dirs) == 1, f"expected exactly one run dir, found {run_dirs}"
    run_dir = run_dirs[0]

    for prefix in ("within_seed", "cross_dataset"):
        assert (run_dir / f"predictions_{prefix}.csv").exists()
        assert (run_dir / f"metrics_{prefix}.json").exists()
        assert (run_dir / f"onset_breakdown_{prefix}.json").exists()
        assert (run_dir / f"calibration_{prefix}.json").exists()

    cross_predictions = pd.read_csv(run_dir / "predictions_cross_dataset.csv")
    assert (cross_predictions["split"] == "test").all()  # every target example is test, none train/validation
    assert cross_predictions["risk_probability"].between(0.0, 1.0).all()

    with open(run_dir / "run_metadata.json") as f:
        metadata = json.load(f)
    assert metadata["source_dataset_id"] == tiny_benchmark.dataset_id
    assert metadata["target_dataset_id"] == tiny_benchmark_b.dataset_id
    assert metadata["experiment_type"] == "D2_true_cross_dataset_generalization"

    baseline_dirs = list(output_dir.glob("*crossdataset*baselines"))
    assert len(baseline_dirs) == 1
    for prefix in ("majority_within_seed", "majority_cross_dataset", "logreg_within_seed", "logreg_cross_dataset"):
        assert (baseline_dirs[0] / f"metrics_{prefix}.json").exists()
