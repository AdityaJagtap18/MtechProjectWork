"""Experiment directory management and reproducibility metadata (plan
§43/§60/§61). Every call to `new_run_dir` makes a fresh, never-overwritten
directory under `experiments/classical_gnn/` (plan §43: "Never overwrite
old experiments.").
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
import torch
import torch_geometric
import yaml

from .. import __version__ as GENERATOR_VERSION
from .config import GraphSAGEConfig
from .data import BenchmarkData
from .features import NodeFeatureFrames


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=os.path.dirname(__file__)
        ).decode().strip()
    except Exception:
        return None


def config_hash(config: GraphSAGEConfig) -> str:
    payload = json.dumps(dataclasses.asdict(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def new_run_dir(experiment_output_dir: str, tag: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{tag}"
    run_dir = os.path.join(experiment_output_dir, run_id)
    if os.path.exists(run_dir):
        raise FileExistsError(f"experiment run directory already exists (never overwritten): {run_dir}")
    os.makedirs(os.path.join(run_dir, "preprocessing"))
    os.makedirs(os.path.join(run_dir, "plots"))
    return run_dir


def save_config(run_dir: str, config: GraphSAGEConfig) -> None:
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(dataclasses.asdict(config), f, sort_keys=False)


def build_run_metadata(
    config: GraphSAGEConfig,
    benchmark: BenchmarkData,
    frames: NodeFeatureFrames,
    seed: int,
    split_summary: dict,
    training_duration_seconds: float | None,
    extra: dict | None = None,
) -> dict:
    """plan §60: dataset_id, generator_version, dataset/model git commit,
    seed, config_hash, prediction_horizon, target, feature list, model
    architecture, hyperparameters, split, class weighting, software
    versions, hardware, training duration -- everything needed to answer
    "what code + config + seed produced this run" without leaving this
    file (mirrors export/metadata.py's `build_provenance` for the dataset
    generator, applied to the modeling stage instead)."""
    feature_list = {
        nt.value: {"numeric": frames.numeric_columns[nt], "categorical": frames.categorical_columns[nt]}
        for nt in frames.frames
    }
    metadata = {
        "dataset_id": benchmark.dataset_id,
        "generator_version": GENERATOR_VERSION,
        "model_git_commit": _git_commit(),
        "dataset_git_commit": _git_commit(),  # same repo/commit produced both in this project
        "seed": seed,
        "config_hash": config_hash(config),
        "prediction_horizon": config.prediction.horizon_periods,
        "target": config.prediction.target,
        "feature_list": feature_list,
        "model_architecture": dataclasses.asdict(config.model),
        "hyperparameters": {
            "training": dataclasses.asdict(config.training),
            "features": dataclasses.asdict(config.features),
        },
        "split": {"strategy": config.split.strategy, "summary": split_summary},
        "class_weighting": config.training.class_weighting,
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "hardware": {
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
            "torch_num_threads": torch.get_num_threads(),
        },
        "training_duration_seconds": training_duration_seconds,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        metadata.update(extra)
    return metadata


def write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
