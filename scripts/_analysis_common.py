"""Shared helpers for the Phase A.5 root-cause diagnostic scripts
(GRAPHSAGE_PHASE_A5_ROOT_CAUSE_INVESTIGATION.md). Read-only: these load
already-generated benchmark data and already-completed experiment run
directories -- nothing here trains a model or writes into
data/benchmark/ or an existing experiments/classical_gnn/<run> directory.
"""

from __future__ import annotations

import glob
import os
import re

import pandas as pd
import yaml


def find_latest_cross_dataset_runs(base_dir: str, source_dataset_id: str, target_dataset_id: str, seeds: list[int]) -> dict[int, str]:
    """Finds the most recent `*_crossdataset_<source>_to_<target>_seed<N>`
    run directory per model seed `N` (picks the latest by directory-name
    timestamp if a seed was run more than once, so this stays correct if
    D2 is ever rerun rather than silently using a stale duplicate).
    `source_dataset_id`/`target_dataset_id` are full dataset ids (e.g.
    `scm_v1_black_swan_seed43`) -- matches scripts/run_cross_dataset_experiment.py's
    own `_short()` tagging."""
    tag = f"crossdataset_{short_dataset_id(source_dataset_id)}_to_{short_dataset_id(target_dataset_id)}"
    result: dict[int, str] = {}
    for seed in seeds:
        pattern = os.path.join(base_dir, f"*_{tag}_seed{seed}")
        matches = sorted(d for d in glob.glob(pattern) if re.fullmatch(rf".*_{tag}_seed{seed}", os.path.basename(d)))
        if not matches:
            raise FileNotFoundError(f"no run directory found for {tag}_seed{seed} under {base_dir}")
        result[seed] = matches[-1]
    return result


def load_cross_dataset_predictions(base_dir: str, source_dataset_id: str, target_dataset_id: str, seeds: list[int], prefix: str = "cross_dataset") -> pd.DataFrame:
    """Concatenates `predictions_<prefix>.csv` across all given model
    seeds for one source->target direction, tagging each row with its
    model_seed."""
    run_dirs = find_latest_cross_dataset_runs(base_dir, source_dataset_id, target_dataset_id, seeds)
    frames = []
    for seed, run_dir in run_dirs.items():
        df = pd.read_csv(os.path.join(run_dir, f"predictions_{prefix}.csv"))
        df["model_seed"] = seed
        df["run_dir"] = run_dir
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def find_latest_cross_dataset_runs_by_feature_mode(
    base_dir: str, source_dataset_id: str, target_dataset_id: str, seeds: list[int], feature_mode: str
) -> dict[int, str]:
    """Like `find_latest_cross_dataset_runs`, but scripts/run_cross_dataset_experiment.py's
    run-directory tag does NOT include feature_mode (only source/target/model
    seed), so re-running the same direction under different --feature-mode
    values produces multiple directories sharing the identical
    `..._crossdataset_<source>_to_<target>_seed<N>` suffix, distinguished
    only by their timestamp prefix and their saved `config.yaml`. This
    disambiguates by reading each candidate's config.yaml and keeping the
    most recent one whose `features.feature_mode` matches."""
    tag = f"crossdataset_{short_dataset_id(source_dataset_id)}_to_{short_dataset_id(target_dataset_id)}"
    result: dict[int, str] = {}
    for seed in seeds:
        pattern = os.path.join(base_dir, f"*_{tag}_seed{seed}")
        candidates = sorted(d for d in glob.glob(pattern) if re.fullmatch(rf".*_{tag}_seed{seed}", os.path.basename(d)))
        matching = []
        for d in candidates:
            config_path = os.path.join(d, "config.yaml")
            if not os.path.exists(config_path):
                continue
            with open(config_path) as f:
                config = yaml.safe_load(f)
            if config.get("features", {}).get("feature_mode", "full") == feature_mode:
                matching.append(d)
        if not matching:
            raise FileNotFoundError(f"no run directory found for {tag}_seed{seed} with feature_mode={feature_mode!r} under {base_dir}")
        result[seed] = matching[-1]
    return result


def short_dataset_id(dataset_id: str) -> str:
    return dataset_id.replace("scm_v1_black_swan_", "")
