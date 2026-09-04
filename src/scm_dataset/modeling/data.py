"""Deterministic benchmark loading layer (plan §46).

Reads one already-assembled dataset directory under `data/benchmark/`
(graph, operations, labels, splits, feature_audit) exactly as the dataset
generator wrote it -- no regeneration, no mutation. This module's only job
is "does the benchmark on disk actually look like what the rest of the
modeling stage assumes," so failures here are loud and early rather than
surfacing as a silent shape mismatch three modules later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from ..export.csv import load_graph, load_labels, load_operations
from ..schema.graph import SupplyChainGraph

REQUIRED_OPERATIONS = ("demand.csv", "production.csv", "inventory.csv", "backlog.csv", "deliveries.csv", "procurement.csv")
REQUIRED_LABEL_KEYS = ("supplier", "material", "plant", "product")
REQUIRED_SPLIT_FILES = ("temporal_split.csv", "scenario_split.csv", "severity_split.csv")

SUPPLIER_LABEL_COLUMNS = {"supplier_id", "time", "supplier_disrupted", "supplier_risk_score"}


@dataclass
class BenchmarkData:
    dataset_id: str
    dataset_dir: str
    graph: SupplyChainGraph
    operations: dict[str, pd.DataFrame]  # keyed by filename, e.g. "procurement.csv"
    labels: dict[str, pd.DataFrame]  # keyed by "supplier"/"material"/"plant"/"product"
    splits: dict[str, pd.DataFrame]  # keyed by "temporal"/"scenario"/"severity" -> columns [time, split]
    feature_audit: pd.DataFrame
    horizon_periods: int


def _load_splits(dataset_dir: str) -> dict[str, pd.DataFrame]:
    splits_dir = os.path.join(dataset_dir, "splits")
    missing = [f for f in REQUIRED_SPLIT_FILES if not os.path.exists(os.path.join(splits_dir, f))]
    if missing:
        raise FileNotFoundError(f"{splits_dir} is missing required split file(s): {missing}")
    return {
        "temporal": pd.read_csv(os.path.join(splits_dir, "temporal_split.csv")),
        "scenario": pd.read_csv(os.path.join(splits_dir, "scenario_split.csv")),
        "severity": pd.read_csv(os.path.join(splits_dir, "severity_split.csv")),
    }


def _validate_operations(operations: dict[str, pd.DataFrame]) -> None:
    missing = [f for f in REQUIRED_OPERATIONS if f not in operations]
    if missing:
        raise ValueError(f"benchmark operations/ is missing required file(s): {missing}")


def _validate_labels(labels: dict[str, pd.DataFrame]) -> None:
    missing = [k for k in REQUIRED_LABEL_KEYS if k not in labels]
    if missing:
        raise ValueError(f"benchmark labels/ is missing required key(s): {missing}")
    actual_cols = set(labels["supplier"].columns)
    if actual_cols != SUPPLIER_LABEL_COLUMNS:
        raise ValueError(
            f"labels/supplier_labels.csv columns {sorted(actual_cols)} != expected {sorted(SUPPLIER_LABEL_COLUMNS)}. "
            "The modeling stage's target/feature logic assumes this exact schema; update modeling/data.py and "
            "modeling/features.py deliberately if the benchmark schema has changed, rather than silently adapting."
        )


def load_benchmark(benchmark_root: str, dataset_id: str) -> BenchmarkData:
    """Loads one dataset directory under `benchmark_root` (e.g.
    `data/benchmark/scm_v1_black_swan_seed43`). Raises loudly (never
    silently drops or coerces) if the on-disk schema doesn't match what
    this modeling stage expects -- plan §46/§75's "the pipeline should fail
    if a dynamic feature violates ..." principle applied to the loader
    itself, not just to the leakage audit downstream."""
    dataset_dir = os.path.join(benchmark_root, dataset_id)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"benchmark dataset directory not found: {dataset_dir}")

    feature_audit_path = os.path.join(benchmark_root, "feature_audit.csv")
    if not os.path.exists(feature_audit_path):
        raise FileNotFoundError(f"feature_audit.csv not found at {feature_audit_path} (plan §11/§27 authoritative reference)")
    feature_audit = pd.read_csv(feature_audit_path)

    graph = load_graph(dataset_dir)
    graph.assert_valid()

    operations = load_operations(dataset_dir)
    _validate_operations(operations)

    labels = load_labels(dataset_dir)
    _validate_labels(labels)

    splits = _load_splits(dataset_dir)

    horizon_periods = int(operations["demand.csv"]["time"].max()) + 1

    return BenchmarkData(
        dataset_id=dataset_id,
        dataset_dir=dataset_dir,
        graph=graph,
        operations=operations,
        labels=labels,
        splits=splits,
        feature_audit=feature_audit,
        horizon_periods=horizon_periods,
    )
