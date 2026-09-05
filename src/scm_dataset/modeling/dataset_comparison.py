"""D0 dataset integrity comparison (GRAPHSAGE_FINAL_GENERALIZATION_AND_IMPROVEMENT_PLAN.md
Sec "D0 -- Dataset Integrity").

Before any cross-dataset training/evaluation, this module checks two
already-loaded `BenchmarkData` objects for structural compatibility --
same node/edge type universe and counts, same feature column names (not
necessarily the same categorical *values*: `FeaturePreprocessor` already
handles an unseen category via its UNKNOWN bucket, so a difference there
is not by itself disqualifying), same label schema, same horizon. It never
silently adapts a mismatch -- it reports one, and
`compatible_for_cross_dataset_evaluation` is only `True` when every
structural check passes.
"""

from __future__ import annotations

import os
from collections import Counter

import pandas as pd

from ..schema.nodes import NodeType
from .data import BenchmarkData
from .features import FEATURE_MODES, apply_feature_mode, build_feature_frames

REQUIRED_FILES = [
    "graph/nodes.csv", "graph/edges.csv",
    "operations/demand.csv", "operations/production.csv", "operations/inventory.csv",
    "operations/backlog.csv", "operations/deliveries.csv", "operations/procurement.csv",
    "labels/supplier_labels.csv", "labels/material_labels.csv", "labels/plant_labels.csv", "labels/product_labels.csv",
    "events/events.csv", "events/event_impact.csv",
    "splits/temporal_split.csv", "splits/scenario_split.csv", "splits/severity_split.csv",
]


def _edge_type_counts(benchmark: BenchmarkData) -> dict:
    return dict(Counter(e.edge_type.value for e in benchmark.graph.edges))


def _node_type_counts(benchmark: BenchmarkData) -> dict:
    return {nt.value: len(benchmark.graph.nodes_of_type(nt)) for nt in NodeType}


def _feature_columns(benchmark: BenchmarkData, rolling_windows: list[int], feature_mode: str) -> dict:
    frames = build_feature_frames(benchmark.graph, benchmark.operations, benchmark.horizon_periods, rolling_windows)
    frames = apply_feature_mode(frames, feature_mode)
    return {
        nt.value: {"numeric": frames.numeric_columns[nt], "categorical": frames.categorical_columns[nt]}
        for nt in NodeType
    }


def _label_columns(benchmark: BenchmarkData) -> dict:
    return {name: list(df.columns) for name, df in benchmark.labels.items()}


def _supplier_disrupted_prevalence(benchmark: BenchmarkData) -> dict:
    df = benchmark.labels["supplier"]
    n_positive = int(df["supplier_disrupted"].sum())
    return {"n_rows": int(len(df)), "n_positive": n_positive, "positive_rate": float(df["supplier_disrupted"].mean())}


def _onset_count(benchmark: BenchmarkData) -> int:
    """Number of raw 0->1 supplier_disrupted transitions -- the count of
    genuinely distinct disruption instances, independent of how many
    prediction examples they generate (see NEXT_PHASE_SPEC.md Sec 9)."""
    df = benchmark.labels["supplier"].sort_values(["supplier_id", "time"])
    prev = df.groupby("supplier_id")["supplier_disrupted"].shift(1).fillna(0)
    return int(((df["supplier_disrupted"] == 1) & (prev == 0)).sum())


def _event_summary(benchmark: BenchmarkData) -> dict | None:
    events_path = os.path.join(benchmark.dataset_dir, "events", "events.csv")
    if not os.path.exists(events_path):
        return None
    events = pd.read_csv(events_path)
    return {
        "n_events": int(len(events)),
        "event_types": events["event_type"].value_counts().to_dict(),
        "severities": {int(k): int(v) for k, v in events["severity"].value_counts().to_dict().items()},
    }


def _split_summary(benchmark: BenchmarkData) -> dict:
    return {name: df["split"].value_counts().to_dict() for name, df in benchmark.splits.items()}


def _required_files_present(benchmark: BenchmarkData) -> dict:
    return {f: os.path.exists(os.path.join(benchmark.dataset_dir, f)) for f in REQUIRED_FILES}


def compare_datasets(benchmark_a: BenchmarkData, benchmark_b: BenchmarkData, rolling_windows: list[int], feature_mode: str = "full") -> dict:
    """Structural comparison of two benchmark datasets, read-only. Does not
    train or evaluate anything. `feature_mode` should match whatever mode
    the cross-dataset experiment will actually use (default "full")."""
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"unknown feature_mode={feature_mode!r}, expected one of {FEATURE_MODES}")

    node_counts_a, node_counts_b = _node_type_counts(benchmark_a), _node_type_counts(benchmark_b)
    edge_counts_a, edge_counts_b = _edge_type_counts(benchmark_a), _edge_type_counts(benchmark_b)
    feature_cols_a = _feature_columns(benchmark_a, rolling_windows, feature_mode)
    feature_cols_b = _feature_columns(benchmark_b, rolling_windows, feature_mode)
    label_cols_a, label_cols_b = _label_columns(benchmark_a), _label_columns(benchmark_b)
    files_a, files_b = _required_files_present(benchmark_a), _required_files_present(benchmark_b)

    feature_columns_match = all(
        feature_cols_a[nt.value]["numeric"] == feature_cols_b[nt.value]["numeric"]
        and feature_cols_a[nt.value]["categorical"] == feature_cols_b[nt.value]["categorical"]
        for nt in NodeType
    )

    result = {
        "dataset_a": benchmark_a.dataset_id,
        "dataset_b": benchmark_b.dataset_id,
        "feature_mode_checked": feature_mode,
        "rolling_windows_checked": rolling_windows,
        "node_counts": {"a": node_counts_a, "b": node_counts_b, "match": node_counts_a == node_counts_b},
        "edge_counts": {"a": edge_counts_a, "b": edge_counts_b, "match": edge_counts_a == edge_counts_b},
        "horizon_periods": {"a": benchmark_a.horizon_periods, "b": benchmark_b.horizon_periods, "match": benchmark_a.horizon_periods == benchmark_b.horizon_periods},
        "feature_columns": {"a": feature_cols_a, "b": feature_cols_b, "match": feature_columns_match},
        "label_columns": {"a": label_cols_a, "b": label_cols_b, "match": label_cols_a == label_cols_b},
        "supplier_disrupted_prevalence": {"a": _supplier_disrupted_prevalence(benchmark_a), "b": _supplier_disrupted_prevalence(benchmark_b)},
        "onset_counts": {"a": _onset_count(benchmark_a), "b": _onset_count(benchmark_b)},
        "events": {"a": _event_summary(benchmark_a), "b": _event_summary(benchmark_b)},
        "splits": {"a": _split_summary(benchmark_a), "b": _split_summary(benchmark_b)},
        "required_files_present": {"a": files_a, "b": files_b, "all_present": all(files_a.values()) and all(files_b.values())},
    }
    result["compatible_for_cross_dataset_evaluation"] = (
        result["node_counts"]["match"]
        and result["edge_counts"]["match"]
        and result["horizon_periods"]["match"]
        and result["feature_columns"]["match"]
        and result["label_columns"]["match"]
        and result["required_files_present"]["all_present"]
    )
    return result
