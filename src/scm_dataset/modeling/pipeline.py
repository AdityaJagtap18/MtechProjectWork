"""Shared setup shared by train.py / evaluate.py / baselines.py (plan §35's
"Load benchmark -> Validate schema -> Construct prediction examples ->
Leakage audit -> Apply benchmark split -> Fit preprocessing on train ->
Construct HeteroData" sequence, factored out once so the GraphSAGE
training script, the evaluation script, and the tabular baselines all run
the exact same leakage-audited pipeline over the exact same examples/split
-- required for the baselines to be a fair comparison (plan §55/§57) and
for the final classical-vs-QGNN comparison to be fair later (plan §72).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..schema.nodes import NodeType
from .config import GraphSAGEConfig
from .data import BenchmarkData, load_benchmark
from .features import (
    NodeFeatureFrames,
    apply_feature_mode,
    audit_feature_sources,
    build_feature_frames,
    build_prediction_examples,
)
from .hetero_graph import HeteroGraphSnapshotBuilder
from .preprocessing import FeaturePreprocessor, full_time_fit_mask, supplier_fit_mask
from .reduction import ReducedGraphSnapshotBuilder, fit_supplier_reducer


def _check_supplier_label_alignment(benchmark: BenchmarkData) -> None:
    """plan §66 Check 1 / §68: every supplier label maps to exactly one
    graph supplier node, and vice versa -- never silently discarded."""
    graph_ids = {n.supplier_id for n in benchmark.graph.nodes_of_type(NodeType.SUPPLIER)}
    label_ids = set(benchmark.labels["supplier"]["supplier_id"].unique())
    if graph_ids != label_ids:
        only_graph = graph_ids - label_ids
        only_labels = label_ids - graph_ids
        raise ValueError(
            "supplier_labels.csv and graph supplier nodes disagree on supplier identity "
            f"(only in graph: {sorted(only_graph)[:5]}{'...' if len(only_graph) > 5 else ''}, "
            f"only in labels: {sorted(only_labels)[:5]}{'...' if len(only_labels) > 5 else ''})"
        )


def _apply_temporal_split(examples: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    merged = examples.merge(split_df, on="time", how="left")
    if merged["split"].isna().any():
        missing = sorted(merged.loc[merged["split"].isna(), "time"].unique())
        raise ValueError(f"temporal split file has no entry for prediction time(s): {missing}")
    return merged


def _apply_generalization_split(examples: pd.DataFrame, split_df: pd.DataFrame, horizon_periods: int, val_frac: float) -> pd.DataFrame:
    """plan §29/§30/§39/§40: `split_df` (scenario_split.csv or
    severity_split.csv) labels each *period* train/test by whether a
    held-out event type / severity band is active then. An example's
    target window is (t, t+H] -- exactly the periods where that held-out
    disruption could occur -- so an example is a "test" example here if
    ANY period in its target window is a "test" period: this is what
    "held out this event type/severity" means for a *future-prediction*
    task, as opposed to the period-level split file itself, which says
    nothing about prediction windows. Documented deviation from a plain
    time-join (plan §92's "document the discrepancy").

    These two split files carry no "validation" category (only
    train/test); a validation slice is carved from the *tail* of the
    resulting train pool by prediction time, purely for early stopping --
    it never touches a period the split file marked "test"."""
    split_lookup = dict(zip(split_df["time"], split_df["split"]))
    horizon = int(examples.attrs["prediction_horizon"])

    def window_is_test(t: int) -> bool:
        return any(split_lookup.get(p) == "test" for p in range(t + 1, min(t + 1 + horizon, horizon_periods)))

    unique_times = sorted(examples["time"].unique())
    time_split = {t: ("test" if window_is_test(t) else "train") for t in unique_times}
    examples = examples.copy()
    examples["split"] = examples["time"].map(time_split)

    train_times = sorted(t for t, s in time_split.items() if s == "train")
    n_val = max(1, round(len(train_times) * val_frac)) if train_times else 0
    val_times = set(train_times[-n_val:]) if n_val else set()
    examples.loc[examples["time"].isin(val_times), "split"] = "validation"
    return examples


def build_split_assignment(examples: pd.DataFrame, benchmark: BenchmarkData, config: GraphSAGEConfig) -> pd.DataFrame:
    strategy = config.split.strategy
    examples = examples.copy()
    examples.attrs["prediction_horizon"] = config.prediction.horizon_periods
    if strategy == "temporal":
        return _apply_temporal_split(examples, benchmark.splits["temporal"])
    if strategy in ("scenario", "severity"):
        return _apply_generalization_split(
            examples, benchmark.splits[strategy], benchmark.horizon_periods, config.split.generalization_val_frac
        )
    raise ValueError(f"unknown split.strategy={strategy!r}, expected 'temporal', 'scenario', or 'severity'")


@dataclass
class PreparedData:
    config: GraphSAGEConfig
    benchmark: BenchmarkData
    frames: NodeFeatureFrames
    examples: pd.DataFrame  # supplier_id, time, target, split
    preprocessor: FeaturePreprocessor
    snapshot_builder: HeteroGraphSnapshotBuilder


def prepare(config: GraphSAGEConfig, preprocessor: FeaturePreprocessor | None = None) -> PreparedData:
    """Loads the benchmark named in `config.dataset` off disk, then defers
    to `prepare_from_benchmark`. See that function for the `preprocessor`
    argument's meaning (train-vs-standalone-evaluation)."""
    benchmark = load_benchmark(config.dataset.benchmark_path, config.dataset.dataset_id)
    return prepare_from_benchmark(config, benchmark, preprocessor)


def prepare_from_benchmark(config: GraphSAGEConfig, benchmark: BenchmarkData, preprocessor: FeaturePreprocessor | None = None) -> PreparedData:
    """The part of `prepare` that doesn't touch disk -- factored out so
    tests can build a `BenchmarkData` in memory (see tests/conftest.py's
    `tiny_prepared_data`) and exercise the *exact* production feature/
    split/preprocessing pipeline, rather than a parallel reimplementation
    of it, without needing a real benchmark directory on disk.

    If `preprocessor` is None (the training path), a fresh one is fit on
    this run's train split. If a `FeaturePreprocessor` is passed (the
    standalone-evaluation path, loaded from a saved experiment's
    `preprocessing/` artifacts), it is used as-is and never refit --
    `scripts/evaluate_graphsage.py` relies on this to guarantee evaluation
    never re-fits preprocessing on a different data mix than training used
    (plan §15's "fit on TRAIN only" would otherwise be silently violated on
    every standalone evaluation run)."""
    audit_feature_sources(benchmark.feature_audit)
    _check_supplier_label_alignment(benchmark)

    frames = build_feature_frames(
        benchmark.graph, benchmark.operations, benchmark.horizon_periods, config.features.rolling_windows
    )
    frames = apply_feature_mode(frames, config.features.feature_mode)

    examples = build_prediction_examples(
        benchmark.labels["supplier"],
        benchmark.horizon_periods,
        config.features.min_history_periods,
        config.prediction.horizon_periods,
    )
    examples = build_split_assignment(examples, benchmark, config)

    if preprocessor is None:
        train_examples = examples[examples["split"] == "train"]
        train_times = set(train_examples["time"].unique())
        fit_masks = {
            NodeType.SUPPLIER: supplier_fit_mask(frames.frames[NodeType.SUPPLIER], train_examples),
            NodeType.MATERIAL: full_time_fit_mask(frames.frames[NodeType.MATERIAL], train_times),
            NodeType.PLANT: full_time_fit_mask(frames.frames[NodeType.PLANT], train_times),
            NodeType.PRODUCT: full_time_fit_mask(frames.frames[NodeType.PRODUCT], train_times),
            NodeType.REGION: pd.Series(True, index=frames.frames[NodeType.REGION].index),
            NodeType.PROCUREMENT: pd.Series(True, index=frames.frames[NodeType.PROCUREMENT].index),
        }
        preprocessor = FeaturePreprocessor().fit(frames.frames, frames.numeric_columns, frames.categorical_columns, fit_masks)

    snapshot_builder = HeteroGraphSnapshotBuilder(benchmark.graph, preprocessor, frames)

    return PreparedData(
        config=config, benchmark=benchmark, frames=frames, examples=examples,
        preprocessor=preprocessor, snapshot_builder=snapshot_builder,
    )


def prepare_for_cross_dataset_eval(config: GraphSAGEConfig, target_benchmark: BenchmarkData, source_preprocessor: FeaturePreprocessor) -> PreparedData:
    """D2 -- true cross-dataset evaluation
    (GRAPHSAGE_FINAL_GENERALIZATION_AND_IMPROVEMENT_PLAN.md): builds a
    `PreparedData` for `target_benchmark` using a `FeaturePreprocessor`
    already fit on a DIFFERENT (source) dataset -- never refits, so
    nothing about the target's own feature distribution ever influences
    scaling/imputation/encoding.

    Every example gets `split="test"`: the whole target dataset is unseen
    from the trained model's perspective, not just its own temporal-test
    tail. This is the precise distinction the plan draws between
    "seed43 train -> seed44 test" (every period of seed44 is held out)
    and "seed44 train -> seed44 test" (only seed44's own temporal-test
    weeks are held out, which is within-seed replication, not cross-
    dataset generalization).

    Node identity cannot leak between datasets here: `HeteroGraphSnapshotBuilder`
    builds its node id -> index maps from `target_benchmark.graph` alone
    (see hetero_graph.py), and the model itself has no per-entity
    parameters -- every learned weight is either per-node-type (input
    projections) or per-relation (SAGEConv), so it is applicable to any
    graph with the same feature dimensionality regardless of which
    dataset's specific entities produced it."""
    audit_feature_sources(target_benchmark.feature_audit)
    _check_supplier_label_alignment(target_benchmark)

    frames = build_feature_frames(
        target_benchmark.graph, target_benchmark.operations, target_benchmark.horizon_periods, config.features.rolling_windows
    )
    frames = apply_feature_mode(frames, config.features.feature_mode)

    examples = build_prediction_examples(
        target_benchmark.labels["supplier"],
        target_benchmark.horizon_periods,
        config.features.min_history_periods,
        config.prediction.horizon_periods,
    )
    examples = examples.copy()
    examples["split"] = "test"

    snapshot_builder = HeteroGraphSnapshotBuilder(target_benchmark.graph, source_preprocessor, frames)

    return PreparedData(
        config=config, benchmark=target_benchmark, frames=frames, examples=examples,
        preprocessor=source_preprocessor, snapshot_builder=snapshot_builder,
    )


def prepare_reduced(
    config: GraphSAGEConfig, reduction_method: str, n_components: int, columns: list[str] | None = None
) -> tuple[PreparedData, object]:
    """Loads the benchmark named in `config.dataset` off disk, then defers
    to `prepare_reduced_from_benchmark`. See that function for details."""
    benchmark = load_benchmark(config.dataset.benchmark_path, config.dataset.dataset_id)
    return prepare_reduced_from_benchmark(config, benchmark, reduction_method, n_components, columns=columns)


def prepare_reduced_from_benchmark(
    config: GraphSAGEConfig, benchmark: BenchmarkData, reduction_method: str, n_components: int, columns: list[str] | None = None
) -> tuple[PreparedData, object]:
    """QGNN_FAIR_BENCHMARK_AND_IMPLEMENTATION_PLAN.md sections 3B/4-6/10-11:
    the disk-free half of `prepare_reduced` (mirrors `prepare`/
    `prepare_from_benchmark`'s own split, so tests can pass an in-memory
    `BenchmarkData` -- see tests/conftest.py's `tiny_benchmark`). Identical
    to `prepare_from_benchmark` in every respect (schema audit, feature
    frames, prediction examples, split assignment, `FeaturePreprocessor`
    fit on train only) except the SUPPLIER node's tensor is additionally
    reduced to `n_components` dims by a `reduction.SupplierReducer` fit on
    the exact same train-split rows. GraphSAGE-Reduced and QGNN-Reduced
    both consume the returned `PreparedData` -- GraphSAGE-Reduced via the
    ordinary, unmodified `HeteroGraphSAGE`/`train_graphsage`/
    `evaluate_experiment` (it duck-types as a normal `PreparedData`, so
    nothing downstream needs to know its snapshot_builder is a
    `ReducedGraphSnapshotBuilder`); QGNN-Reduced via `qgnn.py`.

    Returns `(prepared, reducer)` -- the fitted reducer is returned
    separately (not stashed on `PreparedData`, whose schema is shared
    infrastructure this phase must not modify) so callers can `.save()` it
    and, for cross-dataset evaluation, reuse it on a target benchmark via
    `prepare_reduced_for_cross_dataset_eval` without refitting."""
    prepared = prepare_from_benchmark(config, benchmark)
    reducer = fit_supplier_reducer(
        reduction_method, n_components, prepared.frames,
        prepared.examples[prepared.examples["split"] == "train"], columns=columns,
    )
    reduced_frame = reducer.transform(prepared.frames.frames[NodeType.SUPPLIER])
    reduced_builder = ReducedGraphSnapshotBuilder(prepared.snapshot_builder, reduced_frame, n_components)
    prepared.snapshot_builder = reduced_builder
    return prepared, reducer


def prepare_reduced_for_cross_dataset_eval(
    config: GraphSAGEConfig, target_benchmark: BenchmarkData, source_preprocessor: FeaturePreprocessor, reducer: object
) -> PreparedData:
    """D2-equivalent for the reduced representation (plan §17): builds a
    cross-dataset `PreparedData` for `target_benchmark` exactly like
    `prepare_for_cross_dataset_eval`, then applies `reducer` (already fit
    on the SOURCE dataset's train split by `prepare_reduced`) to the
    target's own supplier frame -- never refit here."""
    prepared = prepare_for_cross_dataset_eval(config, target_benchmark, source_preprocessor)
    reduced_frame = reducer.transform(prepared.frames.frames[NodeType.SUPPLIER])
    reduced_builder = ReducedGraphSnapshotBuilder(prepared.snapshot_builder, reduced_frame, reducer.n_components)
    prepared.snapshot_builder = reduced_builder
    return prepared
