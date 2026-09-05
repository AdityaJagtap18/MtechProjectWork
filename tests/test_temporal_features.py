"""Tests for modeling/features.py: the leakage audit gate, target
construction/alignment, and rolling-window causality (plan §18-22/§66
Checks 2/3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scm_dataset.modeling.features import (
    audit_feature_sources,
    build_prediction_examples,
    _supplier_panel,
)

# ---- audit_feature_sources ----


def _minimal_audit(overrides: dict[tuple[str, str], bool] | None = None) -> pd.DataFrame:
    rows = [
        ("operations/procurement.csv", "quantity_ordered, quantity_fulfilled", "time_ordered", "> time_ordered", True, "ok"),
        ("operations/deliveries.csv", "quantity", "t", "> t", True, "ok"),
        ("operations/inventory.csv", "inventory", "t", "> t", True, "ok"),
        ("operations/production.csv", "production", "t", "> t", True, "ok"),
        ("operations/demand.csv", "demand", "t", "> t", True, "ok"),
        ("operations/backlog.csv", "backlog", "t", "> t", True, "ok"),
    ]
    df = pd.DataFrame(rows, columns=["table", "column", "available_time", "target_time_rule", "allowed", "reason"])
    if overrides:
        for (table, column), allowed in overrides.items():
            df.loc[(df.table == table) & (df.column == column), "allowed"] = allowed
    return df


def test_audit_passes_with_full_allowed_audit():
    audit_feature_sources(_minimal_audit())  # must not raise


def test_audit_splits_comma_joined_column_cell():
    # "quantity_ordered, quantity_fulfilled" is one audit row covering two
    # columns this module reads separately -- must not be treated as a
    # single literal column name.
    audit_feature_sources(_minimal_audit())


def test_audit_raises_when_a_used_source_is_disallowed():
    audit = _minimal_audit()
    audit.loc[audit.table == "operations/inventory.csv", "allowed"] = False
    with pytest.raises(ValueError, match="allowed=False"):
        audit_feature_sources(audit)


def test_audit_raises_when_a_used_source_is_missing_entirely():
    audit = _minimal_audit()
    audit = audit[audit.table != "operations/demand.csv"].reset_index(drop=True)
    with pytest.raises(ValueError, match="not found in feature_audit.csv"):
        audit_feature_sources(audit)


def test_audit_fails_closed_on_events_and_labels_tables_if_ever_referenced():
    # events/events.csv and labels/*.csv are never in FEATURE_SOURCE_COLUMNS
    # at all (this module never reads them) -- the audit only checks what
    # is actually declared, so this documents the invariant rather than
    # testing new behavior: absence from the registry is the leakage
    # safeguard, enforced by code review / the registry itself.
    from scm_dataset.modeling.features import FEATURE_SOURCE_COLUMNS

    tables_used = {source.table for source in FEATURE_SOURCE_COLUMNS}
    assert "events/events.csv" not in tables_used
    assert "events/event_impact.csv" not in tables_used
    assert not any(t.startswith("labels/") for t in tables_used)


# ---- build_prediction_examples ----


def _supplier_labels(pattern: dict[str, list[int]], horizon_periods: int) -> pd.DataFrame:
    """`pattern` maps supplier_id -> sorted list of disrupted periods."""
    rows = []
    for supplier_id, disrupted_periods in pattern.items():
        disrupted = set(disrupted_periods)
        for t in range(horizon_periods):
            rows.append({"supplier_id": supplier_id, "time": t, "supplier_disrupted": int(t in disrupted), "supplier_risk_score": None})
    return pd.DataFrame(rows)


def test_target_is_max_over_future_horizon_not_current_period():
    labels = _supplier_labels({"s0": [20]}, horizon_periods=40)
    examples = build_prediction_examples(labels, horizon_periods=40, min_history_periods=12, prediction_horizon=4)
    s0 = examples[examples.supplier_id == "s0"].set_index("time")["target"]

    # disruption occurs at t=20 -> Y(t)=1 for t in [16, 19] (20 in (t, t+4]), 0 elsewhere
    assert s0.loc[16] == 1
    assert s0.loc[19] == 1
    assert s0.loc[20] == 0  # the disruption period itself is not "future" relative to itself
    assert s0.loc[15] == 0
    assert s0.loc[21] == 0  # already in the past relative to t=21's future window


def test_target_never_uses_current_or_past_periods():
    # a supplier disrupted only at t=0 should never produce a positive
    # target for any usable prediction time (t=0's own disruption is
    # never inside a strictly-future window (t, t+H]).
    labels = _supplier_labels({"s0": [0]}, horizon_periods=40)
    examples = build_prediction_examples(labels, horizon_periods=40, min_history_periods=12, prediction_horizon=4)
    assert (examples[examples.supplier_id == "s0"]["target"] == 0).all()


def test_usable_prediction_time_bounds():
    labels = _supplier_labels({"s0": []}, horizon_periods=40)
    min_history, horizon = 12, 4
    examples = build_prediction_examples(labels, horizon_periods=40, min_history_periods=min_history, prediction_horizon=horizon)
    times = sorted(examples["time"].unique())
    assert times[0] == min_history - 1
    assert times[-1] == 40 - 1 - horizon


def test_build_prediction_examples_raises_if_horizon_too_short():
    labels = _supplier_labels({"s0": []}, horizon_periods=10)
    with pytest.raises(ValueError, match="no usable prediction times"):
        build_prediction_examples(labels, horizon_periods=10, min_history_periods=12, prediction_horizon=4)


# ---- rolling-window causality (_supplier_panel) ----


def test_supplier_panel_feature_at_t_is_unaffected_by_future_records():
    windows = [4]
    procurement = pd.DataFrame(
        [
            {"supplier_id": "s0", "time_ordered": 5, "quantity_ordered": 100.0, "quantity_fulfilled": 100.0},
            {"supplier_id": "s0", "time_ordered": 6, "quantity_ordered": 50.0, "quantity_fulfilled": 25.0},
        ]
    )
    deliveries = pd.DataFrame(columns=["supplier_id", "time", "quantity"])
    operations = {"procurement.csv": procurement, "deliveries.csv": deliveries}

    panel_before = _supplier_panel(operations, ["s0"], horizon_periods=20, windows=windows)
    value_at_t6_before = panel_before.loc[("s0", 6), "fulfillment_ratio_4"]
    order_volume_before = panel_before.loc[("s0", 6), "order_volume_4"]

    # Inject a huge future order at t=15 -- must not change anything computed at t=6.
    procurement_with_future = pd.concat(
        [procurement, pd.DataFrame([{"supplier_id": "s0", "time_ordered": 15, "quantity_ordered": 99999.0, "quantity_fulfilled": 1.0}])],
        ignore_index=True,
    )
    operations_with_future = {"procurement.csv": procurement_with_future, "deliveries.csv": deliveries}
    panel_after = _supplier_panel(operations_with_future, ["s0"], horizon_periods=20, windows=windows)

    assert panel_after.loc[("s0", 6), "fulfillment_ratio_4"] == pytest.approx(value_at_t6_before)
    assert panel_after.loc[("s0", 6), "order_volume_4"] == pytest.approx(order_volume_before)
    # the future record IS visible at its own (later) time, so the panel did register it somewhere
    assert panel_after.loc[("s0", 15), "order_volume_4"] > 0


def test_supplier_panel_fulfillment_ratio_correctness_and_missing_when_no_orders():
    procurement = pd.DataFrame(
        [
            {"supplier_id": "s0", "time_ordered": 2, "quantity_ordered": 100.0, "quantity_fulfilled": 80.0},
            {"supplier_id": "s0", "time_ordered": 3, "quantity_ordered": 100.0, "quantity_fulfilled": 100.0},
        ]
    )
    deliveries = pd.DataFrame(columns=["supplier_id", "time", "quantity"])
    operations = {"procurement.csv": procurement, "deliveries.csv": deliveries}
    panel = _supplier_panel(operations, ["s0"], horizon_periods=10, windows=[4])

    # window [0,3] ending at t=3: 180/200 = 0.9
    assert panel.loc[("s0", 3), "fulfillment_ratio_4"] == pytest.approx(0.9)
    # no orders ever placed by t=0 -> genuinely undefined, must be NaN (not 0)
    assert np.isnan(panel.loc[("s0", 0), "fulfillment_ratio_4"])


# ---- apply_feature_mode (GRAPH_SAGE_IMPROVEMENT_PLAN.md Phase B) ----


def test_feature_mode_full_is_a_no_op(tiny_benchmark):
    from scm_dataset.modeling.features import apply_feature_mode, build_feature_frames

    frames = build_feature_frames(tiny_benchmark.graph, tiny_benchmark.operations, tiny_benchmark.horizon_periods, [4, 8])
    result = apply_feature_mode(frames, "full")
    assert result is frames


def test_feature_mode_dynamic_only_restricts_supplier_but_not_other_types(tiny_benchmark):
    from scm_dataset.modeling.features import STATIC_NUMERIC_FIELDS, apply_feature_mode, build_feature_frames
    from scm_dataset.schema.nodes import NodeType

    frames = build_feature_frames(tiny_benchmark.graph, tiny_benchmark.operations, tiny_benchmark.horizon_periods, [4, 8])
    result = apply_feature_mode(frames, "dynamic_only")

    # supplier: no static fields or categoricals survive
    assert result.categorical_columns[NodeType.SUPPLIER] == []
    for col in result.numeric_columns[NodeType.SUPPLIER]:
        assert col not in STATIC_NUMERIC_FIELDS[NodeType.SUPPLIER]
    assert len(result.numeric_columns[NodeType.SUPPLIER]) > 0

    # other node types are untouched
    assert result.numeric_columns[NodeType.MATERIAL] == frames.numeric_columns[NodeType.MATERIAL]
    assert result.numeric_columns[NodeType.PLANT] == frames.numeric_columns[NodeType.PLANT]
    assert result.numeric_columns[NodeType.PRODUCT] == frames.numeric_columns[NodeType.PRODUCT]


def test_feature_mode_static_only_restricts_supplier_to_static_fields(tiny_benchmark):
    from scm_dataset.modeling.features import STATIC_NUMERIC_FIELDS, apply_feature_mode, build_feature_frames
    from scm_dataset.schema.nodes import NodeType

    frames = build_feature_frames(tiny_benchmark.graph, tiny_benchmark.operations, tiny_benchmark.horizon_periods, [4, 8])
    result = apply_feature_mode(frames, "static_only")
    assert set(result.numeric_columns[NodeType.SUPPLIER]) == set(STATIC_NUMERIC_FIELDS[NodeType.SUPPLIER])
    assert result.categorical_columns[NodeType.SUPPLIER] == ["industry"]
    # the underlying dataframe's columns must match, not just the recorded list
    assert set(result.frames[NodeType.SUPPLIER].columns) == set(result.numeric_columns[NodeType.SUPPLIER]) | {"industry"}


def test_feature_mode_region_risk_only_keeps_only_exposure_fields(tiny_benchmark):
    from scm_dataset.modeling.features import SUPPLIER_RISK_EXPOSURE_FIELDS, apply_feature_mode, build_feature_frames
    from scm_dataset.schema.nodes import NodeType

    frames = build_feature_frames(tiny_benchmark.graph, tiny_benchmark.operations, tiny_benchmark.horizon_periods, [4, 8])
    result = apply_feature_mode(frames, "region_risk_only")
    assert set(result.numeric_columns[NodeType.SUPPLIER]) == set(SUPPLIER_RISK_EXPOSURE_FIELDS)
    assert result.categorical_columns[NodeType.SUPPLIER] == []
    assert "criticality" not in result.numeric_columns[NodeType.SUPPLIER]  # not an "exposure" field


def test_feature_mode_static_plus_graph_restricts_every_node_type(tiny_benchmark):
    from scm_dataset.modeling.features import STATIC_NUMERIC_FIELDS, apply_feature_mode, build_feature_frames
    from scm_dataset.schema.nodes import NodeType

    frames = build_feature_frames(tiny_benchmark.graph, tiny_benchmark.operations, tiny_benchmark.horizon_periods, [4, 8])
    result = apply_feature_mode(frames, "static_plus_graph")
    for node_type in (NodeType.SUPPLIER, NodeType.MATERIAL, NodeType.PLANT, NodeType.PRODUCT):
        assert set(result.numeric_columns[node_type]) == set(STATIC_NUMERIC_FIELDS[node_type])
    # region/procurement had no dynamic component anyway -- unaffected
    assert result.numeric_columns[NodeType.REGION] == frames.numeric_columns[NodeType.REGION]
    assert result.numeric_columns[NodeType.PROCUREMENT] == frames.numeric_columns[NodeType.PROCUREMENT]


def test_feature_mode_rejects_unknown_mode(tiny_benchmark):
    from scm_dataset.modeling.features import apply_feature_mode, build_feature_frames

    frames = build_feature_frames(tiny_benchmark.graph, tiny_benchmark.operations, tiny_benchmark.horizon_periods, [4, 8])
    with pytest.raises(ValueError, match="unknown feature_mode"):
        apply_feature_mode(frames, "made_up_mode")


def test_feature_mode_never_touches_non_supplier_dynamic_computation(tiny_benchmark):
    # Restricting the supplier's own columns must not change what was
    # computed for material/plant/product -- apply_feature_mode only drops
    # columns, never recomputes anything (leakage-safety is orthogonal to
    # which mode is selected).
    from scm_dataset.modeling.features import apply_feature_mode, build_feature_frames
    from scm_dataset.schema.nodes import NodeType

    frames = build_feature_frames(tiny_benchmark.graph, tiny_benchmark.operations, tiny_benchmark.horizon_periods, [4, 8])
    result = apply_feature_mode(frames, "static_only")
    pd.testing.assert_frame_equal(
        result.frames[NodeType.MATERIAL],
        frames.frames[NodeType.MATERIAL][result.numeric_columns[NodeType.MATERIAL] + result.categorical_columns[NodeType.MATERIAL]],
    )
