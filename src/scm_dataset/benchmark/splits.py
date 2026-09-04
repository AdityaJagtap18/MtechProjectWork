"""Train/validation/test split strategies (plan §26) and the data-leakage
feature audit (plan §27).

Each split function assigns every period `t` in `[0, horizon_periods)` to
"train", "validation", or "test", returned as a DataFrame with columns
`time`/`split` — joinable onto any labels or operations table by `time`
(and, for per-entity tables, whatever entity id column it also has).
Avoid random per-row splitting of a simulated time series (plan §26's
warning): these all partition along *time* or *event membership*, never a
random row shuffle, so a train row's neighbors in the same trajectory don't
leak into test.
"""

from __future__ import annotations

import pandas as pd

from ..schema.events import Event, EventType


def temporal_split(horizon_periods: int, train_frac: float = 0.7, val_frac: float = 0.15) -> pd.DataFrame:
    """plan §26 "Temporal split": train on [0, train_end), validation on
    [train_end, val_end), test on [val_end, horizon) -- the only strategy
    that needs no events at all."""
    train_end = int(round(horizon_periods * train_frac))
    val_end = int(round(horizon_periods * (train_frac + val_frac)))
    records = []
    for t in range(horizon_periods):
        if t < train_end:
            split = "train"
        elif t < val_end:
            split = "validation"
        else:
            split = "test"
        records.append({"time": t, "split": split})
    return pd.DataFrame(records)


def _event_windows(events: list[Event]) -> list[tuple[int, int, EventType, int]]:
    return [
        (e.start_time, e.start_time + e.duration + e.recovery_delay + e.recovery_periods, e.event_type, e.severity)
        for e in events
    ]


def scenario_split(horizon_periods: int, events: list[Event], test_event_types: set[EventType]) -> pd.DataFrame:
    """plan §26 "Scenario split" / §49 "Black-Swan Generalization Experiment":
    a period goes to test if an event whose type is in `test_event_types` is
    active then; baseline periods and periods covered only by non-held-out
    event types go to train. This tests whether a model generalizes to an
    unseen disruption family rather than memorizing per-type signatures."""
    windows = _event_windows(events)
    records = []
    for t in range(horizon_periods):
        active_types = {etype for start, end, etype, _sev in windows if start <= t < end}
        records.append({"time": t, "split": "test" if active_types & test_event_types else "train"})
    return pd.DataFrame(records)


def severity_split(horizon_periods: int, events: list[Event], train_max_severity: int = 3) -> pd.DataFrame:
    """plan §26 "Severity generalization": train on severity 1-3, test on
    severity 4-5. A period's severity is the max severity of any event
    active then; baseline (no active event) periods go to train."""
    windows = _event_windows(events)
    records = []
    for t in range(horizon_periods):
        active_severities = [sev for start, end, _etype, sev in windows if start <= t < end]
        max_severity = max(active_severities, default=0)
        records.append({"time": t, "split": "train" if max_severity <= train_max_severity else "test"})
    return pd.DataFrame(records)


FEATURE_AUDIT_RECORDS = [
    ("graph/nodes.csv", "*", "generation (before simulation starts)", "any", True, "static input attribute, fixed before simulation begins"),
    ("operations/demand.csv", "demand", "t", "> t", True, "observed realized demand up to and including t"),
    ("operations/production.csv", "production", "t", "> t", True, "observed realized production up to and including t"),
    ("operations/inventory.csv", "inventory", "t", "> t", True, "observed end-of-period inventory up to and including t"),
    ("operations/backlog.csv", "backlog", "t", "> t", True, "observed backlog entering period t"),
    ("operations/deliveries.csv", "quantity", "t (arrival time)", "> t", True, "observed realized delivery up to and including t"),
    ("operations/procurement.csv", "quantity_ordered, quantity_fulfilled", "time_ordered", "> time_ordered", True, "observed order outcome, available once time_ordered <= t"),
    ("operations/procurement.csv", "expected_delivery_time", "time_ordered", "any", False, "a future timestamp computed from the event's own disruption -- use only realized deliveries up to t, not this"),
    ("events/events.csv", "*", "event.start_time", "any", False, "ground truth about an upcoming disruption; using it to predict that same disruption is leakage by plan §23's target definition"),
    ("events/event_impact.csv", "*", "n/a (computed after the event window resolves)", "any", False, "post-hoc cascade summary, never available at any legitimate prediction time"),
    ("labels/supplier_labels.csv", "*", "n/a (target)", "any", False, "ground-truth label -- only ever a prediction TARGET, never an input feature"),
    ("labels/material_labels.csv", "*", "n/a (target)", "any", False, "ground-truth label -- only ever a prediction TARGET, never an input feature"),
    ("labels/plant_labels.csv", "*", "n/a (target)", "any", False, "ground-truth label -- only ever a prediction TARGET, never an input feature"),
    ("labels/product_labels.csv", "*", "n/a (target)", "any", False, "ground-truth label -- only ever a prediction TARGET, never an input feature"),
]


def build_feature_audit() -> pd.DataFrame:
    """plan §27's template: feature, available_time, target_time, allowed,
    reason. `available_time` is when a column's value is first knowable;
    `target_time` states the rule for how it relates to whatever future
    period a label is being predicted for -- "> t" means the column is safe
    to use for a target strictly after its own available_time, "any" means
    the allowed/disallowed verdict doesn't depend on the target time."""
    return pd.DataFrame(
        FEATURE_AUDIT_RECORDS,
        columns=["table", "column", "available_time", "target_time_rule", "allowed", "reason"],
    )
