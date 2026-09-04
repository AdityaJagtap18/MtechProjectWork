"""Supplier Failure event (plan §15.1).

Directly reduces one supplier's capacity and increases its effective lead
time. Capacity reduction is the dominant channel — this is a failure of the
supplier itself, not of transport — unlike Logistics/Geopolitical, where
lead-time increase dominates instead (plan §16: severity should move
multiple variables coherently, but *how much* each channel moves is
event-type-specific).
"""

from __future__ import annotations

import numpy as np

from ..schema.events import Event, EventType
from ..schema.graph import SupplyChainGraph
from ..schema.nodes import NodeType
from .base import SEVERITY_INTENSITY


def create_event(
    event_id: str, start_time: int, graph: SupplyChainGraph, severity: int, probability_regime: float, rng: np.random.Generator
) -> Event:
    suppliers = graph.nodes_of_type(NodeType.SUPPLIER)
    target = suppliers[int(rng.integers(0, len(suppliers)))]
    intensity = SEVERITY_INTENSITY[severity]

    duration = max(1, int(round(rng.gamma(shape=2.0, scale=1.5 * severity))))
    recovery_periods = max(1, int(round(duration * rng.uniform(1.0, 2.0))))
    recovery_delay = max(0, int(round(severity * 0.3)))

    return Event(
        event_id=event_id,
        event_type=EventType.SUPPLIER_FAILURE,
        start_time=start_time,
        duration=duration,
        severity=severity,
        probability_regime=probability_regime,
        affected_suppliers=[target.supplier_id],
        capacity_reduction_fraction=float(np.clip(intensity * rng.uniform(0.9, 1.1), 0.0, 1.0)),
        lead_time_increase_fraction=float(max(0.0, intensity * 0.5 * rng.uniform(0.8, 1.2))),
        recovery_delay=recovery_delay,
        recovery_periods=recovery_periods,
    )
