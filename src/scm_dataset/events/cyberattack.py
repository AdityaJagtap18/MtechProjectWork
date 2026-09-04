"""Cyberattack event (plan §15.4).

Targets one supplier (two at higher severity — a coordinated attack hitting
multiple firms): order-processing disruption (lead-time increase) plus a
moderate, temporary capacity cut. Duration and recovery are both shorter
than Supplier Failure's — once contained/patched, systems come back online
faster than a physically damaged supplier recovers.
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
    num_targets = 2 if severity >= 4 else 1
    num_targets = min(num_targets, len(suppliers))
    idx = rng.choice(len(suppliers), size=num_targets, replace=False)
    targets = [suppliers[i] for i in idx]
    intensity = SEVERITY_INTENSITY[severity]

    duration = max(1, int(round(rng.gamma(shape=1.5, scale=1.0 * severity))))
    recovery_periods = max(1, int(round(duration * rng.uniform(0.8, 1.5))))
    recovery_delay = max(0, int(round(severity * 0.2)))

    return Event(
        event_id=event_id,
        event_type=EventType.CYBERATTACK,
        start_time=start_time,
        duration=duration,
        severity=severity,
        probability_regime=probability_regime,
        affected_suppliers=[s.supplier_id for s in targets],
        capacity_reduction_fraction=float(np.clip(intensity * 0.5 * rng.uniform(0.8, 1.2), 0.0, 1.0)),
        lead_time_increase_fraction=float(max(0.0, intensity * 0.6 * rng.uniform(0.8, 1.2))),
        recovery_delay=recovery_delay,
        recovery_periods=recovery_periods,
    )
