"""Geopolitical Disruption event (plan §15.3).

Targets a region: trade restrictions, sanctions, and border closures
primarily slow deliveries rather than destroy physical capacity, so
lead-time increase dominates over capacity reduction here — the opposite
weighting from Natural Disaster. Recovery is comparatively slow: political/
diplomatic resolution often outlasts the physical damage a natural disaster
of similar severity would cause. Only suppliers are directly targeted
(plants are affected only indirectly, through disrupted deliveries).
"""

from __future__ import annotations

import numpy as np

from ..schema.edges import EdgeType
from ..schema.events import Event, EventType
from ..schema.graph import SupplyChainGraph
from ..schema.nodes import NodeType
from .base import SEVERITY_INTENSITY, sample_affected_entities


def create_event(
    event_id: str, start_time: int, graph: SupplyChainGraph, severity: int, probability_regime: float, rng: np.random.Generator
) -> Event:
    regions = graph.nodes_of_type(NodeType.REGION)
    region = regions[int(rng.integers(0, len(regions)))]
    intensity = SEVERITY_INTENSITY[severity]

    suppliers_here = [e.source_id for e in graph.edges_of_type(EdgeType.SUPPLIER_REGION) if e.target_id == region.region_id]
    affected_suppliers = sample_affected_entities(suppliers_here, min(1.0, intensity + 0.2), rng)

    # Acute duration matches Natural Disaster's scale (severity, not event
    # type, should drive how long the acute phase lasts); only the recovery
    # multiplier is larger, reflecting "political resolution is slower than
    # physical repair" without making a severity-5 event's mean lifecycle
    # exceed the plan §10 recommended ~104-period simulation horizon (an
    # earlier version's wider 2.0-4.0x recovery multiplier averaged ~125
    # periods at severity 5 -- longer than the whole horizon, which made
    # "recovery" invisible in the exported dataset for that event).
    duration = max(1, int(round(rng.gamma(shape=2.5, scale=2.0 * severity))))
    recovery_periods = max(1, int(round(duration * rng.uniform(1.8, 2.8))))
    recovery_delay = max(0, int(round(severity * 0.6)))

    return Event(
        event_id=event_id,
        event_type=EventType.GEOPOLITICAL,
        start_time=start_time,
        duration=duration,
        severity=severity,
        probability_regime=probability_regime,
        affected_regions=[region.region_id],
        affected_suppliers=affected_suppliers,
        capacity_reduction_fraction=float(np.clip(intensity * 0.6 * rng.uniform(0.8, 1.2), 0.0, 1.0)),
        lead_time_increase_fraction=float(max(0.0, intensity * 1.0 * rng.uniform(0.9, 1.3))),
        recovery_delay=recovery_delay,
        recovery_periods=recovery_periods,
    )
