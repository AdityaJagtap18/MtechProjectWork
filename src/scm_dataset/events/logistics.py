"""Logistics/Transportation Disruption event (plan §15.5).

Targets a region's transport: lead-time increase dominates, with only a
small direct capacity effect — the goods and production capacity still
exist, they just can't move on schedule. Recovery is comparatively fast:
rerouted transport typically resumes well before destroyed infrastructure
(Natural Disaster) or a political standoff (Geopolitical) resolves.
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
    affected_suppliers = sample_affected_entities(suppliers_here, min(1.0, intensity + 0.3), rng)

    duration = max(1, int(round(rng.gamma(shape=2.0, scale=1.5 * severity))))
    recovery_periods = max(1, int(round(duration * rng.uniform(0.8, 1.8))))
    recovery_delay = max(0, int(round(severity * 0.2)))

    return Event(
        event_id=event_id,
        event_type=EventType.LOGISTICS,
        start_time=start_time,
        duration=duration,
        severity=severity,
        probability_regime=probability_regime,
        affected_regions=[region.region_id],
        affected_suppliers=affected_suppliers,
        capacity_reduction_fraction=float(np.clip(intensity * 0.15 * rng.uniform(0.7, 1.3), 0.0, 1.0)),
        lead_time_increase_fraction=float(max(0.0, intensity * 1.0 * rng.uniform(0.9, 1.3))),
        recovery_delay=recovery_delay,
        recovery_periods=recovery_periods,
    )
