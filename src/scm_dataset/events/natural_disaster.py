"""Natural Disaster event (plan §15.2).

Targets a region: reduces capacity and increases lead time for every
supplier *and* plant located there ("regional infrastructure reduction,
supplier capacity reduction, transport delay, temporary shutdown"). This is
the one event type that directly hits plants (a regional shutdown), not
just suppliers — for the others, plants are only affected indirectly,
through reduced material deliveries (plan §19's cascade). The number of
affected entities in the region scales with severity (plan §16), and
recovery is slow relative to Supplier Failure (physical infrastructure
repair, not a single firm recovering).
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
    plants_here = [e.source_id for e in graph.edges_of_type(EdgeType.PLANT_REGION) if e.target_id == region.region_id]

    affected_fraction = min(1.0, intensity + 0.2)
    affected_suppliers = sample_affected_entities(suppliers_here, affected_fraction, rng)
    affected_plants = sample_affected_entities(plants_here, affected_fraction, rng)

    duration = max(1, int(round(rng.gamma(shape=2.5, scale=2.0 * severity))))
    recovery_periods = max(1, int(round(duration * rng.uniform(1.5, 3.0))))
    recovery_delay = max(0, int(round(severity * 0.5)))

    return Event(
        event_id=event_id,
        event_type=EventType.NATURAL_DISASTER,
        start_time=start_time,
        duration=duration,
        severity=severity,
        probability_regime=probability_regime,
        affected_regions=[region.region_id],
        affected_suppliers=affected_suppliers,
        affected_plants=affected_plants,
        capacity_reduction_fraction=float(np.clip(intensity * rng.uniform(0.9, 1.1), 0.0, 1.0)),
        lead_time_increase_fraction=float(max(0.0, intensity * 0.7 * rng.uniform(0.8, 1.2))),
        recovery_delay=recovery_delay,
        recovery_periods=recovery_periods,
    )
