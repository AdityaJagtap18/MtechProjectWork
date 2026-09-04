"""Per-period event generation (plan §17, §38's `events:` config section).

At each simulation period, each event type independently has a small
configured probability of starting a new event — this mirrors SupplySim's
`shock_prob` mechanic (reused as an idea, not as code — see
DATASET_DESIGN_REVIEW.md §2.1). Severity is drawn from
`events_config.severity_weights`, deliberately skewed toward low
severities so severity-5 ("Black Swan") events stay rare in the generated
population, per plan §17's definition of a black swan as low-probability/
high-impact.
"""

from __future__ import annotations

from itertools import count

import numpy as np

from ..generator.config import EventsConfig
from ..schema.events import Event, EventType
from ..schema.graph import SupplyChainGraph
from . import cyberattack, geopolitical, logistics, natural_disaster, supplier_failure

CREATORS = {
    EventType.SUPPLIER_FAILURE: supplier_failure.create_event,
    EventType.NATURAL_DISASTER: natural_disaster.create_event,
    EventType.GEOPOLITICAL: geopolitical.create_event,
    EventType.CYBERATTACK: cyberattack.create_event,
    EventType.LOGISTICS: logistics.create_event,
}

_PROBABILITY_FIELD_BY_TYPE = {
    EventType.SUPPLIER_FAILURE: "supplier_failure",
    EventType.NATURAL_DISASTER: "natural_disaster",
    EventType.GEOPOLITICAL: "geopolitical",
    EventType.CYBERATTACK: "cyberattack",
    EventType.LOGISTICS: "logistics",
}


def maybe_generate_events(
    t: int, graph: SupplyChainGraph, events_config: EventsConfig, rng: np.random.Generator, event_id_counter: "count[int]"
) -> list[Event]:
    if not events_config.enabled:
        return []

    weights = np.array(events_config.severity_weights, dtype=float)
    severity_probs = weights / weights.sum()

    new_events = []
    for event_type, field_name in _PROBABILITY_FIELD_BY_TYPE.items():
        probability = getattr(events_config, field_name).probability
        if rng.random() < probability:
            severity = int(rng.choice([1, 2, 3, 4, 5], p=severity_probs))
            event_id = f"event_{next(event_id_counter)}"
            creator = CREATORS[event_type]
            new_events.append(creator(event_id, t, graph, severity, probability, rng))
    return new_events
