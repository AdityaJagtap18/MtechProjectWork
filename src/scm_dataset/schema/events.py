"""Event schema (plan §14): the black-swan event object.

Five event types (plan §15), a 1-5 severity scale (plan §16: "Minor" to
"Black Swan"), and a recovery profile (plan §21: acute disruption, then a
delayed, gradual ramp back to normal — see `events/base.py`'s
`event_intensity` for how this is evaluated over time).

`cascade_mechanism` documents rather than implements: propagation through
the supply chain (procurement disruption -> material shortage -> production
reduction -> product shortage, plan §19) is not a separate per-event rule
here — it emerges from `simulation/engine.py`'s existing conservation-based
production/inventory/procurement dynamics once an event's capacity/lead-time
multipliers are applied to the entities it directly affects. This field
just records that fact for anyone reading an exported event record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EventType(str, Enum):
    SUPPLIER_FAILURE = "supplier_failure"
    NATURAL_DISASTER = "natural_disaster"
    GEOPOLITICAL = "geopolitical"
    CYBERATTACK = "cyberattack"
    LOGISTICS = "logistics"


@dataclass
class Event:
    event_id: str
    event_type: EventType
    start_time: int
    duration: int
    severity: int  # 1=Minor, 2=Moderate, 3=Major, 4=Extreme, 5=Black Swan
    probability_regime: float  # the configured per-period probability this event was drawn from

    affected_regions: list[str] = field(default_factory=list)
    affected_suppliers: list[str] = field(default_factory=list)
    affected_plants: list[str] = field(default_factory=list)

    # direct effects (plan §14): how much this event reduces capacity /
    # increases lead time for its affected entities at full intensity
    capacity_reduction_fraction: float = 0.0
    lead_time_increase_fraction: float = 0.0

    # recovery model (plan §21)
    recovery_delay: int = 0
    recovery_periods: int = 1

    cascade_mechanism: str = "operations_engine"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not (1 <= self.severity <= 5):
            errors.append(f"severity={self.severity} must be in [1, 5]")
        if self.duration < 1:
            errors.append(f"duration={self.duration} must be >= 1")
        if self.recovery_periods < 1:
            errors.append(f"recovery_periods={self.recovery_periods} must be >= 1")
        if self.recovery_delay < 0:
            errors.append(f"recovery_delay={self.recovery_delay} must be >= 0")
        if not (0.0 <= self.capacity_reduction_fraction <= 1.0):
            errors.append(f"capacity_reduction_fraction={self.capacity_reduction_fraction} out of [0, 1]")
        if self.lead_time_increase_fraction < 0:
            errors.append(f"lead_time_increase_fraction={self.lead_time_increase_fraction} must be >= 0")
        if not (self.affected_suppliers or self.affected_plants or self.affected_regions):
            errors.append("event has no affected entities")
        return errors
