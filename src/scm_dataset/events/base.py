"""Shared severity model and event-intensity timeline (plan §16, §21).

Severity (1=Minor .. 5=Black Swan) drives multiple variables coherently,
not just one multiplier (plan §16's explicit requirement): `SEVERITY_INTENSITY`
sets one shared base fraction per severity level, and each event-type
module (supplier_failure.py, natural_disaster.py, ...) weights that same
base differently per channel (capacity reduction vs. lead-time increase)
and per how many entities it touches — so severity 5 doesn't just mean "one
number is 5x bigger," it means duration, recovery time, affected-entity
count, and both effect channels all move together.

`event_intensity` implements the recovery profile from plan §21 (initial
disruption -> maximum impact -> partial recovery -> normal operation) as a
pure function of time: full effect for `duration` periods, held for
`recovery_delay` more periods (bureaucratic/logistic lag before repair
starts), then a linear ramp back to zero over `recovery_periods`.
"""

from __future__ import annotations

import numpy as np

from ..schema.events import Event

SEVERITY_INTENSITY: dict[int, float] = {1: 0.15, 2: 0.35, 3: 0.55, 4: 0.75, 5: 0.95}


def sample_affected_entities(pool: list[str], frac: float, rng: np.random.Generator) -> list[str]:
    """Pick a severity-scaled fraction of a region's suppliers/plants,
    shared by the region-wide event types (natural disaster, geopolitical,
    logistics) — plan §16's "affected nodes ↑" as severity increases."""
    if not pool:
        return []
    k = min(len(pool), max(1, int(round(len(pool) * frac))))
    idx = rng.choice(len(pool), size=k, replace=False)
    return [pool[i] for i in idx]


def event_intensity(event: Event, t: int) -> float:
    if t < event.start_time:
        return 0.0
    end_of_duration = event.start_time + event.duration
    if t < end_of_duration:
        return 1.0
    recovery_start = end_of_duration + event.recovery_delay
    if t < recovery_start:
        return 1.0
    periods_into_recovery = t - recovery_start
    if periods_into_recovery >= event.recovery_periods:
        return 0.0
    return 1.0 - (periods_into_recovery / event.recovery_periods)


def is_active(event: Event, t: int) -> bool:
    return event.start_time <= t < event.start_time + event.duration + event.recovery_delay + event.recovery_periods


def capacity_multiplier(event: Event, t: int) -> float:
    return 1.0 - event.capacity_reduction_fraction * event_intensity(event, t)


def lead_time_multiplier(event: Event, t: int) -> float:
    return 1.0 + event.lead_time_increase_fraction * event_intensity(event, t)
