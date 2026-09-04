import tempfile
from itertools import count

import numpy as np
import pytest

from scm_dataset.events.base import capacity_multiplier, event_intensity, is_active, lead_time_multiplier
from scm_dataset.events.generate import maybe_generate_events
from scm_dataset.export.csv import export_events, load_events
from scm_dataset.generator.config import EventsConfig, EventTypeConfig, GeneratorConfig, NetworkConfig, SimulationConfig
from scm_dataset.generator.topology import generate_graph
from scm_dataset.schema.events import Event, EventType
from scm_dataset.schema.nodes import NodeType
from scm_dataset.simulation.engine import run_simulation


def _small_config(seed: int = 42, horizon_periods: int = 20, events: EventsConfig | None = None) -> GeneratorConfig:
    return GeneratorConfig(
        seed=seed,
        network=NetworkConfig(suppliers=20, procurement_orders=50, materials=15, plants=6, products=15, regions=4),
        simulation=SimulationConfig(horizon_periods=horizon_periods),
        events=events or EventsConfig(),
    )


def _make_event(**overrides) -> Event:
    defaults = dict(
        event_id="e0",
        event_type=EventType.SUPPLIER_FAILURE,
        start_time=10,
        duration=5,
        severity=3,
        probability_regime=0.02,
        affected_suppliers=["supplier_0"],
        capacity_reduction_fraction=0.8,
        lead_time_increase_fraction=0.4,
        recovery_delay=2,
        recovery_periods=6,
    )
    defaults.update(overrides)
    return Event(**defaults)


def test_event_validate_rejects_bad_severity():
    event = _make_event(severity=7)
    assert any("severity" in e for e in event.validate())


def test_event_validate_rejects_no_affected_entities():
    event = _make_event(affected_suppliers=[], affected_plants=[], affected_regions=[])
    assert any("no affected entities" in e for e in event.validate())


def test_event_validate_accepts_well_formed_event():
    assert _make_event().validate() == []


def test_event_intensity_profile_shape():
    event = _make_event(start_time=10, duration=5, recovery_delay=2, recovery_periods=6)
    assert event_intensity(event, 5) == 0.0  # before start
    assert event_intensity(event, 10) == 1.0  # onset
    assert event_intensity(event, 14) == 1.0  # still in acute duration
    assert event_intensity(event, 15) == 1.0  # in recovery_delay (held at full effect)
    assert event_intensity(event, 16) == 1.0  # recovery starts at 10+5+2=17
    mid_recovery = event_intensity(event, 20)  # 3 periods into a 6-period ramp
    assert 0.0 < mid_recovery < 1.0
    assert event_intensity(event, 23) == 0.0  # fully recovered (17+6=23)
    assert event_intensity(event, 100) == 0.0


def test_is_active_matches_intensity_window():
    event = _make_event(start_time=10, duration=5, recovery_delay=2, recovery_periods=6)
    assert not is_active(event, 9)
    assert is_active(event, 10)
    assert is_active(event, 22)  # last period of recovery ramp
    assert not is_active(event, 23)


def test_capacity_and_lead_time_multipliers_at_trough_and_baseline():
    event = _make_event(capacity_reduction_fraction=0.8, lead_time_increase_fraction=0.4, start_time=10, duration=5)
    assert capacity_multiplier(event, 5) == pytest.approx(1.0)
    assert capacity_multiplier(event, 12) == pytest.approx(0.2)
    assert lead_time_multiplier(event, 5) == pytest.approx(1.0)
    assert lead_time_multiplier(event, 12) == pytest.approx(1.4)


@pytest.mark.parametrize(
    "event_type", [EventType.SUPPLIER_FAILURE, EventType.NATURAL_DISASTER, EventType.GEOPOLITICAL, EventType.CYBERATTACK, EventType.LOGISTICS]
)
def test_each_event_type_creator_produces_a_valid_event_referencing_real_entities(event_type):
    from scm_dataset.events.generate import CREATORS

    graph = generate_graph(_small_config())
    rng = np.random.default_rng(123)
    event = CREATORS[event_type]("e0", start_time=0, graph=graph, severity=5, probability_regime=0.01, rng=rng)

    assert event.validate() == []
    assert event.event_type == event_type

    supplier_ids = {s.supplier_id for s in graph.nodes_of_type(NodeType.SUPPLIER)}
    plant_ids = {p.plant_id for p in graph.nodes_of_type(NodeType.PLANT)}
    region_ids = {r.region_id for r in graph.nodes_of_type(NodeType.REGION)}
    assert set(event.affected_suppliers) <= supplier_ids
    assert set(event.affected_plants) <= plant_ids
    assert set(event.affected_regions) <= region_ids


def test_maybe_generate_events_disabled_returns_nothing():
    graph = generate_graph(_small_config())
    rng = np.random.default_rng(0)
    events = maybe_generate_events(0, graph, EventsConfig(enabled=False), rng, count())
    assert events == []


def test_maybe_generate_events_forced_probability_generates_one_per_type():
    graph = generate_graph(_small_config())
    rng = np.random.default_rng(0)
    events_config = EventsConfig(
        enabled=True,
        supplier_failure=EventTypeConfig(1.0),
        natural_disaster=EventTypeConfig(1.0),
        geopolitical=EventTypeConfig(1.0),
        cyberattack=EventTypeConfig(1.0),
        logistics=EventTypeConfig(1.0),
    )
    events = maybe_generate_events(0, graph, events_config, rng, count())
    assert len(events) == 5
    assert {e.event_type for e in events} == set(EventType)


def test_simulation_with_events_disabled_produces_no_events():
    config = _small_config()
    graph = generate_graph(config)
    result = run_simulation(graph, config)
    assert result.events == []


def test_simulation_with_forced_severe_events_reduces_total_production():
    baseline_config = _small_config(seed=5)
    baseline_graph = generate_graph(baseline_config)
    baseline_result = run_simulation(baseline_graph, baseline_config)
    baseline_production = sum(r["production"] for r in baseline_result.production_records)

    disrupted_config = _small_config(
        seed=5,
        events=EventsConfig(
            enabled=True,
            supplier_failure=EventTypeConfig(0.5),
            natural_disaster=EventTypeConfig(0.2),
            geopolitical=EventTypeConfig(0.0),
            cyberattack=EventTypeConfig(0.0),
            logistics=EventTypeConfig(0.0),
            severity_weights=[0.0, 0.0, 0.0, 0.0, 1.0],  # force severity 5 ("Black Swan") every time
        ),
    )
    disrupted_graph = generate_graph(disrupted_config)
    disrupted_result = run_simulation(disrupted_graph, disrupted_config)
    disrupted_production = sum(r["production"] for r in disrupted_result.production_records)

    assert len(disrupted_result.events) > 0
    assert all(e.validate() == [] for e in disrupted_result.events)
    # a meaningful margin, not a razor's-edge inequality -- forced severity-5
    # events on a fifth to a half of periods should be an overwhelming
    # direct effect relative to any downstream RNG-stream-shift noise
    assert disrupted_production < 0.9 * baseline_production


def test_events_csv_roundtrip():
    events = [
        _make_event(event_id="e0", affected_suppliers=["supplier_0", "supplier_1"]),
        _make_event(event_id="e1", event_type=EventType.NATURAL_DISASTER, affected_regions=["region_0"], affected_plants=["plant_0"]),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        export_events(events, tmp)
        reloaded = load_events(tmp)

    assert len(reloaded) == 2
    assert reloaded[0].affected_suppliers == ["supplier_0", "supplier_1"]
    assert reloaded[1].event_type == EventType.NATURAL_DISASTER
    assert reloaded[1].affected_regions == ["region_0"]
    assert isinstance(reloaded[0].severity, int)


def test_load_events_missing_file_returns_empty_list(tmp_path):
    assert load_events(str(tmp_path)) == []
