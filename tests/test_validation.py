import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from scm_dataset.export.csv import export_events, export_graph, export_operations, load_events, load_graph, load_operations
from scm_dataset.generator.config import EventsConfig, EventTypeConfig, GeneratorConfig, NetworkConfig, SimulationConfig
from scm_dataset.generator.topology import generate_graph
from scm_dataset.simulation.engine import run_simulation
from scm_dataset.validation.constraints import (
    check_deliveries_follow_orders,
    check_event_references,
    check_negative_inventory,
    check_production_within_capacity,
    run_constraint_checks,
)
from scm_dataset.validation.report import (
    validate_geographic_isolation,
    validate_severity_monotonicity,
    validate_single_source_concentration,
)
from scm_dataset.validation.statistical import compare_region_risk_distributions
from scm_dataset.validation.topology import (
    compare_topology_to_nist,
    compute_connectivity_stats,
    compute_graph_connectivity,
    compute_multi_source_rate,
    compute_supplier_concentration,
)

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def _small_config(seed: int = 42, horizon_periods: int = 20, events: EventsConfig | None = None) -> GeneratorConfig:
    return GeneratorConfig(
        seed=seed,
        network=NetworkConfig(suppliers=20, procurement_orders=60, materials=15, plants=6, products=15, regions=4),
        simulation=SimulationConfig(horizon_periods=horizon_periods),
        events=events or EventsConfig(),
    )


def _generate_and_export(tmp_dir: str, events: EventsConfig | None = None):
    config = _small_config(events=events)
    graph = generate_graph(config)
    result = run_simulation(graph, config)
    export_graph(graph, tmp_dir)
    export_operations(result, tmp_dir)
    export_events(result.events, tmp_dir)
    return graph, result


# ---- constraints.py ----


def test_run_constraint_checks_passes_on_a_freshly_generated_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        _generate_and_export(tmp)
        graph = load_graph(tmp)
        operations = load_operations(tmp)
        events = load_events(tmp)
        report = run_constraint_checks(graph, operations, events)

    for check_name, result in report.items():
        assert result == [], f"{check_name} unexpectedly found: {result}"


def test_check_negative_inventory_catches_a_corrupted_value():
    df = pd.DataFrame([{"plant_id": "p0", "material_id": "m0", "time": 3, "inventory": -5.0}])
    errors = check_negative_inventory(df)
    assert len(errors) == 1
    assert "p0/m0" in errors[0]


def test_check_negative_inventory_ignores_healthy_values():
    df = pd.DataFrame([{"plant_id": "p0", "material_id": "m0", "time": 3, "inventory": 12.0}])
    assert check_negative_inventory(df) == []


def test_check_production_within_capacity_catches_overshoot():
    with tempfile.TemporaryDirectory() as tmp:
        graph, _result = _generate_and_export(tmp)
    from scm_dataset.schema.nodes import NodeType

    plant = graph.nodes_of_type(NodeType.PLANT)[0]
    bad_df = pd.DataFrame([{"plant_id": plant.plant_id, "time": 0, "production": plant.production_capacity * 10}])
    errors = check_production_within_capacity(bad_df, graph)
    assert len(errors) == 1
    assert plant.plant_id in errors[0]


def test_check_deliveries_follow_orders_catches_orphan_delivery():
    deliveries = pd.DataFrame([{"plant_id": "p0", "material_id": "m0", "supplier_id": "s0", "time": 5, "quantity": 10.0}])
    procurement = pd.DataFrame([{"plant_id": "px", "material_id": "mx", "supplier_id": "sx", "time_ordered": 1}])
    errors = check_deliveries_follow_orders(deliveries, procurement)
    assert len(errors) == 1
    assert "no matching procurement order" in errors[0]


def test_check_deliveries_follow_orders_catches_delivery_not_after_order():
    deliveries = pd.DataFrame([{"plant_id": "p0", "material_id": "m0", "supplier_id": "s0", "time": 5, "quantity": 10.0}])
    procurement = pd.DataFrame([{"plant_id": "p0", "material_id": "m0", "supplier_id": "s0", "time_ordered": 5}])
    errors = check_deliveries_follow_orders(deliveries, procurement)
    assert len(errors) == 1
    assert "at or before" in errors[0]


def test_check_event_references_catches_unknown_supplier():
    with tempfile.TemporaryDirectory() as tmp:
        graph, _result = _generate_and_export(tmp)
    from scm_dataset.schema.events import Event, EventType

    bad_event = Event(
        event_id="e_bad", event_type=EventType.SUPPLIER_FAILURE, start_time=0, duration=1, severity=1,
        probability_regime=0.01, affected_suppliers=["totally_made_up_supplier"],
    )
    errors = check_event_references(graph, [bad_event])
    assert len(errors) == 1
    assert "totally_made_up_supplier" in errors[0]


# ---- topology.py ----


def test_supplier_concentration_reports_plausible_values():
    graph = generate_graph(_small_config())
    stats = compute_supplier_concentration(graph)
    assert 0.0 <= stats["gini"] <= 1.0
    assert 0.0 <= stats["top_decile_share"] <= 1.0
    assert stats["n_suppliers_with_edges"] > 0


def test_multi_source_rate_in_unit_range():
    graph = generate_graph(_small_config())
    assert 0.0 <= compute_multi_source_rate(graph) <= 1.0


def test_connectivity_stats_show_no_orphans_by_construction():
    graph = generate_graph(_small_config())
    stats = compute_connectivity_stats(graph)
    assert stats["materials_with_no_supplier"] == 0
    assert stats["materials_with_no_plant"] == 0


def test_graph_connectivity_reports_a_large_weakly_connected_component():
    graph = generate_graph(_small_config())
    stats = compute_graph_connectivity(graph)
    assert stats["num_components"] >= 1
    assert 0.0 < stats["largest_component_fraction"] <= 1.0


def test_compare_topology_to_nist_returns_none_for_missing_file(tmp_path):
    graph = generate_graph(_small_config())
    assert compare_topology_to_nist(graph, str(tmp_path / "nope.json")) is None


@pytest.mark.skipif(not os.path.exists(os.path.join(PROCESSED_DIR, "nist_structural_stats.json")), reason="run scripts/calibrate.py first")
def test_compare_topology_to_nist_with_real_stats():
    graph = generate_graph(_small_config())
    result = compare_topology_to_nist(graph, os.path.join(PROCESSED_DIR, "nist_structural_stats.json"))
    assert result is not None
    assert 0.0 <= result["synthetic_multi_source_rate"] <= 1.0


# ---- statistical.py ----


@pytest.mark.skipif(not os.path.exists(os.path.join(PROCESSED_DIR, "region_risk_calibration.csv")), reason="run scripts/calibrate.py first")
def test_compare_region_risk_distributions_shape():
    real_table = pd.read_csv(os.path.join(PROCESSED_DIR, "region_risk_calibration.csv"))
    rng = np.random.default_rng(0)
    comparison = compare_region_risk_distributions(real_table, rng, n_synthetic=50)
    assert len(comparison) == 4
    assert set(comparison.columns) == {"feature", "real_mean", "synthetic_mean", "real_std", "synthetic_std", "ks_statistic", "wasserstein_distance"}
    assert (comparison.ks_statistic >= 0.0).all()
    assert (comparison.wasserstein_distance >= 0.0).all()


# ---- report.py (event validation) ----


def test_severity_monotonicity_holds_on_a_real_graph():
    graph = generate_graph(_small_config())
    rng = np.random.default_rng(0)
    result = validate_severity_monotonicity(graph, rng, n_samples=200)
    assert result["severity_5_exceeds_severity_1"]
    assert result["mean_capacity_reduction_severity_5"] > result["mean_capacity_reduction_severity_1"]


def test_geographic_isolation_has_no_violations():
    graph = generate_graph(_small_config())
    rng = np.random.default_rng(0)
    result = validate_geographic_isolation(graph, rng, n_samples=50)
    assert result["violations"] == []


def test_single_source_concentration_has_no_violations():
    graph = generate_graph(_small_config())
    result = validate_single_source_concentration(graph)
    assert result["violations"] == []
    assert result["num_single_source_materials"] >= 0
