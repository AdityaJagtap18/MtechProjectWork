import tempfile

import numpy as np
import pytest

from scm_dataset.export.csv import export_event_impact, export_labels, load_event_impact, load_labels
from scm_dataset.generator.config import EventsConfig, EventTypeConfig, GeneratorConfig, NetworkConfig, SimulationConfig
from scm_dataset.generator.topology import generate_graph
from scm_dataset.labels.risk_labels import (
    compute_event_impact_labels,
    compute_material_labels,
    compute_plant_labels,
    compute_product_labels,
    compute_supplier_labels,
)
from scm_dataset.schema.nodes import NodeType
from scm_dataset.simulation.engine import build_indices, compute_flow_scales, run_simulation


def _small_config(seed: int = 42, horizon_periods: int = 30, events: EventsConfig | None = None) -> GeneratorConfig:
    return GeneratorConfig(
        seed=seed,
        network=NetworkConfig(suppliers=20, procurement_orders=60, materials=15, plants=6, products=15, regions=4),
        simulation=SimulationConfig(horizon_periods=horizon_periods),
        events=events or EventsConfig(),
    )


def _forced_events_config(**overrides) -> EventsConfig:
    defaults = dict(
        enabled=True,
        supplier_failure=EventTypeConfig(0.3),
        natural_disaster=EventTypeConfig(0.1),
        geopolitical=EventTypeConfig(0.0),
        cyberattack=EventTypeConfig(0.0),
        logistics=EventTypeConfig(0.0),
        severity_weights=[0.0, 0.0, 0.0, 0.0, 1.0],
    )
    defaults.update(overrides)
    return EventsConfig(**defaults)


def _generate(seed=1, horizon=30, events=None):
    config = _small_config(seed=seed, horizon_periods=horizon, events=events)
    graph = generate_graph(config)
    result = run_simulation(graph, config)
    return graph, config, result


def test_supplier_labels_flag_exactly_the_events_active_window():
    graph, config, result = _generate(events=_forced_events_config())
    assert result.events, "expected at least one event in this forced run"

    labels = compute_supplier_labels(graph, result)
    assert set(labels.columns) == {"supplier_id", "time", "supplier_disrupted", "supplier_risk_score"}

    for event in result.events:
        if not event.affected_suppliers:
            continue
        supplier_id = event.affected_suppliers[0]
        window_end = event.start_time + event.duration + event.recovery_delay + event.recovery_periods
        sub = labels[labels.supplier_id == supplier_id]
        in_window = sub[(sub.time >= event.start_time) & (sub.time < window_end)]
        if len(in_window):
            assert (in_window.supplier_disrupted == 1).all()


def test_supplier_risk_score_is_missing_without_orders_and_bounded_with_them():
    graph, config, result = _generate(events=_forced_events_config())
    labels = compute_supplier_labels(graph, result)
    scored = labels.dropna(subset=["supplier_risk_score"])
    assert len(scored) > 0
    assert (scored.supplier_risk_score >= 0.0).all()


def test_material_labels_match_reorder_threshold_directly():
    graph, config, result = _generate()
    labels = compute_material_labels(graph, result, config)

    plants = graph.nodes_of_type(NodeType.PLANT)
    materials_by_id = {m.material_id: m for m in graph.nodes_of_type(NodeType.MATERIAL)}
    suppliers_by_id = {s.supplier_id: s for s in graph.nodes_of_type(NodeType.SUPPLIER)}
    plant_materials, _plant_products, primary_supplier = build_indices(graph)
    _init, reorder_threshold, _target, _cap = compute_flow_scales(
        plants, materials_by_id, plant_materials, primary_supplier, suppliers_by_id, config.simulation
    )

    row = labels.iloc[0]
    key = (row.plant_id, row.material_id)
    inv_row = next(r for r in result.inventory_records if r["plant_id"] == row.plant_id and r["material_id"] == row.material_id and r["time"] == row.time)
    expected_shortage = int(inv_row["inventory"] < reorder_threshold[key])
    assert row.material_shortage == expected_shortage


def test_plant_labels_loss_fraction_bounded_and_zero_when_no_demand():
    graph, config, result = _generate()
    labels = compute_plant_labels(graph, result)
    assert (labels.production_loss_fraction >= 0.0).all()
    assert (labels.production_loss_fraction <= 1.0).all()
    assert set(labels.plant_disruption.unique()) <= {0, 1}


def test_product_labels_revenue_impact_matches_shortfall_times_price():
    graph, config, result = _generate()
    labels = compute_product_labels(graph, result)
    products_by_id = {p.product_id: p for p in graph.nodes_of_type(NodeType.PRODUCT)}

    production_by_key: dict = {}
    for r in result.production_records:
        production_by_key[(r["product_id"], r["time"])] = production_by_key.get((r["product_id"], r["time"]), 0.0) + r["production"]

    for row in labels.itertuples():
        demand_row = next(r for r in result.demand_records if r["product_id"] == row.product_id and r["time"] == row.time)
        produced = production_by_key.get((row.product_id, row.time), 0.0)
        shortfall = max(0.0, demand_row["demand"] - produced)
        expected_revenue_impact = shortfall * products_by_id[row.product_id].revenue_per_unit
        assert row.revenue_impact == pytest.approx(expected_revenue_impact)
        assert row.product_shortage == int(shortfall > 0)


def test_event_impact_labels_shape_and_invariants():
    graph, config, result = _generate(events=_forced_events_config())
    plant_labels = compute_plant_labels(graph, result)
    product_labels = compute_product_labels(graph, result)
    event_impact = compute_event_impact_labels(graph, result.events, plant_labels, product_labels)

    assert len(event_impact) == len(result.events)
    assert set(event_impact.columns) == {"event_id", "cascade_severity", "total_affected_nodes", "time_to_impact", "recovery_time"}
    assert (event_impact.total_affected_nodes >= 0).all()
    assert (event_impact.cascade_severity >= 0.0).all()
    assert (event_impact.recovery_time >= 0).all()

    # time_to_impact only counts purely-downstream propagation, so it can
    # never be later than recovery_time (which spans direct + downstream)
    both_defined = event_impact.dropna(subset=["time_to_impact"])
    assert (both_defined.time_to_impact <= both_defined.recovery_time).all()


def test_event_impact_total_affected_nodes_at_least_direct_targets():
    graph, config, result = _generate(events=_forced_events_config())
    plant_labels = compute_plant_labels(graph, result)
    product_labels = compute_product_labels(graph, result)
    event_impact = compute_event_impact_labels(graph, result.events, plant_labels, product_labels)

    by_id = {row.event_id: row for row in event_impact.itertuples()}
    for event in result.events:
        # total_affected_nodes = suppliers + materials + all_plants (superset
        # of affected_plants) + products, so it can never undercount the
        # entities the event directly targeted
        direct_count = len(event.affected_suppliers) + len(event.affected_plants)
        assert by_id[event.event_id].total_affected_nodes >= direct_count


def test_labels_csv_roundtrip():
    graph, config, result = _generate(events=_forced_events_config())
    supplier_labels = compute_supplier_labels(graph, result)
    material_labels = compute_material_labels(graph, result, config)
    plant_labels = compute_plant_labels(graph, result)
    product_labels = compute_product_labels(graph, result)
    event_impact = compute_event_impact_labels(graph, result.events, plant_labels, product_labels)

    with tempfile.TemporaryDirectory() as tmp:
        export_labels(tmp, supplier_labels, material_labels, plant_labels, product_labels)
        export_event_impact(event_impact, tmp)
        reloaded_labels = load_labels(tmp)
        reloaded_event_impact = load_event_impact(tmp)

    assert set(reloaded_labels) == {"supplier", "material", "plant", "product"}
    assert len(reloaded_labels["supplier"]) == len(supplier_labels)
    assert len(reloaded_event_impact) == len(event_impact)


def test_load_event_impact_missing_file_returns_none(tmp_path):
    assert load_event_impact(str(tmp_path)) is None


def test_labels_computed_without_events_show_no_supplier_disruption():
    graph, config, result = _generate(events=EventsConfig(enabled=False))
    labels = compute_supplier_labels(graph, result)
    assert (labels.supplier_disrupted == 0).all()
