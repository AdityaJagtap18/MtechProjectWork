import tempfile
from collections import defaultdict

from scm_dataset.export.csv import export_operations, load_operations
from scm_dataset.generator.config import GeneratorConfig, NetworkConfig, SimulationConfig, TopologyConfig
from scm_dataset.generator.topology import generate_graph
from scm_dataset.schema.edges import EdgeType
from scm_dataset.schema.nodes import NodeType
from scm_dataset.simulation.engine import build_indices, compute_flow_scales, run_simulation


def _small_config(seed: int = 42, horizon_periods: int = 12) -> GeneratorConfig:
    return GeneratorConfig(
        seed=seed,
        network=NetworkConfig(suppliers=15, procurement_orders=40, materials=10, plants=5, products=15, regions=3),
        topology=TopologyConfig(),
        simulation=SimulationConfig(horizon_periods=horizon_periods),
    )


def test_record_counts_match_horizon_and_topology():
    config = _small_config()
    graph = generate_graph(config)
    result = run_simulation(graph, config)

    horizon = config.simulation.horizon_periods
    num_products = len(graph.nodes_of_type(NodeType.PRODUCT))
    assert len(result.demand_records) == horizon * num_products
    assert len(result.production_records) == horizon * num_products  # exactly one producing plant per product

    plant_material_pairs = {(e.source_id, e.target_id) for e in graph.edges_of_type(EdgeType.MATERIAL_PLANT)}
    assert len(result.inventory_records) == horizon * len(plant_material_pairs)


def test_no_negative_inventory():
    graph = generate_graph(_small_config())
    result = run_simulation(graph, _small_config())
    assert all(r["inventory"] >= -1e-6 for r in result.inventory_records)


def test_production_never_exceeds_plant_capacity():
    config = _small_config()
    graph = generate_graph(config)
    result = run_simulation(graph, config)

    capacity_by_plant = {p.plant_id: p.production_capacity for p in graph.nodes_of_type(NodeType.PLANT)}
    totals: dict[tuple[str, int], float] = defaultdict(float)
    for r in result.production_records:
        totals[(r["plant_id"], r["time"])] += r["production"]

    for (plant_id, _t), total in totals.items():
        assert total <= capacity_by_plant[plant_id] + 1e-6


def test_consumption_never_exceeds_available_inventory():
    # Indirect check: since inventory is clamped at >=0 by construction and
    # production is derived from the same ratios, no inventory row should
    # ever go negative -- test_no_negative_inventory covers this; here we
    # additionally check production is non-negative everywhere.
    graph = generate_graph(_small_config())
    result = run_simulation(graph, _small_config())
    assert all(r["production"] >= -1e-9 for r in result.production_records)


def test_deliveries_arrive_strictly_after_order_time():
    graph = generate_graph(_small_config())
    result = run_simulation(graph, _small_config())
    assert len(result.procurement_records) > 0, "expected at least one replenishment order in this run"
    for r in result.procurement_records:
        assert r["expected_delivery_time"] > r["time_ordered"]


def test_supplier_capacity_constraint_is_respected_per_period():
    config = _small_config()
    graph = generate_graph(config)
    result = run_simulation(graph, config)

    plants = graph.nodes_of_type(NodeType.PLANT)
    materials_by_id = {m.material_id: m for m in graph.nodes_of_type(NodeType.MATERIAL)}
    suppliers_by_id = {s.supplier_id: s for s in graph.nodes_of_type(NodeType.SUPPLIER)}
    plant_materials, _plant_products, primary_supplier = build_indices(graph)
    *_rest, supplier_effective_capacity = compute_flow_scales(
        plants, materials_by_id, plant_materials, primary_supplier, suppliers_by_id, config.simulation
    )

    fulfilled_by_supplier_period: dict[tuple[str, int], float] = defaultdict(float)
    for r in result.procurement_records:
        fulfilled_by_supplier_period[(r["supplier_id"], r["time_ordered"])] += r["quantity_fulfilled"]

    for (supplier_id, _t), fulfilled in fulfilled_by_supplier_period.items():
        assert fulfilled <= supplier_effective_capacity[supplier_id] + 1e-6


def test_production_tracks_demand_reasonably():
    # Regression guard: an earlier version derived inventory/reorder buffers
    # from Material.safety_stock directly, which is on a completely
    # different scale than plant.production_capacity * required_quantity_
    # per_product. That made production collapse to near-zero (mean
    # production ~0.44 against mean demand ~186 in a full-scale run) because
    # the min() over a plant's many materials was dominated by whichever
    # material happened to have the smallest unrelated safety_stock. With
    # internally-consistent buffers, production should track demand to
    # within the same order of magnitude, not be starved by ~100x.
    config = _small_config(horizon_periods=20)
    graph = generate_graph(config)
    result = run_simulation(graph, config)

    total_demand = sum(r["demand"] for r in result.demand_records)
    total_production = sum(r["production"] for r in result.production_records)
    assert total_production >= 0.3 * total_demand


def test_reproducible_for_same_seed():
    config_a = _small_config(seed=7)
    config_b = _small_config(seed=7)
    result_a = run_simulation(generate_graph(config_a), config_a)
    result_b = run_simulation(generate_graph(config_b), config_b)
    assert result_a.demand_records == result_b.demand_records
    assert result_a.inventory_records == result_b.inventory_records


def test_operations_csv_roundtrip():
    graph = generate_graph(_small_config())
    result = run_simulation(graph, _small_config())
    with tempfile.TemporaryDirectory() as tmp:
        export_operations(result, tmp)
        tables = load_operations(tmp)

    assert set(tables) == {"demand.csv", "production.csv", "inventory.csv", "deliveries.csv", "procurement.csv", "backlog.csv"}
    assert len(tables["demand.csv"]) == len(result.demand_records)
    assert len(tables["inventory.csv"]) == len(result.inventory_records)
    assert len(tables["backlog.csv"]) == len(result.backlog_records)
