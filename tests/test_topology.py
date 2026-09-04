import tempfile

from scm_dataset.export.csv import export_graph, load_graph
from scm_dataset.generator.config import GeneratorConfig, NetworkConfig, TopologyConfig
from scm_dataset.generator.topology import generate_graph
from scm_dataset.schema.edges import EdgeType
from scm_dataset.schema.nodes import NodeType


def _small_config(seed: int = 42) -> GeneratorConfig:
    return GeneratorConfig(
        seed=seed,
        network=NetworkConfig(suppliers=20, procurement_orders=50, materials=15, plants=6, products=25, regions=4),
        topology=TopologyConfig(),
    )


def test_generated_graph_is_valid():
    graph = generate_graph(_small_config())
    assert graph.is_valid(), graph.validate()


def test_node_counts_match_config():
    config = _small_config()
    graph = generate_graph(config)
    assert len(graph.nodes_of_type(NodeType.SUPPLIER)) == config.network.suppliers
    assert len(graph.nodes_of_type(NodeType.MATERIAL)) == config.network.materials
    assert len(graph.nodes_of_type(NodeType.PLANT)) == config.network.plants
    assert len(graph.nodes_of_type(NodeType.PRODUCT)) == config.network.products
    assert len(graph.nodes_of_type(NodeType.REGION)) == config.network.regions
    assert len(graph.nodes_of_type(NodeType.PROCUREMENT)) == config.network.procurement_orders


def test_every_material_has_at_least_one_supplier_and_plant():
    graph = generate_graph(_small_config())
    materials = graph.nodes_of_type(NodeType.MATERIAL)
    supplied = {e.target_id for e in graph.edges_of_type(EdgeType.SUPPLIER_MATERIAL)}
    consumed = {e.source_id for e in graph.edges_of_type(EdgeType.MATERIAL_PLANT)}
    for material in materials:
        assert material.material_id in supplied
        assert material.material_id in consumed
        assert material.supplier_count >= 1
        assert 0.0 < material.concentration <= 1.0


def test_single_and_multi_source_materials_both_occur():
    # with the default topology config (p_single_source_material=0.3,
    # multi_source_min=2), a graph with enough materials should produce both
    graph = generate_graph(_small_config())
    counts = {m.material_id: m.supplier_count for m in graph.nodes_of_type(NodeType.MATERIAL)}
    assert any(c == 1 for c in counts.values())
    assert any(c >= 2 for c in counts.values())


def test_generation_is_reproducible_for_same_seed():
    graph_a = generate_graph(_small_config(seed=7))
    graph_b = generate_graph(_small_config(seed=7))
    assert set(graph_a.nodes) == set(graph_b.nodes)
    assert len(graph_a.edges) == len(graph_b.edges)
    for node_id in graph_a.nodes:
        assert graph_a.nodes[node_id] == graph_b.nodes[node_id]


def test_different_seeds_produce_different_graphs():
    graph_a = generate_graph(_small_config(seed=1))
    graph_b = generate_graph(_small_config(seed=2))
    supplier_a = graph_a.nodes["supplier_0"]
    supplier_b = graph_b.nodes["supplier_0"]
    assert supplier_a != supplier_b


def test_procurement_orders_reference_valid_topology():
    graph = generate_graph(_small_config())
    supplier_material_pairs = {
        (e.source_id, e.target_id) for e in graph.edges_of_type(EdgeType.SUPPLIER_MATERIAL)
    }
    material_plant_pairs = {(e.source_id, e.target_id) for e in graph.edges_of_type(EdgeType.MATERIAL_PLANT)}
    for order in graph.nodes_of_type(NodeType.PROCUREMENT):
        assert (order.supplier_id, order.material_id) in supplier_material_pairs
        assert (order.material_id, order.plant_id) in material_plant_pairs


def test_generated_graph_csv_roundtrip():
    graph = generate_graph(_small_config())
    with tempfile.TemporaryDirectory() as tmp:
        export_graph(graph, tmp)
        reloaded = load_graph(tmp)
    assert reloaded.is_valid(), reloaded.validate()
    assert set(reloaded.nodes) == set(graph.nodes)
    assert len(reloaded.edges) == len(graph.edges)
