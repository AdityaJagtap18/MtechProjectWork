"""Tests for modeling/hetero_graph.py: deterministic node id mapping, edge
validation, and snapshot construction (plan §47-50/§66 Check 8)."""

from __future__ import annotations

import pytest

from scm_dataset.modeling.hetero_graph import (
    HeteroGraphSnapshotBuilder,
    build_edge_index_dict,
    build_node_index_maps,
    build_topology,
)
from scm_dataset.schema.edges import Edge, EdgeType
from scm_dataset.schema.graph import SupplyChainGraph
from scm_dataset.schema.nodes import Material, NodeType, Supplier


def _supplier(sid: str) -> Supplier:
    return Supplier(
        supplier_id=sid, tier=1, region_id="region_0", industry="machinery", capacity=100.0,
        capacity_utilization=0.5, reliability=0.5, financial_health=0.5, lead_time_mean=5.0,
        lead_time_variability=1.0, quality_score=0.5, inventory_buffer=10.0, substitution_availability=0.5,
        geopolitical_exposure=0.1, disaster_exposure=0.1, cyber_exposure=0.1, criticality=0.5,
    )


def _material(mid: str) -> Material:
    return Material(
        material_id=mid, material_category="metal", criticality=0.5, substitutability=0.5, demand=10.0,
        unit_cost=1.0, inventory_level=10.0, safety_stock=5.0, supplier_count=1, concentration=1.0,
        required_quantity_per_product=1.0,
    )


def test_node_index_map_uses_natural_numeric_sort_not_lexical():
    graph = SupplyChainGraph()
    for i in [2, 10, 1]:
        graph.add_node(_supplier(f"supplier_{i}"))
    node_index = build_node_index_maps(graph)[NodeType.SUPPLIER]
    # lexical sort would put supplier_10 before supplier_2; natural sort must not
    assert node_index["supplier_1"] < node_index["supplier_2"] < node_index["supplier_10"]


def test_node_index_map_is_a_bijection_over_all_nodes():
    graph = SupplyChainGraph()
    for i in range(5):
        graph.add_node(_supplier(f"supplier_{i}"))
    node_index = build_node_index_maps(graph)[NodeType.SUPPLIER]
    assert set(node_index.values()) == set(range(5))
    assert len(set(node_index.keys())) == 5


def test_build_edge_index_dict_raises_on_missing_endpoint():
    graph = SupplyChainGraph()
    graph.add_node(_supplier("supplier_0"))
    graph.add_node(_material("material_0"))
    # bypass SupplyChainGraph.add_edge's (nonexistent) validation to simulate
    # a corrupt edge list reaching the graph builder directly
    graph.edges.append(Edge(source_id="supplier_0", target_id="material_does_not_exist", edge_type=EdgeType.SUPPLIER_MATERIAL))
    node_index = build_node_index_maps(graph)
    with pytest.raises(ValueError, match="missing from node index map"):
        build_edge_index_dict(graph, node_index)


def test_build_edge_index_dict_adds_reverse_relation_for_every_forward_relation():
    graph = SupplyChainGraph()
    graph.add_node(_supplier("supplier_0"))
    graph.add_node(_material("material_0"))
    graph.add_edge(Edge(source_id="supplier_0", target_id="material_0", edge_type=EdgeType.SUPPLIER_MATERIAL))
    node_index = build_node_index_maps(graph)
    edge_index_dict = build_edge_index_dict(graph, node_index)

    forward = edge_index_dict[("supplier", "supplier_material", "material")]
    reverse = edge_index_dict[("material", "rev_supplier_material", "supplier")]
    assert forward.shape == (2, 1)
    assert reverse.shape == (2, 1)
    # reverse edge must point the opposite direction of the same pair
    assert reverse[0, 0].item() == forward[1, 0].item()
    assert reverse[1, 0].item() == forward[0, 0].item()


def test_build_edge_index_dict_covers_every_edge_type_including_empty_ones():
    graph = SupplyChainGraph()
    graph.add_node(_supplier("supplier_0"))
    graph.add_node(_material("material_0"))
    graph.add_edge(Edge(source_id="supplier_0", target_id="material_0", edge_type=EdgeType.SUPPLIER_MATERIAL))
    node_index = build_node_index_maps(graph)
    edge_index_dict = build_edge_index_dict(graph, node_index)
    # every one of the 9 edge types gets a (forward, reverse) pair, even
    # types with zero edges in this tiny graph -- HeteroConv needs every
    # relation key present, not just the populated ones.
    assert len(edge_index_dict) == 2 * len(EdgeType)
    empty_relation = edge_index_dict[("procurement", "procurement_material", "material")]
    assert empty_relation.shape == (2, 0)


def test_build_topology_num_nodes_matches_graph(tiny_benchmark):
    topology = build_topology(tiny_benchmark.graph)
    for node_type in NodeType:
        assert topology.num_nodes[node_type] == len(tiny_benchmark.graph.nodes_of_type(node_type))


def test_snapshot_builder_produces_correct_shapes_and_valid_edges(tiny_prepared_data):
    builder = tiny_prepared_data.snapshot_builder
    t = int(tiny_prepared_data.examples["time"].min())
    data = builder.build(t)

    feature_dims = builder.feature_dims()
    for node_type in NodeType:
        x = data.x_dict[node_type.value]
        assert x.shape[0] == builder.topology.num_nodes[node_type]
        assert x.shape[1] == feature_dims[node_type]
        assert bool(x.isnan().any()) is False
        assert bool(x.isinf().any()) is False

    for (src_type, _rel, dst_type), edge_index in data.edge_index_dict.items():
        if edge_index.numel() == 0:
            continue
        assert edge_index[0].max().item() < builder.topology.num_nodes[NodeType(src_type)]
        assert edge_index[1].max().item() < builder.topology.num_nodes[NodeType(dst_type)]
        assert edge_index.min().item() >= 0


def test_snapshot_static_node_types_identical_across_time(tiny_prepared_data):
    builder = tiny_prepared_data.snapshot_builder
    times = sorted(tiny_prepared_data.examples["time"].unique())
    data_a = builder.build(int(times[0]))
    data_b = builder.build(int(times[-1]))
    # region/procurement carry no dynamic component (plan §16/§17): their
    # feature tensors must be byte-identical regardless of snapshot time.
    assert bool((data_a.x_dict["region"] == data_b.x_dict["region"]).all())
    assert bool((data_a.x_dict["procurement"] == data_b.x_dict["procurement"]).all())


def test_snapshot_dynamic_node_types_can_vary_across_time(tiny_prepared_data):
    builder = tiny_prepared_data.snapshot_builder
    times = sorted(tiny_prepared_data.examples["time"].unique())
    data_a = builder.build(int(times[0]))
    data_b = builder.build(int(times[-1]))
    # supplier features include rolling operational aggregates -- across the
    # full usable time range they should not be frozen (a bug that collapsed
    # every snapshot to the same tensor would defeat the entire point of
    # per-timestep dynamic features).
    assert not bool((data_a.x_dict["supplier"] == data_b.x_dict["supplier"]).all())


def test_snapshot_builder_raises_on_out_of_range_time(tiny_prepared_data):
    builder = tiny_prepared_data.snapshot_builder
    with pytest.raises(KeyError):
        builder.build(999999)
