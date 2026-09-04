import tempfile

import pytest

from scm_dataset.export.csv import export_graph, load_graph
from scm_dataset.schema.edges import Edge, EdgeType
from scm_dataset.schema.graph import SupplyChainGraph
from scm_dataset.schema.nodes import Material, Plant, ProcurementOrder, Product, Region, Supplier


def make_minimal_graph() -> SupplyChainGraph:
    graph = SupplyChainGraph()
    graph.add_node(
        Region(
            region_id="R1", country_or_region="Testland", geopolitical_risk=0.2,
            natural_disaster_risk=0.1, infrastructure_risk=0.1, trade_risk=0.2,
            cyber_risk=0.1, transport_reliability=0.9,
        )
    )
    graph.add_node(
        Supplier(
            supplier_id="S1", tier=1, region_id="R1", industry="electronics",
            capacity=1000.0, capacity_utilization=0.6, reliability=0.9,
            financial_health=0.8, lead_time_mean=5.0, lead_time_variability=1.0,
            quality_score=0.95, inventory_buffer=50.0, substitution_availability=0.5,
            geopolitical_exposure=0.2, disaster_exposure=0.1, cyber_exposure=0.1,
            criticality=0.7,
        )
    )
    graph.add_node(
        Material(
            material_id="M1", material_category="chip", criticality=0.8,
            substitutability=0.3, demand=200.0, unit_cost=10.0, inventory_level=100.0,
            safety_stock=20.0, supplier_count=1, concentration=1.0,
            required_quantity_per_product=2.0, procurement_type="MTS",
        )
    )
    graph.add_node(
        Plant(
            plant_id="P1", region_id="R1", production_capacity=500.0, utilization=0.7,
            operating_cost=1000.0, inventory_capacity=200.0, resilience_score=0.6,
            downtime_cost=5000.0, recovery_rate=0.2,
        )
    )
    graph.add_node(
        Product(
            product_id="PR1", product_category="widget", demand=300.0,
            revenue_per_unit=20.0, margin=0.3, material_dependency=0.8,
            criticality=0.6, substitution_score=0.4, backlog=10.0,
        )
    )
    graph.add_node(
        ProcurementOrder(
            procurement_id="O1", supplier_id="S1", material_id="M1", plant_id="P1",
            order_quantity=50.0, order_value=500.0, order_frequency=1.0,
            promised_lead_time=5.0, actual_lead_time=6.0, urgency=0.3,
            contract_duration=180.0, priority=0.5,
        )
    )

    graph.add_edge(Edge("S1", "M1", EdgeType.SUPPLIER_MATERIAL))
    graph.add_edge(Edge("S1", "O1", EdgeType.SUPPLIER_PROCUREMENT))
    graph.add_edge(Edge("O1", "M1", EdgeType.PROCUREMENT_MATERIAL))
    graph.add_edge(Edge("O1", "P1", EdgeType.PROCUREMENT_PLANT))
    graph.add_edge(Edge("M1", "P1", EdgeType.MATERIAL_PLANT))
    graph.add_edge(Edge("P1", "PR1", EdgeType.PLANT_PRODUCT))
    graph.add_edge(Edge("PR1", "R1", EdgeType.PRODUCT_REGION))
    graph.add_edge(Edge("S1", "R1", EdgeType.SUPPLIER_REGION))
    graph.add_edge(Edge("P1", "R1", EdgeType.PLANT_REGION))
    return graph


def test_minimal_graph_is_valid():
    graph = make_minimal_graph()
    assert graph.is_valid(), graph.validate()


def test_out_of_range_score_is_rejected():
    supplier = Supplier(
        supplier_id="S2", tier=1, region_id="R1", industry="x", capacity=1.0,
        capacity_utilization=1.5, reliability=0.5, financial_health=0.5,
        lead_time_mean=1.0, lead_time_variability=0.1, quality_score=0.5,
        inventory_buffer=1.0, substitution_availability=0.5, geopolitical_exposure=0.5,
        disaster_exposure=0.5, cyber_exposure=0.5, criticality=0.5,
    )
    errors = supplier.validate()
    assert any("capacity_utilization" in e for e in errors)


def test_invalid_tier_is_rejected():
    supplier = Supplier(
        supplier_id="S3", tier=0, region_id="R1", industry="x", capacity=1.0,
        capacity_utilization=0.5, reliability=0.5, financial_health=0.5,
        lead_time_mean=1.0, lead_time_variability=0.1, quality_score=0.5,
        inventory_buffer=1.0, substitution_availability=0.5, geopolitical_exposure=0.5,
        disaster_exposure=0.5, cyber_exposure=0.5, criticality=0.5,
    )
    errors = supplier.validate()
    assert any("tier" in e for e in errors)


def test_invalid_procurement_type_is_rejected():
    material = Material(
        material_id="M2", material_category="x", criticality=0.5, substitutability=0.5,
        demand=1.0, unit_cost=1.0, inventory_level=1.0, safety_stock=1.0,
        supplier_count=1, concentration=0.5, required_quantity_per_product=1.0,
        procurement_type="BOGUS",
    )
    errors = material.validate()
    assert any("procurement_type" in e for e in errors)


def test_edge_with_wrong_endpoint_type_is_rejected():
    graph = make_minimal_graph()
    graph.add_edge(Edge("S1", "P1", EdgeType.SUPPLIER_MATERIAL))  # P1 is a Plant, not a Material
    errors = graph.validate()
    assert any("P1" in e and "expected" in e for e in errors)


def test_edge_referencing_missing_node_is_rejected():
    graph = make_minimal_graph()
    graph.add_edge(Edge("S1", "GHOST", EdgeType.SUPPLIER_MATERIAL))
    errors = graph.validate()
    assert any("GHOST" in e for e in errors)


def test_duplicate_node_id_is_rejected():
    graph = make_minimal_graph()
    with pytest.raises(ValueError):
        graph.add_node(
            Material(
                material_id="M1", material_category="dup", criticality=0.1,
                substitutability=0.1, demand=1.0, unit_cost=1.0, inventory_level=1.0,
                safety_stock=1.0, supplier_count=1, concentration=1.0,
                required_quantity_per_product=1.0,
            )
        )


def test_csv_roundtrip_preserves_graph():
    graph = make_minimal_graph()
    with tempfile.TemporaryDirectory() as tmp:
        export_graph(graph, tmp)
        reloaded = load_graph(tmp)

    assert reloaded.is_valid(), reloaded.validate()
    assert set(reloaded.nodes) == set(graph.nodes)
    assert len(reloaded.edges) == len(graph.edges)

    supplier = reloaded.nodes["S1"]
    assert supplier.node_type.value == "supplier"
    assert supplier.tier == 1
    assert isinstance(supplier.tier, int)

    material = reloaded.nodes["M1"]
    assert material.procurement_type == "MTS"
