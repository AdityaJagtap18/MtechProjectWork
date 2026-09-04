"""Topology generator: assembles a full heterogeneous `SupplyChainGraph`
(plan §42 Phase 2, extended in Phase 4 for real-data region calibration).
Orchestrates node generation (regions/suppliers/materials/plants/products/
procurement orders) and non-uniform, preferential-attachment-based edge
generation (plan §9), instead of uniform random edges. Operational time
series (Phase 3), black-swan events (Phase 5), and cascade/labels (Phase 6)
are out of scope here.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..real_data.calibrator import load_region_risk_table
from ..schema.edges import Edge, EdgeType
from ..schema.graph import SupplyChainGraph
from ..schema.nodes import Material, Plant, Product, Region, Supplier
from .config import GeneratorConfig
from .materials import generate_materials
from .plants import generate_plants
from .procurement import generate_procurement_orders
from .products import generate_products
from .regions import generate_regions
from .suppliers import generate_suppliers
from .utils import preferential_sample


def _assign_supplier_material_edges(
    config: GeneratorConfig, suppliers: list[Supplier], materials: list[Material], rng: np.random.Generator
) -> tuple[list[Edge], list[tuple[str, str]]]:
    """Supplier -> Material: single- vs multi-source per plan §9, biased
    toward suppliers in the material's own industry, with preferential
    attachment (usage-weighted) driving which specific suppliers get picked
    — this is what produces supplier concentration / critical suppliers."""
    by_industry: dict[str, list[str]] = defaultdict(list)
    for s in suppliers:
        by_industry[s.industry].append(s.supplier_id)
    all_supplier_ids = [s.supplier_id for s in suppliers]

    usage: dict[str, int] = {}
    edges: list[Edge] = []
    pairs: list[tuple[str, str]] = []
    topo = config.topology

    for material in materials:
        if rng.random() < topo.p_single_source_material:
            k = 1
        else:
            k = int(rng.integers(topo.multi_source_min, topo.multi_source_max + 1))

        same_industry_pool = by_industry.get(material.material_category, [])
        pool = same_industry_pool if same_industry_pool and rng.random() < topo.same_industry_bias else all_supplier_ids

        for supplier_id in preferential_sample(rng, pool, usage, k):
            edges.append(Edge(supplier_id, material.material_id, EdgeType.SUPPLIER_MATERIAL))
            pairs.append((supplier_id, material.material_id))

    return edges, pairs


def _assign_material_plant_edges(
    config: GeneratorConfig, materials: list[Material], plants: list[Plant], rng: np.random.Generator
) -> tuple[list[Edge], dict[str, list[str]]]:
    """Material -> Plant: each plant draws its BOM preferentially, weighted
    by material criticality and by how often a material has already been
    picked by other plants (plan §9's "bottleneck materials")."""
    usage: dict[str, int] = {}
    edges: list[Edge] = []
    material_plant_map: dict[str, list[str]] = defaultdict(list)
    topo = config.topology

    material_ids = [m.material_id for m in materials]
    base_weights = [m.criticality + 0.1 for m in materials]

    for plant in plants:
        k = int(rng.integers(topo.materials_per_plant_min, topo.materials_per_plant_max + 1))
        for material_id in preferential_sample(rng, material_ids, usage, k, base_weights=base_weights):
            edges.append(Edge(material_id, plant.plant_id, EdgeType.MATERIAL_PLANT))
            material_plant_map[material_id].append(plant.plant_id)

    return edges, material_plant_map


def _repair_orphan_materials(
    materials: list[Material], plants: list[Plant], material_plant_map: dict[str, list[str]], rng: np.random.Generator
) -> list[Edge]:
    """Guarantee every material is consumed by at least one plant, so it can
    later be procured for — without this, a material with zero draws in
    `_assign_material_plant_edges` (low-probability but possible for large
    material counts / small plant counts) would be a dead end."""
    edges = []
    plant_ids = [p.plant_id for p in plants]
    for material in materials:
        if not material_plant_map.get(material.material_id):
            plant_id = plant_ids[rng.integers(0, len(plant_ids))]
            material_plant_map[material.material_id].append(plant_id)
            edges.append(Edge(material.material_id, plant_id, EdgeType.MATERIAL_PLANT))
    return edges


def _assign_plant_product_edges(
    plants: list[Plant], products: list[Product], rng: np.random.Generator
) -> tuple[list[Edge], dict[str, list[str]]]:
    """Plant -> Product: each product's producing plant is chosen with
    probability proportional to plant production capacity, not uniformly."""
    edges: list[Edge] = []
    plant_product_map: dict[str, list[str]] = defaultdict(list)
    plant_ids = [p.plant_id for p in plants]
    weights = np.array([p.production_capacity for p in plants], dtype=float)
    probs = weights / weights.sum()

    for product in products:
        plant_id = plant_ids[rng.choice(len(plant_ids), p=probs)]
        edges.append(Edge(plant_id, product.product_id, EdgeType.PLANT_PRODUCT))
        plant_product_map[plant_id].append(product.product_id)

    return edges, plant_product_map


def _assign_product_region_edges(
    config: GeneratorConfig,
    products: list[Product],
    plants: list[Plant],
    regions: list[Region],
    plant_product_map: dict[str, list[str]],
    rng: np.random.Generator,
) -> list[Edge]:
    """Product -> Region: a product's market is usually its producing
    plant's own region, occasionally elsewhere (export), biased toward
    regions with more reliable transport."""
    plant_region = {p.plant_id: p.region_id for p in plants}
    product_plant = {pid: plant_id for plant_id, pids in plant_product_map.items() for pid in pids}

    region_ids = [r.region_id for r in regions]
    transport = np.array([r.transport_reliability + 0.1 for r in regions], dtype=float)
    transport_probs = transport / transport.sum()

    edges = []
    for product in products:
        home_plant = product_plant.get(product.product_id)
        if home_plant is not None and rng.random() < config.topology.home_region_bias:
            region_id = plant_region[home_plant]
        else:
            region_id = region_ids[rng.choice(len(region_ids), p=transport_probs)]
        edges.append(Edge(product.product_id, region_id, EdgeType.PRODUCT_REGION))
    return edges


def _direct_region_edges(suppliers: list[Supplier], plants: list[Plant]) -> list[Edge]:
    """Supplier -> Region and Plant -> Region follow directly from each
    node's own `region_id`, assigned at generation time."""
    edges = [Edge(s.supplier_id, s.region_id, EdgeType.SUPPLIER_REGION) for s in suppliers]
    edges += [Edge(p.plant_id, p.region_id, EdgeType.PLANT_REGION) for p in plants]
    return edges


def _procurement_edges(orders) -> list[Edge]:
    edges = []
    for order in orders:
        edges.append(Edge(order.supplier_id, order.procurement_id, EdgeType.SUPPLIER_PROCUREMENT))
        edges.append(Edge(order.procurement_id, order.material_id, EdgeType.PROCUREMENT_MATERIAL))
        edges.append(Edge(order.procurement_id, order.plant_id, EdgeType.PROCUREMENT_PLANT))
    return edges


def _finalize_material_derived_fields(materials: list[Material], supplier_material_pairs: list[tuple[str, str]]) -> None:
    """supplier_count and concentration are computed from the generated
    edges, per DATASET_DESIGN_REVIEW.md §5, not sampled independently.
    concentration uses the equal-share Herfindahl index (1/n) as a simple,
    principled default for "how concentrated is this material's supply.\""""
    counts: dict[str, int] = defaultdict(int)
    for _supplier_id, material_id in supplier_material_pairs:
        counts[material_id] += 1
    for material in materials:
        n = counts.get(material.material_id, 0)
        material.supplier_count = n
        material.concentration = (1.0 / n) if n > 0 else 0.0


def _finalize_product_material_dependency(
    products: list[Product],
    plant_product_map: dict[str, list[str]],
    material_plant_map: dict[str, list[str]],
) -> None:
    """Blend the category-level prior set in generate_products() with a
    signal derived from the actual topology: products made by plants with
    larger, more complex BOMs get a higher material_dependency."""
    product_plant = {pid: plant_id for plant_id, pids in plant_product_map.items() for pid in pids}
    plant_material_count: dict[str, int] = defaultdict(int)
    for _material_id, plant_ids in material_plant_map.items():
        for plant_id in plant_ids:
            plant_material_count[plant_id] += 1
    max_count = max(plant_material_count.values(), default=1) or 1

    for product in products:
        plant_id = product_plant.get(product.product_id)
        topology_signal = plant_material_count.get(plant_id, 0) / max_count if plant_id else 0.0
        product.material_dependency = float(np.clip(0.3 * 0.5 + 0.7 * topology_signal, 0.0, 1.0))


def generate_graph(config: GeneratorConfig) -> SupplyChainGraph:
    rng = np.random.default_rng(config.seed)
    net = config.network

    region_risk_table = None
    if config.real_data.use_real_region_calibration:
        region_risk_table = load_region_risk_table(config.real_data.region_risk_table_path)

    regions = generate_regions(net.regions, rng, region_risk_table=region_risk_table)
    suppliers = generate_suppliers(net.suppliers, regions, rng)
    materials = generate_materials(net.materials, rng)
    plants = generate_plants(net.plants, regions, rng)
    products = generate_products(net.products, rng)

    supplier_material_edges, supplier_material_pairs = _assign_supplier_material_edges(config, suppliers, materials, rng)
    material_plant_edges, material_plant_map = _assign_material_plant_edges(config, materials, plants, rng)
    material_plant_edges += _repair_orphan_materials(materials, plants, material_plant_map, rng)
    plant_product_edges, plant_product_map = _assign_plant_product_edges(plants, products, rng)
    product_region_edges = _assign_product_region_edges(config, products, plants, regions, plant_product_map, rng)
    region_edges = _direct_region_edges(suppliers, plants)

    _finalize_material_derived_fields(materials, supplier_material_pairs)
    _finalize_product_material_dependency(products, plant_product_map, material_plant_map)

    suppliers_by_id = {s.supplier_id: s for s in suppliers}
    materials_by_id = {m.material_id: m for m in materials}
    orders = generate_procurement_orders(
        net.procurement_orders, suppliers_by_id, materials_by_id, supplier_material_pairs, material_plant_map, rng
    )
    procurement_edges = _procurement_edges(orders)

    graph = SupplyChainGraph()
    for node in (*regions, *suppliers, *materials, *plants, *products, *orders):
        graph.add_node(node)
    for edge in (
        *supplier_material_edges,
        *material_plant_edges,
        *plant_product_edges,
        *product_region_edges,
        *region_edges,
        *procurement_edges,
    ):
        graph.add_edge(edge)

    return graph
