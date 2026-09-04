"""Structural / topology validation (plan §35 "Structural validation").

There is no real supply-chain *network* dataset available to compare
degree distributions against (DATASET_DESIGN_REVIEW.md decision 1 — the
NIST sample data is a schema exemplar, not a statistically usable network).
This module therefore reports the generated graph's own structural
properties, plus the one real structural comparison Phase 4 did produce:
NIST's observed multi-source rate (`compute_structural_stats`, computed
from real, if small-N, SRM export data).
"""

from __future__ import annotations

import json
import os

import networkx as nx
import numpy as np

from ..schema.edges import EdgeType
from ..schema.graph import SupplyChainGraph
from ..schema.nodes import NodeType


def _gini(values: np.ndarray) -> float:
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    total = sorted_vals.sum()
    if n == 0 or total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_vals)) / (n * total) - (n + 1) / n)


def compute_supplier_concentration(graph: SupplyChainGraph) -> dict:
    """Gini coefficient and top-decile share of Supplier -> Material
    out-degree (plan §9's "supplier concentration" / "critical suppliers",
    now measured rather than just designed for)."""
    degrees: dict[str, int] = {}
    for e in graph.edges_of_type(EdgeType.SUPPLIER_MATERIAL):
        degrees[e.source_id] = degrees.get(e.source_id, 0) + 1
    if not degrees:
        return {"gini": None, "top_decile_share": None, "n_suppliers_with_edges": 0, "mean_degree": None, "max_degree": None}

    values = np.array(sorted(degrees.values()))
    total = values.sum()
    top_decile_n = max(1, len(values) // 10)
    return {
        "gini": _gini(values),
        "top_decile_share": float(values[-top_decile_n:].sum() / total) if total > 0 else None,
        "n_suppliers_with_edges": len(values),
        "mean_degree": float(values.mean()),
        "max_degree": int(values.max()),
    }


def compute_multi_source_rate(graph: SupplyChainGraph) -> float:
    materials = graph.nodes_of_type(NodeType.MATERIAL)
    if not materials:
        return float("nan")
    return float(np.mean([m.supplier_count > 1 for m in materials]))


def compute_connectivity_stats(graph: SupplyChainGraph) -> dict:
    """Orphan checks that should always read zero by construction (Phase 2's
    repair step guarantees every material has >=1 supplier and >=1 plant) —
    a regression signal if this generator is ever changed and breaks that
    guarantee."""
    materials = graph.nodes_of_type(NodeType.MATERIAL)
    materials_with_plant = {e.source_id for e in graph.edges_of_type(EdgeType.MATERIAL_PLANT)}
    return {
        "num_materials": len(materials),
        "materials_with_no_supplier": sum(1 for m in materials if m.supplier_count == 0),
        "materials_with_no_plant": sum(1 for m in materials if m.material_id not in materials_with_plant),
    }


def compute_graph_connectivity(graph: SupplyChainGraph) -> dict:
    """Weakly-connected-component structure and clustering (plan §35's
    "component sizes"/"clustering") over the full heterogeneous graph — a
    reimplementation of the idea behind SupplySim's `get_stats_on_firm_network`
    (see DATASET_DESIGN_REVIEW.md §2.1), not its code."""
    g = nx.DiGraph()
    g.add_nodes_from(graph.nodes.keys())
    g.add_edges_from((e.source_id, e.target_id) for e in graph.edges)

    if g.number_of_nodes() == 0:
        return {"num_components": 0, "largest_component_fraction": None, "average_clustering": None}

    components = list(nx.weakly_connected_components(g))
    largest = max(len(c) for c in components)
    return {
        "num_components": len(components),
        "largest_component_fraction": float(largest / g.number_of_nodes()),
        "average_clustering": float(nx.average_clustering(g.to_undirected())),
    }


def compare_topology_to_nist(graph: SupplyChainGraph, nist_stats_path: str) -> dict | None:
    if not os.path.exists(nist_stats_path):
        return None
    with open(nist_stats_path) as f:
        nist_stats = json.load(f)

    synthetic_rate = compute_multi_source_rate(graph)
    real_rate = nist_stats.get("overall_multi_source_rate")
    return {
        "synthetic_multi_source_rate": synthetic_rate,
        "nist_multi_source_rate": real_rate,
        "difference": (synthetic_rate - real_rate) if real_rate is not None else None,
        "caveat": nist_stats.get("caveat"),
    }


def build_topology_report(graph: SupplyChainGraph, nist_stats_path: str = "data/processed/nist_structural_stats.json") -> dict:
    return {
        "supplier_concentration": compute_supplier_concentration(graph),
        "multi_source_rate": compute_multi_source_rate(graph),
        "connectivity": compute_connectivity_stats(graph),
        "graph_connectivity": compute_graph_connectivity(graph),
        "nist_comparison": compare_topology_to_nist(graph, nist_stats_path),
    }
