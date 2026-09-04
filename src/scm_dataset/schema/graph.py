"""Heterogeneous graph container for the SCM ontology (plan §6, Phase 1 of §42).

Holds typed nodes and edges and validates referential integrity and edge-type
endpoint conformance. This is deliberately just a container + validator: it
does not generate content (see generator/, Phase 2) and does not do the
statistical/topology/event validation of Phase 7 (see validation/) — this is
Phase 1's narrower "valid SCM graph" deliverable.
"""

from __future__ import annotations

from typing import Union

from .edges import EDGE_ENDPOINTS, Edge
from .nodes import Material, NodeType, Plant, ProcurementOrder, Product, Region, Supplier

Node = Union[Supplier, ProcurementOrder, Material, Plant, Product, Region]

NODE_ID_FIELD: dict[NodeType, str] = {
    NodeType.SUPPLIER: "supplier_id",
    NodeType.PROCUREMENT: "procurement_id",
    NodeType.MATERIAL: "material_id",
    NodeType.PLANT: "plant_id",
    NodeType.PRODUCT: "product_id",
    NodeType.REGION: "region_id",
}

NODE_CLASS_BY_TYPE: dict[NodeType, type] = {
    NodeType.SUPPLIER: Supplier,
    NodeType.PROCUREMENT: ProcurementOrder,
    NodeType.MATERIAL: Material,
    NodeType.PLANT: Plant,
    NodeType.PRODUCT: Product,
    NodeType.REGION: Region,
}


def node_id(node: Node) -> str:
    return getattr(node, NODE_ID_FIELD[node.node_type])


class SupplyChainGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.node_types: dict[str, NodeType] = {}
        self.edges: list[Edge] = []

    def add_node(self, node: Node) -> None:
        nid = node_id(node)
        if nid in self.nodes:
            raise ValueError(
                f"Duplicate node id {nid!r} (existing type={self.node_types[nid]}, "
                f"new type={node.node_type})"
            )
        self.nodes[nid] = node
        self.node_types[nid] = node.node_type

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def nodes_of_type(self, node_type: NodeType) -> list[Node]:
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def edges_of_type(self, edge_type) -> list[Edge]:
        return [e for e in self.edges if e.edge_type == edge_type]

    def validate(self) -> list[str]:
        errors: list[str] = []
        for nid, node in self.nodes.items():
            for msg in node.validate():
                errors.append(f"node {nid}: {msg}")

        for i, edge in enumerate(self.edges):
            expected = EDGE_ENDPOINTS.get(edge.edge_type)
            if expected is None:
                errors.append(f"edge[{i}]: unknown edge_type {edge.edge_type!r}")
                continue
            src_type, tgt_type = expected

            if edge.source_id not in self.nodes:
                errors.append(f"edge[{i}] ({edge.edge_type}): source_id {edge.source_id!r} does not exist")
            elif self.node_types[edge.source_id] != src_type:
                errors.append(
                    f"edge[{i}] ({edge.edge_type}): source_id {edge.source_id!r} has type "
                    f"{self.node_types[edge.source_id]}, expected {src_type}"
                )

            if edge.target_id not in self.nodes:
                errors.append(f"edge[{i}] ({edge.edge_type}): target_id {edge.target_id!r} does not exist")
            elif self.node_types[edge.target_id] != tgt_type:
                errors.append(
                    f"edge[{i}] ({edge.edge_type}): target_id {edge.target_id!r} has type "
                    f"{self.node_types[edge.target_id]}, expected {tgt_type}"
                )

        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0

    def assert_valid(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError(f"Graph has {len(errors)} validation error(s):\n" + "\n".join(errors))
