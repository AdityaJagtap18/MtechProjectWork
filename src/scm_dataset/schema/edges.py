"""Edge schema for the heterogeneous SCM graph (plan §6-7).

Nine edge types, each with a fixed (source node type, target node type) pair
and an explicit operational meaning. `PLANT_REGION`'s meaning is not spelled
out in plan §7 (only its existence is listed in §6's edge table); it is
treated analogously to SUPPLIER_REGION as "plant operates in region."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .nodes import NodeType


class EdgeType(str, Enum):
    SUPPLIER_MATERIAL = "supplier_material"
    SUPPLIER_PROCUREMENT = "supplier_procurement"
    PROCUREMENT_MATERIAL = "procurement_material"
    PROCUREMENT_PLANT = "procurement_plant"
    MATERIAL_PLANT = "material_plant"
    PLANT_PRODUCT = "plant_product"
    PRODUCT_REGION = "product_region"
    SUPPLIER_REGION = "supplier_region"
    PLANT_REGION = "plant_region"


EDGE_ENDPOINTS: dict[EdgeType, tuple[NodeType, NodeType]] = {
    EdgeType.SUPPLIER_MATERIAL: (NodeType.SUPPLIER, NodeType.MATERIAL),
    EdgeType.SUPPLIER_PROCUREMENT: (NodeType.SUPPLIER, NodeType.PROCUREMENT),
    EdgeType.PROCUREMENT_MATERIAL: (NodeType.PROCUREMENT, NodeType.MATERIAL),
    EdgeType.PROCUREMENT_PLANT: (NodeType.PROCUREMENT, NodeType.PLANT),
    EdgeType.MATERIAL_PLANT: (NodeType.MATERIAL, NodeType.PLANT),
    EdgeType.PLANT_PRODUCT: (NodeType.PLANT, NodeType.PRODUCT),
    EdgeType.PRODUCT_REGION: (NodeType.PRODUCT, NodeType.REGION),
    EdgeType.SUPPLIER_REGION: (NodeType.SUPPLIER, NodeType.REGION),
    EdgeType.PLANT_REGION: (NodeType.PLANT, NodeType.REGION),
}

EDGE_SEMANTICS: dict[EdgeType, str] = {
    EdgeType.SUPPLIER_MATERIAL: "Supplier provides material.",
    EdgeType.SUPPLIER_PROCUREMENT: "Supplier participates in procurement order.",
    EdgeType.PROCUREMENT_MATERIAL: "Order requests material.",
    EdgeType.PROCUREMENT_PLANT: "Order is intended for plant.",
    EdgeType.MATERIAL_PLANT: "Plant consumes material.",
    EdgeType.PLANT_PRODUCT: "Plant produces product.",
    EdgeType.PRODUCT_REGION: "Product demand/market is associated with region.",
    EdgeType.SUPPLIER_REGION: "Supplier operates in region.",
    EdgeType.PLANT_REGION: "Plant operates in region.",
}


@dataclass
class Edge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    attributes: dict[str, Any] = field(default_factory=dict)
