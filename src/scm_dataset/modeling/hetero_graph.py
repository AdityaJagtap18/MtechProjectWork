"""Heterogeneous PyG graph construction (plan §9/§10/§47-50).

The topology is static -- built once from `graph/nodes.csv`/`graph/
edges.csv` -- and reused for every prediction-time snapshot; only node
feature tensors change per snapshot (plan §74's "static topology + time-
indexed features rather than rebuilding arbitrary future graphs").

Node id -> index mapping (plan §48) is a deterministic natural sort (by
each id's trailing integer, e.g. `supplier_2` before `supplier_10`) so the
same benchmark always produces the same mapping, independent of row order
in `nodes.csv`.

Edge validation (plan §47/§49) piggybacks on `SupplyChainGraph.assert_valid`
(already called in `data.py`) for referential integrity, and additionally
checks here that every endpoint actually made it into the id->index map --
this file never silently drops an edge; a missing endpoint is a hard error.

Reverse relations (plan §50): for each of the 9 existing edge types, a
`rev_<edge_type>` relation with source/target swapped is added *only* to
the PyG representation returned here, so a 2-layer GraphSAGE can propagate
information in both directions (e.g. plant -> material -> supplier, not
only supplier -> material -> plant). This is deterministic, adds no new
semantic information (every reverse edge is implied by an existing forward
edge), and never touches the on-disk benchmark.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from ..schema.edges import EDGE_ENDPOINTS, EdgeType
from ..schema.graph import SupplyChainGraph
from ..schema.nodes import NodeType
from .features import ID_FIELD, NodeFeatureFrames
from .preprocessing import FeaturePreprocessor

_TRAILING_INT = re.compile(r"(\d+)$")


def _natural_sort_key(node_id: str):
    m = _TRAILING_INT.search(node_id)
    return (node_id[: m.start()], int(m.group(1))) if m else (node_id, -1)


def build_node_index_maps(graph: SupplyChainGraph) -> dict[NodeType, dict[str, int]]:
    maps: dict[NodeType, dict[str, int]] = {}
    for node_type in NodeType:
        ids = sorted((getattr(n, ID_FIELD[node_type]) for n in graph.nodes_of_type(node_type)), key=_natural_sort_key)
        maps[node_type] = {nid: i for i, nid in enumerate(ids)}
    return maps


def _ids_in_order(index_map: dict[str, int]) -> list[str]:
    return [nid for nid, _ in sorted(index_map.items(), key=lambda kv: kv[1])]


def build_edge_index_dict(
    graph: SupplyChainGraph, node_index: dict[NodeType, dict[str, int]]
) -> dict[tuple[str, str, str], torch.Tensor]:
    by_type: dict[EdgeType, list[tuple[int, int]]] = {et: [] for et in EdgeType}
    for i, edge in enumerate(graph.edges):
        src_type, dst_type = EDGE_ENDPOINTS[edge.edge_type]
        src_map, dst_map = node_index[src_type], node_index[dst_type]
        if edge.source_id not in src_map:
            raise ValueError(f"edge[{i}] ({edge.edge_type}): source_id {edge.source_id!r} missing from node index map")
        if edge.target_id not in dst_map:
            raise ValueError(f"edge[{i}] ({edge.edge_type}): target_id {edge.target_id!r} missing from node index map")
        by_type[edge.edge_type].append((src_map[edge.source_id], dst_map[edge.target_id]))

    edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = {}
    for edge_type, pairs in by_type.items():
        src_type, dst_type = EDGE_ENDPOINTS[edge_type]
        arr = np.array(pairs, dtype=np.int64).T if pairs else np.zeros((2, 0), dtype=np.int64)
        forward = torch.from_numpy(arr)
        edge_index_dict[(src_type.value, edge_type.value, dst_type.value)] = forward
        edge_index_dict[(dst_type.value, f"rev_{edge_type.value}", src_type.value)] = forward.flip(0)
    return edge_index_dict


@dataclass
class GraphTopology:
    node_index: dict[NodeType, dict[str, int]]
    edge_index_dict: dict[tuple[str, str, str], torch.Tensor]
    num_nodes: dict[NodeType, int]


def build_topology(graph: SupplyChainGraph) -> GraphTopology:
    node_index = build_node_index_maps(graph)
    edge_index_dict = build_edge_index_dict(graph, node_index)
    num_nodes = {nt: len(node_index[nt]) for nt in NodeType}
    return GraphTopology(node_index=node_index, edge_index_dict=edge_index_dict, num_nodes=num_nodes)


class HeteroGraphSnapshotBuilder:
    """Builds one `HeteroData` per prediction time `t`. Feature tensors for
    every node type are precomputed once (transformed through the already-
    fit `FeaturePreprocessor`), so `build(t)` is just an index lookup + a
    dict assembly -- cheap enough to call once per distinct `t` used by any
    split without caching concerns (plan §54: sparse edge_index throughout,
    no dense N x N adjacency is ever constructed)."""

    def __init__(self, graph: SupplyChainGraph, preprocessor: FeaturePreprocessor, frames: NodeFeatureFrames) -> None:
        self.topology = build_topology(graph)
        self._id_order = {nt: _ids_in_order(self.topology.node_index[nt]) for nt in NodeType}
        self._static_x: dict[NodeType, torch.Tensor] = {}
        self._dynamic_transformed: dict[NodeType, pd.DataFrame] = {}

        for node_type in (NodeType.REGION, NodeType.PROCUREMENT):
            transformed = preprocessor.transform(node_type, frames.frames[node_type])
            ordered = transformed.reindex(self._id_order[node_type])
            self._static_x[node_type] = torch.tensor(ordered.values.astype(np.float32))

        for node_type in (NodeType.SUPPLIER, NodeType.MATERIAL, NodeType.PLANT, NodeType.PRODUCT):
            transformed = preprocessor.transform(node_type, frames.frames[node_type])
            self._dynamic_transformed[node_type] = transformed

    def build(self, t: int) -> HeteroData:
        data = HeteroData()
        for node_type in (NodeType.REGION, NodeType.PROCUREMENT):
            data[node_type.value].x = self._static_x[node_type]
        for node_type in (NodeType.SUPPLIER, NodeType.MATERIAL, NodeType.PLANT, NodeType.PRODUCT):
            transformed = self._dynamic_transformed[node_type]
            sub = transformed.xs(t, level="time")
            ordered = sub.reindex(self._id_order[node_type])
            if ordered.isna().any().any():
                raise ValueError(f"snapshot t={t}, node_type={node_type}: reindex produced NaN -- id/time not found in feature frame")
            data[node_type.value].x = torch.tensor(ordered.values.astype(np.float32))
        for rel, edge_index in self.topology.edge_index_dict.items():
            data[rel].edge_index = edge_index
        return data

    def feature_dims(self) -> dict[NodeType, int]:
        return {nt: t.shape[1] for nt, t in self._static_x.items()} | {
            nt: df.shape[1] for nt, df in self._dynamic_transformed.items()
        }

    def supplier_id_order(self) -> list[str]:
        return self._id_order[NodeType.SUPPLIER]
