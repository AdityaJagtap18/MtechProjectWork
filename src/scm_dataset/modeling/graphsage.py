"""2-layer heterogeneous GraphSAGE (plan §17-20/§27-32).

    type-specific input projection
                |
    HeteroGraphSAGE layer 1 (mean aggregation)
                |
    HeteroGraphSAGE layer 2 (mean aggregation)
                |
        supplier embeddings
                |
              MLP
                |
             logit

No sigmoid inside the model: `forward` returns raw logits so the training
loop can use `BCEWithLogitsLoss` directly (plan §20/§33); callers that want
a probability call `torch.sigmoid` on the logits themselves (see
`evaluate.py`). No attention, gating, or per-relation weighting is used --
deliberately, per plan §18/§29, to keep this a "clean classical baseline."
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroGraphSAGE(nn.Module):
    def __init__(
        self,
        in_dims: dict[str, int],
        edge_types: list[tuple[str, str, str]],
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.20,
        aggregation: str = "mean",
        readout_node_type: str = "supplier",
    ) -> None:
        super().__init__()
        self.readout_node_type = readout_node_type
        self.input_proj = nn.ModuleDict({nt: nn.Linear(dim, hidden_dim) for nt, dim in in_dims.items()})

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {rel: SAGEConv(hidden_dim, hidden_dim, aggr=aggregation) for rel in edge_types},
                aggr="mean",  # how a node type combines messages arriving via *different* relations
            )
            self.convs.append(conv)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[tuple[str, str, str], torch.Tensor]) -> dict[str, torch.Tensor]:
        h_dict = {nt: self.input_proj[nt](x) for nt, x in x_dict.items()}
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {nt: self.dropout(F.relu(h)) for nt, h in h_dict.items()}
        return h_dict

    def forward(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[tuple[str, str, str], torch.Tensor]) -> torch.Tensor:
        """Returns raw supplier logits, shape [num_suppliers]."""
        h_dict = self.encode(x_dict, edge_index_dict)
        logits = self.classifier(h_dict[self.readout_node_type]).squeeze(-1)
        return logits


def build_model(
    in_dims: dict[str, int],
    edge_types: list[tuple[str, str, str]],
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.20,
    aggregation: str = "mean",
) -> HeteroGraphSAGE:
    return HeteroGraphSAGE(
        in_dims=in_dims, edge_types=edge_types, hidden_dim=hidden_dim, num_layers=num_layers,
        dropout=dropout, aggregation=aggregation, readout_node_type="supplier",
    )
