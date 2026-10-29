"""QGNN-v2: frozen GraphSAGE-Full supplier-embedding extraction + PCA
(QGNN_V2_HYBRID_IMPLEMENTATION_AND_DEPTH_BENCHMARK.md sections 3-4,
QGNN_V2_ARCHITECTURE_ANALYSIS.md's recommended Option C).

    existing, unmodified GraphSAGE-Full checkpoint
        -> loaded, frozen (requires_grad_(False)), eval()
        -> HeteroGraphSAGE.encode(...)["supplier"]   (already a separate
           method on the existing model -- no change to graphsage.py)
        -> [n_suppliers, hidden_dim] embedding per timestep
        -> assembled into a (supplier_id, time)-indexed DataFrame
        -> reduction.PCASupplierReducer (reused UNCHANGED -- it only
           needs a DataFrame + boolean fit_mask + column list, and does
           not care whether those columns are raw supplier features or
           embedding dimensions) fit on train-split rows only

No gradient ever flows into the frozen encoder: extraction runs under
`torch.no_grad()` and every parameter has `requires_grad_(False)` set
explicitly (verified by test, not just asserted in a docstring).
"""

from __future__ import annotations

import pandas as pd
import torch

from .config import GraphSAGEConfig
from .data import BenchmarkData
from .graphsage import HeteroGraphSAGE, build_model
from .pipeline import PreparedData, prepare_from_benchmark
from .preprocessing import FeaturePreprocessor, supplier_fit_mask
from .reduction import PCASupplierReducer


def load_frozen_graphsage_encoder(checkpoint_path: str, prepared: PreparedData) -> HeteroGraphSAGE:
    """Loads an existing GraphSAGE-Full `model.pt`, freezes every
    parameter, and sets eval mode. `prepared` must come from the SAME
    config (full feature mode, primary temporal split, matching
    hidden_dim/num_layers) that originally produced the checkpoint --
    only used here to supply `in_dims`/`edge_types`, never refit."""
    feature_dims = prepared.snapshot_builder.feature_dims()
    in_dims = {nt.value: dim for nt, dim in feature_dims.items()}
    edge_types = list(prepared.snapshot_builder.topology.edge_index_dict.keys())
    model = build_model(
        in_dims=in_dims, edge_types=edge_types, hidden_dim=prepared.config.model.hidden_dim,
        num_layers=prepared.config.model.num_layers, dropout=prepared.config.model.dropout,
        aggregation=prepared.config.model.aggregation,
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    return model


def prepare_frozen_encoder_input(config: GraphSAGEConfig, benchmark: BenchmarkData, preprocessing_dir: str) -> PreparedData:
    """Builds the full (unreduced, 60-dim supplier, full feature mode)
    `PreparedData` needed to feed the frozen encoder -- reusing the
    EXACT `FeaturePreprocessor` that was fit when the checkpoint was
    originally trained (loaded from its saved `preprocessing/`
    directory), never refit."""
    preprocessor = FeaturePreprocessor.load(preprocessing_dir)
    return prepare_from_benchmark(config, benchmark, preprocessor=preprocessor)


def extract_supplier_embeddings(model: HeteroGraphSAGE, prepared: PreparedData, times: list[int]) -> pd.DataFrame:
    """For each `t` in `times`, runs the frozen encoder's `encode()` over
    the full heterogeneous snapshot and records `h_dict["supplier"]`.
    Returns a `(supplier_id, time)`-indexed DataFrame with columns
    `emb_0..emb_{hidden_dim-1}` -- the same indexing convention
    `reduction.py`'s raw supplier frame already uses, so
    `preprocessing.supplier_fit_mask` applies unchanged."""
    supplier_order = prepared.snapshot_builder.supplier_id_order()
    rows = []
    with torch.no_grad():
        for t in times:
            data = prepared.snapshot_builder.build(int(t))
            h = model.encode(data.x_dict, data.edge_index_dict)["supplier"]
            h_np = h.numpy()
            for i, supplier_id in enumerate(supplier_order):
                row = {"supplier_id": supplier_id, "time": int(t)}
                row.update({f"emb_{j}": float(h_np[i, j]) for j in range(h_np.shape[1])})
                rows.append(row)
    return pd.DataFrame(rows).set_index(["supplier_id", "time"])


def fit_embedding_pca(embedding_frame: pd.DataFrame, train_examples: pd.DataFrame, n_components: int) -> PCASupplierReducer:
    """`train_examples` must already be filtered to `split == "train"` --
    mirrors `reduction.fit_supplier_reducer`'s own contract. Reuses
    `PCASupplierReducer` unchanged; its `fit` signature only needs a
    DataFrame, a boolean fit mask, and a column-name list, regardless of
    what those columns represent."""
    fit_mask = supplier_fit_mask(embedding_frame, train_examples)
    reducer = PCASupplierReducer(n_components)
    reducer.fit(embedding_frame, fit_mask, list(embedding_frame.columns))
    return reducer
