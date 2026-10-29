"""Dimensionality reduction of the SUPPLIER node's feature representation
(QGNN_FAIR_BENCHMARK_AND_IMPLEMENTATION_PLAN.md sections 4-6, 10-11).

Fits on train-split (supplier, time) rows only -- using the exact same
`supplier_fit_mask` leakage boundary as `FeaturePreprocessor` -- and is
frozen before touching validation/test/target data. Operates on the
SUPPLIER node's raw, pre-`FeaturePreprocessor` combined static+dynamic
frame (`NodeFeatureFrames.frames[NodeType.SUPPLIER]`), not the already
one-hot/missing-indicator-expanded 60-dim representation `FeaturePreprocessor`
produces -- a completely different reducer is needed here (not a reused
`FeaturePreprocessor` instance) because it outputs a different column
schema (`n_components` reduced columns) that `FeaturePreprocessor`'s own
saved fit (column names, one-hot vocab) has no way to represent.

`ReducedGraphSnapshotBuilder` then composes a fitted reducer with an
ordinary, unmodified `HeteroGraphSnapshotBuilder`: every node type other
than SUPPLIER, every edge_index, and the graph topology are exactly what
the base builder already produces -- only the SUPPLIER tensor is
replaced. This is what makes GraphSAGE-Reduced a true isolated control
(plan §3B): it is the *existing* `HeteroGraphSAGE` architecture, doing
full heterogeneous message passing, with nothing changed except the
width of the supplier node's own input.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch_geometric.data import HeteroData

from ..schema.nodes import NodeType
from .features import NodeFeatureFrames
from .hetero_graph import HeteroGraphSnapshotBuilder
from .preprocessing import supplier_fit_mask

# QGNN_FAIR_BENCHMARK_AND_IMPLEMENTATION_PLAN.md §5A's example list, checked
# against the real column names in features.py::STATIC_NUMERIC_FIELDS[SUPPLIER]
# -- these are genuinely present in the repository, not invented.
DOMAIN_SELECTED_COLUMNS: dict[int, list[str]] = {
    4: ["reliability", "financial_health", "capacity_utilization", "criticality"],
    6: ["reliability", "financial_health", "capacity_utilization", "criticality", "geopolitical_exposure", "cyber_exposure"],
    8: [
        "reliability", "financial_health", "capacity_utilization", "criticality",
        "geopolitical_exposure", "cyber_exposure", "lead_time_mean", "quality_score",
    ],
}


class PCASupplierReducer:
    """PCA fit on train-split supplier rows only (standardized first,
    train mean/std), frozen for validation/test/target -- never fit on
    the complete or combined dataset."""

    method = "pca"

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.columns = [f"pca_{i}" for i in range(n_components)]
        self._input_columns: list[str] | None = None
        self._impute_median: pd.Series | None = None
        self._mean: pd.Series | None = None
        self._std: pd.Series | None = None
        self._pca: PCA | None = None

    def fit(self, frame: pd.DataFrame, fit_mask: pd.Series, numeric_columns: list[str]) -> "PCASupplierReducer":
        self._input_columns = list(numeric_columns)
        raw = frame[self._input_columns]
        self._impute_median = raw.loc[fit_mask].median()
        imputed = raw.fillna(self._impute_median)
        fit_imputed = imputed.loc[fit_mask]
        self._mean = fit_imputed.mean()
        self._std = fit_imputed.std().replace(0, 1.0)
        standardized_fit = (fit_imputed - self._mean) / self._std
        self._pca = PCA(n_components=self.n_components, random_state=0)
        self._pca.fit(standardized_fit.values)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        imputed = frame[self._input_columns].fillna(self._impute_median)
        standardized = (imputed - self._mean) / self._std
        reduced = self._pca.transform(standardized.values)
        return pd.DataFrame(reduced, index=frame.index, columns=self.columns)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        payload = {
            "method": self.method, "n_components": self.n_components, "columns": self.columns,
            "input_columns": self._input_columns,
            "impute_median": self._impute_median.to_dict(),
            "mean": self._mean.to_dict(), "std": self._std.to_dict(),
            "pca_components": self._pca.components_.tolist(),
            "pca_mean": self._pca.mean_.tolist(),
            "explained_variance_ratio": self._pca.explained_variance_ratio_.tolist(),
        }
        with open(os.path.join(path, "reducer.json"), "w") as f:
            json.dump(payload, f, indent=2)


class DomainSelectedReducer:
    """Selects a small, fixed, human-interpretable subset of the
    supplier's own existing static fields (no PCA) and standardizes them
    with train-only statistics. Does not invent any feature -- every
    column must already exist in the supplier frame."""

    method = "domain_selected"

    def __init__(self, n_components: int, columns: list[str] | None = None):
        chosen = columns or DOMAIN_SELECTED_COLUMNS.get(n_components)
        if chosen is None:
            raise ValueError(f"no default domain-selected column set for n_components={n_components}; pass `columns` explicitly")
        if len(chosen) != n_components:
            raise ValueError(f"columns has {len(chosen)} entries, expected n_components={n_components}")
        self.n_components = n_components
        self.columns = list(chosen)
        self._impute_median: pd.Series | None = None
        self._mean: pd.Series | None = None
        self._std: pd.Series | None = None

    def fit(self, frame: pd.DataFrame, fit_mask: pd.Series, numeric_columns: list[str]) -> "DomainSelectedReducer":
        missing = [c for c in self.columns if c not in numeric_columns]
        if missing:
            raise ValueError(f"domain-selected columns not found in supplier numeric columns: {missing}")
        raw = frame[self.columns]
        self._impute_median = raw.loc[fit_mask].median()
        imputed = raw.fillna(self._impute_median)
        fit_imputed = imputed.loc[fit_mask]
        self._mean = fit_imputed.mean()
        self._std = fit_imputed.std().replace(0, 1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        imputed = frame[self.columns].fillna(self._impute_median)
        return (imputed - self._mean) / self._std

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        payload = {
            "method": self.method, "n_components": self.n_components, "columns": self.columns,
            "impute_median": self._impute_median.to_dict(),
            "mean": self._mean.to_dict(), "std": self._std.to_dict(),
        }
        with open(os.path.join(path, "reducer.json"), "w") as f:
            json.dump(payload, f, indent=2)


def build_reducer(method: str, n_components: int, columns: list[str] | None = None):
    if method == "pca":
        return PCASupplierReducer(n_components)
    if method == "domain_selected":
        return DomainSelectedReducer(n_components, columns=columns)
    raise ValueError(f"unknown reduction method {method!r}, expected 'pca' or 'domain_selected'")


def fit_supplier_reducer(method: str, n_components: int, frames: NodeFeatureFrames, train_examples: pd.DataFrame, columns: list[str] | None = None):
    """`train_examples` must already be filtered to `split == "train"` --
    mirrors `preprocessing.supplier_fit_mask`'s own contract exactly, so
    fitting the reducer respects the identical train-only boundary as the
    existing `FeaturePreprocessor`."""
    frame = frames.frames[NodeType.SUPPLIER]
    fit_mask = supplier_fit_mask(frame, train_examples)
    reducer = build_reducer(method, n_components, columns=columns)
    reducer.fit(frame, fit_mask, frames.numeric_columns[NodeType.SUPPLIER])
    return reducer


class ReducedGraphSnapshotBuilder:
    """Wraps an existing, unmodified `HeteroGraphSnapshotBuilder`,
    replacing only the SUPPLIER node's tensor with a reduced `n_components`-
    dim version. Every other node type, every edge_index, and the graph
    topology come from the base builder untouched."""

    def __init__(self, base_builder: HeteroGraphSnapshotBuilder, reduced_supplier_frame: pd.DataFrame, n_components: int):
        self._base = base_builder
        self._reduced = reduced_supplier_frame
        self._n_components = n_components
        self.topology = base_builder.topology

    def supplier_id_order(self) -> list[str]:
        return self._base.supplier_id_order()

    def feature_dims(self) -> dict[NodeType, int]:
        dims = dict(self._base.feature_dims())
        dims[NodeType.SUPPLIER] = self._n_components
        return dims

    def build(self, t: int) -> HeteroData:
        data = self._base.build(t)
        order = self.supplier_id_order()
        sub = self._reduced.xs(t, level="time").reindex(order)
        if sub.isna().any().any():
            raise ValueError(f"reduced supplier snapshot t={t}: reindex produced NaN -- id/time not found in reduced frame")
        data["supplier"].x = torch.tensor(sub.values.astype(np.float32))
        return data
