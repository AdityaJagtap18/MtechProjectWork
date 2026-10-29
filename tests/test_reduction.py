"""Tests for supplier-feature dimensionality reduction
(QGNN_FAIR_BENCHMARK_AND_IMPLEMENTATION_PLAN.md sections 4-6, 10-11):
train-only fitting, correct output shape, and that GraphSAGE-Reduced's
snapshot builder changes nothing except the supplier node's width."""

from __future__ import annotations

import numpy as np
import pytest

from scm_dataset.modeling.pipeline import prepare_from_benchmark, prepare_reduced_from_benchmark
from scm_dataset.modeling.reduction import (
    DOMAIN_SELECTED_COLUMNS,
    DomainSelectedReducer,
    PCASupplierReducer,
    fit_supplier_reducer,
)
from scm_dataset.schema.nodes import NodeType

from conftest import make_tiny_config


def test_pca_reducer_output_shape_and_no_nan(tiny_benchmark):
    config = make_tiny_config()
    prepared = prepare_from_benchmark(config, tiny_benchmark)
    train_examples = prepared.examples[prepared.examples["split"] == "train"]

    reducer = fit_supplier_reducer("pca", 4, prepared.frames, train_examples)
    reduced = reducer.transform(prepared.frames.frames[NodeType.SUPPLIER])

    assert reduced.shape[1] == 4
    assert list(reduced.columns) == reducer.columns
    assert not reduced.isna().any().any()
    assert reduced.index.equals(prepared.frames.frames[NodeType.SUPPLIER].index)


def test_domain_selected_reducer_uses_real_existing_columns(tiny_benchmark):
    config = make_tiny_config()
    prepared = prepare_from_benchmark(config, tiny_benchmark)
    train_examples = prepared.examples[prepared.examples["split"] == "train"]

    for n in (4, 6, 8):
        assert set(DOMAIN_SELECTED_COLUMNS[n]) <= set(prepared.frames.numeric_columns[NodeType.SUPPLIER])

    reducer = fit_supplier_reducer("domain_selected", 6, prepared.frames, train_examples)
    reduced = reducer.transform(prepared.frames.frames[NodeType.SUPPLIER])
    assert list(reduced.columns) == DOMAIN_SELECTED_COLUMNS[6]
    assert reduced.shape[1] == 6


def test_domain_selected_reducer_rejects_unknown_columns(tiny_benchmark):
    config = make_tiny_config()
    prepared = prepare_from_benchmark(config, tiny_benchmark)
    train_examples = prepared.examples[prepared.examples["split"] == "train"]

    with pytest.raises(ValueError, match="not found"):
        fit_supplier_reducer("domain_selected", 2, prepared.frames, train_examples, columns=["not_a_real_column", "also_fake"])


def test_reducer_fit_never_uses_validation_or_test_rows(tiny_benchmark):
    """Leakage test (plan section 19): tamper the validation/test-period
    rows of the raw supplier frame after fitting normally, refit an
    identical reducer with only the train-period rows swapped out for
    garbage, and confirm the fitted mean/std/PCA components are
    unaffected -- i.e. fitting truly only reads rows the fit_mask marks
    True."""
    config = make_tiny_config()
    prepared = prepare_from_benchmark(config, tiny_benchmark)
    train_examples = prepared.examples[prepared.examples["split"] == "train"]

    frame = prepared.frames.frames[NodeType.SUPPLIER]
    baseline = fit_supplier_reducer("pca", 4, prepared.frames, train_examples)

    from scm_dataset.modeling.preprocessing import supplier_fit_mask

    fit_mask = supplier_fit_mask(frame, train_examples)
    tampered = frame.copy()
    tampered.loc[~fit_mask, prepared.frames.numeric_columns[NodeType.SUPPLIER]] = 1e9  # garbage in non-train rows only

    tampered_reducer = PCASupplierReducer(4)
    tampered_reducer.fit(tampered, fit_mask, prepared.frames.numeric_columns[NodeType.SUPPLIER])

    assert np.allclose(baseline._mean.values, tampered_reducer._mean.values)
    assert np.allclose(baseline._std.values, tampered_reducer._std.values)
    assert np.allclose(baseline._pca.components_, tampered_reducer._pca.components_)


def test_prepare_reduced_only_changes_supplier_dimension(tiny_benchmark):
    config = make_tiny_config()
    full = prepare_from_benchmark(config, tiny_benchmark)
    reduced, reducer = prepare_reduced_from_benchmark(config, tiny_benchmark, "domain_selected", 4)

    full_dims = full.snapshot_builder.feature_dims()
    reduced_dims = reduced.snapshot_builder.feature_dims()

    assert reduced_dims[NodeType.SUPPLIER] == 4
    for node_type in (NodeType.REGION, NodeType.PROCUREMENT, NodeType.MATERIAL, NodeType.PLANT, NodeType.PRODUCT):
        assert reduced_dims[node_type] == full_dims[node_type]

    assert reduced.snapshot_builder.supplier_id_order() == full.snapshot_builder.supplier_id_order()
    assert reduced.snapshot_builder.topology.edge_index_dict.keys() == full.snapshot_builder.topology.edge_index_dict.keys()

    t = sorted(reduced.examples["time"].unique())[0]
    reduced_data = reduced.snapshot_builder.build(int(t))
    full_data = full.snapshot_builder.build(int(t))
    assert reduced_data.x_dict["supplier"].shape == (len(reduced.snapshot_builder.supplier_id_order()), 4)
    for node_type in ("region", "procurement", "material", "plant", "product"):
        assert (reduced_data.x_dict[node_type] == full_data.x_dict[node_type]).all()
    for rel in full_data.edge_index_dict:
        assert (reduced_data.edge_index_dict[rel] == full_data.edge_index_dict[rel]).all()
