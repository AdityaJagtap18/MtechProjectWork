"""Tests for modeling/graphsage.py (plan §17-20/§27-32)."""

from __future__ import annotations

import torch

from scm_dataset.modeling.graphsage import build_model
from scm_dataset.schema.nodes import NodeType


def _build_model_for(prepared, **overrides):
    in_dims = {nt.value: dim for nt, dim in prepared.snapshot_builder.feature_dims().items()}
    edge_types = list(prepared.snapshot_builder.topology.edge_index_dict.keys())
    kwargs = dict(hidden_dim=16, num_layers=2, dropout=0.2, aggregation="mean")
    kwargs.update(overrides)
    return build_model(in_dims=in_dims, edge_types=edge_types, **kwargs)


def test_forward_returns_one_logit_per_supplier(tiny_prepared_data):
    model = _build_model_for(tiny_prepared_data)
    t = int(tiny_prepared_data.examples["time"].min())
    data = tiny_prepared_data.snapshot_builder.build(t)
    model.eval()
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)
    n_suppliers = tiny_prepared_data.snapshot_builder.topology.num_nodes[NodeType.SUPPLIER]
    assert logits.shape == (n_suppliers,)
    assert logits.dtype == torch.float32


def test_model_never_applies_sigmoid_internally(tiny_prepared_data):
    # BCEWithLogitsLoss requires raw logits (plan §20/§33). Checking the
    # architecture directly (rather than the numeric range of one randomly-
    # initialized forward pass, which can coincidentally land in [0, 1] with
    # a small hidden_dim) is what actually verifies this.
    model = _build_model_for(tiny_prepared_data)
    assert not any(isinstance(m, torch.nn.Sigmoid) for m in model.modules())

    # And, mechanically: forward() must return exactly classifier(h_supplier)
    # squeezed, with nothing applied after it.
    t = int(tiny_prepared_data.examples["time"].min())
    data = tiny_prepared_data.snapshot_builder.build(t)
    model.eval()
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)
        h_dict = model.encode(data.x_dict, data.edge_index_dict)
        expected = model.classifier(h_dict["supplier"]).squeeze(-1)
    assert torch.equal(logits, expected)


def test_dropout_zero_is_deterministic_in_eval_mode(tiny_prepared_data):
    model = _build_model_for(tiny_prepared_data, dropout=0.0)
    model.eval()
    t = int(tiny_prepared_data.examples["time"].min())
    data = tiny_prepared_data.snapshot_builder.build(t)
    with torch.no_grad():
        logits_a = model(data.x_dict, data.edge_index_dict)
        logits_b = model(data.x_dict, data.edge_index_dict)
    assert torch.allclose(logits_a, logits_b)


def test_hidden_dim_is_honored_in_encoder_output(tiny_prepared_data):
    model = _build_model_for(tiny_prepared_data, hidden_dim=32)
    t = int(tiny_prepared_data.examples["time"].min())
    data = tiny_prepared_data.snapshot_builder.build(t)
    model.eval()
    with torch.no_grad():
        h_dict = model.encode(data.x_dict, data.edge_index_dict)
    for node_type, h in h_dict.items():
        assert h.shape[1] == 32


def test_num_layers_controls_number_of_conv_modules(tiny_prepared_data):
    model = _build_model_for(tiny_prepared_data, num_layers=3)
    assert len(model.convs) == 3


def test_forward_handles_isolated_node_type_without_crashing(tiny_prepared_data):
    # A node type with an edge relation that happens to have zero edges in
    # a given (tiny) graph must not crash HeteroConv/SAGEConv.
    model = _build_model_for(tiny_prepared_data)
    t = int(tiny_prepared_data.examples["time"].min())
    data = tiny_prepared_data.snapshot_builder.build(t)
    model.train()
    logits = model(data.x_dict, data.edge_index_dict)
    loss = logits.sum()
    loss.backward()  # gradient must flow without error even through zero-edge relations
    assert any(p.grad is not None for p in model.parameters())
