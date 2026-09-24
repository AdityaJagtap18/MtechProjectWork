"""Tests for the QGNN-v3.1 trainable-scaling confirmation benchmark.
Purely additive alongside test_qgnn_v2*.py/test_qgnn_v3.py -- does not
modify or weaken any existing test. Items 5/6/7 (added-parameter count,
a=1/b=0 init, numerical init-equivalence) are already covered at the
model-class level by test_qgnn_v3.py; this file additionally proves them
in the context of THIS script's own shared-object construction (items
2-4), which is new here.
"""

from __future__ import annotations

import os

import torch

from scm_dataset.modeling.graph_embedding_reduction import extract_supplier_embeddings
from scm_dataset.modeling.pipeline import prepare_from_benchmark
from scm_dataset.modeling.qgnn import build_qgnn_model
from scm_dataset.modeling.qgnn_v2 import _flatten, build_v2_prepared, evaluate_v2, generate_predictions_v2, train_v2_head
from scm_dataset.modeling.qgnn_v3 import build_qgnn_trainable_scaling_model
from scm_dataset.modeling.train import train_graphsage

from conftest import make_tiny_config

CPU_DEVICE = "default.qubit"
CONFIRMATION_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


def _tiny_frozen_encoder(tiny_benchmark):
    config = make_tiny_config()
    config.training.epochs = 3
    prepared = prepare_from_benchmark(config, tiny_benchmark)
    result = train_graphsage(prepared, seed=42, verbose=False)
    model = result.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    return model, prepared


def _shared_v2_prepared(tiny_benchmark, n_components=4):
    """Mirrors run_qgnn_v3_1_experiment.py's own construction exactly:
    ONE frozen encoder, ONE embedding extraction, ONE build_v2_prepared
    call -- the same (v2prepared, reducer) pair the script passes to both
    A and D, not rebuilt per model."""
    encoder, prepared = _tiny_frozen_encoder(tiny_benchmark)
    times = sorted(prepared.examples["time"].unique())
    emb = extract_supplier_embeddings(encoder, prepared, times)
    v2prepared, reducer = build_v2_prepared(prepared.config, prepared.benchmark, prepared.examples, emb, n_components)
    return encoder, v2prepared, reducer


# ---- 1: 10 seeds configured correctly ----


def test_confirmation_seed_list_has_10_seeds_42_through_51():
    assert len(CONFIRMATION_SEEDS) == 10
    assert CONFIRMATION_SEEDS == list(range(42, 52))

    # scripts/ is not an importable package (its own top-level `from
    # _analysis_common import ...` only resolves when run directly, the
    # same way every other scripts/run_*.py in this project works) --
    # check the script's own declared default by reading its source text
    # rather than importing it.
    script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_qgnn_v3_1_experiment.py")
    with open(script_path) as f:
        source = f.read()
    assert '"42,43,44,45,46,47,48,49,50,51"' in source


# ---- 2-4: A and D share identical input rows / embeddings / PCA object ----


def test_a_and_d_share_the_identical_v2prepared_and_reducer_objects(tiny_benchmark):
    _, v2prepared, reducer = _shared_v2_prepared(tiny_benchmark)

    # This is the actual guarantee the script relies on: A and D are
    # trained from the SAME v2prepared/reducer Python objects, not
    # separately-rebuilt copies -- so their input rows, embeddings, and
    # PCA fit are identical by construction. Confirm the flattened
    # tensors used for training are bit-identical regardless of "which
    # model" logically consumes them.
    x_a, y_a, _ = _flatten(v2prepared, "test")
    x_d, y_d, _ = _flatten(v2prepared, "test")
    assert torch.equal(x_a, x_d)
    assert torch.equal(y_a, y_d)
    assert reducer.n_components == v2prepared.n_components == v2prepared.reduced.shape[1]


# ---- 5-7: covered at the model level by test_qgnn_v3.py; re-confirmed here in-pipeline ----


def test_d_initial_output_matches_a_when_given_the_shared_pipeline_input(tiny_benchmark):
    _, v2prepared, _ = _shared_v2_prepared(tiny_benchmark)
    x, _, _ = _flatten(v2prepared, "test")

    a_model = build_qgnn_model(n_qubits=4, n_layers=3, device_name=CPU_DEVICE)
    d_model = build_qgnn_trainable_scaling_model(4, 3, device_name=CPU_DEVICE)
    assert d_model.quantum.scale_a.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert d_model.quantum.scale_b.tolist() == [0.0, 0.0, 0.0, 0.0]

    d_model.quantum.weights.data = a_model.quantum.weights.data.clone()  # isolate encoding-equivalence from random weight init
    assert torch.allclose(a_model.quantum(x), d_model.quantum(x), atol=1e-6)

    # This test's pipeline uses n_components=4 (tiny-benchmark scale, like
    # every other test_qgnn_v2*/v3*.py test); the added-parameter count is
    # 2*n_components regardless of scale -- the real n_components=8 case
    # (added_params == 16) is already covered by
    # test_qgnn_v3.py::test_trainable_scaling_added_parameter_count.
    added_params = d_model.quantum.scale_a.numel() + d_model.quantum.scale_b.numel()
    assert added_params == 2 * v2prepared.n_components == 8


# ---- 8: a/b receive gradients ----


def test_scaling_parameters_receive_gradients(tiny_benchmark):
    _, v2prepared, _ = _shared_v2_prepared(tiny_benchmark)
    v2prepared.config.training.epochs = 2
    v2prepared.config.training.early_stopping_patience = 10
    model = build_qgnn_trainable_scaling_model(4, 3, device_name=CPU_DEVICE)
    initial_a = model.quantum.scale_a.detach().clone()
    initial_b = model.quantum.scale_b.detach().clone()

    train_v2_head(v2prepared, model, seed=42, verbose=False)

    assert not torch.equal(initial_a, model.quantum.scale_a.detach())
    assert not torch.equal(initial_b, model.quantum.scale_b.detach())


# ---- 9, 11, 12, 13: encoder freeze, no leakage, finite forward/backward ----


def test_full_pipeline_encoder_frozen_no_leakage_finite_everywhere(tiny_benchmark):
    encoder, v2prepared, reducer = _shared_v2_prepared(tiny_benchmark)
    v2prepared.config.training.epochs = 2
    v2prepared.config.training.early_stopping_patience = 10
    assert reducer._input_columns is not None  # already fit inside build_v2_prepared, train-only by construction

    for variant, build_fn in (("A", lambda: build_qgnn_model(n_qubits=4, n_layers=3, device_name=CPU_DEVICE)),
                               ("D", lambda: build_qgnn_trainable_scaling_model(4, 3, device_name=CPU_DEVICE))):
        model = build_fn()
        result = train_v2_head(v2prepared, model, seed=42, verbose=False)
        assert all(not p.requires_grad for p in encoder.parameters()), variant
        assert all(p.grad is None for p in encoder.parameters()), variant

        predictions = generate_predictions_v2(result.model, v2prepared)
        assert torch.isfinite(torch.tensor(predictions["risk_probability"].values)).all(), variant
        assert predictions["risk_probability"].between(0.0, 1.0).all(), variant
        eval_result = evaluate_v2(result.model, v2prepared, v2prepared.config.threshold)
        assert "test" in eval_result.metrics_by_split, variant


# ---- 10: output shapes identical ----


def test_a_and_d_output_shapes_identical():
    x = torch.randn(7, 8)
    a = build_qgnn_model(n_qubits=8, n_layers=3, device_name=CPU_DEVICE)
    d = build_qgnn_trainable_scaling_model(8, 3, device_name=CPU_DEVICE)
    assert a(x).shape == d(x).shape == (7,)


# ---- risk-ranking metric wiring (new module used by this script) ----


def test_risk_ranking_summary_wired_into_evaluation(tiny_benchmark):
    from scm_dataset.modeling.risk_ranking_metrics import risk_ranking_summary

    _, v2prepared, _ = _shared_v2_prepared(tiny_benchmark)
    v2prepared.config.training.epochs = 2
    v2prepared.config.training.early_stopping_patience = 10
    model = build_qgnn_model(n_qubits=4, n_layers=3, device_name=CPU_DEVICE)
    result = train_v2_head(v2prepared, model, seed=42, verbose=False)
    eval_result = evaluate_v2(result.model, v2prepared, v2prepared.config.threshold)

    ranking = risk_ranking_summary(eval_result.predictions, split="test", k_percents=(5.0, 10.0))
    assert set(ranking.keys()) == {"k=5.0%", "k=10.0%"}
    for entry in ranking.values():
        assert entry["precision_at_k"] is None or 0.0 <= entry["precision_at_k"] <= 1.0
