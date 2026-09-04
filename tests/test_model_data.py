"""Tests for modeling/data.py and modeling/pipeline.py's benchmark loading
and supplier-label-alignment sanity check (plan §46/§66 Check 1/§68)."""

from __future__ import annotations

import pytest

from scm_dataset.modeling.data import load_benchmark
from scm_dataset.modeling.pipeline import _check_supplier_label_alignment

BENCHMARK_ROOT = "data/benchmark"
DATASET_ID = "scm_v1_black_swan_seed43"


def _benchmark_available() -> bool:
    import os

    return os.path.isdir(os.path.join(BENCHMARK_ROOT, DATASET_ID))


pytestmark = pytest.mark.skipif(not _benchmark_available(), reason="generated benchmark not present under data/benchmark/")


def test_load_benchmark_reads_expected_shapes():
    benchmark = load_benchmark(BENCHMARK_ROOT, DATASET_ID)
    assert benchmark.dataset_id == DATASET_ID
    assert benchmark.horizon_periods == 104
    assert set(benchmark.splits.keys()) == {"temporal", "scenario", "severity"}
    assert set(benchmark.labels.keys()) == {"supplier", "material", "plant", "product"}
    assert "procurement.csv" in benchmark.operations
    assert len(benchmark.graph.nodes) > 0


def test_load_benchmark_missing_dataset_raises():
    with pytest.raises(FileNotFoundError):
        load_benchmark(BENCHMARK_ROOT, "does_not_exist")


def test_supplier_label_alignment_passes_on_real_benchmark():
    benchmark = load_benchmark(BENCHMARK_ROOT, DATASET_ID)
    _check_supplier_label_alignment(benchmark)  # must not raise


def test_supplier_label_alignment_raises_on_mismatch():
    benchmark = load_benchmark(BENCHMARK_ROOT, DATASET_ID)
    tampered = benchmark.labels["supplier"].copy()
    tampered["supplier_id"] = tampered["supplier_id"].replace({"supplier_0": "supplier_does_not_exist"})
    benchmark.labels["supplier"] = tampered
    with pytest.raises(ValueError, match="disagree on supplier identity"):
        _check_supplier_label_alignment(benchmark)
