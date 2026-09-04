"""Shared fixtures for the modeling test suite.

`tiny_prepared_data` builds a small, fast, in-memory benchmark (topology ->
simulation -> events -> labels -> splits -> feature audit, all through the
real generator/simulation/labels/splits code -- nothing here is a
reimplementation of the dataset framework) and runs it through the real
`modeling.pipeline.prepare_from_benchmark` so every modeling test exercises
production code end to end without needing `data/benchmark/` on disk or
paying the cost of the full-size dataset.

Seed 3 with this network size was picked empirically (see the session that
introduced this fixture) for producing a non-trivial, non-degenerate
positive rate in all three temporal-split partitions (train ~6.8%,
validation ~17.2%, test ~20%) -- rare enough to be representative of the
real benchmark's class imbalance, but not so rare that a 20-supplier/
60-period graph has zero positives in some split.
"""

from __future__ import annotations

import pytest

from scm_dataset.benchmark.splits import build_feature_audit, scenario_split, severity_split, temporal_split
from scm_dataset.generator.config import EventsConfig, EventTypeConfig, GeneratorConfig, NetworkConfig, SimulationConfig
from scm_dataset.generator.topology import generate_graph
from scm_dataset.labels.risk_labels import (
    compute_material_labels,
    compute_plant_labels,
    compute_product_labels,
    compute_supplier_labels,
)
from scm_dataset.modeling.config import GraphSAGEConfig
from scm_dataset.modeling.data import BenchmarkData
from scm_dataset.modeling.pipeline import prepare_from_benchmark
from scm_dataset.schema.events import EventType
from scm_dataset.simulation.engine import run_simulation

TINY_HORIZON = 60
TINY_SEED = 3


def make_tiny_config(**overrides) -> GraphSAGEConfig:
    config = GraphSAGEConfig()
    config.dataset.dataset_id = "tiny_test"
    config.features.rolling_windows = [4, 8]
    config.features.min_history_periods = 12
    config.prediction.horizon_periods = 4
    config.model.hidden_dim = 16
    config.training.epochs = 3
    config.training.early_stopping_patience = 3
    config.seed = TINY_SEED
    for key, value in overrides.items():
        parts = key.split("__")
        target = config
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    return config


def build_tiny_benchmark() -> BenchmarkData:
    gen_config = GeneratorConfig(
        seed=TINY_SEED,
        network=NetworkConfig(suppliers=20, procurement_orders=80, materials=10, plants=5, products=10, regions=3),
        simulation=SimulationConfig(horizon_periods=TINY_HORIZON),
        events=EventsConfig(
            enabled=True,
            supplier_failure=EventTypeConfig(0.02),
            logistics=EventTypeConfig(0.015),
            severity_weights=[0.35, 0.30, 0.20, 0.10, 0.05],
        ),
    )
    graph = generate_graph(gen_config)
    graph.assert_valid()
    result = run_simulation(graph, gen_config)

    operations = {
        "demand.csv": _df(result.demand_records),
        "production.csv": _df(result.production_records),
        "inventory.csv": _df(result.inventory_records),
        "backlog.csv": _df(result.backlog_records),
        "deliveries.csv": _df(result.deliveries_records),
        "procurement.csv": _df(result.procurement_records),
    }
    labels = {
        "supplier": compute_supplier_labels(graph, result),
        "material": compute_material_labels(graph, result, gen_config),
        "plant": compute_plant_labels(graph, result),
        "product": compute_product_labels(graph, result),
    }
    splits = {
        "temporal": temporal_split(TINY_HORIZON, train_frac=0.7, val_frac=0.15),
        "scenario": scenario_split(TINY_HORIZON, result.events, {EventType.CYBERATTACK}),
        "severity": severity_split(TINY_HORIZON, result.events, train_max_severity=3),
    }

    return BenchmarkData(
        dataset_id="tiny_test",
        dataset_dir="<in-memory, no disk path>",
        graph=graph,
        operations=operations,
        labels=labels,
        splits=splits,
        feature_audit=build_feature_audit(),
        horizon_periods=TINY_HORIZON,
    )


def _df(records: list[dict]):
    import pandas as pd

    return pd.DataFrame(records)


@pytest.fixture(scope="session")
def tiny_benchmark() -> BenchmarkData:
    return build_tiny_benchmark()


@pytest.fixture
def tiny_prepared_data(tiny_benchmark: BenchmarkData):
    config = make_tiny_config()
    return prepare_from_benchmark(config, tiny_benchmark)
