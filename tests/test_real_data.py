import os

import numpy as np
import pandas as pd
import pytest

from scm_dataset.generator.config import GeneratorConfig, NetworkConfig, RealDataConfig, SimulationConfig, TopologyConfig
from scm_dataset.generator.regions import generate_regions
from scm_dataset.generator.topology import generate_graph
from scm_dataset.real_data.calibrator import build_region_risk_table, load_region_risk_table, sample_calibrated_region_risk
from scm_dataset.real_data.loader import load_all_nist_scenarios, load_inform_risk, load_wgi_indicator
from scm_dataset.real_data.mapper import FIELD_MAPPING, compute_multi_source_rate, compute_structural_stats
from scm_dataset.schema.nodes import NodeType

RAW_DIR = "data/raw"


def _raw_data_available() -> bool:
    return (
        os.path.exists(os.path.join(RAW_DIR, "wgi", "political_stability.json"))
        and os.path.exists(os.path.join(RAW_DIR, "inform", "inform_risk.xlsx"))
        and os.path.exists(os.path.join(RAW_DIR, "nist", "SampleDataSets"))
    )


requires_raw_data = pytest.mark.skipif(
    not _raw_data_available(), reason="data/raw/ not present -- run scripts/download_real_data.py first"
)


@requires_raw_data
def test_load_wgi_indicator_returns_plausible_values():
    values = load_wgi_indicator(RAW_DIR, "political_stability.json")
    assert len(values) > 100
    assert all(-3.5 <= v <= 3.5 for v in values.values())


@requires_raw_data
def test_load_inform_risk_has_expected_columns():
    df = load_inform_risk(RAW_DIR)
    assert {"country", "iso3", "natural_hazard", "infrastructure"} <= set(df.columns)
    assert len(df) > 100


@requires_raw_data
def test_build_region_risk_table_values_in_unit_range():
    table = build_region_risk_table(RAW_DIR)
    assert len(table) > 100
    for col in ("geopolitical_risk", "trade_risk", "natural_disaster_risk", "infrastructure_risk"):
        assert table[col].between(0.0, 1.0).all()


@requires_raw_data
def test_compute_multi_source_rate_in_unit_range():
    scenarios = load_all_nist_scenarios(RAW_DIR)
    rate = compute_multi_source_rate(scenarios["Small Government Entity"]["products"])
    assert 0.0 <= rate <= 1.0


@requires_raw_data
def test_compute_structural_stats_has_expected_shape():
    stats = compute_structural_stats(RAW_DIR)
    assert "field_mapping" in stats
    assert "per_scenario" in stats
    assert 0.0 <= stats["overall_multi_source_rate"] <= 1.0


def test_field_mapping_covers_all_six_node_types():
    assert set(FIELD_MAPPING) == {"Supplier", "Procurement Order", "Material", "Plant", "Product", "Region"}


def _toy_region_risk_table(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "country": f"Country{i}",
                "iso3": f"C{i:02d}",
                "geopolitical_risk": 0.1 + 0.05 * i,
                "trade_risk": 0.2,
                "natural_disaster_risk": 0.15,
                "infrastructure_risk": 0.25,
            }
            for i in range(n)
        ]
    )


def test_sample_calibrated_region_risk_returns_all_fields():
    table = _toy_region_risk_table(1)
    rng = np.random.default_rng(0)
    result = sample_calibrated_region_risk(table, rng)
    assert result["country_or_region"] == "Country0"
    assert result["geopolitical_risk"] == pytest.approx(0.1)
    for key in ("cyber_risk", "transport_reliability"):
        assert 0.0 <= result[key] <= 1.0


def test_load_region_risk_table_roundtrip(tmp_path):
    table = _toy_region_risk_table(3)
    path = tmp_path / "region_risk.csv"
    table.to_csv(path, index=False)
    reloaded = load_region_risk_table(str(path))
    assert list(reloaded.columns) == list(table.columns)
    assert len(reloaded) == 3


def test_generate_regions_calibrated_mode_uses_real_countries():
    table = _toy_region_risk_table(2)
    rng = np.random.default_rng(1)
    regions = generate_regions(5, rng, region_risk_table=table)
    assert len(regions) == 5
    assert {r.country_or_region for r in regions} <= {"Country0", "Country1"}
    for r in regions:
        assert 0.0 <= r.geopolitical_risk <= 1.0
        assert 0.0 <= r.transport_reliability <= 1.0


def test_generate_regions_synthetic_mode_unaffected_by_default():
    rng = np.random.default_rng(2)
    regions = generate_regions(5, rng)
    assert len(regions) == 5
    assert all("_" in r.country_or_region for r in regions)  # "<archetype>_<i>", not a real country name


def test_generate_graph_with_calibration_enabled(tmp_path):
    table = _toy_region_risk_table(10)
    path = tmp_path / "region_risk.csv"
    table.to_csv(path, index=False)

    config = GeneratorConfig(
        seed=3,
        network=NetworkConfig(suppliers=10, procurement_orders=20, materials=8, plants=4, products=8, regions=5),
        topology=TopologyConfig(),
        simulation=SimulationConfig(horizon_periods=4),
        real_data=RealDataConfig(use_real_region_calibration=True, region_risk_table_path=str(path)),
    )
    graph = generate_graph(config)
    assert graph.is_valid(), graph.validate()

    region_names = {r.country_or_region for r in graph.nodes_of_type(NodeType.REGION)}
    assert region_names <= set(table["country"])


def test_generate_graph_default_config_unaffected_by_real_data_module():
    config = GeneratorConfig(
        seed=4,
        network=NetworkConfig(suppliers=10, procurement_orders=20, materials=8, plants=4, products=8, regions=5),
    )
    graph = generate_graph(config)
    assert graph.is_valid(), graph.validate()
