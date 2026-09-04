import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from scm_dataset.benchmark.splits import build_feature_audit, scenario_split, severity_split, temporal_split
from scm_dataset.export.csv import export_events, export_graph, export_operations
from scm_dataset.export.metadata import build_dataset_info, build_provenance, write_metadata
from scm_dataset.generator.config import EventsConfig, EventTypeConfig, GeneratorConfig, NetworkConfig, SimulationConfig
from scm_dataset.generator.topology import generate_graph
from scm_dataset.schema.events import Event, EventType
from scm_dataset.simulation.engine import run_simulation

REPO_ROOT = Path(__file__).resolve().parents[1]


def _small_config(seed: int = 42, horizon_periods: int = 20, events: EventsConfig | None = None) -> GeneratorConfig:
    return GeneratorConfig(
        seed=seed,
        network=NetworkConfig(suppliers=15, procurement_orders=40, materials=10, plants=5, products=10, regions=3),
        simulation=SimulationConfig(horizon_periods=horizon_periods),
        events=events or EventsConfig(),
    )


def _make_event(**overrides) -> Event:
    defaults = dict(
        event_id="e0", event_type=EventType.SUPPLIER_FAILURE, start_time=5, duration=2, severity=3,
        probability_regime=0.02, affected_suppliers=["s0"], capacity_reduction_fraction=0.5,
        lead_time_increase_fraction=0.2, recovery_delay=0, recovery_periods=3,
    )
    defaults.update(overrides)
    return Event(**defaults)


# ---- splits.py ----


def test_temporal_split_boundaries():
    result = temporal_split(100, train_frac=0.7, val_frac=0.15)
    assert len(result) == 100
    assert (result[result.time < 70].split == "train").all()
    assert (result[(result.time >= 70) & (result.time < 85)].split == "validation").all()
    assert (result[result.time >= 85].split == "test").all()


def test_scenario_split_assigns_held_out_event_type_periods_to_test():
    train_event = _make_event(event_id="e_train", event_type=EventType.SUPPLIER_FAILURE, start_time=2, duration=2, recovery_delay=0, recovery_periods=0)
    test_event = _make_event(event_id="e_test", event_type=EventType.CYBERATTACK, start_time=10, duration=2, recovery_delay=0, recovery_periods=0)
    result = scenario_split(20, [train_event, test_event], {EventType.CYBERATTACK})

    assert (result[(result.time >= 2) & (result.time < 4)].split == "train").all()
    assert (result[(result.time >= 10) & (result.time < 12)].split == "test").all()
    assert (result[result.time == 0].split == "train").all()  # baseline period, no active event


def test_severity_split_assigns_high_severity_periods_to_test():
    low = _make_event(event_id="e_low", severity=2, start_time=0, duration=3, recovery_delay=0, recovery_periods=0)
    high = _make_event(event_id="e_high", severity=5, start_time=10, duration=3, recovery_delay=0, recovery_periods=0)
    result = severity_split(20, [low, high], train_max_severity=3)

    assert (result[(result.time >= 0) & (result.time < 3)].split == "train").all()
    assert (result[(result.time >= 10) & (result.time < 13)].split == "test").all()


def test_build_feature_audit_marks_labels_as_disallowed_and_static_attributes_as_allowed():
    audit = build_feature_audit()
    assert set(audit.columns) == {"table", "column", "available_time", "target_time_rule", "allowed", "reason"}

    labels_rows = audit[audit.table.str.startswith("labels/")]
    assert (~labels_rows.allowed).all()

    node_rows = audit[audit.table == "graph/nodes.csv"]
    assert node_rows.allowed.all()

    events_rows = audit[audit.table == "events/events.csv"]
    assert (~events_rows.allowed).all()


# ---- metadata.py ----


def test_build_dataset_info_without_result_marks_no_operations():
    config = _small_config()
    graph = generate_graph(config)
    info = build_dataset_info("test_dataset", config, graph)
    assert info["has_operations"] is False
    assert info["horizon_periods"] is None
    assert info["num_nodes"] == len(graph.nodes)


def test_build_dataset_info_with_result():
    config = _small_config()
    graph = generate_graph(config)
    result = run_simulation(graph, config)
    info = build_dataset_info("test_dataset", config, graph, result)
    assert info["has_operations"] is True
    assert info["horizon_periods"] == config.simulation.horizon_periods
    assert info["num_events"] == len(result.events)


def test_build_provenance_has_expected_fields():
    config = _small_config()
    provenance = build_provenance("test_dataset", config, real_data_sources=["WGI"])
    assert provenance["dataset_id"] == "test_dataset"
    assert provenance["seed"] == config.seed
    assert len(provenance["config_hash"]) == 16
    assert provenance["real_data_sources"] == ["WGI"]
    assert "python" in provenance["software_versions"]


def test_write_metadata_creates_all_four_files():
    config = _small_config()
    graph = generate_graph(config)
    result = run_simulation(graph, config)
    with tempfile.TemporaryDirectory() as tmp:
        write_metadata(tmp, "test_dataset", config, graph, result)
        metadata_dir = Path(tmp) / "metadata"
        for filename in ("dataset_info.json", "generation_config.yaml", "feature_schema.json", "provenance.json"):
            assert (metadata_dir / filename).exists(), filename

        with open(metadata_dir / "dataset_info.json") as f:
            info = json.load(f)
        assert info["dataset_id"] == "test_dataset"


# ---- scripts/build_benchmark.py (end-to-end) ----


def test_build_benchmark_script_end_to_end():
    config = _small_config(
        events=EventsConfig(enabled=True, supplier_failure=EventTypeConfig(0.5), severity_weights=[0.2, 0.2, 0.2, 0.2, 0.2])
    )
    graph = generate_graph(config)
    result = run_simulation(graph, config)

    with tempfile.TemporaryDirectory() as tmp:
        input_dir = Path(tmp) / "generated"
        output_dir = Path(tmp) / "benchmark"
        dataset_dir = input_dir / "scm_test_seed42"
        dataset_dir.mkdir(parents=True)

        export_graph(graph, str(dataset_dir))
        export_operations(result, str(dataset_dir))
        export_events(result.events, str(dataset_dir))
        write_metadata(str(dataset_dir), "scm_test_seed42", config, graph, result)

        proc = subprocess.run(
            [sys.executable, "scripts/build_benchmark.py", "--input", str(input_dir), "--output", str(output_dir)],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr

        assert (output_dir / "feature_audit.csv").exists()
        manifest_path = output_dir / "benchmark_manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["datasets"][0]["dataset_id"] == "scm_test_seed42"

        splits_dir = output_dir / "scm_test_seed42" / "splits"
        assert (splits_dir / "temporal_split.csv").exists()
        assert (splits_dir / "scenario_split.csv").exists()
        assert (splits_dir / "severity_split.csv").exists()

        temporal = pd.read_csv(splits_dir / "temporal_split.csv")
        assert len(temporal) == config.simulation.horizon_periods
        assert set(temporal.split.unique()) <= {"train", "validation", "test"}
