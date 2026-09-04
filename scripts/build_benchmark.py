#!/usr/bin/env python3
"""Assemble a reproducible benchmark package (plan §42 Phase 8 deliverable)
from one or more already-generated dataset directories.

For each dataset found under `--input` (any subdirectory containing a
`graph/`), copies it into `--output`, computes the three plan §26 split
strategies into `<output>/<dataset>/splits/`, and ensures `metadata/` is
present (reconstructing a minimal, honestly-partial one if an older
dataset predates `scripts/generate_*.py` writing its own — those scripts
now call `write_metadata` at generation time, so this fallback only matters
for pre-existing runs). Also writes a shared `feature_audit.csv` (plan §27)
and a top-level `benchmark_manifest.json` listing every included dataset,
so multiple seeds of the same scenario (plan §28: report mean ± std, never
just the best run) are easy to enumerate together.

Usage:
    python scripts/build_benchmark.py --input data/generated --output data/benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

from scm_dataset.benchmark.splits import build_feature_audit, scenario_split, severity_split, temporal_split
from scm_dataset.export.csv import load_events, load_graph, load_operations
from scm_dataset.schema.events import EventType

# plan §49's "Black-Swan Generalization Experiment" holdout set
SCENARIO_SPLIT_TEST_TYPES = {EventType.CYBERATTACK, EventType.GEOPOLITICAL}


def _find_dataset_dirs(input_dir: str) -> list[str]:
    if not os.path.isdir(input_dir):
        return []
    return sorted(name for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name, "graph")))


def _infer_horizon(dataset_dir: str) -> int:
    info_path = os.path.join(dataset_dir, "metadata", "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        if info.get("horizon_periods"):
            return int(info["horizon_periods"])
    operations = load_operations(dataset_dir)
    demand = operations.get("demand.csv")
    if demand is not None and not demand.empty:
        return int(demand.time.max()) + 1
    return 0


def _ensure_metadata(dataset_dir: str, dataset_name: str, graph, events) -> None:
    metadata_dir = os.path.join(dataset_dir, "metadata")
    info_path = os.path.join(metadata_dir, "dataset_info.json")
    if os.path.exists(info_path):
        return
    os.makedirs(metadata_dir, exist_ok=True)
    with open(info_path, "w") as f:
        json.dump(
            {
                "dataset_id": dataset_name,
                "num_nodes": len(graph.nodes),
                "num_edges": len(graph.edges),
                "num_events": len(events),
                "note": "reconstructed by build_benchmark.py -- this dataset predates "
                "generate_*.py writing its own metadata, so generation_config/provenance are unavailable",
            },
            f,
            indent=2,
        )


def _build_splits(dataset_dir: str, horizon: int, events: list) -> None:
    splits_dir = os.path.join(dataset_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    temporal_split(horizon).to_csv(os.path.join(splits_dir, "temporal_split.csv"), index=False)
    if events:
        scenario_split(horizon, events, SCENARIO_SPLIT_TEST_TYPES).to_csv(os.path.join(splits_dir, "scenario_split.csv"), index=False)
        severity_split(horizon, events).to_csv(os.path.join(splits_dir, "severity_split.csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/generated")
    parser.add_argument("--output", default="data/benchmark")
    args = parser.parse_args()

    dataset_names = _find_dataset_dirs(args.input)
    if not dataset_names:
        print(f"No dataset directories found under {args.input} (expected subdirectories containing graph/)")
        return

    os.makedirs(args.output, exist_ok=True)
    manifest_datasets = []

    for name in dataset_names:
        src = os.path.join(args.input, name)
        dst = os.path.join(args.output, name)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

        graph = load_graph(dst)
        events = load_events(dst)
        _ensure_metadata(dst, name, graph, events)

        horizon = _infer_horizon(dst)
        _build_splits(dst, horizon, events)

        manifest_datasets.append(
            {"dataset_id": name, "num_nodes": len(graph.nodes), "num_edges": len(graph.edges), "num_events": len(events), "horizon_periods": horizon}
        )
        print(f"  {name}: {len(graph.nodes)} nodes, {len(events)} event(s), horizon={horizon}")

    feature_audit_path = os.path.join(args.output, "feature_audit.csv")
    build_feature_audit().to_csv(feature_audit_path, index=False)

    manifest_path = os.path.join(args.output, "benchmark_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"datasets": manifest_datasets}, f, indent=2)

    print(f"Assembled {len(dataset_names)} dataset(s) into {args.output}/")
    print(f"  - {feature_audit_path}")
    print(f"  - {manifest_path}")


if __name__ == "__main__":
    main()
