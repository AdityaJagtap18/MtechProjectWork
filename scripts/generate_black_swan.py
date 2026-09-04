#!/usr/bin/env python3
"""Generate a black-swan-augmented operations dataset on top of an existing
normal-operations graph (plan §25 Dataset D, §41 pipeline).

Loads the graph from `--input`'s `graph/` directory (produced by
generate_topology.py or generate_normal.py) rather than regenerating
topology — the point is the *same* underlying supply chain with a
*different* disruption realization, so experiments comparing normal vs.
black-swan training data (plan §32 Experiment 3) aren't confounded by also
changing the graph. Then re-runs the Phase 3 simulator with Phase 5's event
engine enabled, and computes Phase 6's ground-truth cascade labels.

Usage:
    python scripts/generate_black_swan.py \\
        --input data/generated/scm_v1_normal_seed42 \\
        --config configs/black_swan.yaml \\
        --output data/generated/scm_v1_black_swan_seed42
"""

from __future__ import annotations

import argparse
import os
import sys

from scm_dataset.export.csv import export_event_impact, export_events, export_graph, export_labels, export_operations, load_graph
from scm_dataset.export.metadata import write_metadata
from scm_dataset.generator.config import load_config
from scm_dataset.labels.risk_labels import (
    compute_event_impact_labels,
    compute_material_labels,
    compute_plant_labels,
    compute_product_labels,
    compute_supplier_labels,
)
from scm_dataset.simulation.engine import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Directory with an existing graph/ (from generate_topology.py or generate_normal.py)")
    parser.add_argument("--config", default="configs/black_swan.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=None, help="Override the config's seed (event/operations randomness only, not the topology).")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed
    if not config.events.enabled:
        print("WARNING: config has events.enabled=false; this run will look identical to a normal run.", file=sys.stderr)

    graph = load_graph(args.input)
    errors = graph.validate()
    if errors:
        print(f"Loaded graph FAILED validation with {len(errors)} error(s):", file=sys.stderr)
        for err in errors[:20]:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    result = run_simulation(graph, config)

    supplier_labels = compute_supplier_labels(graph, result)
    material_labels = compute_material_labels(graph, result, config)
    plant_labels = compute_plant_labels(graph, result)
    product_labels = compute_product_labels(graph, result)
    event_impact = compute_event_impact_labels(graph, result.events, plant_labels, product_labels)

    dataset_id = os.path.basename(os.path.normpath(args.output))
    real_data_sources = ["World Bank WGI", "EU JRC INFORM Risk Index"] if config.real_data.use_real_region_calibration else []

    export_graph(graph, args.output)
    export_operations(result, args.output)
    export_events(result.events, args.output)
    export_labels(args.output, supplier_labels, material_labels, plant_labels, product_labels)
    export_event_impact(event_impact, args.output)
    write_metadata(args.output, dataset_id, config, graph, result, real_data_sources=real_data_sources)
    print(
        f"Loaded graph ({len(graph.nodes)} nodes) from {args.input}; generated {len(result.events)} event(s) and "
        f"{config.simulation.horizon_periods} periods of operations -> {args.output}"
    )


if __name__ == "__main__":
    main()
