#!/usr/bin/env python3
"""Generate a complete "normal operations" SCM dataset (plan §25 Dataset A):
topology (Phase 2) plus time-varying demand/production/inventory/
procurement/deliveries (Phase 3). `configs/normal.yaml` has events disabled,
so `events/events.csv` and `events/event_impact.csv` are written but empty
— kept for a consistent directory structure across every dataset variant
(see scripts/generate_black_swan.py for the Phase 5 event-augmented version
of this same graph). Phase 6's ground-truth labels are still computed here:
with no events, `*_disrupted`/`*_shortage` flags will mostly be 0, which is
itself a legitimate (if less interesting) label set for this scenario.

Usage:
    python scripts/generate_normal.py --config configs/normal.yaml \\
        --output data/generated/scm_v1_normal_seed42
"""

from __future__ import annotations

import argparse
import os
import sys

from scm_dataset.export.csv import export_event_impact, export_events, export_graph, export_labels, export_operations
from scm_dataset.export.metadata import write_metadata
from scm_dataset.generator.config import load_config
from scm_dataset.generator.topology import generate_graph
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
    parser.add_argument("--config", default="configs/normal.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=None, help="Override the config's seed.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed

    graph = generate_graph(config)
    errors = graph.validate()
    if errors:
        print(f"Graph FAILED validation with {len(errors)} error(s):", file=sys.stderr)
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
        f"Generated valid graph ({len(graph.nodes)} nodes, {len(graph.edges)} edges) and "
        f"{config.simulation.horizon_periods} periods of operations -> {args.output}"
    )


if __name__ == "__main__":
    main()
