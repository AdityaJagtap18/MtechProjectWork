#!/usr/bin/env python3
"""Generate the Phase 2 topology-only SCM graph and write it to disk.

Usage:
    python scripts/generate_topology.py --config configs/base.yaml \\
        --output data/generated/scm_v1_topology_seed42

This only builds the graph (nodes + edges) — no time-varying operations,
events, or labels yet (those are Phases 3, 5, 6).
"""

from __future__ import annotations

import argparse
import sys

import os

from scm_dataset.export.csv import export_graph
from scm_dataset.export.metadata import write_metadata
from scm_dataset.generator.config import load_config
from scm_dataset.generator.topology import generate_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
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

    dataset_id = os.path.basename(os.path.normpath(args.output))
    export_graph(graph, args.output)
    write_metadata(args.output, dataset_id, config, graph)
    print(f"Generated valid graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges -> {args.output}")


if __name__ == "__main__":
    main()
