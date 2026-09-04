"""Reproducibility metadata (plan §39 Reproducibility, §37's export/metadata.py).

Every generated dataset should be able to answer "what code + config + seed
produced this, and can it be regenerated deterministically?" without
needing anything beyond what's written into its own `metadata/` directory.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from .. import __version__ as GENERATOR_VERSION
from ..generator.config import GeneratorConfig
from ..schema.events import Event
from ..schema.graph import NODE_CLASS_BY_TYPE, SupplyChainGraph
from ..schema.nodes import NodeType
from ..simulation.engine import OperationsResult

FEATURE_SCHEMA = {
    "graph/nodes.csv": ["node_id", "node_type"]
    + sorted({f.name for cls in NODE_CLASS_BY_TYPE.values() for f in dataclasses.fields(cls)}),
    "graph/edges.csv": ["edge_id", "source_id", "target_id", "edge_type"],
    "graph/edge_features.csv": ["edge_id", "<edge-specific attributes, if any>"],
    "operations/demand.csv": ["product_id", "time", "demand"],
    "operations/production.csv": ["plant_id", "product_id", "time", "production"],
    "operations/inventory.csv": ["plant_id", "material_id", "time", "inventory"],
    "operations/backlog.csv": ["product_id", "time", "backlog"],
    "operations/deliveries.csv": ["plant_id", "material_id", "supplier_id", "time", "quantity"],
    "operations/procurement.csv": [
        "plant_id", "material_id", "supplier_id", "time_ordered",
        "quantity_ordered", "quantity_fulfilled", "expected_delivery_time",
    ],
    "events/events.csv": [f.name for f in dataclasses.fields(Event)],
    "events/event_impact.csv": ["event_id", "cascade_severity", "total_affected_nodes", "time_to_impact", "recovery_time"],
    "labels/supplier_labels.csv": ["supplier_id", "time", "supplier_disrupted", "supplier_risk_score"],
    "labels/material_labels.csv": ["plant_id", "material_id", "time", "material_shortage", "material_risk_score"],
    "labels/plant_labels.csv": ["plant_id", "time", "production_loss", "production_loss_fraction", "plant_disruption"],
    "labels/product_labels.csv": ["product_id", "time", "product_shortage", "revenue_impact"],
}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=os.path.dirname(__file__)
        ).decode().strip()
    except Exception:
        return None


def _config_hash(config: GeneratorConfig) -> str:
    payload = json.dumps(dataclasses.asdict(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_dataset_info(
    dataset_id: str, config: GeneratorConfig, graph: SupplyChainGraph, result: OperationsResult | None = None
) -> dict:
    return {
        "dataset_id": dataset_id,
        "num_nodes": len(graph.nodes),
        "num_edges": len(graph.edges),
        "node_counts": {nt.value: len(graph.nodes_of_type(nt)) for nt in NodeType},
        "horizon_periods": config.simulation.horizon_periods if result is not None else None,
        "num_events": len(result.events) if result is not None else None,
        "events_enabled": config.events.enabled,
        "real_region_calibration": config.real_data.use_real_region_calibration,
        "has_operations": result is not None,
    }


def build_provenance(dataset_id: str, config: GeneratorConfig, real_data_sources: list[str] | None = None) -> dict:
    return {
        "dataset_id": dataset_id,
        "generator_version": GENERATOR_VERSION,
        "git_commit": _git_commit(),
        "seed": config.seed,
        "config_hash": _config_hash(config),
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "real_data_sources": real_data_sources or [],
        "software_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }


def write_metadata(
    output_dir: str,
    dataset_id: str,
    config: GeneratorConfig,
    graph: SupplyChainGraph,
    result: OperationsResult | None = None,
    real_data_sources: list[str] | None = None,
) -> None:
    """Writes plan §24's metadata/{dataset_info,generation_config,
    feature_schema,provenance}.{json,yaml}."""
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    with open(os.path.join(metadata_dir, "dataset_info.json"), "w") as f:
        json.dump(build_dataset_info(dataset_id, config, graph, result), f, indent=2)

    with open(os.path.join(metadata_dir, "generation_config.yaml"), "w") as f:
        yaml.safe_dump(dataclasses.asdict(config), f, sort_keys=False)

    with open(os.path.join(metadata_dir, "feature_schema.json"), "w") as f:
        json.dump(FEATURE_SCHEMA, f, indent=2)

    with open(os.path.join(metadata_dir, "provenance.json"), "w") as f:
        json.dump(build_provenance(dataset_id, config, real_data_sources), f, indent=2)
