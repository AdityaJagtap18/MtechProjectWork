"""Event-engine validation (plan §35 "Event validation") and the Phase 7
report orchestrator.

Event validation checks are statistical/structural properties of the event
*engine* (does severity monotonically increase impact? do region-scoped
events stay inside their region?) rather than properties of one specific
generated dataset, so — like `validation/statistical.py`'s region-risk
comparison — they're run fresh against a graph rather than read off
already-exported operations records.
"""

from __future__ import annotations

import numpy as np

from ..events.natural_disaster import create_event as create_natural_disaster_event
from ..events.supplier_failure import create_event as create_supplier_failure_event
from ..schema.edges import EdgeType
from ..schema.graph import SupplyChainGraph
from ..schema.nodes import NodeType


def validate_severity_monotonicity(graph: SupplyChainGraph, rng: np.random.Generator, n_samples: int = 100) -> dict:
    """Plan §36: "supplier failure severity 5 should not result in less
    disruption than severity 1." Checked in expectation over many draws
    (any single pair can differ by chance — the noise multipliers in
    events/supplier_failure.py are only ±10-20%), not for one pair."""
    low = [
        create_supplier_failure_event(f"val_low_{i}", 0, graph, severity=1, probability_regime=0.01, rng=rng).capacity_reduction_fraction
        for i in range(n_samples)
    ]
    high = [
        create_supplier_failure_event(f"val_high_{i}", 0, graph, severity=5, probability_regime=0.01, rng=rng).capacity_reduction_fraction
        for i in range(n_samples)
    ]
    return {
        "mean_capacity_reduction_severity_1": float(np.mean(low)),
        "mean_capacity_reduction_severity_5": float(np.mean(high)),
        "severity_5_exceeds_severity_1": float(np.mean(high)) > float(np.mean(low)),
        "n_samples": n_samples,
    }


def validate_geographic_isolation(graph: SupplyChainGraph, rng: np.random.Generator, n_samples: int = 50) -> dict:
    """Plan §35: "geographically isolated events do not affect unrelated
    regions without a modeled path" — every supplier/plant a Natural
    Disaster event lists as affected must actually be located (via
    SUPPLIER_REGION/PLANT_REGION edges) in the region it targeted."""
    supplier_region = {e.source_id: e.target_id for e in graph.edges_of_type(EdgeType.SUPPLIER_REGION)}
    plant_region = {e.source_id: e.target_id for e in graph.edges_of_type(EdgeType.PLANT_REGION)}

    violations = []
    for i in range(n_samples):
        event = create_natural_disaster_event(f"val_nd_{i}", 0, graph, severity=5, probability_regime=0.01, rng=rng)
        target_region = event.affected_regions[0]
        for supplier_id in event.affected_suppliers:
            if supplier_region.get(supplier_id) != target_region:
                violations.append(f"{event.event_id}: supplier {supplier_id} not located in targeted region {target_region}")
        for plant_id in event.affected_plants:
            if plant_region.get(plant_id) != target_region:
                violations.append(f"{event.event_id}: plant {plant_id} not located in targeted region {target_region}")

    return {"n_samples": n_samples, "violations": violations}


def validate_single_source_concentration(graph: SupplyChainGraph) -> dict:
    """Plan §35: "higher dependency increases vulnerability" / "multi-source
    supply reduces vulnerability" — checked via the structural invariant a
    single-sourced material should always carry: `concentration == 1.0`
    (the equal-share HHI collapses to 1 for exactly one supplier)."""
    materials = graph.nodes_of_type(NodeType.MATERIAL)
    single_source = [m for m in materials if m.supplier_count == 1]
    violations = [f"{m.material_id}: supplier_count=1 but concentration={m.concentration}" for m in single_source if abs(m.concentration - 1.0) > 1e-6]
    return {"num_single_source_materials": len(single_source), "violations": violations}


def build_event_validation_report(graph: SupplyChainGraph, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "severity_monotonicity": validate_severity_monotonicity(graph, rng),
        "geographic_isolation": validate_geographic_isolation(graph, rng),
        "single_source_concentration": validate_single_source_concentration(graph),
    }
