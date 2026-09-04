"""Node schema for the six-node SCM ontology (plan §5).

The plan names each node's attributes but does not always specify a type or
value range. Where it doesn't, this module follows the plan's own convention
for the fields it *does* pin down: [0, 1] for normalized scores/probabilities/
utilization, non-negative floats for physical quantities (capacity, cost,
inventory, lead time). These are Phase 1 design decisions, flagged as such,
and may be revisited once real-data calibration (Phase 4) is available.

`Material.procurement_type` is an addition beyond plan §5.3, adopted per the
DATASET_DESIGN_REVIEW.md decision log (item 4): a real field observed in the
NIST sample data (MTS/OTS), optional and defaulted to None so it never blocks
construction of a Material that doesn't set it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class NodeType(str, Enum):
    SUPPLIER = "supplier"
    PROCUREMENT = "procurement"
    MATERIAL = "material"
    PLANT = "plant"
    PRODUCT = "product"
    REGION = "region"


def _check_range(value: Optional[float], lo: float, hi: float, name: str, errors: list[str]) -> None:
    if value is None:
        return
    if not (lo <= value <= hi):
        errors.append(f"{name}={value!r} is out of range [{lo}, {hi}]")


def _check_nonneg(value: Optional[float], name: str, errors: list[str]) -> None:
    if value is None:
        return
    if value < 0:
        errors.append(f"{name}={value!r} must be >= 0")


def _check_nonempty(value: str, name: str, errors: list[str]) -> None:
    if not value:
        errors.append(f"{name} must be non-empty")


_UNIT_SCORES = (
    "capacity_utilization", "reliability", "financial_health", "quality_score",
    "substitution_availability", "geopolitical_exposure", "disaster_exposure",
    "cyber_exposure", "criticality",
)


@dataclass
class Supplier:
    supplier_id: str
    tier: int
    region_id: str
    industry: str
    capacity: float
    capacity_utilization: float
    reliability: float
    financial_health: float
    lead_time_mean: float
    lead_time_variability: float
    quality_score: float
    inventory_buffer: float
    substitution_availability: float
    geopolitical_exposure: float
    disaster_exposure: float
    cyber_exposure: float
    criticality: float

    node_type = NodeType.SUPPLIER

    def validate(self) -> list[str]:
        errors: list[str] = []
        _check_nonempty(self.supplier_id, "supplier_id", errors)
        _check_nonempty(self.region_id, "region_id", errors)
        if self.tier < 1:
            errors.append(f"tier={self.tier} must be >= 1")
        _check_nonneg(self.capacity, "capacity", errors)
        _check_nonneg(self.lead_time_mean, "lead_time_mean", errors)
        _check_nonneg(self.lead_time_variability, "lead_time_variability", errors)
        _check_nonneg(self.inventory_buffer, "inventory_buffer", errors)
        for name in _UNIT_SCORES:
            _check_range(getattr(self, name), 0.0, 1.0, name, errors)
        return errors


@dataclass
class ProcurementOrder:
    procurement_id: str
    supplier_id: str
    material_id: str
    plant_id: str
    order_quantity: float
    order_value: float
    order_frequency: float
    promised_lead_time: float
    actual_lead_time: float
    urgency: float
    contract_duration: float
    priority: float

    node_type = NodeType.PROCUREMENT

    def validate(self) -> list[str]:
        errors: list[str] = []
        for name in ("procurement_id", "supplier_id", "material_id", "plant_id"):
            _check_nonempty(getattr(self, name), name, errors)
        for name in ("order_quantity", "order_value", "order_frequency",
                      "promised_lead_time", "actual_lead_time", "contract_duration"):
            _check_nonneg(getattr(self, name), name, errors)
        _check_range(self.urgency, 0.0, 1.0, "urgency", errors)
        _check_range(self.priority, 0.0, 1.0, "priority", errors)
        return errors


@dataclass
class Material:
    material_id: str
    material_category: str
    criticality: float
    substitutability: float
    demand: float
    unit_cost: float
    inventory_level: float
    safety_stock: float
    supplier_count: int
    concentration: float
    required_quantity_per_product: float
    procurement_type: Optional[str] = None

    node_type = NodeType.MATERIAL

    def validate(self) -> list[str]:
        errors: list[str] = []
        _check_nonempty(self.material_id, "material_id", errors)
        _check_range(self.criticality, 0.0, 1.0, "criticality", errors)
        _check_range(self.substitutability, 0.0, 1.0, "substitutability", errors)
        _check_range(self.concentration, 0.0, 1.0, "concentration", errors)
        for name in ("demand", "unit_cost", "inventory_level", "safety_stock",
                      "required_quantity_per_product"):
            _check_nonneg(getattr(self, name), name, errors)
        if self.supplier_count < 0:
            errors.append(f"supplier_count={self.supplier_count} must be >= 0")
        if self.procurement_type is not None and self.procurement_type not in ("MTS", "OTS"):
            errors.append(f"procurement_type={self.procurement_type!r} must be one of 'MTS', 'OTS', or None")
        return errors


@dataclass
class Plant:
    plant_id: str
    region_id: str
    production_capacity: float
    utilization: float
    operating_cost: float
    inventory_capacity: float
    resilience_score: float
    downtime_cost: float
    recovery_rate: float

    node_type = NodeType.PLANT

    def validate(self) -> list[str]:
        errors: list[str] = []
        _check_nonempty(self.plant_id, "plant_id", errors)
        _check_nonempty(self.region_id, "region_id", errors)
        _check_range(self.utilization, 0.0, 1.0, "utilization", errors)
        _check_range(self.resilience_score, 0.0, 1.0, "resilience_score", errors)
        for name in ("production_capacity", "operating_cost", "inventory_capacity", "downtime_cost"):
            _check_nonneg(getattr(self, name), name, errors)
        # recovery_rate semantics are underspecified by plan §5.4/§21; only
        # constrained to be strictly positive here (see nodes.py docstring).
        if self.recovery_rate <= 0:
            errors.append(f"recovery_rate={self.recovery_rate!r} must be > 0")
        return errors


@dataclass
class Product:
    product_id: str
    product_category: str
    demand: float
    revenue_per_unit: float
    margin: float
    material_dependency: float
    criticality: float
    substitution_score: float
    backlog: float

    node_type = NodeType.PRODUCT

    def validate(self) -> list[str]:
        errors: list[str] = []
        _check_nonempty(self.product_id, "product_id", errors)
        _check_nonneg(self.demand, "demand", errors)
        _check_nonneg(self.revenue_per_unit, "revenue_per_unit", errors)
        _check_nonneg(self.backlog, "backlog", errors)
        _check_range(self.margin, -1.0, 1.0, "margin", errors)
        _check_range(self.material_dependency, 0.0, 1.0, "material_dependency", errors)
        _check_range(self.criticality, 0.0, 1.0, "criticality", errors)
        _check_range(self.substitution_score, 0.0, 1.0, "substitution_score", errors)
        return errors


@dataclass
class Region:
    region_id: str
    country_or_region: str
    geopolitical_risk: float
    natural_disaster_risk: float
    infrastructure_risk: float
    trade_risk: float
    cyber_risk: float
    transport_reliability: float

    node_type = NodeType.REGION

    def validate(self) -> list[str]:
        errors: list[str] = []
        _check_nonempty(self.region_id, "region_id", errors)
        for name in ("geopolitical_risk", "natural_disaster_risk", "infrastructure_risk",
                     "trade_risk", "cyber_risk", "transport_reliability"):
            _check_range(getattr(self, name), 0.0, 1.0, name, errors)
        return errors
