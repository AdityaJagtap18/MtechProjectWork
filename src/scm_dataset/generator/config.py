"""Generation config (plan §38's config-first design / §8's scaling
parameters). Covers Phase 2 (topology), Phase 3 (normal operations
simulation), Phase 4 (real-data calibration), and Phase 5 (black-swan
events).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class NetworkConfig:
    suppliers: int = 300
    procurement_orders: int = 2000
    materials: int = 100
    plants: int = 50
    products: int = 200
    regions: int = 20


@dataclass
class TopologyConfig:
    # Supplier -> Material assignment (plan §9: single-source vs multi-source)
    p_single_source_material: float = 0.3
    multi_source_min: int = 2
    multi_source_max: int = 5
    same_industry_bias: float = 0.8  # prob. a material's supplier is drawn from its own industry pool

    # Material -> Plant assignment (BOM size per plant)
    materials_per_plant_min: int = 5
    materials_per_plant_max: int = 20

    # Product -> Region market assignment
    home_region_bias: float = 0.7  # prob. a product's market region is its producing plant's own region


@dataclass
class SimulationConfig:
    # Weekly resolution by default (plan §10: "weekly for the first
    # prototype"); ~2 simulated years, within plan §10's 2-3 year range.
    horizon_periods: int = 104
    periods_per_year: int = 52
    days_per_period: int = 7  # converts a supplier's lead_time_mean (days) into periods

    demand_seasonal_amplitude: float = 0.3
    demand_drift_std: float = 0.05

    # Inventory/reorder buffers are sized in "periods of full-capacity
    # material consumption" -- plant.production_capacity *
    # material.required_quantity_per_product -- NOT from the Material
    # node's own safety_stock/inventory_level. Those were generated
    # independently of any specific plant's throughput (materials exist
    # before the material->plant topology does), so using them directly as
    # absolute simulation buffers created a units mismatch that starved
    # production near zero: see simulation/engine.py's module docstring.
    initial_inventory_periods: float = 4.0
    reorder_threshold_periods: float = 1.5
    # reorder target = replenishment_multiplier * reorder_threshold_periods
    replenishment_multiplier: float = 2.0

    # plan §11.3: a supplier can't deliver more than its available capacity
    # in a single period. Effective per-period capacity is likewise derived
    # from the same reference scale (see engine.py's compute_flow_scales),
    # not from supplier.capacity directly, for the same units-mismatch
    # reason; this headroom multiplies that derived reference.
    supplier_capacity_headroom: float = 1.5


@dataclass
class RealDataConfig:
    # Off by default (Phase 2/3 configs stay pure-synthetic/backward
    # compatible); configs/real_calibration.yaml turns this on. Requires
    # `scripts/download_real_data.py` + `scripts/calibrate.py` to have been
    # run first to produce region_risk_table_path.
    use_real_region_calibration: bool = False
    region_risk_table_path: str = "data/processed/region_risk_calibration.csv"


@dataclass
class EventTypeConfig:
    # per-period probability that a new event of this type starts (plan
    # §38's example values, reused as defaults here)
    probability: float = 0.0


@dataclass
class EventsConfig:
    # Off by default: configs/base.yaml/normal.yaml stay event-free (plan
    # §25 Dataset A); configs/black_swan.yaml turns this on.
    enabled: bool = False
    supplier_failure: EventTypeConfig = field(default_factory=lambda: EventTypeConfig(0.02))
    natural_disaster: EventTypeConfig = field(default_factory=lambda: EventTypeConfig(0.01))
    geopolitical: EventTypeConfig = field(default_factory=lambda: EventTypeConfig(0.005))
    cyberattack: EventTypeConfig = field(default_factory=lambda: EventTypeConfig(0.01))
    logistics: EventTypeConfig = field(default_factory=lambda: EventTypeConfig(0.015))
    # P(severity=1..5) when an event fires -- skewed low so severity-5
    # ("Black Swan") events stay rare in the generated population (plan §17)
    severity_weights: list[float] = field(default_factory=lambda: [0.35, 0.30, 0.20, 0.10, 0.05])


@dataclass
class GeneratorConfig:
    seed: int = 42
    network: NetworkConfig = field(default_factory=NetworkConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    real_data: RealDataConfig = field(default_factory=RealDataConfig)
    events: EventsConfig = field(default_factory=EventsConfig)


def _parse_events_config(raw: dict) -> EventsConfig:
    kwargs: dict = {}
    for key in ("supplier_failure", "natural_disaster", "geopolitical", "cyberattack", "logistics"):
        if key in raw:
            kwargs[key] = EventTypeConfig(**raw[key])
    if "enabled" in raw:
        kwargs["enabled"] = raw["enabled"]
    if "severity_weights" in raw:
        kwargs["severity_weights"] = raw["severity_weights"]
    return EventsConfig(**kwargs)


def load_config(path: str) -> GeneratorConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return GeneratorConfig(
        seed=raw.get("seed", 42),
        network=NetworkConfig(**raw.get("network", {})),
        topology=TopologyConfig(**raw.get("topology", {})),
        simulation=SimulationConfig(**raw.get("simulation", {})),
        real_data=RealDataConfig(**raw.get("real_data", {})),
        events=_parse_events_config(raw.get("events", {})),
    )
