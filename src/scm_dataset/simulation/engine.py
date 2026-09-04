"""Discrete-time operations simulator (plan §42 Phases 3 & 5, §10-11, §19).

Builds a "normal operations" time series on top of the Phase 2 topology:
demand, production, inventory, procurement orders, and deliveries per
period, connected through the constraints in plan §11 (bounded production,
disruption-free inventory, lead-time-gated delivery).

Phase 5's black-swan events plug into this same loop rather than needing
separate cascade machinery: each active event just multiplies the capacity
and lead-time inputs of the entities it directly affects (see
`events/base.py`) before this loop's existing production/inventory/
procurement logic runs. A disrupted supplier delivers less and later, which
starves the plants that depend on it, which cuts production, which grows
backlog — plan §19's cascade sequence falls out of the conservation
constraints already enforced here, not from event-specific propagation
rules. With `config.events.enabled = False` (the default), this reduces
exactly to the undisturbed baseline (plan §25's Dataset A).

**Why inventory/reorder/supplier-capacity scales are derived, not read
directly off the graph:** `Material.required_quantity_per_product`,
`Material.safety_stock`, `Plant.production_capacity`, and
`Supplier.capacity` are all generated independently in Phase 2 — materials
exist before the material->plant topology does, so nothing at generation
time can calibrate a material's buffer against any particular plant's
throughput. Using those Phase-2 attributes directly as absolute simulation
quantities creates a units mismatch: a plant's capacity (~thousands/period)
against a material's independently-sampled safety_stock (~tens) means
production is starved to near-zero by whichever of a plant's ~10+ materials
happened to be sampled with the least relative buffer (`min()` over many
near-arbitrary numbers collapses toward the smallest one). `compute_flow_scales`
fixes this by deriving every (plant, material) pair's buffer from that
pair's own implied consumption rate — `plant.production_capacity *
material.required_quantity_per_product` — so every material a plant
depends on runs out in the same number of periods by construction, and
supplier capacity is sized against the same reference scale rather than an
unrelated random number. The static `Material.safety_stock`/
`inventory_level` and `Supplier.capacity` fields remain as-is on the graph
(useful as descriptive/ML-feature attributes); they're just not what the
simulation uses for its own internal buffers.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import count

import numpy as np

from ..events.base import capacity_multiplier, is_active, lead_time_multiplier
from ..events.generate import maybe_generate_events
from ..generator.config import GeneratorConfig, SimulationConfig
from ..operations.demand import generate_demand_schedule
from ..operations.inventory import consume_materials
from ..operations.procurement import decide_orders, fulfill_orders
from ..operations.production import compute_plant_production
from ..schema.edges import EdgeType
from ..schema.events import Event
from ..schema.graph import SupplyChainGraph
from ..schema.nodes import Material, NodeType, Plant, Supplier


@dataclass
class OperationsResult:
    demand_records: list[dict] = field(default_factory=list)
    production_records: list[dict] = field(default_factory=list)
    inventory_records: list[dict] = field(default_factory=list)
    deliveries_records: list[dict] = field(default_factory=list)
    procurement_records: list[dict] = field(default_factory=list)
    backlog_records: list[dict] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)


def build_indices(graph: SupplyChainGraph):
    plant_materials: dict[str, list[str]] = defaultdict(list)
    for e in graph.edges_of_type(EdgeType.MATERIAL_PLANT):
        plant_materials[e.target_id].append(e.source_id)

    plant_products: dict[str, list[str]] = defaultdict(list)
    for e in graph.edges_of_type(EdgeType.PLANT_PRODUCT):
        plant_products[e.source_id].append(e.target_id)

    material_suppliers: dict[str, list[str]] = defaultdict(list)
    for e in graph.edges_of_type(EdgeType.SUPPLIER_MATERIAL):
        material_suppliers[e.target_id].append(e.source_id)

    # Which supplier does each (plant, material) pair order from "by default"?
    # Prefer whichever supplier Phase 2 already anchored a static procurement
    # order to for that pair; otherwise fall back to a deterministic choice
    # among the material's suppliers.
    order_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for order in graph.nodes_of_type(NodeType.PROCUREMENT):
        order_counts[(order.plant_id, order.material_id)][order.supplier_id] += 1

    primary_supplier: dict[tuple[str, str], str | None] = {}
    for plant_id, material_ids in plant_materials.items():
        for material_id in material_ids:
            key = (plant_id, material_id)
            candidates = order_counts.get(key)
            if candidates:
                primary_supplier[key] = max(candidates.items(), key=lambda kv: kv[1])[0]
            else:
                options = sorted(material_suppliers.get(material_id, []))
                primary_supplier[key] = options[0] if options else None

    return plant_materials, plant_products, primary_supplier


def compute_flow_scales(
    plants: list[Plant],
    materials_by_id: dict[str, Material],
    plant_materials: dict[str, list[str]],
    primary_supplier: dict[tuple[str, str], str | None],
    suppliers_by_id: dict[str, Supplier],
    sim: SimulationConfig,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float], dict[tuple[str, str], float], dict[str, float]]:
    """Returns (initial_inventory, reorder_threshold, reorder_target,
    supplier_effective_capacity), all derived from each (plant, material)
    pair's own `plant.production_capacity * material.required_quantity_per_product`
    reference rate — see this module's docstring."""
    pair_rate: dict[tuple[str, str], float] = {}
    for plant in plants:
        for material_id in plant_materials.get(plant.plant_id, []):
            material = materials_by_id[material_id]
            pair_rate[(plant.plant_id, material_id)] = plant.production_capacity * max(
                material.required_quantity_per_product, 1e-6
            )

    initial_inventory = {key: rate * sim.initial_inventory_periods for key, rate in pair_rate.items()}
    reorder_threshold = {key: rate * sim.reorder_threshold_periods for key, rate in pair_rate.items()}
    reorder_target = {
        key: rate * sim.reorder_threshold_periods * sim.replenishment_multiplier for key, rate in pair_rate.items()
    }

    supplier_pair_total: dict[str, float] = defaultdict(float)
    for key, rate in pair_rate.items():
        supplier_id = primary_supplier.get(key)
        if supplier_id is not None:
            supplier_pair_total[supplier_id] += rate

    active_capacities = [suppliers_by_id[sid].capacity for sid in supplier_pair_total]
    mean_capacity = float(np.mean(active_capacities)) if active_capacities else 1.0
    supplier_effective_capacity = {
        sid: total * sim.supplier_capacity_headroom * (suppliers_by_id[sid].capacity / mean_capacity)
        for sid, total in supplier_pair_total.items()
    }

    return initial_inventory, reorder_threshold, reorder_target, supplier_effective_capacity


def run_simulation(graph: SupplyChainGraph, config: GeneratorConfig, rng: np.random.Generator | None = None) -> OperationsResult:
    rng = rng if rng is not None else np.random.default_rng(config.seed)
    sim = config.simulation

    plants = graph.nodes_of_type(NodeType.PLANT)
    products = graph.nodes_of_type(NodeType.PRODUCT)
    materials_by_id = {m.material_id: m for m in graph.nodes_of_type(NodeType.MATERIAL)}
    suppliers_by_id = {s.supplier_id: s for s in graph.nodes_of_type(NodeType.SUPPLIER)}

    plant_materials, plant_products, primary_supplier = build_indices(graph)
    initial_inventory, reorder_threshold, reorder_target, supplier_effective_capacity = compute_flow_scales(
        plants, materials_by_id, plant_materials, primary_supplier, suppliers_by_id, sim
    )

    demand_schedule = generate_demand_schedule(products, sim.horizon_periods, sim.periods_per_year, sim, rng)

    inventory: dict[tuple[str, str], float] = dict(initial_inventory)
    pending: dict[tuple[str, str], list[tuple[int, float, str]]] = defaultdict(list)
    backlog: dict[str, float] = {p.product_id: 0.0 for p in products}

    result = OperationsResult()
    events: list[Event] = []
    event_id_counter = count()

    for t in range(sim.horizon_periods):
        # Snapshot backlog as it stands entering this period -- this is what
        # demand_cap below actually uses, and Phase 6's labels need it (not
        # the post-update value) to reconstruct demand_cap(t) exactly.
        backlog_entering = dict(backlog)

        events.extend(maybe_generate_events(t, graph, config.events, rng, event_id_counter))
        active_events = [e for e in events if is_active(e, t)]

        supplier_cap_mult: dict[str, float] = defaultdict(lambda: 1.0)
        supplier_lead_mult: dict[str, float] = defaultdict(lambda: 1.0)
        plant_cap_mult: dict[str, float] = defaultdict(lambda: 1.0)
        for e in active_events:
            cap_m = capacity_multiplier(e, t)
            lead_m = lead_time_multiplier(e, t)
            for supplier_id in e.affected_suppliers:
                supplier_cap_mult[supplier_id] *= cap_m
                supplier_lead_mult[supplier_id] *= lead_m
            for plant_id in e.affected_plants:
                plant_cap_mult[plant_id] *= cap_m

        for product in products:
            result.demand_records.append(
                {"product_id": product.product_id, "time": t, "demand": demand_schedule[(product.product_id, t)]}
            )

        # Deliveries whose lead time has elapsed arrive before today's production.
        for key, entries in list(pending.items()):
            plant_id, material_id = key
            still_pending = []
            for arrival_t, qty, supplier_id in entries:
                if arrival_t <= t:
                    inventory[key] = inventory.get(key, 0.0) + qty
                    result.deliveries_records.append(
                        {"plant_id": plant_id, "material_id": material_id, "supplier_id": supplier_id, "time": t, "quantity": qty}
                    )
                else:
                    still_pending.append((arrival_t, qty, supplier_id))
            pending[key] = still_pending

        product_production_total: dict[str, float] = defaultdict(float)
        for plant in plants:
            material_ids = plant_materials.get(plant.plant_id, [])
            product_ids = plant_products.get(plant.plant_id, [])
            demand_cap = sum(demand_schedule[(pid, t)] + backlog[pid] for pid in product_ids)
            production_qty = compute_plant_production(
                plant, material_ids, materials_by_id, inventory, demand_cap=demand_cap,
                capacity_multiplier=plant_cap_mult[plant.plant_id],
            )

            weights = [max(demand_schedule[(pid, t)] + backlog[pid], 1e-6) for pid in product_ids]
            total_weight = sum(weights)
            for pid, weight in zip(product_ids, weights):
                share = production_qty * (weight / total_weight) if total_weight > 0 else 0.0
                result.production_records.append({"plant_id": plant.plant_id, "product_id": pid, "time": t, "production": share})
                product_production_total[pid] += share

            consume_materials(inventory, plant.plant_id, material_ids, materials_by_id, production_qty)

        for product in products:
            result.backlog_records.append({"product_id": product.product_id, "time": t, "backlog": backlog_entering[product.product_id]})
            produced = product_production_total.get(product.product_id, 0.0)
            backlog[product.product_id] = max(
                0.0, backlog[product.product_id] + demand_schedule[(product.product_id, t)] - produced
            )

        for key, level in inventory.items():
            result.inventory_records.append({"plant_id": key[0], "material_id": key[1], "time": t, "inventory": level})

        orders_this_period = []
        for plant in plants:
            orders_this_period += decide_orders(
                plant.plant_id, plant_materials.get(plant.plant_id, []), inventory, reorder_threshold, reorder_target,
                primary_supplier, rng,
            )
        result.procurement_records.extend(
            fulfill_orders(
                orders_this_period, suppliers_by_id, supplier_effective_capacity, sim, t, pending, rng,
                supplier_capacity_multiplier=supplier_cap_mult, supplier_lead_time_multiplier=supplier_lead_mult,
            )
        )

    result.events = events
    return result
