"""Material generation (plan §5.3).

`supplier_count` and `concentration` are set to 0 / 0.0 here — per
DATASET_DESIGN_REVIEW.md §5, these must be *derived from the generated
topology*, not sampled independently, so `generator/topology.py` overwrites
them once Supplier->Material edges exist (see `_finalize_material_derived_fields`).

`substitutability` is deliberately anti-correlated with `criticality`
(plan §13's supplier-dependency example generalizes to materials: highly
critical materials tend to be harder to substitute), not drawn independently.

`procurement_type` (MTS/OTS) is the field adopted from the NIST GPS
Manufacturer table per DATASET_DESIGN_REVIEW.md decision 4; materials
without a clear real-world analogue get `None`.
"""

from __future__ import annotations

import numpy as np

from ..schema.nodes import Material

MATERIAL_CATEGORIES = ["electronics", "metals", "chemicals", "textiles", "machinery", "software", "logistics"]
PROCUREMENT_TYPES: list[str | None] = ["MTS", "OTS", None]


def generate_materials(num_materials: int, rng: np.random.Generator) -> list[Material]:
    materials = []
    for i in range(num_materials):
        category = MATERIAL_CATEGORIES[rng.integers(0, len(MATERIAL_CATEGORIES))]
        criticality = float(rng.beta(2.5, 3.0))
        substitutability = float(np.clip(rng.beta(2.0, 2.0) * (1 - 0.5 * criticality), 0.0, 1.0))
        demand = float(rng.lognormal(mean=np.log(200), sigma=0.6))
        unit_cost = float(rng.lognormal(mean=np.log(20), sigma=0.8))
        safety_stock = float(demand * rng.uniform(0.1, 0.3))
        inventory_level = float(safety_stock * rng.uniform(1.0, 2.5))
        required_quantity_per_product = float(rng.gamma(shape=2.0, scale=1.5))
        procurement_type = PROCUREMENT_TYPES[rng.integers(0, len(PROCUREMENT_TYPES))]

        materials.append(
            Material(
                material_id=f"material_{i}",
                material_category=category,
                criticality=criticality,
                substitutability=substitutability,
                demand=demand,
                unit_cost=unit_cost,
                inventory_level=inventory_level,
                safety_stock=safety_stock,
                supplier_count=0,
                concentration=0.0,
                required_quantity_per_product=required_quantity_per_product,
                procurement_type=procurement_type,
            )
        )
    return materials
