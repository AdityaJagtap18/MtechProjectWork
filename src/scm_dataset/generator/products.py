"""Product generation (plan §5.5).

`material_dependency` is set to a category-level prior (0.5) here and
refined once each product's producing plant (and that plant's BOM size) is
known — see `topology._finalize_product_material_dependency`. Full
consistency between `material_dependency` and actual material consumption
only becomes possible once Phase 3 (operations/production) exists; this is
a Phase 2 approximation, not the final value.
"""

from __future__ import annotations

import numpy as np

from ..schema.nodes import Product

PRODUCT_CATEGORIES = [
    "consumer_electronics", "industrial_equipment", "automotive", "medical", "apparel", "software",
]


def generate_products(num_products: int, rng: np.random.Generator) -> list[Product]:
    products = []
    for i in range(num_products):
        category = PRODUCT_CATEGORIES[rng.integers(0, len(PRODUCT_CATEGORIES))]
        demand = float(rng.lognormal(mean=np.log(150), sigma=0.7))
        revenue_per_unit = float(rng.lognormal(mean=np.log(50), sigma=0.9))
        margin = float(np.clip(rng.normal(0.25, 0.15), -0.5, 0.9))
        criticality = float(rng.beta(2.0, 3.0))
        substitution_score = float(rng.beta(2.0, 2.0))
        backlog = float(rng.gamma(shape=1.5, scale=max(demand * 0.05, 1e-3)))

        products.append(
            Product(
                product_id=f"product_{i}",
                product_category=category,
                demand=demand,
                revenue_per_unit=revenue_per_unit,
                margin=margin,
                material_dependency=0.5,
                criticality=criticality,
                substitution_score=substitution_score,
                backlog=backlog,
            )
        )
    return products
