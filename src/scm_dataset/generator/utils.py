"""Shared generation utilities.

`preferential_sample` reimplements the preferential-attachment idea used by
SupplySim's buyer-supplier assignment (see DATASET_DESIGN_REVIEW.md §2.1:
this is an independent reimplementation of the *algorithm*, not code taken
from that repo, which carries no license). Probability of picking an
already-popular candidate grows with its current usage count, which produces
the supplier-concentration / power-law-degree behaviour plan §9 asks for
instead of uniform random edges. An optional static `base_weights` lets a
node property (e.g. material criticality) bias the draw on top of usage.
"""

from __future__ import annotations

import numpy as np


def preferential_sample(
    rng: np.random.Generator,
    candidates: list[str],
    usage: dict[str, int],
    k: int,
    base_weights: list[float] | None = None,
) -> list[str]:
    k = min(k, len(candidates))
    pool = list(candidates)
    base = dict(zip(candidates, base_weights)) if base_weights is not None else {}
    chosen = []
    for _ in range(k):
        weights = np.array([(usage.get(c, 0) + 1) * base.get(c, 1.0) for c in pool], dtype=float)
        probs = weights / weights.sum()
        idx = rng.choice(len(pool), p=probs)
        picked = pool[idx]
        chosen.append(picked)
        usage[picked] = usage.get(picked, 0) + 1
        pool.pop(idx)
    return chosen
