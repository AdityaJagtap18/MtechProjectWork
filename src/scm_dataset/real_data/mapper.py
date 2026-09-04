"""Maps NIST fields onto the six-node ontology and computes structural
statistics for the plan §42 Phase 4 "topology comparison" deliverable.

This does NOT inject NIST's literal values into the generated graph — per
DATASET_DESIGN_REVIEW.md §3.1, NIST's content is synthetic placeholder text
at too small a scale (max 56 rows/table) to calibrate distributions
against. What *is* reusable is structural: field vocabulary and
relationship patterns. `FIELD_MAPPING` documents the former (mirrors
DATASET_DESIGN_REVIEW.md §4); `compute_structural_stats` measures the
latter, e.g. how often a product lists more than one supplier.
"""

from __future__ import annotations

import pandas as pd

from .cleaner import split_supplier_ids
from .loader import load_all_nist_scenarios

FIELD_MAPPING = {
    "Supplier": {
        "source": "suppliers.csv",
        "fields_used": ["ID", "Name", "City/State/Zip"],
        "note": "identity + coarse location only; all risk/capacity/financial attributes are simulated",
    },
    "Material": {
        "source": "products.csv (NIST's 'Product' is actually a BOM line item / component)",
        "fields_used": ["ID", "Name", "Supplier ID", "BOM Level", "Procurement Type"],
        "note": "Procurement Type (MTS/OTS) adopted as an optional Material attribute (design review decision 4)",
    },
    "Product": {
        "source": "projects.csv, top-of-hierarchy rows only",
        "fields_used": ["ID", "Level"],
        "note": "used only as a template for branching-factor shape, not literal Product records",
    },
    "Plant": {"source": None, "note": "no NIST analogue whatsoever; fully simulated"},
    "Region": {
        "source": "suppliers.csv",
        "fields_used": ["City", "State", "Zip"],
        "note": "raw place names only, frequently blank; risk indices come from WGI/INFORM instead (calibrator.py)",
    },
    "Procurement Order": {
        "source": None,
        "note": "NIST tables are static relationship tables, not a transaction/order log; fully simulated",
    },
}


def compute_multi_source_rate(products_df: pd.DataFrame, column: str = "Supplier ID") -> float:
    """Fraction of product rows listing more than one supplier ID."""
    if column not in products_df.columns:
        return float("nan")
    supplier_lists = split_supplier_ids(products_df, column)
    non_empty = supplier_lists[supplier_lists.apply(len) > 0]
    if len(non_empty) == 0:
        return float("nan")
    return float((non_empty.apply(len) > 1).mean())


def compute_bom_level_distribution(products_df: pd.DataFrame, column: str = "BOM Level") -> dict[int, int]:
    if column not in products_df.columns:
        return {}
    return {int(k): int(v) for k, v in products_df[column].dropna().astype(int).value_counts().sort_index().items()}


def compute_project_hierarchy_depth(projects_df: pd.DataFrame, column: str = "Level") -> dict[int, int]:
    if column not in projects_df.columns:
        return {}
    depths = projects_df[column].dropna().astype(str).apply(lambda s: s.count(".") + 1)
    return {int(k): int(v) for k, v in depths.value_counts().sort_index().items()}


def compute_structural_stats(raw_dir: str) -> dict:
    """Aggregate structural statistics across all 5 NIST scenarios. Small-N
    caveat applies throughout (DATASET_DESIGN_REVIEW.md §3.1): illustrative,
    not a statistically powered calibration target."""
    scenarios = load_all_nist_scenarios(raw_dir)
    per_scenario = {}
    multi_source_rates = []
    for name, tables in scenarios.items():
        rate = compute_multi_source_rate(tables["products"])
        per_scenario[name] = {
            "num_suppliers": len(tables["suppliers"]),
            "num_products": len(tables["products"]),
            "num_projects": len(tables["projects"]),
            "multi_source_rate": rate,
            "bom_level_distribution": compute_bom_level_distribution(tables["products"]),
            "project_hierarchy_depth": compute_project_hierarchy_depth(tables["projects"]),
        }
        if rate == rate:  # not NaN
            multi_source_rates.append(rate)

    return {
        "field_mapping": FIELD_MAPPING,
        "per_scenario": per_scenario,
        "overall_multi_source_rate": sum(multi_source_rates) / len(multi_source_rates) if multi_source_rates else None,
        "caveat": "Small-N (max 56 rows/table across 5 toy scenarios); illustrative only, not statistically powered.",
    }
