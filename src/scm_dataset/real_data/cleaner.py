"""Cleaning helpers for raw NIST tables (plan §3.2)."""

from __future__ import annotations

import pandas as pd


def split_supplier_ids(products_df: pd.DataFrame, column: str = "Supplier ID") -> pd.Series:
    """Split a possibly semicolon-delimited Supplier ID field (observed in
    SGE_products.csv — see DATASET_DESIGN_REVIEW.md §3.3) into a list of IDs
    per row. Rows with a single supplier just get a one-element list."""
    return products_df[column].fillna("").astype(str).apply(lambda s: [x.strip() for x in s.split(";") if x.strip()])
