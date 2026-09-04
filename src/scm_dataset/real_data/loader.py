"""Loaders for external real-data sources (plan §3, Phase 4).

Pure file I/O — no network calls here (that's `scripts/download_real_data.py`).
Everything here reads from an already-downloaded `data/raw/` directory.
"""

from __future__ import annotations

import json
import os

import openpyxl
import pandas as pd

NIST_SCENARIOS: dict[str, str] = {
    "01 Basic Sample Data Set Small": "",
    "02 Interconnected Sample Data Set Small": "",
    "GPS Manufacturer": "GPS_",
    "Medical Software": "medical_",
    "Small Government Entity": "SGE_",
}


def load_wgi_indicator(raw_dir: str, filename: str) -> dict[str, float]:
    """Returns {iso3: value} for a downloaded WGI indicator JSON file."""
    path = os.path.join(raw_dir, "wgi", filename)
    with open(path) as f:
        data = json.load(f)
    result = {}
    for row in data[1]:
        iso3 = row.get("countryiso3code")
        value = row.get("value")
        if iso3 and value is not None:
            result[iso3] = float(value)
    return result


def load_inform_risk(raw_dir: str, sheet_name: str = "INFORM Risk 2026 (a-z)") -> pd.DataFrame:
    """Returns a DataFrame with columns country/iso3/natural_hazard/
    infrastructure, both sub-scores on INFORM's native [0, 10] scale."""
    path = os.path.join(raw_dir, "inform", "inform_risk.xlsx")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet_name]

    records = []
    for row in ws.iter_rows(min_row=4, values_only=True):  # rows 1-3 are header/units
        country, iso3 = row[0], row[1]
        if country is None or iso3 is None:
            continue
        records.append({"country": country, "iso3": iso3, "natural_hazard": row[7], "infrastructure": row[34]})
    return pd.DataFrame(records)


def load_nist_scenario(raw_dir: str, scenario_dir: str, prefix: str = "") -> dict[str, pd.DataFrame]:
    """Load one NIST sample scenario's suppliers/products/projects.csv.

    Most files are UTF-8 with a BOM; `GPS_products.csv` has a handful of
    non-UTF-8 bytes in free-text description fields (not in any field this
    project uses), so fall back to cp1252 rather than fail the whole load."""
    base = os.path.join(raw_dir, "nist", "SampleDataSets", scenario_dir)

    def _read(name: str) -> pd.DataFrame:
        path = os.path.join(base, f"{prefix}{name}.csv")
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            # latin-1 maps every byte 0-255, so it never raises -- acceptable
            # here since the bad bytes are in free-text fields we don't use
            return pd.read_csv(path, encoding="latin-1")

    return {"suppliers": _read("suppliers"), "products": _read("products"), "projects": _read("projects")}


def load_all_nist_scenarios(raw_dir: str) -> dict[str, dict[str, pd.DataFrame]]:
    return {name: load_nist_scenario(raw_dir, name, prefix) for name, prefix in NIST_SCENARIOS.items()}
