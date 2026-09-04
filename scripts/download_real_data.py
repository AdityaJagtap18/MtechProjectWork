#!/usr/bin/env python3
"""Download and persist the external real-data sources used for Phase 4
calibration (plan §3, §42 Phase 4), each with a provenance record per plan
§3.3's template.

Sources:
  - NIST Sample Purchasing / Supply Chain Data (public; schema/relationship
    reference only per DATASET_DESIGN_REVIEW.md §3.1 -- too small and
    synthetic-looking to calibrate distributions against)
  - World Bank Worldwide Governance Indicators, WGI (CC-BY-4.0) -- feeds
    region.geopolitical_risk (Political Stability) and region.trade_risk
    (Regulatory Quality)
  - INFORM Risk Index, EU JRC (CC-BY-4.0) -- feeds
    region.natural_disaster_risk ("Natural" hazard&exposure) and
    region.infrastructure_risk ("Infrastructure" lack-of-coping-capacity)

Usage:
    python scripts/download_real_data.py --output data/raw
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
import zipfile
from datetime import date

NIST_URL = "https://data.nist.gov/od/ds/mds2-2183/Sample_Data_Sets.zip"
WGI_BASE_URL = "https://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&date={year}&per_page=400"
INFORM_URL = "https://drmkc.jrc.ec.europa.eu/inform-index/Portals/0/InfoRM/2026/INFORM_Risk_Mid_2026_v073.xlsx"

WGI_YEAR = 2023
INFORM_VERSION = "Mid 2026 (v073)"


def _download(url: str, dest_path: str, headers: dict[str, str] | None = None) -> None:
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())


def _write_provenance(dir_path: str, **fields) -> None:
    fields.setdefault("access_date", date.today().isoformat())
    with open(os.path.join(dir_path, "provenance.json"), "w") as f:
        json.dump(fields, f, indent=2)


def download_nist(output_dir: str) -> None:
    nist_dir = os.path.join(output_dir, "nist")
    os.makedirs(nist_dir, exist_ok=True)
    zip_path = os.path.join(nist_dir, "Sample_Data_Sets.zip")
    print(f"Downloading NIST sample data -> {zip_path}")
    _download(NIST_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(nist_dir)

    _write_provenance(
        nist_dir,
        source_name="NIST Sample Purchasing / Supply Chain Data",
        source_url="https://catalog.data.gov/dataset/sample-purchasing-supply-chain-data",
        download_url=NIST_URL,
        license="https://www.nist.gov/open/license (public)",
        data_type="schema exemplar (5 toy SRM scenarios, suppliers/products/projects.csv)",
        observed_or_synthetic="synthetic placeholder content (Lorem-Ipsum-style), real schema",
        fields_used="supplier ID/name/location; product/BOM Level/Procurement Type/Supplier ID; project ID/Level hierarchy",
        fields_not_used="contact name/email/phone, website (no modeling value)",
        transformations="none (used only for schema/structural comparison, not literal values -- see DATASET_DESIGN_REVIEW.md §3.1)",
    )
    print(f"NIST data extracted and provenance written to {nist_dir}")


def download_wgi(output_dir: str) -> None:
    wgi_dir = os.path.join(output_dir, "wgi")
    os.makedirs(wgi_dir, exist_ok=True)

    indicators = {"GOV_WGI_PV.EST": "political_stability.json", "GOV_WGI_RQ.EST": "regulatory_quality.json"}
    for indicator, filename in indicators.items():
        url = WGI_BASE_URL.format(indicator=indicator, year=WGI_YEAR)
        dest = os.path.join(wgi_dir, filename)
        print(f"Downloading WGI {indicator} ({WGI_YEAR}) -> {dest}")
        _download(url, dest)

    _write_provenance(
        wgi_dir,
        source_name="World Bank Worldwide Governance Indicators (WGI)",
        source_url="https://databank.worldbank.org/source/worldwide-governance-indicators",
        download_url=WGI_BASE_URL.format(indicator="GOV_WGI_{PV,RQ}.EST", year=WGI_YEAR),
        license="CC-BY-4.0",
        data_type="observed (survey-based governance estimates), per-country, per-year",
        observed_or_synthetic="observed",
        fields_used=f"GOV_WGI_PV.EST (Political Stability), GOV_WGI_RQ.EST (Regulatory Quality), year={WGI_YEAR}",
        fields_not_used="Government Effectiveness, Rule of Law, Control of Corruption, Voice & Accountability (not mapped to our region schema)",
        transformations="none at download time -- normalization to [0,1] risk happens in scripts/calibrate.py",
    )
    print(f"WGI data downloaded and provenance written to {wgi_dir}")


def download_inform(output_dir: str) -> None:
    inform_dir = os.path.join(output_dir, "inform")
    os.makedirs(inform_dir, exist_ok=True)
    dest = os.path.join(inform_dir, "inform_risk.xlsx")
    print(f"Downloading INFORM Risk Index -> {dest}")
    _download(INFORM_URL, dest)

    _write_provenance(
        inform_dir,
        source_name="INFORM Risk Index",
        source_url="https://drmkc.jrc.ec.europa.eu/inform-index/INFORM-Risk/Results-and-data",
        download_url=INFORM_URL,
        license="CC-BY-4.0",
        data_type="composite index (hazard/exposure, vulnerability, lack of coping capacity), per-country",
        observed_or_synthetic="observed/modeled composite index",
        fields_used="'Natural' (Hazard & Exposure), 'Infrastructure' (Lack of Coping Capacity), from sheet 'INFORM Risk 2026 (a-z)'",
        fields_not_used="Human hazard/conflict, Vulnerability, Institutional/DRR/Governance/Communication sub-scores (not mapped to our region schema)",
        transformations="none at download time -- normalization to [0,1] risk happens in scripts/calibrate.py",
        version=INFORM_VERSION,
    )
    print(f"INFORM data downloaded and provenance written to {inform_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--skip", nargs="*", choices=["nist", "wgi", "inform"], default=[])
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if "nist" not in args.skip:
        download_nist(args.output)
    if "wgi" not in args.skip:
        download_wgi(args.output)
    if "inform" not in args.skip:
        download_inform(args.output)


if __name__ == "__main__":
    main()
