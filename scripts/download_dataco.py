#!/usr/bin/env python3
"""Download the DataCo Smart Supply Chain dataset (Constante, Silva &
Pereira, 2019, Mendeley Data V5, CC-BY-4.0), used as an external
real-world benchmark for the classical-vs-QGNN comparison (separate from
the Phase 4 region-risk calibration sources handled by
scripts/download_real_data.py).

The Mendeley bundle also ships a 91MB clickstream file
(tokenized_access_logs.csv) that has no bearing on order/shipping risk
prediction -- discarded here rather than tracked.

Usage:
    python scripts/download_dataco.py --output data/raw/dataco
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
import zipfile
from datetime import date

MENDELEY_ZIP_URL = "https://data.mendeley.com/public-api/zip/8gx2fvg2k6/download/5"
DOI = "10.17632/8gx2fvg2k6.5"


def _download(url: str, dest_path: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())


def download_dataco(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "dataco.zip")
    print(f"Downloading DataCo bundle -> {zip_path}")
    _download(MENDELEY_ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("DataCoSupplyChainDataset.csv", output_dir)
        zf.extract("DescriptionDataCoSupplyChain.csv", output_dir)
    os.remove(zip_path)

    provenance = {
        "source_name": "DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS",
        "source_url": "https://data.mendeley.com/datasets/8gx2fvg2k6/5",
        "download_url": MENDELEY_ZIP_URL,
        "doi": DOI,
        "citation": "Constante, Fabian; Silva, Fernando; Pereira, Antonio (2019), \"DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS\", Mendeley Data, V5, doi: 10.17632/8gx2fvg2k6.5",
        "license": "CC-BY-4.0",
        "data_type": "order-item level e-commerce/logistics transactions, 2015-2019",
        "observed_or_synthetic": "observed",
        "rows_columns": "180519 rows x 53 columns (DataCoSupplyChainDataset.csv)",
        "fields_used": "see notebooks/dataco_exploration.ipynb for the feature/leakage audit -- no fixed list here since that audit is the point",
        "fields_not_used": "tokenized_access_logs.csv (91MB clickstream, no risk-modeling relevance) -- downloaded by Mendeley's bundle but discarded, not extracted",
        "known_data_quality_issues": "'Customer Password' column contains plaintext-looking values and 'Customer Email'/'Customer Fname'/'Customer Lname'/'Customer Street' are PII-shaped -- all excluded as features, not just because they're non-predictive but because they should never be modeled on even for a public toy dataset",
        "transformations": "none at download time",
        "access_date": date.today().isoformat(),
    }
    with open(os.path.join(output_dir, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"DataCo dataset extracted and provenance written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/dataco")
    args = parser.parse_args()
    download_dataco(args.output)


if __name__ == "__main__":
    main()
