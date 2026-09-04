#!/usr/bin/env python3
"""Build the Phase 4 calibration artifacts from downloaded raw data
(`scripts/download_real_data.py` must be run first).

Writes:
  - data/processed/region_risk_calibration.csv: real per-country region
    risk table (WGI + INFORM), consumed by generator/regions.py when
    `real_data.use_real_region_calibration` is enabled (configs/real_calibration.yaml)
  - data/processed/nist_structural_stats.json: NIST field mapping +
    structural statistics (plan §42 Phase 4's "topology comparison")

Usage:
    python scripts/calibrate.py --raw-dir data/raw --output data/processed
"""

from __future__ import annotations

import argparse
import json
import os

from scm_dataset.real_data.calibrator import build_region_risk_table
from scm_dataset.real_data.mapper import compute_structural_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", default="data/processed")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    region_risk = build_region_risk_table(args.raw_dir)
    region_risk_path = os.path.join(args.output, "region_risk_calibration.csv")
    region_risk.to_csv(region_risk_path, index=False)
    print(f"Wrote {len(region_risk)} countries' calibrated region risk -> {region_risk_path}")

    stats = compute_structural_stats(args.raw_dir)
    stats_path = os.path.join(args.output, "nist_structural_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote NIST structural stats -> {stats_path}")
    print(f"Overall multi-source rate observed in NIST data: {stats['overall_multi_source_rate']}")


if __name__ == "__main__":
    main()
