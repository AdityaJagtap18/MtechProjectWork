#!/usr/bin/env python3
"""Validate an exported dataset directory (plan §42 Phase 7 deliverable).

Runs four checks and writes each to its own report file under
`<input>/validation/` (plan §24's tree names `statistical_report.json` and
`topology_report.json` explicitly; `constraints_report.json` and
`event_validation_report.json` are added here to give plan §35's other two
validation categories — operational and event — an equally well-defined
home instead of forcing them into one of the named files):

  - constraints_report.json   -- §35 operational validation, §36 sanity tests
  - topology_report.json      -- §35 structural validation
  - statistical_report.json   -- §35 statistical validation (§12's template)
  - event_validation_report.json -- §35 event validation

No plots/ directory is generated here — see notebooks/walkthrough.ipynb for
visual inspection instead (matplotlib is a notebook-only dependency, not a
core one).

Usage:
    python scripts/validate_dataset.py --input data/generated/scm_v1_black_swan_seed42
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from scm_dataset.export.csv import load_events, load_graph, load_operations
from scm_dataset.validation.constraints import run_constraint_checks
from scm_dataset.validation.report import build_event_validation_report
from scm_dataset.validation.statistical import build_statistical_report
from scm_dataset.validation.topology import build_topology_report


def _collect_violations(constraints_report: dict) -> list[str]:
    violations = []
    for check_name, result in constraints_report.items():
        if isinstance(result, list) and result:
            violations.append(f"{check_name}: {len(result)} violation(s)")
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Dataset directory to validate (must contain graph/)")
    parser.add_argument("--region-risk-table", default="data/processed/region_risk_calibration.csv")
    parser.add_argument("--nist-stats", default="data/processed/nist_structural_stats.json")
    args = parser.parse_args()

    graph = load_graph(args.input)
    operations = load_operations(args.input)
    events = load_events(args.input)

    constraints_report = run_constraint_checks(graph, operations, events)
    topology_report = build_topology_report(graph, nist_stats_path=args.nist_stats)
    statistical_report = build_statistical_report(args.region_risk_table)
    event_validation_report = build_event_validation_report(graph)

    validation_dir = os.path.join(args.input, "validation")
    os.makedirs(validation_dir, exist_ok=True)
    reports = {
        "constraints_report.json": constraints_report,
        "topology_report.json": topology_report,
        "statistical_report.json": statistical_report,
        "event_validation_report.json": event_validation_report,
    }
    for filename, report in reports.items():
        with open(os.path.join(validation_dir, filename), "w") as f:
            json.dump(report, f, indent=2, default=str)

    violations = _collect_violations(constraints_report)
    violations += [f"geographic_isolation: {len(event_validation_report['geographic_isolation']['violations'])} violation(s)"
                   for v in [event_validation_report["geographic_isolation"]["violations"]] if v]
    violations += [f"single_source_concentration: {len(event_validation_report['single_source_concentration']['violations'])} violation(s)"
                   for v in [event_validation_report["single_source_concentration"]["violations"]] if v]
    if not event_validation_report["severity_monotonicity"]["severity_5_exceeds_severity_1"]:
        violations.append("severity_monotonicity: severity 5 did not exceed severity 1 in expectation")

    print(f"Validation reports written to {validation_dir}/")
    if violations:
        print(f"FAILED with {len(violations)} issue(s):", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
