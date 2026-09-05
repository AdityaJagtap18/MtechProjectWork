#!/usr/bin/env python3
"""D0 dataset integrity check (GRAPHSAGE_FINAL_GENERALIZATION_AND_IMPROVEMENT_PLAN.md).

Read-only structural comparison of two benchmark datasets -- trains
nothing, evaluates nothing. Writes dataset_comparison_<a>_<b>.json.

Usage:
    python scripts/compare_datasets.py --dataset-a scm_v1_black_swan_seed43 --dataset-b scm_v1_black_swan_seed44
"""

from __future__ import annotations

import argparse
import json
import sys

from scm_dataset.modeling.data import load_benchmark
from scm_dataset.modeling.dataset_comparison import compare_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-path", default="data/benchmark")
    parser.add_argument("--dataset-a", default="scm_v1_black_swan_seed43")
    parser.add_argument("--dataset-b", default="scm_v1_black_swan_seed44")
    parser.add_argument("--rolling-windows", default="4,8,12")
    parser.add_argument("--feature-mode", default="full")
    parser.add_argument("--output", default=None, help="Defaults to dataset_comparison_<a>_<b>.json in the repo root.")
    args = parser.parse_args()

    benchmark_a = load_benchmark(args.benchmark_path, args.dataset_a)
    benchmark_b = load_benchmark(args.benchmark_path, args.dataset_b)
    windows = [int(w) for w in args.rolling_windows.split(",")]

    result = compare_datasets(benchmark_a, benchmark_b, windows, args.feature_mode)

    output_path = args.output or f"dataset_comparison_{args.dataset_a.replace('scm_v1_black_swan_', '')}_{args.dataset_b.replace('scm_v1_black_swan_', '')}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Wrote {output_path}")
    print(f"node_counts match: {result['node_counts']['match']}")
    print(f"edge_counts match: {result['edge_counts']['match']}")
    print(f"horizon_periods match: {result['horizon_periods']['match']}")
    print(f"feature_columns match: {result['feature_columns']['match']}")
    print(f"label_columns match: {result['label_columns']['match']}")
    print(f"required_files present (both): {result['required_files_present']['all_present']}")
    print(f"\nsupplier_disrupted prevalence: a={result['supplier_disrupted_prevalence']['a']}")
    print(f"                                b={result['supplier_disrupted_prevalence']['b']}")
    print(f"onset counts: a={result['onset_counts']['a']}  b={result['onset_counts']['b']}")
    print(f"\nCOMPATIBLE FOR CROSS-DATASET EVALUATION: {result['compatible_for_cross_dataset_evaluation']}")

    if not result["compatible_for_cross_dataset_evaluation"]:
        print("\nIncompatibility detected -- do NOT proceed with D2 cross-dataset training/evaluation "
              "until this is resolved or explicitly documented as an accepted, understood difference.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
