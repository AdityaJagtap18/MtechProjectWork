#!/usr/bin/env python3
"""A5-D5 -- Event distribution comparison
(GRAPHSAGE_PHASE_A5_ROOT_CAUSE_INVESTIGATION.md §9).

Read-only diagnostic. seed43 has only 6 total underlying events and
seed44 has 9 -- no broad statistical claims are drawn from this sample,
per the plan's own explicit caution.

Usage:
    python scripts/analyze_event_distribution.py
"""

from __future__ import annotations

import os

import pandas as pd

from scm_dataset.modeling.data import load_benchmark
from scm_dataset.schema.nodes import NodeType

BENCHMARK_ROOT = "data/benchmark"
SEED43 = "scm_v1_black_swan_seed43"
SEED44 = "scm_v1_black_swan_seed44"
OUT_DIR = "experiments/analysis/event_shift"


def _load_events(dataset_id: str) -> pd.DataFrame:
    events = pd.read_csv(os.path.join(BENCHMARK_ROOT, dataset_id, "events", "events.csv"))
    impact = pd.read_csv(os.path.join(BENCHMARK_ROOT, dataset_id, "events", "event_impact.csv"))
    return events.merge(impact, on="event_id", how="left")


def _affected_regions_for_event(row, supplier_region: dict[str, str]) -> set[str]:
    regions = set()
    if isinstance(row.get("affected_regions"), str):
        regions.update(row["affected_regions"].split(";"))
    if isinstance(row.get("affected_suppliers"), str):
        for sup in row["affected_suppliers"].split(";"):
            if sup in supplier_region:
                regions.add(supplier_region[sup])
    return regions


def _onset_fresh_ongoing_counts(benchmark) -> dict:
    labels = benchmark.labels["supplier"].sort_values(["supplier_id", "time"]).copy()
    labels["prev_disrupted"] = labels.groupby("supplier_id")["supplier_disrupted"].shift(1).fillna(0)
    positive = labels[labels["supplier_disrupted"] == 1]
    n_fresh = int((positive["prev_disrupted"] == 0).sum())
    n_ongoing = int((positive["prev_disrupted"] == 1).sum())
    return {"n_fresh_positive_rows": n_fresh, "n_ongoing_positive_rows": n_ongoing, "fresh_fraction": n_fresh / max(n_fresh + n_ongoing, 1)}


def _summarize(dataset_id: str, benchmark) -> tuple[dict, pd.DataFrame]:
    events = _load_events(dataset_id)
    supplier_region = {s.supplier_id: s.region_id for s in benchmark.graph.nodes_of_type(NodeType.SUPPLIER)}

    events = events.sort_values("start_time").reset_index(drop=True)
    events["end_time"] = events["start_time"] + events["duration"]
    events["n_affected_suppliers"] = events["affected_suppliers"].apply(lambda x: len(x.split(";")) if isinstance(x, str) else 0)
    events["affected_region_set"] = events.apply(lambda r: _affected_regions_for_event(r, supplier_region), axis=1)
    events["n_affected_regions"] = events["affected_region_set"].apply(len)

    overlaps = 0
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            a, b = events.iloc[i], events.iloc[j]
            if a["start_time"] < b["end_time"] and b["start_time"] < a["end_time"]:
                overlaps += 1

    onset_stats = _onset_fresh_ongoing_counts(benchmark)

    summary = {
        "dataset_id": dataset_id,
        "n_events": len(events),
        "event_type_counts": events["event_type"].value_counts().to_dict(),
        "severity_values": sorted(events["severity"].tolist()),
        "duration_values": sorted(events["duration"].tolist()),
        "onset_weeks": sorted(events["start_time"].tolist()),
        "n_affected_suppliers_per_event": events["n_affected_suppliers"].tolist(),
        "n_affected_regions_per_event": events["n_affected_regions"].tolist(),
        "total_unique_affected_regions": len(set().union(*events["affected_region_set"])) if len(events) else 0,
        "simultaneous_event_pairs": overlaps,
        "mean_cascade_severity": float(events["cascade_severity"].mean()) if "cascade_severity" in events else None,
        "mean_recovery_time": float(events["recovery_time"].mean()) if "recovery_time" in events else None,
        **onset_stats,
    }
    return summary, events


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    b43 = load_benchmark(BENCHMARK_ROOT, SEED43)
    b44 = load_benchmark(BENCHMARK_ROOT, SEED44)

    summary43, events43 = _summarize(SEED43, b43)
    summary44, events44 = _summarize(SEED44, b44)

    md = ["# A5-D5 -- Event Distribution Comparison\n"]
    md.append("**seed43 has only 6 total underlying events and seed44 has 9. No broad statistical claims are drawn from this sample -- reported for context alongside A5-D1/D2/D3, not as independent statistical evidence.**\n")

    for name, s, ev in (("seed43", summary43, events43), ("seed44", summary44, events44)):
        md.append(f"## {name}\n")
        md.append(f"- Total events: **{s['n_events']}**")
        md.append(f"- Event types: {s['event_type_counts']}")
        md.append(f"- Severities: {s['severity_values']}")
        md.append(f"- Durations (periods): {s['duration_values']}")
        md.append(f"- Onset weeks (start_time): {s['onset_weeks']}")
        md.append(f"- Affected suppliers per event: {s['n_affected_suppliers_per_event']}")
        md.append(f"- Affected regions per event: {s['n_affected_regions_per_event']}")
        md.append(f"- Total unique regions touched by any event: **{s['total_unique_affected_regions']} / 20**")
        md.append(f"- Simultaneous (overlapping-window) event pairs: {s['simultaneous_event_pairs']}")
        md.append(f"- Mean cascade severity: {s['mean_cascade_severity']}")
        md.append(f"- Mean recovery time: {s['mean_recovery_time']}")
        md.append(f"- Fresh-onset positive rows: {s['n_fresh_positive_rows']}, already-ongoing positive rows: {s['n_ongoing_positive_rows']} (fresh fraction: {s['fresh_fraction']:.4f})")
        md.append("")
        md.append("```\n" + ev[["event_id", "event_type", "start_time", "duration", "severity", "n_affected_suppliers", "n_affected_regions"]].to_string(index=False) + "\n```\n")

    md.append("## Relation to A5-D1's disjoint disrupted-region finding\n")
    md.append(f"seed43's {summary43['n_events']} events touch {summary43['total_unique_affected_regions']} of 20 regions; seed44's {summary44['n_events']} events touch {summary44['total_unique_affected_regions']} of 20 regions.")
    md.append("With so few events relative to 20 regions, and no evidence the generator's region assignment is conditioned on any risk feature (established in the prior improvement-plan phase's reading of the event-generation code), two independent draws landing on non-overlapping region sets is the expected behavior of the generator, not an anomaly requiring a separate explanation.\n")
    md.append("## Limitations\n")
    md.append("- n=6 and n=9 events respectively; every statistic above is a full-population count, not a sample estimate, but the population itself is tiny and any single additional/different event would materially change these numbers.")

    md_path = os.path.join(OUT_DIR, "EVENT_DISTRIBUTION_ANALYSIS.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
