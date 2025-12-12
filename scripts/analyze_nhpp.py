#!/usr/bin/env python3
"""Standalone NHPP comparison for a specific borough/zone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import plotly.graph_objects as go
import pandas as pd


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        src = candidate / "src"
        if src.exists():
            if str(src) not in sys.path:
                sys.path.append(str(src))
            return
    raise FileNotFoundError("Could not locate src/ directory for imports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NHPP comparison for taxi arrivals.")
    parser.add_argument("--trips", type=Path, required=True, help="Yellow taxi parquet file")
    parser.add_argument("--zones", type=Path, required=True, help="TLC zone lookup")
    parser.add_argument("--group", default="Borough", help="Grouping column (Zone or Borough)")
    parser.add_argument("--target", required=True, help="Name of the group (e.g., Manhattan)")
    parser.add_argument("--freq", default="15min", help="Bucket size")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap")
    parser.add_argument("--output", type=Path, default=Path("outputs/nhpp"), help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_src_on_path()

    from modeling.poisson_zone import (
        load_taxi_pickups,
        attach_zone_metadata,
        bucket_counts_by_group,
    )
    from modeling.nhpp import build_nhpp, cumulative_counts

    args.output.mkdir(parents=True, exist_ok=True)

    trips = load_taxi_pickups(args.trips, max_rows=args.max_rows)
    trips = attach_zone_metadata(trips, args.zones)
    trips = trips.dropna(subset=[args.group]).assign(
        event_time=lambda d: d["event_time"].dt.tz_convert(None)
    )

    counts = bucket_counts_by_group(trips, freq=args.freq, group_cols=args.group)
    if args.target not in counts.columns:
        raise ValueError(f"Target {args.target} not found in counts columns")
    series = counts[args.target]

    nhpp = build_nhpp(series, freq=args.freq)
    empirical = cumulative_counts(series)
    expected = nhpp.expected_counts(series.index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=empirical.index, y=empirical.values, name="Empirical N(t)"))
    fig.add_trace(go.Scatter(x=expected.index, y=expected.values, name="NHPP E[N(t)]"))
    fig.update_layout(
        title=f"NHPP vs Empirical – {args.target}",
        xaxis_title="Time",
        yaxis_title="Cumulative trips",
    )
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", args.target)
    fig_path = args.output / f"nhpp_{safe}.html"
    fig.write_html(fig_path)
    print(f"Saved NHPP comparison plot to {fig_path}")

    rates_path = args.output / f"nhpp_rates_{safe}.json"
    rates_path.write_text(json.dumps(nhpp.rates, indent=2))
    print(f"Saved hourly rate estimates to {rates_path}")


if __name__ == "__main__":
    main()
