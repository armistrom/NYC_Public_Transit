#!/usr/bin/env python3
"""
Compute Poisson diagnostics for taxi pickups and visualize empirical vs Poisson fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import re
import pandas as pd
import plotly.express as px
from scipy.stats import poisson as sp_poisson

import sys


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        src = candidate / "src"
        if src.exists():
            if str(src) not in sys.path:
                sys.path.append(str(src))
            return
    raise FileNotFoundError("Could not locate src/ directory to import project modules.")


_ensure_src_on_path()

from modeling.poisson_zone import (  # type: ignore  # noqa: E402
    load_taxi_pickups,
    attach_zone_metadata,
    bucket_counts_by_group,
    poisson_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poisson diagnostics for taxi arrivals.")
    parser.add_argument("--trips", type=Path, required=True, help="Yellow taxi parquet file.")
    parser.add_argument("--zones", type=Path, required=True, help="TLC zone lookup CSV.")
    parser.add_argument("--group", default="Borough", help="Grouping column(s), comma-separated.")
    parser.add_argument("--freq", default="15min", help="Bucket size (e.g., 5min, 30min).")
    parser.add_argument("--top-k", type=int, default=5, help="Number of groups to analyze.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for testing.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="P-value threshold for accepting Poisson fit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/poisson"),
        help="Directory for summary JSON and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    trips = load_taxi_pickups(args.trips, max_rows=args.max_rows)
    trips = attach_zone_metadata(trips, args.zones)
    group_cols = [col.strip() for col in args.group.split(",") if col.strip()]
    trips = trips.dropna(subset=group_cols).assign(
        event_time=lambda d: d["event_time"].dt.tz_convert(None)
    )

    counts = bucket_counts_by_group(trips, freq=args.freq, group_cols=group_cols)
    totals = counts.sum().sort_values(ascending=False)
    top_groups: List = list(totals.head(args.top_k).index)
    summaries = {str(group): poisson_summary(counts[group]) for group in top_groups}

    summary_path = args.output / f"poisson_summary_{args.group.replace(',', '_')}.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"Wrote summary stats to {summary_path}")

    total_trips = counts[top_groups].sum().sum()
    good_groups = [g for g in top_groups if summaries[str(g)]["chi2_pvalue"] >= args.alpha]
    good_trips = counts[good_groups].sum().sum() if good_groups else 0
    fraction = good_trips / total_trips if total_trips else 0
    print(f"Fraction of trips in top groups with p >= {args.alpha}: {fraction:.4f}")

    for group in top_groups:
        series = counts[group]
        lam = series.mean()
        max_count = int(series.quantile(0.99)) + 10
        grid = np.arange(0, max(series.max(), max_count) + 1)
        observed = series.value_counts().reindex(grid, fill_value=0).values
        expected = sp_poisson.pmf(grid, mu=lam) * len(series)

        df_plot = pd.DataFrame(
            {"count": grid, "observed": observed, "expected": expected}
        )
        fig = px.bar(
            df_plot,
            x="count",
            y="observed",
            title=f"Observed vs Poisson – {group}",
        )
        fig.add_scatter(
            x=df_plot["count"], y=df_plot["expected"], mode="lines", name="Poisson expectation"
        )
        fig.update_layout(xaxis_title="Arrivals per bucket", yaxis_title="Frequency")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(group))
        fig_path = args.output / f"hist_{safe_name}.html"
        fig.write_html(fig_path)
        print(f"Wrote plot to {fig_path}")


if __name__ == "__main__":
    main()
