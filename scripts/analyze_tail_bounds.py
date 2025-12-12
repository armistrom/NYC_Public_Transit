#!/usr/bin/env python3
"""Tail-risk diagnostics comparing empirical exceedance vs Poisson/NB bounds."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly import io as pio
from scipy.stats import nbinom as sp_nbinom
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
    raise FileNotFoundError("Could not locate src/ directory for imports.")


_ensure_src_on_path()

from modeling.poisson_zone import (  # type: ignore  # noqa: E402
    attach_zone_metadata,
    bucket_counts_by_group,
    load_taxi_pickups,
)


def parse_hour_ranges(spec: str) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f"Invalid range '{part}', expected start-end")
        start_s, end_s = part.split("-", 1)
        start, end = int(start_s), int(end_s)
        if not (0 <= start <= 23 and 1 <= end <= 24 and end > start):
            raise ValueError(f"Hour range must satisfy 0<=start<end<=24: '{part}'")
        ranges.append((start, end))
    return ranges


def is_rush_hour(hour: int, ranges: Iterable[Tuple[int, int]]) -> bool:
    return any(start <= hour < end for start, end in ranges)


def cohort_label(is_weekend: bool, is_rush: bool) -> str:
    week = "weekend" if is_weekend else "weekday"
    rush = "rush" if is_rush else "offpeak"
    return f"{week}_{rush}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute tail probabilities and compare Markov/Chebyshev/Chernoff/Hoeffding bounds "
            "against empirical and Poisson/NB expectations."
        )
    )
    parser.add_argument("--trips", type=Path, required=True, help="Yellow taxi parquet file.")
    parser.add_argument("--zones", type=Path, required=True, help="TLC taxi_zone_lookup.csv.")
    parser.add_argument("--freq", default="15min", help="Bucket size for arrival counts.")
    parser.add_argument(
        "--rush-hours",
        default="7-10,16-19",
        help="Comma-separated hour ranges (24h) defining rush hour, e.g. '7-10,16-19'.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for testing.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tail_bounds"),
        help="Destination directory for tables and plots.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Absolute threshold for arrivals per bucket. If omitted, use relative multiplier.",
    )
    parser.add_argument(
        "--threshold-multiplier",
        type=float,
        default=2.0,
        help="If --threshold is omitted, use `multiplier * mean` as the tail threshold.",
    )
    parser.add_argument(
        "--min-buckets",
        type=int,
        default=200,
        help="Skip zones with fewer buckets than this value.",
    )
    parser.add_argument(
        "--top-zones",
        type=int,
        default=5,
        help="Number of zones to visualize (highest empirical tail probability).",
    )
    return parser.parse_args()


def estimate_nb_params(series: pd.Series) -> Tuple[float, float]:
    mean = series.mean()
    var = series.var(ddof=0)
    if var <= mean or mean <= 0:
        return np.nan, np.nan
    r = mean ** 2 / (var - mean)
    p = r / (r + mean)
    return r, p


def poisson_chernoff(lam: float, threshold: float) -> float:
    if lam <= 0 or threshold <= lam:
        return 1.0
    delta = threshold / lam - 1.0
    return float(np.exp(-lam * ((1 + delta) * np.log(1 + delta) - delta)))


def nb_chernoff(r: float, p: float, threshold: float) -> float:
    if not (np.isfinite(r) and np.isfinite(p) and r > 0 and 0 < p < 1):
        return float("nan")
    mean = r * (1 - p) / p
    if threshold <= mean:
        return 1.0
    t_max = -np.log(1 - p) - 1e-6
    ts = np.linspace(1e-6, t_max, 400)
    mgf = (p / (1 - (1 - p) * np.exp(ts))) ** r
    bounds = np.exp(-ts * threshold) * mgf
    return float(np.min(bounds))


def hoeffding_bound(mean: float, threshold: float, max_val: float) -> float:
    if threshold <= mean or max_val <= 0:
        return 1.0
    width = max_val
    return float(np.exp(-2 * ((threshold - mean) / width) ** 2))


def evaluate_zone(series: pd.Series, threshold: float) -> Dict[str, float]:
    mean = series.mean()
    var = series.var(ddof=0)
    empirical = float((series >= threshold).mean())
    lam = mean
    poisson_tail = float(sp_poisson.sf(np.floor(threshold) - 1, lam)) if lam > 0 else float("nan")

    nb_r, nb_p = estimate_nb_params(series)
    if np.isfinite(nb_r) and np.isfinite(nb_p) and nb_r > 0 and 0 < nb_p < 1:
        nb_tail = float(sp_nbinom.sf(np.floor(threshold) - 1, n=nb_r, p=nb_p))
    else:
        nb_tail = float("nan")

    markov = 1.0 if threshold <= 0 else min(1.0, mean / threshold)
    cantelli = (
        1.0
        if (threshold <= mean or var <= 0)
        else float(var / (var + (threshold - mean) ** 2))
    )
    poisson_ch = poisson_chernoff(lam, threshold) if lam > 0 else float("nan")
    nb_ch = nb_chernoff(nb_r, nb_p, threshold)
    hoeff = hoeffding_bound(mean, threshold, float(series.max()))

    return {
        "mean": float(mean),
        "variance": float(var),
        "threshold": float(threshold),
        "empirical_tail": empirical,
        "poisson_tail": poisson_tail,
        "nb_tail": nb_tail,
        "markov_bound": markov,
        "cantelli_bound": cantelli,
        "poisson_chernoff": poisson_ch,
        "nb_chernoff": nb_ch,
        "hoeffding_bound": hoeff,
        "nb_r": float(nb_r),
        "nb_p": float(nb_p),
    }


def make_tail_plot(entries: List[Dict[str, float]], output_dir: Path, title: str) -> None:
    df = pd.DataFrame(entries)
    fig = go.Figure()
    for metric, color in [
        ("empirical_tail", "#4B6BFB"),
        ("poisson_tail", "#FFA500"),
        ("nb_tail", "#D62728"),
        ("markov_bound", "#999999"),
        ("cantelli_bound", "#6A3D9A"),
        ("poisson_chernoff", "#2CA02C"),
        ("nb_chernoff", "#E377C2"),
        ("hoeffding_bound", "#8C564B"),
    ]:
        if metric in df:
            fig.add_scatter(
                x=df["zone"],
                y=df[metric],
                mode="lines+markers",
                name=metric,
                line=dict(color=color),
            )
    fig.update_layout(
        title=title,
        xaxis_title="Zone",
        yaxis_title="P(N ≥ threshold)",
        yaxis_type="log",
        template="plotly_white",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base = title.replace(" ", "_")
    html_path = output_dir / f"{base}.html"
    pdf_path = output_dir / f"{base}.pdf"
    fig.write_html(html_path)
    try:
        pio.write_image(fig, pdf_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not export tail plot {title}: {exc}")


def main() -> None:
    args = parse_args()
    rush_ranges = parse_hour_ranges(args.rush_hours)
    args.output.mkdir(parents=True, exist_ok=True)

    trips = load_taxi_pickups(args.trips, max_rows=args.max_rows)
    trips = attach_zone_metadata(trips, args.zones)
    trips = trips.dropna(subset=["Zone", "Borough"])
    trips = trips[trips["Borough"] == "Manhattan"].copy()
    trips["event_time"] = trips["event_time"].dt.tz_convert(None)
    trips["hour"] = trips["event_time"].dt.hour
    trips["is_weekend"] = trips["event_time"].dt.weekday >= 5
    trips["is_rush"] = trips["hour"].apply(lambda h: is_rush_hour(h, rush_ranges))
    trips["cohort"] = trips.apply(
        lambda row: cohort_label(bool(row["is_weekend"]), bool(row["is_rush"])), axis=1
    )

    results: List[Dict[str, float]] = []
    plots: Dict[str, List[Dict[str, float]]] = {}

    for cohort in sorted(trips["cohort"].unique()):
        cohort_trips = trips[trips["cohort"] == cohort]
        if cohort_trips.empty:
            continue
        counts = bucket_counts_by_group(cohort_trips, freq=args.freq, group_cols="Zone")
        for zone in counts.columns:
            series = counts[zone].dropna()
            if len(series) < args.min_buckets:
                continue
            mean = series.mean()
            if args.threshold is not None:
                threshold = args.threshold
            else:
                threshold = args.threshold_multiplier * mean
            stats = evaluate_zone(series, threshold)
            stats.update(
                {
                    "zone": zone,
                    "cohort": cohort,
                    "buckets": int(series.count()),
                    "empirical_max": float(series.max()),
                }
            )
            results.append(stats)
            plots.setdefault(cohort, []).append(
                {"zone": zone, "empirical_tail": stats["empirical_tail"], "poisson_tail": stats["poisson_tail"],
                 "nb_tail": stats["nb_tail"], "markov_bound": stats["markov_bound"],
                 "cantelli_bound": stats["cantelli_bound"], "poisson_chernoff": stats["poisson_chernoff"],
                 "nb_chernoff": stats["nb_chernoff"], "hoeffding_bound": stats["hoeffding_bound"]}
            )

    if not results:
        raise RuntimeError("No zones survived filtering; try lowering --min-buckets.")

    df = pd.DataFrame(results)
    table_path = args.output / "tail_bounds.csv"
    df.to_csv(table_path, index=False)
    print(f"Wrote tail diagnostics to {table_path}")

    for cohort, entries in plots.items():
        entries_sorted = sorted(entries, key=lambda d: d["empirical_tail"], reverse=True)[: args.top_zones]
        if entries_sorted:
            make_tail_plot(entries_sorted, args.output, f"{cohort}_tail_bounds")


if __name__ == "__main__":
    main()
