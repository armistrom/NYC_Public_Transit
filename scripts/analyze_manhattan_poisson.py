#!/usr/bin/env python3
"""Evaluate Poisson fit for Manhattan zones split by rush hours and weekend."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly import io as pio
from scipy.stats import chisquare
from scipy.stats import nbinom as sp_nbinom
from scipy.stats import poisson as sp_poisson


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
    poisson_summary,
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
        if not (0 <= start <= 23 and 1 <= end <= 24):
            raise ValueError(f"Hours must be in [0, 24]: '{part}'")
        if end <= start:
            raise ValueError(f"End must be greater than start: '{part}'")
        ranges.append((start, end))
    return ranges


def is_rush_hour(hour: int, ranges: Iterable[Tuple[int, int]]) -> bool:
    return any(start <= hour < end for start, end in ranges)


def cohort_label(is_weekend: bool, is_rush: bool) -> str:
    week = "weekend" if is_weekend else "weekday"
    rush = "rush" if is_rush else "offpeak"
    return f"{week}_{rush}"


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Poisson fit for Manhattan zones (rush vs off-peak, weekend vs weekday)."
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
        default=Path("outputs/manhattan_poisson"),
        help="Destination directory for stats and plots.",
    )
    parser.add_argument(
        "--hist-best",
        type=int,
        default=2,
        help="Number of zones (per cohort) with dispersion closest to 1 to plot.",
    )
    parser.add_argument(
        "--hist-worst",
        type=int,
        default=2,
        help="Number of zones (per cohort) with highest dispersion to plot.",
    )
    parser.add_argument(
        "--aggregate-manhattan",
        action="store_true",
        help="If set, aggregate counts across all Manhattan zones instead of per-zone stats.",
    )
    return parser.parse_args()


def estimate_nb_params(series: pd.Series) -> Tuple[float, float]:
    """Return (r, p) parameters for scipy's nbinom if variance > mean."""
    mean = series.mean()
    var = series.var(ddof=0)
    if var <= mean or mean == 0:
        return np.nan, np.nan
    r = mean ** 2 / (var - mean)
    p = r / (r + mean)
    return r, p


def histogram_arrays(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    max_count = int(max(series.max(), 0))
    pad = int(series.quantile(0.99)) + 5
    grid_end = max(max_count, pad, 5)
    grid = np.arange(0, grid_end + 1)
    observed = series.value_counts().reindex(grid, fill_value=0).values
    return grid, observed


def chi_square_gof(observed: np.ndarray, expected: np.ndarray) -> Tuple[float, float]:
    mask = expected > 0
    if not mask.any():
        return float("nan"), float("nan")
    observed = observed[mask]
    expected = expected[mask]
    mask = expected >= 5
    if mask.sum() < 2:
        return float("nan"), float("nan")
    observed = observed[mask]
    expected = expected[mask]
    total_obs = observed.sum()
    total_exp = expected.sum()
    if total_exp == 0 or total_obs == 0:
        return float("nan"), float("nan")
    expected = expected * (total_obs / total_exp)
    stat, pval = chisquare(f_obs=observed, f_exp=expected)
    return float(stat), float(pval)


def finite_mean(values: Iterable[float]) -> float:
    arr = [v for v in values if np.isfinite(v)]
    return float(np.mean(arr)) if arr else float("nan")


def finite_median(values: Iterable[float]) -> float:
    arr = [v for v in values if np.isfinite(v)]
    return float(np.median(arr)) if arr else float("nan")


def make_histogram(
    series: pd.Series, label: str, output_dir: Path, freq: str, nb_params: Tuple[float, float]
) -> Tuple[Path, Path]:
    lam = series.mean()
    grid = np.arange(0, max(series.max(), int(series.quantile(0.99)) + 5) + 1)
    observed = series.value_counts().reindex(grid, fill_value=0).values
    expected = sp_poisson.pmf(grid, mu=lam) * len(series)
    df = pd.DataFrame({"count": grid, "observed": observed, "poisson": expected})
    fig = go.Figure()
    fig.add_bar(
        x=df["count"],
        y=df["observed"],
        name="Observed",
        marker=dict(color="#4B6BFB"),
        opacity=0.8,
    )
    fig.add_scatter(
        x=df["count"],
        y=df["poisson"],
        mode="lines",
        name="Poisson expectation",
        line=dict(color="#FFA500", width=2),
    )
    r, p = nb_params
    if np.isfinite(r) and np.isfinite(p) and r > 0 and 0 < p < 1:
        nb_expected = sp_nbinom.pmf(grid, n=r, p=p) * len(series)
        fig.add_scatter(
            x=df["count"],
            y=nb_expected,
            mode="lines",
            name="NegBin expectation",
            line=dict(color="#D62728", width=3),
        )
    fig.update_layout(
        title=f"{label} ({freq})",
        xaxis_title="Arrivals per bucket",
        yaxis_title="Frequency",
        template="plotly_white",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = safe_filename(label)
    html_path = output_dir / f"{base_name}.html"
    pdf_path = output_dir / f"{base_name}.pdf"
    fig.write_html(html_path)
    try:
        pio.write_image(fig, pdf_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not write PDF for {label}: {exc}")
    return html_path, pdf_path


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

    histogram_dir = args.output / "histograms"
    cohorts = sorted(trips["cohort"].unique())

    if args.aggregate_manhattan:
        summaries: Dict[str, Dict[str, Union[float, int]]] = {}
        for cohort in cohorts:
            cohort_trips = trips[trips["cohort"] == cohort]
            if cohort_trips.empty:
                continue
            series = (
                cohort_trips.set_index("event_time")
                .resample(args.freq)
                .size()
                .rename("arrivals")
            )
            stats = poisson_summary(series)
            stats["total_trips"] = int(series.sum())
            stats["buckets"] = int(series.count())
            nb_r, nb_p = estimate_nb_params(series)
            stats["nb_r"] = float(nb_r)
            stats["nb_p"] = float(nb_p)
            if np.isfinite(nb_r) and np.isfinite(nb_p) and nb_r > 0 and 0 < nb_p < 1:
                stats["nb_mean"] = float(nb_r * (1 - nb_p) / nb_p)
                stats["nb_var"] = float(nb_r * (1 - nb_p) / (nb_p ** 2))
                grid, observed = histogram_arrays(series)
                nb_expected = sp_nbinom.pmf(grid, n=nb_r, p=nb_p) * len(series)
                nb_chi2_stat, nb_chi2_pvalue = chi_square_gof(observed, nb_expected)
                stats["nb_chi2_stat"] = nb_chi2_stat
                stats["nb_chi2_pvalue"] = nb_chi2_pvalue
            else:
                stats["nb_mean"] = float("nan")
                stats["nb_var"] = float("nan")
                stats["nb_chi2_stat"] = float("nan")
                stats["nb_chi2_pvalue"] = float("nan")
            summaries[cohort] = stats
            if series.sum() > 0:
                make_histogram(series, cohort, histogram_dir, args.freq, (nb_r, nb_p))
        summary_path = args.output / "manhattan_poisson.json"
        summary_path.write_text(json.dumps(summaries, indent=2))
        print(f"Wrote summaries to {summary_path}")
        print(f"Histograms saved to {histogram_dir} (one per cohort)")
    else:
        summaries: Dict[str, Dict[str, Dict[str, Union[float, int]]]] = {}
        for cohort in cohorts:
            cohort_trips = trips[trips["cohort"] == cohort]
            if cohort_trips.empty:
                continue
            counts = bucket_counts_by_group(cohort_trips, freq=args.freq, group_cols="Zone")
            summaries[cohort] = {}
            dispersion_map: Dict[str, float] = {}
            for zone in counts.columns:
                series = counts[zone]
                if series.sum() == 0:
                    continue
                stats = poisson_summary(series)
                nb_r, nb_p = estimate_nb_params(series)
                stats["nb_r"] = float(nb_r)
                stats["nb_p"] = float(nb_p)
                if np.isfinite(nb_r) and np.isfinite(nb_p) and nb_r > 0 and 0 < nb_p < 1:
                    stats["nb_mean"] = float(nb_r * (1 - nb_p) / nb_p)
                    stats["nb_var"] = float(nb_r * (1 - nb_p) / (nb_p ** 2))
                    grid, observed = histogram_arrays(series)
                    nb_expected = sp_nbinom.pmf(grid, n=nb_r, p=nb_p) * len(series)
                    nb_chi2_stat, nb_chi2_pvalue = chi_square_gof(observed, nb_expected)
                    stats["nb_chi2_stat"] = nb_chi2_stat
                    stats["nb_chi2_pvalue"] = nb_chi2_pvalue
                else:
                    stats["nb_mean"] = float("nan")
                    stats["nb_var"] = float("nan")
                    stats["nb_chi2_stat"] = float("nan")
                    stats["nb_chi2_pvalue"] = float("nan")
                stats["total_trips"] = int(series.sum())
                stats["buckets"] = int(series.count())
                summaries[cohort][zone] = stats
                dispersion_map[zone] = float(stats["dispersion_index"])

            def select_zones(count: int, reverse: bool) -> List[str]:
                if count <= 0:
                    return []
                valid = [
                    (zone, val)
                    for zone, val in dispersion_map.items()
                    if np.isfinite(val) and val > 0
                ]
                if not valid:
                    return []
                if reverse:
                    ranked = sorted(valid, key=lambda item: item[1], reverse=True)
                else:
                    ranked = sorted(valid, key=lambda item: abs(item[1] - 1.0))
                return [zone for zone, _ in ranked[:count]]

            zones_to_plot = select_zones(args.hist_best, reverse=False) + select_zones(
                args.hist_worst, reverse=True
            )
            seen = set()
            for zone in zones_to_plot:
                if zone in seen:
                    continue
                seen.add(zone)
                series = counts[zone]
                nb_params = (summaries[cohort][zone]["nb_r"], summaries[cohort][zone]["nb_p"])
                make_histogram(series, f"{zone} – {cohort}", histogram_dir, args.freq, nb_params)
            zone_records = [
                stats
                for zone, stats in summaries[cohort].items()
                if not zone.startswith("__")
            ]
            poisson_ps = [float(stats["chi2_pvalue"]) for stats in zone_records]
            nb_ps = [float(stats["nb_chi2_pvalue"]) for stats in zone_records]
            dispersion_vals = [float(stats["dispersion_index"]) for stats in zone_records]
            poisson_pass = [
                p >= 0.05 for p in poisson_ps if np.isfinite(p)
            ]
            nb_pass = [
                p >= 0.05 for p in nb_ps if np.isfinite(p)
            ]
            summaries[cohort]["__overall__"] = {
                "zone_count": len(zone_records),
                "avg_dispersion": finite_mean(dispersion_vals),
                "median_dispersion": finite_median(dispersion_vals),
                "poisson_pass_rate": float(np.mean(poisson_pass)) if poisson_pass else float("nan"),
                "nb_pass_rate": float(np.mean(nb_pass)) if nb_pass else float("nan"),
            }
        summary_path = args.output / "manhattan_poisson.json"
        summary_path.write_text(json.dumps(summaries, indent=2))
        print(f"Wrote summaries to {summary_path}")
        print(
            f"Histograms saved to {histogram_dir} "
            f"(best {args.hist_best} + worst {args.hist_worst} dispersion zones per cohort)"
        )


if __name__ == "__main__":
    main()
