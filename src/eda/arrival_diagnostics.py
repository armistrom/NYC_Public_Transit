#!/usr/bin/env python3
"""
Quick-start EDA pipeline for NYC mobility arrivals.

Example
-------
python src/eda/arrival_diagnostics.py \
    --input yellow_tripdata_2025-01.parquet \
    --mode taxi --freq 15min --max-rows 200000

Outputs
-------
1. arrivals_{mode}.csv: bucketed arrival counts.
2. summary_{mode}.json: key descriptive statistics.
3. *_plot.png figures illustrating arrivals, LLN behavior, and inter-arrival histograms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def _infer_datetime_column(df: pd.DataFrame) -> str:
    """Return the most likely pickup/start timestamp column."""
    candidates = [
        "tpep_pickup_datetime",
        "pickup_datetime",
        "pickup_time",
        "started_at",
        "starttime",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    if len(datetime_cols) == 0:
        raise ValueError("Could not find a timestamp column. Please rename to tpep_pickup_datetime.")
    return datetime_cols[0]


def load_trips(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load parquet or csv file with optional row cap for quick experiments."""
    if not path.exists():
        raise FileNotFoundError(path)
    read_kwargs: Dict[str, int] = {}
    if max_rows is not None:
        read_kwargs["nrows"] = max_rows

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        if max_rows is not None:
            df = df.head(max_rows)
    else:
        df = pd.read_csv(path, **read_kwargs)
    datetime_col = _infer_datetime_column(df)
    df = df.rename(columns={datetime_col: "event_time"})
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["event_time"]).sort_values("event_time")
    return df


def bucket_counts(df: pd.DataFrame, freq: str = "15min") -> pd.Series:
    """Count trips per `freq` interval."""
    counts = (
        df.set_index("event_time")
        .resample(freq)
        .size()
        .rename("arrivals")
        .tz_localize(None)
    )
    return counts


def inter_arrival_stats(counts: pd.Series) -> Dict[str, float]:
    """Compute metrics useful for Poisson diagnostics."""
    lam = counts.mean()
    variance = counts.var()
    dispersion_index = variance / lam if lam > 0 else np.nan
    zero_prob = (counts == 0).mean()
    poisson_zero = np.exp(-lam)
    return {
        "mean_arrivals_per_bucket": lam,
        "variance_arrivals_per_bucket": variance,
        "dispersion_index": dispersion_index,
        "emp_zero_prob": zero_prob,
        "poisson_zero_prob": poisson_zero,
    }


def lln_rate(counts: pd.Series, freq_minutes: float) -> pd.Series:
    """Cumulative average arrival rate (events per hour) to illustrate LLN."""
    cumulative = counts.cumsum()
    elapsed_hours = (np.arange(1, len(counts) + 1) * freq_minutes) / 60.0
    rate = (cumulative / elapsed_hours).rename("cum_rate_per_hour")
    rate.index = counts.index
    return rate


def save_summary(path: Path, counts: pd.Series, stats: Dict[str, float]) -> None:
    summary = {
        "total_observations": int(counts.sum()),
        "num_buckets": int(len(counts)),
        "freq": counts.index.inferred_freq or "irregular",
        **stats,
    }
    path.write_text(json.dumps(summary, indent=2))


def plot_arrivals(counts: pd.Series, rate: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    counts.plot(ax=ax[0], color="#1f77b4")
    ax[0].set_title("Arrivals per Bucket")
    ax[0].set_ylabel("Arrivals")
    rate.plot(ax=ax[1], color="#ff7f0e")
    ax[1].set_title("Cumulative Average Rate (LLN diagnostic)")
    ax[1].set_ylabel("Trips / hour")
    ax[1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_interarrival_hist(counts: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="hist", bins=30, alpha=0.7, ax=ax, color="#2ca02c")
    ax.set_title("Distribution of Arrivals per Bucket")
    ax.set_xlabel("Arrivals")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run arrival diagnostics on NYC mobility data.")
    parser.add_argument("--input", required=True, type=Path, help="CSV or Parquet file.")
    parser.add_argument("--mode", default="taxi", help="Label for output artifacts.")
    parser.add_argument("--freq", default="15min", help="Bucket size (e.g., 5min, 30min, 1H).")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit.")
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/eda"),
        type=Path,
        help="Directory to save plots and summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_trips(args.input, max_rows=args.max_rows)
    counts = bucket_counts(df, freq=args.freq)
    freq_minutes = pd.to_timedelta(args.freq).total_seconds() / 60.0
    rate = lln_rate(counts, freq_minutes=freq_minutes)
    stats = inter_arrival_stats(counts)

    arrivals_path = args.output_dir / f"arrivals_{args.mode}.csv"
    counts.to_csv(arrivals_path, header=True)
    summary_path = args.output_dir / f"summary_{args.mode}.json"
    save_summary(summary_path, counts, stats)
    plot_arrivals(counts, rate, args.output_dir / f"arrivals_lln_{args.mode}.png")
    plot_interarrival_hist(counts, args.output_dir / f"interarrival_hist_{args.mode}.png")
    print(f"Saved bucket counts to {arrivals_path}")
    print(f"Saved summary stats to {summary_path}")


if __name__ == "__main__":
    main()
