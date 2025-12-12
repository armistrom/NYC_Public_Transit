"""
Utilities for modeling taxi pickup arrivals as Poisson processes per zone or borough.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


TAXI_BASE_COLUMNS = [
    "tpep_pickup_datetime",
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "PULocationID",
]


def load_taxi_pickups(
    path: Path,
    max_rows: Optional[int] = None,
    columns: Iterable[str] = TAXI_BASE_COLUMNS,
) -> pd.DataFrame:
    """Load taxi trips and prepare event_time + pickup zone."""
    df = pd.read_parquet(path, columns=list(columns))
    if max_rows is not None:
        df = df.head(max_rows)
    df = df.rename(columns={"tpep_pickup_datetime": "event_time"})
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df = df.dropna(subset=["event_time", "PULocationID"])
    return df


def attach_zone_metadata(
    trips: pd.DataFrame,
    lookup_csv: Path,
    zone_col: str = "PULocationID",
) -> pd.DataFrame:
    """Join TLC zone lookup to add Borough/Zone labels."""
    zones = pd.read_csv(lookup_csv)
    zones = zones.rename(columns={"LocationID": zone_col})
    merged = trips.merge(zones[[zone_col, "Borough", "Zone"]], on=zone_col, how="left")
    return merged


def bucket_counts_by_group(
    trips: pd.DataFrame,
    freq: str = "15min",
    group_cols: Sequence[str] | str = "Borough",
) -> pd.DataFrame:
    """Return counts per time bucket per group (wide format)."""
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    grouped = (
        trips.groupby([*group_cols, pd.Grouper(key="event_time", freq=freq)])
        .size()
        .rename("arrivals")
        .reset_index()
    )
    pivot_cols = group_cols[0] if len(group_cols) == 1 else list(group_cols)
    pivot = grouped.pivot_table(
        index="event_time",
        columns=pivot_cols,
        values="arrivals",
        fill_value=0,
        aggfunc="sum",
    )
    if isinstance(pivot.columns, pd.MultiIndex) and len(pivot.columns.names) == 1:
        pivot.columns = pivot.columns.get_level_values(0)
    if pivot.index.tz is not None:
        pivot.index = pivot.index.tz_convert(None)
    return pivot


def poisson_summary(counts: pd.Series) -> dict:
    """Compute mean/variance ratio and chi-square goodness-of-fit."""
    lam = counts.mean()
    variance = counts.var(ddof=0)
    dispersion = variance / lam if lam > 0 else np.nan

    # Bin counts for chi-square test
    value_counts = counts.value_counts()
    xs = value_counts.index.values
    obs = value_counts.values
    poisson_probs = stats.poisson.pmf(xs, mu=lam)
    expected = poisson_probs * len(counts)
    mask = expected > 5  # rule of thumb for chi-square validity
    chi2_stat, chi2_p = np.nan, np.nan
    if mask.any():
        obs_masked = obs[mask]
        exp_masked = expected[mask]
        scale = obs_masked.sum() / exp_masked.sum()
        exp_masked = exp_masked * scale
        res = stats.chisquare(f_obs=obs_masked, f_exp=exp_masked)
        chi2_stat, chi2_p = res.statistic, res.pvalue

    return {
        "mean": lam,
        "variance": variance,
        "dispersion_index": dispersion,
        "chi2_stat": chi2_stat,
        "chi2_pvalue": chi2_p,
    }


def export_zone_model(
    counts: pd.DataFrame,
    out_path: Path,
) -> None:
    """Persist per-zone Poisson summaries as JSON."""
    meta = {}
    for col in counts.columns:
        meta[col] = poisson_summary(counts[col])
    out_path.write_text(json.dumps(meta, indent=2))


__all__ = [
    "load_taxi_pickups",
    "attach_zone_metadata",
    "bucket_counts_by_group",
    "poisson_summary",
    "export_zone_model",
]
