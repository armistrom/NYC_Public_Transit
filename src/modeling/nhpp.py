"""
Utilities to estimate non-homogeneous Poisson rates and compare against observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class NHPPModel:
    rates: Dict[int, float]
    freq_minutes: float

    def expected_counts(self, timestamps: pd.DatetimeIndex) -> pd.Series:
        hours = pd.Series(timestamps.hour, index=timestamps)
        rate_per_bucket = hours.map(self.rates).fillna(0.0) * (self.freq_minutes / 60.0)
        return rate_per_bucket.cumsum()


def estimate_hourly_rates(counts: pd.Series) -> Dict[int, float]:
    df = counts.to_frame("arrivals")
    df["hour"] = df.index.hour
    hourly = df.groupby("hour")["arrivals"].mean()
    return hourly.to_dict()


def cumulative_counts(counts: pd.Series) -> pd.Series:
    return counts.cumsum()


def build_nhpp(counts: pd.Series, freq: str = "15min") -> NHPPModel:
    freq_minutes = pd.to_timedelta(freq).total_seconds() / 60.0
    rates = estimate_hourly_rates(counts)
    return NHPPModel(rates=rates, freq_minutes=freq_minutes)
