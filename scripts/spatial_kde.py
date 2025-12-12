#!/usr/bin/env python3
"""Estimate spatial pickup intensity using KDE (Databuckets-style)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import plotly.express as px


def locate_src() -> Path:
    base = Path.cwd()
    for candidate in [base, *base.parents]:
        if (candidate / "src").exists():
            if str(candidate / "src") not in sys.path:
                sys.path.append(str(candidate / "src"))
            return candidate
    raise FileNotFoundError("Could not find src/ directory")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spatial KDE for taxi pickups")
    parser.add_argument("--trips", type=Path, required=True, help="Yellow taxi parquet file")
    parser.add_argument("--lookup", type=Path, required=True, help="CSV with LocationID, lon, lat, service_zone")
    parser.add_argument("--hour-start", type=int, default=7, help="Start hour for filter")
    parser.add_argument("--hour-end", type=int, default=9, help="End hour (inclusive)")
    parser.add_argument("--bandwidth", type=float, default=0.01, help="KDE bandwidth")
    parser.add_argument("--output", type=Path, default=Path("outputs/kde.html"))
    parser.add_argument("--max-rows", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    locate_src()
    from modeling.poisson_zone import load_taxi_pickups

    lookup = pd.read_csv(args.lookup)
    if not {"LocationID", "lon", "lat"}.issubset(lookup.columns):
        raise ValueError("Lookup CSV must contain LocationID, lon, lat columns")

    trips = load_taxi_pickups(args.trips, max_rows=args.max_rows)
    trips = trips.merge(lookup[["LocationID", "lon", "lat", "service_zone"]],
                        left_on="PULocationID", right_on="LocationID", how="left")
    trips = trips.dropna(subset=["lon", "lat"])
    trips["hour"] = trips["event_time"].dt.hour
    mask = (trips["hour"] >= args.hour_start) & (trips["hour"] <= args.hour_end)
    subset = trips[mask]

    coords = subset[["lon", "lat"]].to_numpy()
    kde = KernelDensity(bandwidth=args.bandwidth, kernel="gaussian").fit(coords)

    lon_lin = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 200)
    lat_lin = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 200)
    xx, yy = np.meshgrid(lon_lin, lat_lin)
    grid = np.vstack([xx.ravel(), yy.ravel()]).T
    dens = np.exp(kde.score_samples(grid)).reshape(xx.shape)

    fig = px.imshow(dens, origin="lower", x=lon_lin, y=lat_lin, color_continuous_scale="OrRd")
    fig.update_layout(title=f"KDE intensity {args.hour_start}-{args.hour_end}h", xaxis_title="lon", yaxis_title="lat")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output)
    print(f"Saved KDE heatmap to {args.output}")


if __name__ == "__main__":
    main()
