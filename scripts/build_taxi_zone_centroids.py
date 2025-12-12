#!/usr/bin/env python3
"""Create a lookup table that maps TLC LocationIDs to lon/lat centroids."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build taxi zone centroid lookup CSV.")
    parser.add_argument(
        "--shapefile",
        type=Path,
        default=Path("data/raw/taxi_zones_shp/taxi_zones.shp"),
        help="Path to taxi_zones.shp extracted from taxi_zones.zip.",
    )
    parser.add_argument(
        "--lookup",
        type=Path,
        default=Path("data/raw/taxi_zone_lookup.csv"),
        help="CSV shipped by TLC with Borough/Zone/service_zone columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/taxi_zone_centroids.csv"),
        help="Destination CSV with LocationID + lon/lat.",
    )
    parser.add_argument(
        "--target-crs",
        default="EPSG:4326",
        help="Coordinate system for the exported centroids (default: WGS84 lon/lat).",
    )
    return parser.parse_args()


def build_centroids(shapefile: Path, lookup_csv: Path, target_crs: str) -> pd.DataFrame:
    if not shapefile.exists():
        raise FileNotFoundError(f"Shapefile not found: {shapefile}")
    if not lookup_csv.exists():
        raise FileNotFoundError(f"Lookup CSV not found: {lookup_csv}")

    gdf = gpd.read_file(shapefile)
    if gdf.crs is None:
        raise ValueError("Shapefile has no CRS metadata; cannot reproject.")
    centroid_series = gdf.geometry.centroid.to_crs(target_crs)

    if "LocationID" not in gdf.columns:
        raise ValueError("Shapefile missing LocationID column.")

    centroids = pd.DataFrame(
        {
            "LocationID": gdf["LocationID"].astype(int),
            "lon": centroid_series.geometry.x,
            "lat": centroid_series.geometry.y,
        }
    )
    lookup = pd.read_csv(lookup_csv)
    if "LocationID" not in lookup.columns:
        raise ValueError("Lookup CSV must include LocationID column.")
    merged = centroids.merge(lookup, on="LocationID", how="left")
    merged = merged.sort_values("LocationID").reset_index(drop=True)
    return merged


def main() -> None:
    args = parse_args()
    centroids = build_centroids(args.shapefile, args.lookup, args.target_crs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    centroids.to_csv(args.output, index=False)
    print(f"Wrote {len(centroids)} centroid rows to {args.output}")


if __name__ == "__main__":
    main()
