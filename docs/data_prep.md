# Data Prep Notes

## Taxi Zone Centroids (lon/lat)
Yellow Taxi trips reference pickup/dropoff `LocationID`s rather than raw GPS coordinates. To translate each zone to representative lon/lat values for KDE plots or map joins:

1. **Download shapefile**  
   Grab the official TLC Taxi Zone shapefile (`taxi_zones.zip`) from <https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip> and unzip it into `data/raw/taxi_zones_shp/`. The folder should contain `taxi_zones.shp`, `.dbf`, `.prj`, etc.

2. **Generate centroid lookup**  
   Run the helper script, which reprojects to WGS84 (EPSG:4326) before emitting centroids and appending borough/zone metadata from the TLC lookup CSV:
   ```bash
   python scripts/build_taxi_zone_centroids.py \
     --shapefile data/raw/taxi_zones_shp/taxi_zones.shp \
     --lookup data/raw/taxi_zone_lookup.csv \
     --output data/raw/taxi_zone_centroids.csv
   ```

3. **Use in analyses**  
   The resulting `taxi_zone_centroids.csv` (columns: `LocationID, lon, lat, Borough, Zone, service_zone`) can be merged with trip tables on `PULocationID`/`DOLocationID` to add approximate pickup/dropoff coordinates.

## Citi Bike Trip Data
- Monthly trip CSVs are published by Citi Bike’s AWS S3 bucket: <https://s3.amazonaws.com/tripdata/>. Each file is named `<YYYYMM>-citibike-tripdata.csv.zip`.
- Example (Jan 2024): download `https://s3.amazonaws.com/tripdata/202401-citibike-tripdata.csv.zip` into `data/raw/202401-citibike-tripdata.zip`, then unzip to `data/raw/citibike/`.
- Station/scooter metadata lives under the same portal; we saved `data/raw/citibike_system_data.html` and `data/raw/citibike_bucket_list.xml` as references.

## MTA Subway Turnstile Data
- Weekly turnstile counts are linked from <https://www.mta.info/developers/turnstile.html> (CSV archives back to 2010).
- Download desired weeks into `data/raw/subway/` and use the HTML snapshot (`mta_turnstile.html`) as documentation.

This workflow keeps data provenance explicit and mirrors the sources cited in the report.
