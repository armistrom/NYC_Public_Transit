# Data Prep Notes

## Taxi Zone Centroids (lon/lat)
Yellow Taxi trips reference pickup/dropoff `LocationID`s rather than raw GPS coordinates. To translate each zone to representative lon/lat values for KDE plots or map joins:

1. **Download shapefile**  
   Grab the official TLC Taxi Zone shapefile (`taxi_zones.zip`) from https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip and unzip it into `data/raw/taxi_zones_shp/`. The folder should contain `taxi_zones.shp`, `.dbf`, `.prj`, etc.

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

This workflow keeps geo calculations reproducible inside the repo and answers the “where do lon/lat come from?” question for the course project.
