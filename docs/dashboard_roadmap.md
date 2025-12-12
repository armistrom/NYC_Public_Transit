# Interactive Dashboard Roadmap

## Goals
- Provide an intuitive interface to explore trip-arrival dynamics across Taxi, Citi Bike, and Subway systems.
- Allow students to connect probability concepts (Poisson tests, LLN convergence, concentration bounds) with real data.
- Support lightweight deployment (Streamlit Cloud/Heroku/Render) for class demo.

## User Journeys
1. **Compare modes** – choose transport mode and date range, see arrivals per bucket, overlaid rate estimates, and burstiness metrics.
2. **Stress period deep-dive** – highlight rush hour vs off-peak, inspect when empirical arrivals violate Chernoff/Hoeffding bounds.
3. **Spatial slice** – pick borough/zone/station to display localized counts and map overlays.

## Architecture
```
data/
  processed/{mode}.parquet   # pre-aggregated counts + metadata
src/
  data.py                    # loading, caching, filters
  stats.py                   # Poisson tests, bounds, MGFs
  dashboard/app.py           # Streamlit UI (or Dash)

streamlit_app.py             # entry point for deployment
```

### Front-end Stack
- **Streamlit** for rapid development (widgets, caching, Plotly integration).
- **Plotly Express** for interactive time-series and histogram visuals.
- **Kepler.gl/pydeck** (optional) for geospatial heatmaps if time allows.

### Back-end/Data Prep
- Nightly scripts convert raw trips into bucketed counts per mode (`scripts/prep_{mode}.py`).
- Store derived metrics (mean λ, variance, dispersion, tail probabilities) to avoid recomputation on load.
- Provide `@st.cache_data` wrappers to keep the dashboard responsive.

## Feature Breakdown
| Feature | Description | Dependencies |
|---------|-------------|--------------|
| Dataset selector | Dropdown for Taxi/Citi Bike/Subway, dynamic descriptions | `data.load_dataset` |
| Time filter | Date range slider + bucket-size radio (5/15/30/60 min) | pandas resampling |
| LLN view | Plot `N(t)/t` vs time, show stabilized λ estimate and confidence band | `stats.cumulative_rate` |
| Poisson diagnostics | Display dispersion index, zero-prob comparison, optional chi-square test p-value | `stats.poisson_test` |
| Concentration bounds | Compute Chernoff/Hoeffding tails for selected bucket, overlay actual counts | `stats.bounds` |
| Burstiness comparison | Panel comparing MGFs / Fano factor across modes | aggregated dataset |
| Export | Button to download filtered counts as CSV for report | Streamlit `st.download_button` |

## Implementation Sprints
1. **MVP UI** – dataset/time selectors, arrivals + LLN plots, summary stats.
2. **Stats Module** – add Poisson tests, bounds, MGFs; integrate into UI.
3. **Spatial Enhancements** – map view for taxi zones or stations.
4. **Narrative Layer** – tooltips explaining what each statistical concept means.
5. **Deployment & Docs** – README instructions, environment.yml, demo video.

## Next Steps
1. Finalize processed datasets for all three transport modes (consistent schema with `event_time`, `mode`, `zone`).
2. Implement `src/stats.py` helpers mirroring the metrics from `arrival_diagnostics`.
3. Scaffold `streamlit_app.py` with placeholder data to verify layout.
4. Incrementally plug analytics + visualizations, validating each against notebooks.
