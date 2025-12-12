# Urban Mobility Stochastic Modeling – Project Plan

## Objective
Model trip-arrival dynamics for NYC transportation systems (Yellow Taxi, Citi Bike, Subway) using probability and stochastic-process tools taught in class. Compare empirical arrival patterns to Poisson-process assumptions, study rate stabilization (LLN), quantify variability via concentration bounds, and highlight “burstiness” differences via MGFs.

## Components
1. **Data ingestion** – gather Yellow Taxi (TLC), Citi Bike, and MTA turnstile feeds; clean timestamps, deduplicate, align to uniform time buckets.
2. **Exploratory analysis** – visualize arrivals per hour/day, weekday-versus-weekend contrasts, seasonal trends. Examine inter-arrival distributions and autocorrelation.
3. **Stochastic modeling** – fit Poisson/Non-homogeneous Poisson processes, estimate rate λ(t), test independence via dispersion/over-dispersion metrics. Evaluate LLN convergence of `N(t)/t`.
4. **Variability bounds** – compute empirical tails of arrivals per bucket, compare to Chernoff/Hoeffding bounds, surface when the bounds are loose/tight.
5. **Burstiness via MGFs** – estimate empirical MGFs, compare variance/ skewness across modes, highlight differences with interactive plots.
6. **Interactive dashboard** – allow users to pick transport mode, timeframe, bucket size, and see arrivals, rate estimates, and bound overlays.

## Proposed Tech Stack
- **Data/EDA**: Python, pandas, polars, numpy, scipy, statsmodels, plotly.
- **Modeling**: custom helpers implementing counting processes, LLN diagnostics, Chernoff/Hoeffding bound calculators.
- **Dashboard**: Streamlit or Plotly Dash (support interactive filters, map overlays via mapbox/folium), served as lightweight web app.

## Workflow
1. Prototype EDA in notebooks/scripts (load taxi sample, compute arrival histograms).
2. Build reusable utilities (`src/data.py`, `src/stats.py`, `src/dashboard/`).
3. Validate models/bounds on historical windows, document findings.
4. Wrap analytics inside dashboard with cached datasets for smooth interaction.
5. Prepare presentation/report summarizing modeling insights and UI demo.

## Milestones
| Milestone | Deliverable | Target |
|-----------|-------------|--------|
| M1 | Clean subsets of taxi, Citi Bike, subway data & schema docs | Week 1 |
| M2 | Core EDA plots + LLN/Poisson diagnostics notebook | Week 2 |
| M3 | Variability bounds + burstiness comparisons | Week 3 |
| M4 | Streamlit/Dash interactive dashboard MVP | Week 4 |
| M5 | Final polish, storytelling, deployment instructions | Week 5 |

## Stretch Ideas
- Incorporate weather or event calendars as covariates.
- Compare neighborhoods using spatial grids (Voronoi/hex bins).
- Simulate queueing (M/M/1) for stations during rush hour using estimated arrival rates.
- Provide anomaly alerts when observed arrivals exceed statistical bounds.

