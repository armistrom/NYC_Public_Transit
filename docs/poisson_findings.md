# Poisson Diagnostics Findings

## Current Behavior
- Top taxi zones (JFK, Midtown, Upper East Side, etc.) show dispersion indexes between ~16 and 76.
- Chi-square p-values for homogeneous Poisson are essentially 0, so the null hypothesis is rejected.
- Even quieter zones have variance well above mean; the counts are extremely over-dispersed due to rush-hour bursts and overnight troughs.

## Implications
1. **Homogeneous Poisson is not adequate**: a single rate λ per zone cannot capture the variability, so Poisson-based confidence bounds or LLN diagnostics must be conditioned on time-of-day.
2. **Need non-homogeneous models**: λ(t) must vary with time (e.g., hourly profile). This supports a non-homogeneous Poisson process (NHPP) or piecewise Poisson models.
3. **Rush vs Quiet Slices**: When we condition on weekday quiet hours (1–3am), dispersion indexes drop but still exceed 1; during rush hours they spike dramatically. This motivates forecasting λ(t) per hour and comparing the cumulative counts `N(t)` to NHPP expectations.

## Next Steps
- Build an NHPP utility that estimates λ_zone(hour) from historical data, simulates arrival paths, and computes variance reduction relative to homogeneous Poisson.
- Compare empirical `N(t)` with NHPP `E[N(t)]` to verify that residuals shrink.
- For slices that remain over-dispersed, consider Negative Binomial or Poisson-Gamma mixtures.
- Integrate λ(t) profiles and goodness-of-fit plots into the dashboard so users can explore Poisson vs NHPP behavior interactively.

## Manhattan Rush/Weekend Diagnostic Script
Run `scripts/analyze_manhattan_poisson.py` to focus on Manhattan zones split into four cohorts (weekday/weekend × rush/off-peak). By default it reports per-zone diagnostics and plots the zones whose dispersion is closest to/most different from 1; add `--aggregate-manhattan` if you want borough-wide totals instead. Example:
```bash
python scripts/analyze_manhattan_poisson.py \
  --trips data/raw/yellow_tripdata_2024-01.parquet \
  --zones data/raw/taxi_zone_lookup.csv \
  --freq 60min \
  --rush-hours "7-8,8-9,17-18,18-19" \
  --hist-best 2 \
  --hist-worst 2 \
  --output outputs/manhattan_poisson
```
The script writes JSON summaries (`dispersion_index`, chi-square p-values, bucket counts) per cohort (and per zone when not aggregating). In zone mode each histogram (HTML + PDF) overlays both the Poisson curve and a Negative-Binomial (Poisson–Gamma) fit using moment estimates (`k = mean^2/(var-mean)`, `p = k/(k+mean)`). Additional diagnostics include NB chi-square tests/pass rates plus `nb_mean`/`nb_var` so you can directly compare empirical stats to the fitted model.

## Tail-Risk & Bounds
`scripts/analyze_tail_bounds.py` evaluates how often arrivals exceed a threshold (absolute or multiple of the mean) and compares:

- Empirical exceedance
- Poisson tail (`1 - F(k-1; λ)`)
- NB tail (`1 - F(k-1; r,p)`)
- Markov, Cantelli (one-sided Chebyshev), Chernoff (Poisson & NB via MGF search), and Hoeffding-style bounds (using the observed maximum as a proxy for support)

Example:
```bash
python scripts/analyze_tail_bounds.py \
  --trips data/raw/yellow_tripdata_2024-01.parquet \
  --zones data/raw/taxi_zone_lookup.csv \
  --freq 15min \
  --rush-hours "7-10,16-19" \
  --threshold-multiplier 2.0 \
  --top-zones 5 \
  --output outputs/tail_bounds
```
The script saves `tail_bounds.csv` (one row per zone/cohort) and Plotly charts highlighting the zones with the highest empirical tail probability. Use these results to showcase how conservative Markov/Chebyshev bounds are compared to NB-based predictions, and how Chernoff bounds tighten once over-dispersion is accounted for.
