# Stochastic Modelling of Urban Mobility in New York City (Dec 2025)
**End-to-end trip time modeling (wait + travel) using 35M+ NYC Yellow Taxi + Citi Bike trips**

This project builds a **parametric stochastic model** to estimate **end-to-end trip time** for urban mobility in NYC by combining:

- **Lognormal travel-time regression** (predicts trip duration given distance + temporal context)
- **Exponential wait-time estimation** from **inter-arrival gaps** across **space–time grids**

The goal is a clean, interpretable model that supports direct comparison between **NYC Yellow Taxi** and **Citi Bike** for typical intra-city trips.

---

## Highlights (what I built)

### Parametric end-to-end trip time model
For each mode (Taxi / Bike), total time is modeled as:

\[
T_{total} = T_{wait} + T_{travel}
\]

- **Travel time** is modeled using **lognormal regression**, achieving **~0.62 R² on held-out data**
- **Wait time** is estimated via **exponential inter-arrival modeling**, computing arrival rates on **space–time grids** and converting them into expected waits

This produces a single estimate that’s directly usable for decision-making: *“How long will this trip take if I choose taxi vs bike right now?”*

---

## Modeling approach

### 1) Travel-time model: lognormal regression
Trip times are right-skewed and heavy-tailed, so travel time is modeled as lognormal:

\[
\log(T_{travel}) = f(\text{distance}, \text{time-of-day}, \text{weekday/weekend}, \ldots) + \epsilon
\]

Key points:
- Modeled separately for **Taxi** and **Bike**
- Built to generalize across trip lengths and avoid brittle binning
- Achieved **≈ 0.62 R² on held-out data**

---

### 2) Wait-time model: exponential inter-arrival estimation
Rider wait isn’t always recorded directly, so wait time is estimated from **inter-arrival gaps** (time between consecutive pickups / bike checkouts).

Assumption:
- Arrivals in a given region/time bucket are approximately **Poisson**, implying exponential inter-arrival gaps.

For each **space–time cell**, estimate arrival rate \( \lambda \) and compute expected wait:

\[
\mathbb{E}[T_{wait}] = \frac{1}{\lambda}
\]

This is done across **space–time grids** to capture:
- geographic variation (zones/stations/areas)
- time-of-day dynamics
- weekday vs weekend patterns

---

## Exploratory analysis: grouping strategy + stability
A major part of the project was an **EDA-driven search for the best grouping strategy** to improve fit and stability:

- distance banding vs continuous distance
- time-of-day buckets (rush vs off-peak)
- weekday vs weekend splits
- alternative aggregation granularities for spatial cells

This analysis improved:
- **distribution fit** for both travel-time and wait-time components
- **model stability** (less sensitivity to sparsity/outliers in low-volume segments)

---

## Data
- **35M+ trips total** across two mobility modes:
  - NYC **Yellow Taxi** trip records
  - NYC **Citi Bike** trip history

Both datasets were cleaned and filtered to remove invalid durations/distances and improve comparability between modes.

---

## Outputs
- End-to-end time estimates (**wait + travel**) for taxi vs bike
- Mode-specific travel-time models (lognormal regression)
- Space–time arrival rate tables for wait-time estimation
- Diagnostic plots comparing empirical vs fitted distributions (by cohort)

---

## How to run (typical)
```bash
pip install -r requirements.txt
streamlit run app/app.py
