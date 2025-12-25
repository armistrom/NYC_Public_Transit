# Stochastic Modelling of Urban Mobility in New York City (Dec 2025)
**End-to-end trip time modeling (wait + travel) using 35M+ NYC Yellow Taxi + Citi Bike trips**

This project builds an interpretable **parametric stochastic model** to estimate **end-to-end trip time** for urban mobility in NYC by combining:

- **Lognormal travel-time regression** (travel duration as a function of trip distance + temporal context)
- **Exponential wait-time estimation** from **inter-arrival gaps** across **space–time grids**

The model enables direct comparison between **NYC Yellow Taxi** and **Citi Bike** for typical urban trips.

---

## Project highlights (resume-focused)
- Developed a **parametric end-to-end trip time model** (wait + travel) using **35M+ trips** from **NYC Yellow Taxi** and **Citi Bike**.
- Combined **lognormal travel-time regression** (**~0.62 R² on held-out data**) with **exponential inter-arrival-based wait-time estimates** computed over **space–time grids**.
- Performed extensive **exploratory analysis** to identify the best **grouping strategy** (distance bands vs continuous distance, time-of-day, weekday/weekend, spatial granularity), improving **distribution fit** and **model stability** for both travel-time and wait-time components.

---

## Problem statement
For a given origin, destination, and departure context (time-of-day / weekday), estimate:

**Total trip time = Wait time + Travel time**

…and compare results for **Taxi vs Bike** under the same conditions.

---

## Approach

### 1) Travel-time model (Lognormal regression)
Travel times are positive and right-skewed, so we model travel time using a lognormal form:

- Fit a regression in log-space:
  - log(T_travel) = f(distance, time-of-day, weekday/weekend, …) + noise
- Predict travel-time in minutes by converting back from log-space.

**Performance:** ~0.62 R² on held-out data (reported for both modes).

---

### 2) Wait-time model (Exponential inter-arrival estimation)
Direct rider wait is not always observed, so we estimate wait time from **inter-arrival gaps** (time between consecutive pickups/checkouts).

Assumption:
- Arrivals within a given location/time bucket are approximately Poisson → inter-arrival gaps are implied by an exponential rate.

Procedure:
- Build **space–time grids** (spatial cell × time bucket × weekday/weekend, etc.)
- Estimate arrival rate **λ** per grid cell from inter-arrival gaps
- Convert λ to expected wait time

**GitHub-safe formula (renders everywhere):**
- Expected wait time: **E[T_wait] = 1 / λ**

---

## Exploratory analysis: grouping strategy + stability
A core part of the work was systematically testing cohort definitions to improve fit and robustness, including:

- **Distance**: continuous vs distance bands
- **Time**: time-of-day buckets (rush/off-peak), weekday vs weekend
- **Space**: different spatial granularities for zones/stations/cells
- **Sparsity handling**: smoothing / fallback aggregation for low-volume cohorts

This improved:
- distribution fit (empirical vs fitted alignment)
- parameter stability across cohorts and time windows

---

## Data
- **35M+ trips total**, spanning:
  - **NYC Yellow Taxi** trip record data
  - **NYC Citi Bike** trip history

Standard cleaning/filters were applied to remove invalid records and ensure consistent comparisons across modes.

---

## Outputs
- End-to-end trip time estimates (**wait + travel**) for taxi vs bike
- Mode-specific travel-time regression models (lognormal)
- Space–time arrival rate tables (λ) and expected wait-time estimates
- Diagnostics (fit plots / cohort distribution checks)

---

## How to run (template)
> Update paths/commands to match your repo.

```bash
pip install -r requirements.txt
streamlit run app/app.py
