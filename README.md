# Stochastic Modeling of Urban Mobility in New York City (NYC Taxi vs Citi Bike)

A lightweight, distribution-first mobility explainer that compares **NYC yellow taxi** vs **Citi Bike** for medium-length trips by estimating:

- **Wait time** (how long until a taxi arrives / a bike becomes available nearby)
- **Travel time** (how long the trip takes once moving)

The project fits simple stochastic models directly to public datasets (Jan–Jun 2024) and serves predictions in an interactive **Streamlit** dashboard.

**Live demo:** https://nycpublictransit.streamlit.app/  
**Report:** `ECE225_Project.pdf`  
**Code:** https://github.com/AtharvRN/NYC_Public_Transit

---

## What this project does

New Yorkers often choose between a taxi (faster pickup, higher cost) and Citi Bike (cheaper/greener, sometimes more reliable in traffic). This tool quantifies that tradeoff using probabilistic models:

### 1) Wait time model (Exponential via inter-arrival rates)
Because actual rider waits are not directly observed, we use **inter-arrival gaps** as a proxy (time between consecutive pickups/checkouts). For each location and time bucket, we estimate an arrival rate **λ** and convert it to a mean wait:

> **Estimated wait (minutes) = 60 / λ**

Although Poisson/Negative-Binomial arrival counts are explored for diagnostics, the deployed wait estimator uses the exponential mean implied by the observed gap histogram.

### 2) Travel time model (Lognormal regression / GLM)
Travel minutes are positively skewed with long right tails, so we model travel time with a **lognormal regression**:

\[
\log T = \beta_0 + \beta_1 d + \beta_2 d^2 + \beta_3 I_{rush} + \beta_4 I_{weekend} + \epsilon
\]

At inference time we predict the log mean and convert back to minutes using the lognormal mean correction:

\[
\hat{T} = \exp(\hat{\mu} + 0.5\hat{\sigma}^2)
\]

The app uses one model per mode (taxi vs bike) for smooth distance-based predictions without brittle binning.

---

## Data

Trained on **Jan–Jun 2024** public datasets:

- **NYC TLC Yellow Taxi Trip Records** (pickup/dropoff timestamps, trip distance, TLC zone IDs)
- **Citi Bike Historical Trip Data** (start/end timestamps, station coordinates, rideable type, membership flag)

Key scale (used in deployed models):
- ~17.6M taxi trips (after filtering)
- ~18.3M Citi Bike rides

---

## App experience (Streamlit)

A typical session:
1. **Select origin/destination** on the map (or by address input).
2. The app snaps to the nearest:
   - TLC taxi zone centroid
   - Citi Bike station
3. **View wait estimates** for each mode (with fallbacks when a cohort is sparse).
4. **Compare travel time + total journey time** (wait + travel + walking).
5. Optionally **inspect diagnostics** that explain the fitted distributions.

---

## Assumptions & modeling choices

- **Wait time proxy:** Inter-arrival gaps approximate rider wait (imperfect but measurable).
- **Comparable cohorts:** Trips are filtered to **1–120 minutes** and **≤ 12 km** for comparability; outside this range the app falls back to coarser heuristics.
- **Features:** Only historically available features are used (distance + rush/weekend indicators). Weather, incidents, pricing, and subway headways are not yet integrated.

---

## Repository structure (suggested)

> Adjust names to match your repo if they differ.

