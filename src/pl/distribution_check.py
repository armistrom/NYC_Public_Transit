from scipy.stats import norm,poisson,nbinom,chi2,weibull_min, kstest
from math import lgamma
import numpy as np
import matplotlib.pyplot as plt


def chi_squared(observed: list, expected: list):
    l = len(observed)
    chi = 0.0
    for i in range(l):
        if expected[i] <= 0:
            # skip bins with zero expected count to avoid division by zero
            continue
        dummy = ((observed[i] - expected[i])**2 / expected[i])
        chi += dummy
    return chi

def pick_quantile(distribution: list):
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    q = [p / 100 for p in percentiles]
    qs = np.quantile(distribution, q)
    return qs
def poisson_pmf(k:int, lam:float):
    log_p = -lam + k * np.log(lam) - lgamma(k + 1)
    return np.exp(log_p)



def poisson_check(emp_dist: np.array,bin_n = 50):
    dist_mean = np.mean(emp_dist)
    dist_var = np.var(emp_dist, ddof=1)
    dispersion = dist_var / dist_mean

    # bin_n = 50
    obs_mean = 0
    obs, bin_edges = np.histogram(emp_dist, bins=bin_n)
    values = bin_edges[:-1]
    observed = obs.tolist()
    obs_mean = np.mean(np.array(observed))
    bin_n -= 2
    while obs_mean < 5 and bin_n > 1:
        obs, bin_edges = np.histogram(emp_dist, bins=bin_n)
        values = bin_edges[:-1]
        observed = obs.tolist()
        obs_mean = np.mean(np.array(observed))
        bin_n -= 2
    print(f'Final Bin values: {bin_n+2}')

    n = emp_dist.shape[0]
    expected = []
    for val in values:
        expected.append(n * poisson_pmf(val, dist_mean))

    DoF = len(observed)-2
    chi_score = chi_squared(observed, expected)
    p_value = chi2.sf(chi_score, DoF)
    emp_quantile = pick_quantile(emp_dist)
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    q = [p / 100 for p in percentiles]
    poisson_quantile = [poisson.ppf(i, dist_mean) for i in q]

    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution comparison plot
    x_range = np.arange(0, int(np.max(emp_dist)) + 1)
    poisson_pmf_vals = poisson.pmf(x_range, dist_mean)

    ax1.hist(emp_dist, bins=30, density=True, alpha=0.6, label='Empirical', color='blue')
    ax1.plot(x_range, poisson_pmf_vals, 'r-', lw=2, label='Poisson')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Probability/Density')
    ax1.set_title('Empirical vs Poisson Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # QQ plot
    ax2.plot(emp_quantile, poisson_quantile, 'o', label='Q-Q points')
    # Add diagonal reference line
    min_val = min(min(emp_quantile), min(poisson_quantile))
    max_val = max(max(emp_quantile), max(poisson_quantile))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect fit')
    ax2.set_xlabel('Empirical Quantiles')
    ax2.set_ylabel('Poisson Quantiles')
    ax2.set_title('Poisson Q-Q Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'chi_score': chi_score,
        'dispersion': dispersion,
        'emp_quantile': emp_quantile,
        'poisson_quantile': poisson_quantile,
        'p_value': p_value
    }
def nb_param(emp_dist:np.array):
    n = emp_dist.shape[0]
    dist_mean = np.mean(emp_dist)
    dist_var = np.var(emp_dist,ddof=1)
    dispersion = dist_var / dist_mean
    if dist_var <= dist_mean:
        raise ValueError('nb does not fit undispersion')
    k = dist_mean**2 / (dist_var - dist_mean)
    n_param = k
    p_param = n_param / (n_param + dist_mean)
    return n_param, p_param

def nb_check(emp_dist: np.array,bin_n = 50):

    n_param, p_param = nb_param(emp_dist)

    dist_mean = np.mean(emp_dist)
    dist_var = np.var(emp_dist, ddof=1)
    dispersion = dist_var / dist_mean

    # bin_n = 50
    obs_mean = 0
    while obs_mean < 5 and bin_n > 1:
        obs, bin_edges = np.histogram(emp_dist, bins=bin_n)
        values = bin_edges[:-1]
        observed = obs.tolist()
        obs_mean = np.mean(np.array(observed))
        bin_n -= 2
    print(f"Final Bin values (NB): {bin_n+2}")

    n = emp_dist.shape[0]
    # Expected counts per bin under fitted NB
    expected = []
    for i in range(len(values)):
        low = values[i]
        high = bin_edges[i+1]
        lo_int = int(np.floor(low))
        hi_int = int(np.floor(high)) - 1

        if hi_int < lo_int:
            expected.append(0.0)
            continue

        cdf_low = nbinom.cdf(lo_int - 1, n_param, p_param) if lo_int > 0 else 0.0
        cdf_high = nbinom.cdf(hi_int, n_param, p_param)
        prob_bin = max(cdf_high - cdf_low, 0.0)

        expected.append(prob_bin * n)

    DoF = len(observed)-3
    chi_score = chi_squared(observed, expected)
    p_value = chi2.sf(chi_score, DoF)
    emp_quantile = pick_quantile(emp_dist.tolist())
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    q = [p / 100 for p in percentiles]
    nb_quantile = [nbinom.ppf(p, n_param, p_param) for p in q]

    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution comparison plot
    x_range = np.arange(0, int(np.max(emp_dist)) + 1)
    nb_pmf = nbinom.pmf(x_range, n_param, p_param)

    ax1.hist(emp_dist, bins=30, density=True, alpha=0.6, label='Empirical', color='blue')
    ax1.plot(x_range, nb_pmf, 'r-', lw=2, label='Negative Binomial')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Probability/Density')
    ax1.set_title('Empirical vs Negative Binomial Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # QQ plot
    ax2.plot(emp_quantile, nb_quantile, 'o', label='Q-Q points')
    # Add diagonal reference line
    min_val = min(min(emp_quantile), min(nb_quantile))
    max_val = max(max(emp_quantile), max(nb_quantile))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect fit')
    ax2.set_xlabel('Empirical Quantiles')
    ax2.set_ylabel('NB Quantiles')
    ax2.set_title('Negative Binomial Q-Q Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        "chi_score": chi_score,
        "dispersion": dispersion,
        "emp_quantile": emp_quantile,
        "nb_quantile": nb_quantile,
        "n_param": n_param,
        "p_param": p_param,
        "p_value": p_value
    }
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, chi2, kstest

def weibull_check(emp_dist: np.array, bin_n: int = 50, floc: float = 0.0):
    emp_dist = np.asarray(emp_dist)

    # Empirical stats
    dist_mean = np.mean(emp_dist)
    dist_var = np.var(emp_dist, ddof=1)
    dispersion = dist_var / dist_mean  # var / mean

    # --- fit Weibull (2-param if floc fixed, 3-param otherwise) ---
    # c = shape, scale = scale parameter
    if floc is not None:
        c, loc, scale = weibull_min.fit(emp_dist, floc=floc)
    else:
        c, loc, scale = weibull_min.fit(emp_dist)

    # Theoretical expectation (mean) of fitted Weibull
    weibull_mean = weibull_min.mean(c, loc=loc, scale=scale)

    # --- adaptive binning (same style as poisson_check) ---
    obs, bin_edges = np.histogram(emp_dist, bins=bin_n)
    observed = obs.tolist()
    obs_mean = np.mean(np.array(observed))
    bin_n -= 2
    while obs_mean < 5 and bin_n > 1:
        obs, bin_edges = np.histogram(emp_dist, bins=bin_n)
        observed = obs.tolist()
        obs_mean = np.mean(np.array(observed))
        bin_n -= 2
    print(f"Final Bin values: {bin_n+2}")
    n = emp_dist.shape[0]

    # --- expected counts per bin using Weibull CDF ---
    cdf_vals = weibull_min.cdf(bin_edges, c, loc=loc, scale=scale)
    expected = []
    for i in range(len(observed)):
        p_bin = cdf_vals[i+1] - cdf_vals[i]
        expected.append(n * p_bin)

    # --- chi-square GOF ---
    DoF = max(len(observed) - 2, 1)  # 2 params (shape, scale)
    chi_score = chi_squared(observed, expected)
    chi_p_value = chi2.sf(chi_score, DoF)

    # --- KS test on original data ---
    ks_stat, ks_p_value = kstest(emp_dist, 'weibull_min', args=(c, loc, scale))

    # --- quantiles for Q-Q plot ---
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    q = [p / 100 for p in percentiles]
    emp_quantile = np.quantile(emp_dist, q)
    weibull_quantile = weibull_min.ppf(q, c, loc=loc, scale=scale)

    # --- plots: histogram + pdf, and Q-Q, side by side ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution comparison: empirical vs Weibull PDF
    x_range = np.linspace(emp_dist.min(), emp_dist.max(), 400)
    weibull_pdf_vals = weibull_min.pdf(x_range, c, loc=loc, scale=scale)

    ax1.hist(emp_dist, bins=30, density=True, alpha=0.6, label='Empirical')
    ax1.plot(x_range, weibull_pdf_vals, 'r-', lw=2, label='Weibull')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Probability/Density')
    ax1.set_title('Empirical vs Weibull Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Q-Q plot
    ax2.plot(emp_quantile, weibull_quantile, 'o', label='Q-Q points')
    min_val = min(emp_quantile.min(), weibull_quantile.min())
    max_val = max(emp_quantile.max(), weibull_quantile.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect fit')
    ax2.set_xlabel('Empirical Quantiles')
    ax2.set_ylabel('Weibull Quantiles')
    ax2.set_title('Weibull Q-Q Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'shape': c,
        'loc': loc,
        'scale': scale,
        'weibull_mean': weibull_mean,   # theoretical expectation of fitted Weibull
        'chi_score': chi_score,
        'chi_p_value': chi_p_value,
        'ks_stat': ks_stat,
        'ks_p_value': ks_p_value,
        'dispersion': dispersion,
        'emp_quantile': emp_quantile,
        'weibull_quantile': weibull_quantile,
    }


