import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation, chisquare, t, genpareto
from collections.abc import Sequence
from typing import Union, Any


def calculate_iqr(data: Union[np.ndarray, Sequence[float]]) -> tuple[float, float, float]:
    """
    Calculate IQR and return quartiles.
    
    Parameters:
    -----------
    data : array-like
        The input data.
        
    Returns:
    --------
    tuple (iqr, q1, q3)
        The interquartile range, 25th percentile, and 75th percentile.
    """
    arr = np.array(data)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    return iqr, q1, q3

def build_robust_baseline(data: Union[np.ndarray, Sequence[float]], name: str = 'metric') -> dict[str, float]:
    """
    Build a robust baseline using median and MAD, which resists the outliers and skewed distributions common in security telemetry.
    
    Parameters:
    -----------
    data : array-like
        The baseline data.
    name : str
        The name of the metric.
        
    Returns:
    --------
    dict
        A dictionary containing robust baseline metrics and thresholds.
    """
    arr = np.array(data)
    
    center = float(np.median(arr))
    spread = float(median_abs_deviation(arr))
    
    # Scale MAD to be comparable to std dev for normal distributions
    # (multiply by 1.4826)
    scaled_spread = spread * 1.4826
    
    warn_upper = center + 2 * scaled_spread
    alert_upper = center + 3 * scaled_spread
    
    print(f'Robust Baseline for {name}:')
    print('=' * 40)
    print(f'Center (median):     {center:.2f}')
    print(f'Spread (MAD):        {spread:.2f}')
    print(f'Scaled spread:       {scaled_spread:.2f}\n')
    print('Suggested thresholds:')
    print(f'  Warning (±2 spreads):  [{max(0.0, center - 2*scaled_spread):.2f}, {warn_upper:.2f}]')
    print(f'  Alert (±3 spreads):    [{max(0.0, center - 3*scaled_spread):.2f}, {alert_upper:.2f}]\n')
    
    return {
        'center': center,
        'spread': spread,
        'scaled_spread': scaled_spread,
        'warn_upper': warn_upper,
        'alert_upper': alert_upper
    }

def calculate_zscore(value: Union[float, np.ndarray], data: Union[np.ndarray, Sequence[float]]) -> Union[float, np.ndarray]:
    """
    Calculate the z-score of a value relative to a dataset.
    
    Parameters:
    -----------
    value : float or array-like
        Value(s) to score.
    data : array-like
        Baseline dataset.
        
    Returns:
    --------
    float or ndarray
        The Z-score.
    """
    arr = np.array(data)
    mean = np.mean(arr)
    std = np.std(arr)
    return (value - mean) / std if std > 0 else 0.0

def modified_zscore(values: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """
    Calculate modified z-scores using median and MAD, a robust scoring method for security anomaly detection that handles heavy-tailed distributions better than standard Z-scores.
    
    Interpretation:
    - |score| > 3.5 is commonly used as the outlier threshold
    - This is equivalent to about 3 standard deviations for normal data
    
    Parameters:
    -----------
    values : array-like
        The input values to score.
        
    Returns:
    --------
    ndarray
        Modified Z-scores for each input value.
    """
    arr = np.array(values, dtype=float)
    
    median = np.median(arr)
    mad = median_abs_deviation(arr)
    
    # Handle edge case where MAD is 0 (all values identical)
    if mad == 0:
        diff = arr - median
        res = np.zeros_like(diff, dtype=float)
        res[diff > 0] = np.inf
        res[diff < 0] = -np.inf
        return res
    
    # Calculate modified z-score
    # 0.6745 is the scaling factor to match standard z-scores for normal data
    m_zscore = 0.6745 * (arr - median) / mad
    
    return m_zscore

def calculate_modified_z(val: float, median: float, mad: float) -> float:
    """
    Calculate individual modified Z-score using precomputed median and MAD.
    
    Parameters:
    -----------
    val : float
        Value to calculate.
    median : float
        Cohort/baseline median.
    mad : float
        Cohort/baseline MAD.
        
    Returns:
    --------
    float
        Modified Z-score.
    """
    if mad == 0:
        if val == median:
            return 0.0
        return float('inf') if val > median else float('-inf')
    return 0.6745 * (val - median) / mad

def detect_outliers_robust(data: Union[np.ndarray, Sequence[float]], threshold: float = 3.5) -> pd.DataFrame:
    """
    Detect outliers using the modified z-score method.
    
    Parameters:
    -----------
    data : array-like
        Your data
    threshold : float
        Modified z-score threshold (default 3.5 is common)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with original values, scores, and outlier flags.
    """
    arr = np.array(data)
    scores = modified_zscore(arr)
    
    results = pd.DataFrame({
        'value': arr,
        'modified_zscore': scores,
        'is_outlier': np.abs(scores) > threshold
    })
    
    return results

def benford_analysis(data: Union[np.ndarray, Sequence[float]]) -> dict[str, Any]:
    """
    Analyze how well a dataset conforms to Benford's Law.
    
    Benford's Law states that in many naturally occurring datasets, 
    the leading significant digit (1-9) follows a specific logarithmic 
    distribution: P(d) = log10(1 + 1/d).
    
    Anomalous or artificial data (e.g., generated by botnets, ransomware, 
    or manual tampering) often deviates from this distribution.
    
    Parameters:
    -----------
    data : array-like
        The numerical values to analyze (e.g., connection byte counts, 
        file sizes, event frequencies).
        
    Returns:
    --------
    dict
        A dictionary containing observed vs expected distributions, 
        the Mean Absolute Error (MAE), and whether it deviates significantly.
    """
    arr = np.array(data)
    # Remove zeros and negative values
    arr = arr[arr > 0]
    
    if len(arr) < 50:
        return {'error': 'Need at least 50 positive values for reliable Benford analysis'}
        
    # Extract leading digits
    digits = []
    for val in arr:
        s = str(abs(val)).lstrip('0.')
        if s:
            digits.append(int(s[0]))
            
    digits = np.array(digits)
    # Keep only digits 1-9
    digits = digits[(digits >= 1) & (digits <= 9)]
    
    if len(digits) < 50:
        return {'error': 'Need at least 50 valid leading digits (1-9)'}
        
    total_count = len(digits)
    
    # Calculate observed frequencies
    observed_counts = np.zeros(9)
    for d in range(1, 10):
        observed_counts[d-1] = np.sum(digits == d)
        
    observed_pct = (observed_counts / total_count) * 100
    
    # Expected Benford frequencies (%)
    expected_pct = np.array([np.log10(1 + 1/d) * 100 for d in range(1, 10)])
    
    # Calculate Mean Absolute Error (MAE)
    mae = float(np.mean(np.abs(observed_pct - expected_pct)))
    
    # Perform Chi-Square goodness-of-fit test
    expected_counts = (expected_pct / 100.0) * total_count
    stat, p_val = chisquare(f_obs=observed_counts, f_exp=expected_counts)
    
    return {
        'observed_distribution': observed_pct.tolist(),
        'expected_distribution': expected_pct.tolist(),
        'mae': mae,
        'chi2_stat': float(stat),
        'p_value': float(p_val),
        'is_anomalous_deviation': bool(p_val < 0.05)
    }

def calculate_ewma(data: Union[np.ndarray, Sequence[float]], alpha: float = 0.1) -> np.ndarray:
    """
    Calculate the Exponentially Weighted Moving Average (EWMA) for a sequence.
    
    Formula:
    y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
    
    Parameters:
    -----------
    data : array-like
        The input time-series data.
    alpha : float
        The smoothing factor, strictly between 0 and 1 (default 0.1).
        
    Returns:
    --------
    ndarray
        The EWMA values.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be strictly between 0 and 1.")
        
    arr = np.array(data, dtype=float)
    if len(arr) == 0:
        return np.array([], dtype=float)
        
    ewma = np.zeros_like(arr)
    ewma[0] = arr[0]
    for i in range(1, len(arr)):
        ewma[i] = alpha * arr[i] + (1.0 - alpha) * ewma[i-1]
    return ewma

def detect_cusum(
    data: Union[np.ndarray, Sequence[float]], 
    threshold: float = 5.0, 
    drift: float = 0.5
) -> dict[str, Any]:
    """
    Detect anomalies using the Cumulative Sum (CUSUM) algorithm.
    
    This is highly effective for detecting 'low-and-slow' persistent shifts
    that do not trigger single-point Z-score alerts.
    
    Formula:
    S_high[t] = max(0, S_high[t-1] + (x[t] - mean - drift * std))
    S_low[t] = max(0, S_low[t-1] - (x[t] - mean + drift * std))
    
    Parameters:
    -----------
    data : array-like
        The input time-series data.
    threshold : float
        The cumulative alert threshold (in standard deviations, default 5.0).
    drift : float
        The slack parameter to ignore small normal variations (default 0.5).
        
    Returns:
    --------
    dict
        A dictionary containing cumulative sums, alert indices, and decision states.
    """
    arr = np.array(data, dtype=float)
    if len(arr) < 2:
        return {'error': 'Need at least 2 data points for CUSUM calculation.'}
        
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        std = 1e-9
        
    z_scores = (arr - mean) / std
    
    s_high = np.zeros_like(z_scores)
    s_low = np.zeros_like(z_scores)
    
    alerts_high = []
    alerts_low = []
    
    for i in range(1, len(z_scores)):
        s_high[i] = max(0.0, s_high[i-1] + z_scores[i] - drift)
        s_low[i] = max(0.0, s_low[i-1] - z_scores[i] - drift)
        
        if s_high[i] > threshold:
            alerts_high.append(i)
        if s_low[i] > threshold:
            alerts_low.append(i)
            
    return {
        's_high': s_high.tolist(),
        's_low': s_low.tolist(),
        'alerts_high': alerts_high,
        'alerts_low': alerts_low,
        'alert_triggered': len(alerts_high) > 0 or len(alerts_low) > 0
    }

def chi_square_shift(observed: dict[str, int], expected: dict[str, float]) -> dict[str, Any]:
    """
    Perform a Chi-Square goodness-of-fit test to detect categorical distribution shifts.
    
    For example, testing if HTTP status codes (200, 302, 404, 500) deviate
    significantly from expected historical baseline distributions.
    
    Parameters:
    -----------
    observed : dict
        Observed counts for each category.
    expected : dict
        Expected probabilities (or percentages summing to 1.0/100) for each category.
        
    Returns:
    --------
    dict
        Chi-Square statistic, p-value, and alert status (anomaly if p_value < 0.05).
    """
    categories = sorted(list(set(observed.keys()).union(set(expected.keys()))))
    
    total_observed = sum(observed.values())
    if total_observed == 0:
        return {'error': 'Observed counts cannot be empty.'}
        
    total_expected = sum(expected.values())
    if total_expected == 0:
        return {'error': 'Expected distribution cannot sum to 0.'}
        
    obs_counts = []
    exp_counts = []
    
    for cat in categories:
        obs_counts.append(observed.get(cat, 0))
        prop = expected.get(cat, 0.0) / total_expected
        exp_counts.append(prop * total_observed)
        
    obs_counts = np.array(obs_counts, dtype=float)
    exp_counts = np.array(exp_counts, dtype=float)
    
    exp_counts = np.where(exp_counts == 0, 1e-9, exp_counts)
    
    stat, p_val = chisquare(f_obs=obs_counts, f_exp=exp_counts)
    
    return {
        'chi2_stat': float(stat),
        'p_value': float(p_val),
        'is_shift_detected': bool(p_val < 0.05),
        'aligned_categories': categories,
        'observed_counts': obs_counts.tolist(),
        'expected_counts': exp_counts.tolist()
    }


def dynamic_time_warping_distance(s1: Union[np.ndarray, Sequence[float]], s2: Union[np.ndarray, Sequence[float]]) -> float:
    """
    Calculate Dynamic Time Warping (DTW) distance between two 1D arrays.
    """
    a1 = np.array(s1, dtype=float)
    a2 = np.array(s2, dtype=float)
    n = len(a1)
    m = len(a2)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a1[i - 1] - a2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


def _generalized_esd(x: np.ndarray, k: int, alpha: float) -> list[int]:
    """
    Internal helper implementing Generalized Extreme Studentized Deviate test.
    """
    n = len(x)
    arr = x.copy().astype(float)
    indices = list(range(n))
    R = []
    lam = []
    removed_indices = []
    
    for i in range(1, k + 1):
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        if std == 0:
            break
        abs_diff = np.abs(arr - mean)
        max_idx = np.argmax(abs_diff)
        R_i = abs_diff[max_idx] / std
        
        df = n - i - 1
        if df <= 0:
            break
        p = 1.0 - alpha / (2.0 * (n - i + 1))
        t_val = t.ppf(p, df)
        num = (n - i) * t_val
        den = np.sqrt((n - i - 1 + t_val**2) * (n - i + 1))
        lam_i = num / den
        
        R.append(R_i)
        lam.append(lam_i)
        
        orig_idx = indices[max_idx]
        removed_indices.append(orig_idx)
        
        arr = np.delete(arr, max_idx)
        indices.pop(max_idx)
        
    num_anomalies = 0
    for i in range(len(R)):
        if R[i] > lam[i]:
            num_anomalies = i + 1
            
    return removed_indices[:num_anomalies]


def detect_s_h_esd(data: Union[np.ndarray, Sequence[float]], period: int = 24, alpha: float = 0.05, max_anomalies: float = 0.1) -> dict:
    """
    Perform Seasonal Hybrid Extreme Studentized Deviate (S-H-ESD) anomaly detection.
    """
    arr = np.array(data, dtype=float)
    n = len(arr)
    if n < period * 2:
        return {'error': 'Data length must be at least twice the period.'}
        
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomp = seasonal_decompose(arr, period=period, model='additive', extrapolate_trend='freq')
    
    resid = decomp.resid
    median = np.median(resid)
    mad = np.median(np.abs(resid - median))
    if mad == 0:
        mad = 1.0
    robust_resid = (resid - median) / mad
    
    k = max(1, int(n * max_anomalies))
    anomaly_indices = _generalized_esd(robust_resid, k, alpha)
    
    anomalies_mask = np.zeros(n, dtype=bool)
    anomalies_mask[anomaly_indices] = True
    
    return {
        'anomaly_indices': sorted(anomaly_indices),
        'anomalies_mask': anomalies_mask.tolist(),
        'residual': resid.tolist(),
        'robust_residual': robust_resid.tolist(),
        'trend': decomp.trend.tolist(),
        'seasonal': decomp.seasonal.tolist()
    }


def calculate_evt_threshold(data: Union[np.ndarray, Sequence[float]], extreme_quantile: float = 0.98) -> float:
    """
    Calculate threshold under Peak-Over-Threshold (POT) framework using Generalized Pareto Distribution (GPD).
    """
    arr = np.array(data, dtype=float)
    u = float(np.percentile(arr, 90))
    excesses = arr[arr > u] - u
    if len(excesses) < 5:
        return float(np.percentile(arr, extreme_quantile * 100))
        
    c, _, scale = genpareto.fit(excesses, floc=0)
    
    n = len(arr)
    N = len(excesses)
    prob = (n / N) * (1.0 - extreme_quantile)
    
    if abs(c) < 1e-9:
        t_val = u - scale * np.log(prob)
    else:
        t_val = u + (scale / c) * (prob**(-c) - 1.0)
        
    return float(t_val)


def calculate_mad(data: Union[np.ndarray, Sequence[float]], nan_policy: str = 'propagate') -> float:
    """
    Calculate the Median Absolute Deviation (MAD) of a dataset.
    """
    from scipy.stats import median_abs_deviation
    return float(median_abs_deviation(np.array(data), nan_policy=nan_policy))


def cohort_zscore(value: float, median: float, mad: float) -> float:
    """
    Calculate modified Z-score using precomputed median and MAD.
    """
    SCALE = 0.6745
    if mad == 0:
        return 0.0 if value == median else float('inf')
    return SCALE * (value - median) / mad


def calculate_skew_kurtosis(data: Union[np.ndarray, Sequence[float]]) -> tuple[float, float]:
    """
    Calculate skewness and kurtosis of a dataset.
    """
    from scipy.stats import skew, kurtosis
    arr = np.array(data)
    return float(skew(arr)), float(kurtosis(arr))


def plot_qq(data: Union[np.ndarray, Sequence[float]], ax=None, dist: str = "norm") -> None:
    """
    Generate a Q-Q plot of data.
    """
    from scipy import stats
    stats.probplot(np.array(data), dist=dist, plot=ax)


def calculate_poisson_pmf(events: Union[np.ndarray, Sequence[int]], lam: float) -> np.ndarray:
    """
    Calculate the Probability Mass Function (PMF) of a Poisson distribution.
    """
    from scipy.stats import poisson
    return poisson.pmf(np.array(events), lam)


def calculate_mahalanobis_distance(pt: Union[np.ndarray, Sequence[float]], mean: np.ndarray, inv_cov: np.ndarray) -> float:
    """
    Calculate the Mahalanobis distance of a point.
    """
    from scipy.spatial.distance import mahalanobis
    return float(mahalanobis(pt, mean, inv_cov))


def calculate_chi2_threshold(probability: float, df: int) -> float:
    """
    Calculate threshold under Chi-Square distribution using Percent Point Function (PPF).
    """
    from scipy.stats import chi2
    return float(chi2.ppf(probability, df=df))


def decompose_time_series(data: pd.Series, model: str = 'additive', period: int = 24):
    """
    Decompose a time series into trend, seasonal, and residual components.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    return seasonal_decompose(data, model=model, period=period)


def run_adfuller_test(data: Union[np.ndarray, Sequence[float]]):
    """
    Run the Augmented Dickey-Fuller (ADF) test for stationarity.
    """
    from statsmodels.tsa.stattools import adfuller
    return adfuller(np.array(data))


def calculate_rfft(data: Union[np.ndarray, Sequence[float]], d: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Real Fast Fourier Transform (RFFT) and frequencies.
    """
    from scipy.fft import rfft, rfftfreq
    arr = np.array(data)
    yf = np.abs(rfft(arr))
    xf = rfftfreq(len(arr), d=d)
    return yf, xf


def apply_median_filter(data: Union[np.ndarray, Sequence[float]], kernel_size: int = 5) -> np.ndarray:
    """
    Apply a 1D median filter to the input data.
    """
    from scipy.signal import medfilt
    return medfilt(np.array(data), kernel_size=kernel_size)


def apply_savgol_filter(data: Union[np.ndarray, Sequence[float]], window_length: int = 15, polyorder: int = 3) -> np.ndarray:
    """
    Apply a Savitzky-Golay filter to the input data.
    """
    from scipy.signal import savgol_filter
    return savgol_filter(np.array(data), window_length=window_length, polyorder=polyorder)


def run_ks_test(d1: Union[np.ndarray, Sequence[float]], d2: Union[np.ndarray, Sequence[float]]) -> tuple[float, float]:
    """
    Run the two-sample Kolmogorov-Smirnov test.
    """
    from scipy.stats import ks_2samp
    res = ks_2samp(np.array(d1), np.array(d2))
    return float(res.statistic), float(res.pvalue)


def calculate_wasserstein_distance(d1: Union[np.ndarray, Sequence[float]], d2: Union[np.ndarray, Sequence[float]]) -> float:
    """
    Calculate the Earth Mover's (Wasserstein) distance between two distributions.
    """
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(np.array(d1), np.array(d2)))


def run_exponential_smoothing(data: pd.Series, trend: str = 'additive', seasonal: str = 'additive', seasonal_periods: int = 7):
    """
    Fit Holt-Winters Exponential Smoothing model.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    return ExponentialSmoothing(data, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods).fit()


def run_chisquare_test(observed: Union[np.ndarray, Sequence[float]], expected: Union[np.ndarray, Sequence[float]]) -> tuple[float, float]:
    """
    Perform a Chi-Square goodness-of-fit test.
    """
    from scipy.stats import chisquare
    res = chisquare(np.array(observed), f_exp=np.array(expected))
    return float(res.statistic), float(res.pvalue)



