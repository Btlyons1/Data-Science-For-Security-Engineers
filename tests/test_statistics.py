import numpy as np
from detection_baseline.statistics import (
    calculate_iqr,
    build_robust_baseline,
    calculate_zscore,
    modified_zscore,
    calculate_modified_z,
    detect_outliers_robust,
    benford_analysis,
    calculate_ewma,
    detect_cusum,
    chi_square_shift,
    dynamic_time_warping_distance,
    detect_s_h_esd,
    calculate_evt_threshold,
    calculate_mad,
    cohort_zscore,
    calculate_skew_kurtosis,
    plot_qq,
    calculate_poisson_pmf,
    calculate_mahalanobis_distance,
    calculate_chi2_threshold,
    decompose_time_series,
    run_adfuller_test,
    calculate_rfft,
    apply_median_filter,
    apply_savgol_filter,
    run_ks_test,
    calculate_wasserstein_distance,
    run_exponential_smoothing,
    run_chisquare_test
)

def test_calculate_iqr():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    iqr, q1, q3 = calculate_iqr(data)
    assert q1 == 3.25
    assert q3 == 7.75
    assert iqr == 4.5

def test_build_robust_baseline():
    data = [10, 12, 11, 15, 9, 10, 14, 11, 45]
    res = build_robust_baseline(data, 'api_calls')
    assert res['center'] == np.median(data)
    assert 'spread' in res
    assert 'warn_upper' in res

def test_calculate_zscore():
    data = [2, 4, 4, 4, 5, 5, 7, 9]
    # Mean: 5.0, Std: 2.0
    # For value 9: (9-5)/2 = 2.0
    z = calculate_zscore(9, data)
    assert abs(z - 2.0) < 1e-5

def test_modified_zscore():
    data = [10, 12, 11, 15, 9, 10, 14, 11, 45]
    scores = modified_zscore(data)
    assert len(scores) == 9
    # The last value (45) should be highly anomalous
    assert scores[-1] > 3.5

def test_calculate_modified_z():
    median = 11.0
    mad = 2.0
    z = calculate_modified_z(15, median, mad)
    expected = 0.6745 * (15 - 11.0) / 2.0
    assert abs(z - expected) < 1e-5

def test_modified_zscore_zero_mad():
    data = [5.0, 5.0, 5.0, 5.0, 5.0, 10.0, 2.0]
    scores = modified_zscore(data)
    assert scores[0] == 0.0
    assert scores[5] == float('inf')
    assert scores[6] == float('-inf')

def test_calculate_modified_z_zero_mad():
    median = 5.0
    mad = 0.0
    assert calculate_modified_z(5.0, median, mad) == 0.0
    assert calculate_modified_z(10.0, median, mad) == float('inf')
    assert calculate_modified_z(2.0, median, mad) == float('-inf')

def test_detect_outliers_robust():
    data = [10, 12, 11, 15, 9, 10, 14, 11, 45]
    df = detect_outliers_robust(data)
    assert df.loc[8, 'value'] == 45
    assert df.loc[8, 'is_outlier'] == True
    assert df.loc[0, 'is_outlier'] == False

def test_benford_analysis_error():
    data = [1.0, 2.0, 3.0]
    res = benford_analysis(data)
    assert 'error' in res

def test_benford_analysis_conforming():
    np.random.seed(42)
    data = 10 ** np.random.uniform(1, 5, 1000)
    res = benford_analysis(data)
    assert 'error' not in res
    assert res['mae'] < 3.0
    assert res['is_anomalous_deviation'] == False

def test_benford_analysis_deviating():
    np.random.seed(42)
    data = np.random.uniform(100, 999, 1000)
    res = benford_analysis(data)
    assert 'error' not in res
    assert res['mae'] > 3.0
    assert res['is_anomalous_deviation'] == True

def test_calculate_ewma():
    data = [10.0, 20.0, 30.0]
    res = calculate_ewma(data, alpha=0.5)
    # y[0] = 10
    # y[1] = 0.5 * 20 + 0.5 * 10 = 15
    # y[2] = 0.5 * 30 + 0.5 * 15 = 22.5
    assert len(res) == 3
    assert abs(res[0] - 10.0) < 1e-5
    assert abs(res[1] - 15.0) < 1e-5
    assert abs(res[2] - 22.5) < 1e-5

def test_detect_cusum():
    # CUSUM anomalies: normal sequence with sudden exfiltration at the end
    data = [10.0] * 50 + [25.0] * 10
    res = detect_cusum(data, threshold=4.0, drift=0.5)
    assert 'error' not in res
    assert res['alert_triggered'] == True
    # Alerts should be triggered in s_high (at the end)
    assert len(res['alerts_high']) > 0

def test_chi_square_shift():
    # Normal web status baseline
    expected = {'200': 90.0, '302': 7.0, '404': 2.0, '500': 1.0}
    # Case A: conforming observations
    observed_normal = {'200': 904, '302': 64, '404': 21, '500': 11}
    res_normal = chi_square_shift(observed_normal, expected)
    assert res_normal['is_shift_detected'] == False
    
    # Case B: directory scan exfiltration / shift (many 404s)
    observed_scan = {'200': 598, '302': 42, '404': 319, '500': 41}
    res_scan = chi_square_shift(observed_scan, expected)
    assert res_scan['is_shift_detected'] == True


def test_dynamic_time_warping_distance():
    s1 = [1, 2, 3, 4]
    s2 = [1, 1, 2, 2, 3, 3, 4, 4]
    dist = dynamic_time_warping_distance(s1, s2)
    # The two sequences are identical under time warping, distance should be 0.0
    assert abs(dist - 0.0) < 1e-5
    
    s3 = [1, 5, 3, 4]
    dist2 = dynamic_time_warping_distance(s1, s3)
    assert dist2 > 0.0


def test_detect_s_h_esd():
    # Generate seasonal data: daily pattern of 24 points
    np.random.seed(42)
    base = np.sin(np.linspace(0, 4 * np.pi, 96)) * 10
    noise = np.random.normal(0, 1, 96)
    data = base + noise
    
    # Add an anomaly
    data[50] = 50.0
    
    res = detect_s_h_esd(data, period=24, alpha=0.05, max_anomalies=0.1)
    assert 'error' not in res
    assert 50 in res['anomaly_indices']
    assert res['anomalies_mask'][50] == True


def test_calculate_evt_threshold():
    np.random.seed(42)
    data = np.random.exponential(scale=10.0, size=200)
    thresh = calculate_evt_threshold(data, extreme_quantile=0.98)
    assert thresh > np.percentile(data, 90)
    assert thresh < 100.0


def test_calculate_mad():
    data = [1, 2, 3, 4, 5]
    assert calculate_mad(data) == 1.0


def test_cohort_zscore():
    assert cohort_zscore(15, 11, 2) == 0.6745 * 2.0
    assert cohort_zscore(5, 5, 0) == 0.0
    assert cohort_zscore(10, 5, 0) == float('inf')


def test_calculate_skew_kurtosis():
    np.random.seed(42)
    data = np.random.normal(0, 1, 100)
    s, k = calculate_skew_kurtosis(data)
    assert abs(s) < 0.5
    assert abs(k) < 1.0


def test_plot_qq():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plot_qq([1, 2, 3], ax=ax)
    plt.close(fig)


def test_calculate_poisson_pmf():
    probs = calculate_poisson_pmf([0, 1], lam=1.0)
    assert len(probs) == 2
    assert abs(probs[0] - np.exp(-1.0)) < 1e-5


def test_calculate_mahalanobis_distance():
    mean = [0.0, 0.0]
    cov = [[1.0, 0.0], [0.0, 1.0]]
    inv_cov = np.linalg.inv(cov)
    dist = calculate_mahalanobis_distance([1.0, 0.0], mean, inv_cov)
    assert abs(dist - 1.0) < 1e-5


def test_calculate_chi2_threshold():
    val = calculate_chi2_threshold(0.95, df=2)
    assert val > 5.0


def test_decompose_time_series():
    import pandas as pd
    data = pd.Series(np.sin(np.linspace(0, 10, 100)))
    res = decompose_time_series(data, period=10)
    assert res.seasonal is not None


def test_run_adfuller_test():
    data = np.random.normal(0, 1, 100)
    res = run_adfuller_test(data)
    assert len(res) >= 2


def test_calculate_rfft():
    yf, xf = calculate_rfft([1, 2, 3, 4], d=1.0)
    assert len(yf) == 3
    assert len(xf) == 3


def test_apply_median_filter():
    res = apply_median_filter([1, 10, 1], kernel_size=3)
    assert len(res) == 3
    assert res[1] == 1.0


def test_apply_savgol_filter():
    res = apply_savgol_filter(np.ones(20), window_length=5, polyorder=2)
    assert len(res) == 20
    assert abs(res[10] - 1.0) < 1e-5


def test_run_ks_test():
    stat, p = run_ks_test([1, 2, 3], [1, 2, 3])
    assert stat == 0.0
    assert p == 1.0


def test_calculate_wasserstein_distance():
    dist = calculate_wasserstein_distance([1, 2], [1, 2])
    assert dist == 0.0


def test_run_exponential_smoothing():
    import pandas as pd
    data = pd.Series(np.sin(np.linspace(0, 10, 100)))
    model = run_exponential_smoothing(data, trend=None, seasonal=None)
    assert model is not None


def test_run_chisquare_test():
    stat, p = run_chisquare_test([10, 10], [10, 10])
    assert stat == 0.0
    assert p == 1.0




