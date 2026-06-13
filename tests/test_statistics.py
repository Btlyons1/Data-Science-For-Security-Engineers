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
    chi_square_shift
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



