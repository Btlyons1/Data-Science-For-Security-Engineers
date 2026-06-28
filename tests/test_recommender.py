import numpy as np
import pandas as pd
from detection_baseline.recommender import recommend_anomaly_detector

def test_recommend_skewed_1d():
    # Exponential data is highly skewed
    np.random.seed(42)
    data = np.random.exponential(scale=10.0, size=100)
    res = recommend_anomaly_detector(data)
    assert res['dimension'] == 'univariate'
    assert res['data_type'] == 'numerical'
    assert any(r['detector'] == 'modified_zscore' for r in res['recommendations'])
    assert any(t['transformation'] == 'log_transform' for t in res['recommended_transformations'])

def test_recommend_symmetrical_1d():
    # Normal data is symmetrical
    np.random.seed(42)
    data = np.random.normal(loc=10.0, scale=2.0, size=100)
    res = recommend_anomaly_detector(data)
    assert res['dimension'] == 'univariate'
    assert res['data_type'] == 'numerical'
    assert any(r['detector'] == 'calculate_zscore' for r in res['recommendations'])

def test_recommend_categorical_high_cardinality():
    # Domain names are categorical high cardinality
    data = [f"domain_{i}.com" for i in range(100)]
    res = recommend_anomaly_detector(data)
    assert res['dimension'] == 'univariate'
    assert res['data_type'] == 'categorical'
    assert any(r['detector'] == 'shannon_entropy' for r in res['recommendations'])

def test_recommend_categorical_low_cardinality():
    # HTTP status codes are categorical low cardinality
    data = ['200']*90 + ['404']*5 + ['500']*5
    res = recommend_anomaly_detector(data)
    assert res['dimension'] == 'univariate'
    assert res['data_type'] == 'categorical'
    assert any(r['detector'] == 'chi_square_shift' for r in res['recommendations'])

def test_recommend_multivariate_low_dim():
    # 2D numerical data
    np.random.seed(42)
    data = np.random.normal(size=(100, 3))
    res = recommend_anomaly_detector(data)
    assert res['dimension'] == 'multivariate'
    assert res['num_features'] == 3
    assert any(r['detector'] == 'mahalanobis_distance' for r in res['recommendations'])

def test_recommend_multivariate_high_dim():
    # 6D numerical data
    np.random.seed(42)
    data = np.random.normal(size=(100, 6))
    res = recommend_anomaly_detector(data)
    assert res['dimension'] == 'multivariate'
    assert res['num_features'] == 6
    assert any(r['detector'] == 'isolation_forest' for r in res['recommendations'])

def test_recommend_multivariate_mixed():
    # Mixed type / transactional data
    df = pd.DataFrame({
        'user': ['alice', 'bob', 'charlie']*33,
        'action': ['login', 'read', 'write']*33
    })
    res = recommend_anomaly_detector(df)
    assert res['dimension'] == 'multivariate'
    assert any(r['detector'] == 'find_association_rules' for r in res['recommendations'])
    assert any(r['detector'] == 'detect_pagerank_spikes' for r in res['recommendations'])
