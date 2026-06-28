import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from typing import Union, Any, Sequence

def recommend_anomaly_detector(data: Any) -> dict[str, Any]:
    """
    Analyze the shape, distribution, stationarity, and cardinality of the input dataset
    and recommend the most appropriate anomaly detection model(s) and preprocessing steps.
    
    Parameters:
    -----------
    data : array-like, pd.Series, or pd.DataFrame
        The dataset to analyze.
        
    Returns:
    --------
    dict
        Profiling metrics and a ranked list of recommended anomaly detectors.
    """
    if isinstance(data, (list, tuple, np.ndarray)):
        arr = np.array(data)
        if len(arr.shape) == 1:
            series = pd.Series(arr)
            df = pd.DataFrame(series)
        else:
            df = pd.DataFrame(arr)
    elif isinstance(data, pd.Series):
        series = data
        df = pd.DataFrame(series)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError("Input data must be a numpy array, pandas Series, or pandas DataFrame.")
        
    num_rows, num_cols = df.shape
    
    if num_cols == 1:
        col = df.columns[0]
        series = df[col]
        
        if pd.api.types.is_numeric_dtype(series):
            clean_series = series.dropna()
            if len(clean_series) < 5:
                return {'error': 'Not enough data points to analyze.'}
                
            skew = float(stats.skew(clean_series))
            kurt = float(stats.kurtosis(clean_series))
            
            try:
                adf_res = adfuller(clean_series.values, maxlag=min(10, len(clean_series)//5))
                p_val = float(adf_res[1])
                is_stationary = bool(p_val < 0.05)
            except Exception:
                p_val = 1.0
                is_stationary = False
                
            cardinality = clean_series.nunique()
            
            recs = []
            transformations = []
            
            # 1. Analyze skewness
            if abs(skew) > 1.0:
                recs.append({
                    'detector': 'modified_zscore',
                    'confidence': 'High',
                    'reason': f'Data is highly skewed (skew={skew:.2f}). Robust statistics (Median/MAD) are recommended over standard Z-score.'
                })
                transformations.append({
                    'transformation': 'log_transform',
                    'reason': 'Data is highly skewed. Applying a logarithmic transform (e.g. np.log1p) will help stabilize variance.'
                })
            else:
                recs.append({
                    'detector': 'calculate_zscore',
                    'confidence': 'High',
                    'reason': f'Data is roughly symmetrical (skew={skew:.2f}). Standard parametric Z-scores are appropriate.'
                })
                
            # 2. Analyze stationarity
            if not is_stationary:
                recs.append({
                    'detector': 'detect_s_h_esd',
                    'confidence': 'Medium',
                    'reason': f'Data is non-stationary (ADF p-value={p_val:.4f}). Seasonal Hybrid ESD is recommended to isolate anomalies over temporal/seasonal baselines.'
                })
                recs.append({
                    'detector': 'detect_cusum',
                    'confidence': 'Medium',
                    'reason': 'Data is non-stationary. CUSUM is recommended to aggregate minor deviations and detect low-and-slow drift.'
                })
                recs.append({
                    'detector': 'calculate_ewma',
                    'confidence': 'Medium',
                    'reason': 'Data shows trend/non-stationarity. Exponentially Weighted Moving Average (EWMA) will adapt to baseline drift.'
                })
            else:
                recs.append({
                    'detector': 'calculate_evt_threshold',
                    'confidence': 'Medium',
                    'reason': 'Data is stationary. Extreme Value Theory (EVT/POT) can determine mathematically optimal thresholds for extreme tail events.'
                })
                
            # 3. Check cardinality
            if cardinality < 10:
                recs.append({
                    'detector': 'chi_square_shift',
                    'confidence': 'High',
                    'reason': 'Data has low cardinality (integer count or categorical-like codes). Chi-Square Shift Test is ideal for auditing ratio shifts.'
                })
                
            return {
                'dimension': 'univariate',
                'data_type': 'numerical',
                'num_samples': num_rows,
                'skewness': skew,
                'kurtosis': kurt,
                'cardinality': cardinality,
                'adf_p_value': p_val,
                'is_stationary': is_stationary,
                'recommendations': recs,
                'recommended_transformations': transformations
            }
        else:
            # Categorical 1D
            cardinality = series.nunique()
            recs = []
            
            if cardinality > 50:
                recs.append({
                    'detector': 'shannon_entropy',
                    'confidence': 'High',
                    'reason': f'High-cardinality categorical data (cardinality={cardinality}). Shannon Entropy will identify highly random strings (e.g., DGA or obfuscation).'
                })
            else:
                recs.append({
                    'detector': 'chi_square_shift',
                    'confidence': 'High',
                    'reason': f'Categorical data with medium/low cardinality (cardinality={cardinality}). Chi-Square goodness-of-fit is ideal for auditing ratio drift.'
                })
                
            return {
                'dimension': 'univariate',
                'data_type': 'categorical',
                'num_samples': num_rows,
                'cardinality': cardinality,
                'recommendations': recs,
                'recommended_transformations': []
            }
    else:
        # Multivariate
        recs = []
        numerical_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        
        if len(numerical_cols) == num_cols:
            if num_cols < 5:
                recs.append({
                    'detector': 'mahalanobis_distance',
                    'confidence': 'High',
                    'reason': f'Low-dimensional multivariate data (dimensions={num_cols}). Mahalanobis Distance is recommended to account for feature covariance.'
                })
            else:
                recs.append({
                    'detector': 'isolation_forest',
                    'confidence': 'High',
                    'reason': f'High-dimensional multivariate data (dimensions={num_cols}). Isolation Forest handles high dimensionality effectively without distance metric collapse.'
                })
        else:
            recs.append({
                'detector': 'find_association_rules',
                'confidence': 'Medium',
                'reason': 'Multivariate mixed/categorical transactions. Association Rule Mining (Apriori) will discover normal connection maps (e.g. File/SaaS access profiling).'
            })
            recs.append({
                'detector': 'detect_pagerank_spikes',
                'confidence': 'Medium',
                'reason': 'Graph/Network relational transactions. PageRank centrality spike detection is recommended to find lateral movement anomalies.'
            })
            
        return {
            'dimension': 'multivariate',
            'num_samples': num_rows,
            'num_features': num_cols,
            'num_numerical_features': len(numerical_cols),
            'recommendations': recs,
            'recommended_transformations': []
        }
