import pandas as pd
import numpy as np
from scipy import stats
from collections.abc import Sequence
from typing import Any, Union


def perform_eda(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """
    Perform basic reconnaissance on a data column.
    This should be your FIRST step before any analysis.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Your data
    column : str
        The column to analyze
        
    Returns:
    --------
    dict
        A dictionary containing the summary metrics.
    """
    print(f'=== Reconnaissance: {column} ===\n')
    
    data = df[column]
    
    # Basic info that applies to all types
    total_records = len(data)
    missing_count = int(data.isna().sum())
    missing_pct = (missing_count / total_records) * 100 if total_records > 0 else 0.0
    unique_count = int(data.nunique())
    
    print(f'Total records:  {total_records:,}')
    print(f'Missing values: {missing_count:,} ({missing_pct:.1f}%)')
    print(f'Unique values:  {unique_count:,}\n')
    
    result = {
        'total_records': total_records,
        'missing_count': missing_count,
        'missing_pct': missing_pct,
        'unique_count': unique_count
    }
    
    # Type-specific analysis
    if pd.api.types.is_bool_dtype(data):
        print('Type: Binary')
        true_count = int(data.sum())
        false_count = total_records - true_count
        true_pct = (true_count / total_records) * 100 if total_records > 0 else 0.0
        false_pct = (false_count / total_records) * 100 if total_records > 0 else 0.0
        
        print(f'  True:  {true_count:,} ({true_pct:.1f}%)')
        print(f'  False: {false_count:,} ({false_pct:.1f}%)')
        
        result.update({
            'type': 'Binary',
            'true_count': true_count,
            'false_count': false_count,
            'true_pct': true_pct,
            'false_pct': false_pct
        })
        
    elif pd.api.types.is_numeric_dtype(data):
        print('Type: Numerical')
        min_val = float(data.min())
        max_val = float(data.max())
        mean_val = float(data.mean())
        median_val = float(data.median())
        std_val = float(data.std()) if len(data.dropna()) > 1 else 0.0
        
        print(f'  Min:    {min_val:.2f}')
        print(f'  Max:    {max_val:.2f}')
        print(f'  Mean:   {mean_val:.2f}')
        print(f'  Median: {median_val:.2f}')
        print(f'  Std:    {std_val:.2f}')
        
        # Check for skewness
        skew = float(stats.skew(data.dropna())) if len(data.dropna()) > 2 else 0.0
        if abs(skew) > 1:
            print(f'  ⚠️  Highly skewed (skewness = {skew:.2f})')
            print(f'       Consider using median instead of mean!')
            
        result.update({
            'type': 'Numerical',
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'skewness': skew
        })
        
    else:
        print('Type: Categorical')
        print('  Top 5 values:')
        top_values = {}
        for val, count in data.value_counts().head(5).items():
            pct = (count / total_records) * 100 if total_records > 0 else 0.0
            print(f'    {val}: {count:,} ({pct:.1f}%)')
            top_values[str(val)] = int(count)
            
        result.update({
            'type': 'Categorical',
            'top_values': top_values
        })
        
    print()
    return result

def choose_center_measure(data: Union[np.ndarray, pd.Series, list[float], Sequence[float]], name: str = 'values') -> float:
    """
    Help decide whether to use mean or median for a baseline.
    
    This function examines your data and provides guidance.
    
    Parameters:
    -----------
    data : array-like
        The array of numerical data to analyze.
    name : str
        The name of the metric/column.
        
    Returns:
    --------
    float
        The recommended baseline value (either mean or median).
    """
    arr = np.array(data)
    
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    ratio = mean_val / median_val if median_val != 0 else float('inf')
    
    # Calculate skewness
    skewness = float(stats.skew(arr)) if len(arr) > 2 else 0.0
    
    print(f'Analysis of {name}:')
    print('=' * 50)
    print(f'Mean:            {mean_val:.2f}')
    print(f'Median:          {median_val:.2f}')
    print(f'Mean/Median:     {ratio:.2f}')
    print(f'Skewness:        {skewness:.2f}\n')
    
    # Provide recommendation
    if abs(skewness) > 1 or ratio > 1.3 or ratio < 0.7:
        print('📊 RECOMMENDATION: Use the MEDIAN')
        print('   Reason: Data is skewed or has outliers')
        print(f'   Baseline value: {median_val:.2f}\n')
        return median_val
    else:
        print('📊 RECOMMENDATION: Mean is acceptable')
        print('   Reason: Data is relatively symmetric')
        print(f'   Baseline value: {mean_val:.2f}\n')
        return mean_val
