import pandas as pd
import numpy as np
from detection_baseline.eda import perform_eda, choose_center_measure

def test_perform_eda_numeric():
    df = pd.DataFrame({'val': [1, 2, 3, 4, 100]})
    res = perform_eda(df, 'val')
    assert res['type'] == 'Numerical'
    assert res['min'] == 1
    assert res['max'] == 100
    assert res['median'] == 3
    assert res['total_records'] == 5

def test_perform_eda_bool():
    df = pd.DataFrame({'flag': [True, True, False, True]})
    res = perform_eda(df, 'flag')
    assert res['type'] == 'Binary'
    assert res['true_count'] == 3
    assert res['false_count'] == 1

def test_perform_eda_categorical():
    df = pd.DataFrame({'cat': ['a', 'a', 'b', 'c', 'a']})
    res = perform_eda(df, 'cat')
    assert res['type'] == 'Categorical'
    assert res['top_values']['a'] == 3

def test_choose_center_measure():
    # Symmetric data
    data_symmetric = [10, 11, 10, 12, 9, 10, 11, 10, 10, 9]
    res_sym = choose_center_measure(data_symmetric)
    assert abs(res_sym - np.mean(data_symmetric)) < 0.1
    
    # Skewed data
    data_skewed = [10, 11, 10, 12, 9, 10, 11, 10, 10, 1000]
    res_skew = choose_center_measure(data_skewed)
    assert res_skew == np.median(data_skewed)
