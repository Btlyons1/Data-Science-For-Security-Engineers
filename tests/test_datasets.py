import pandas as pd
from detection_baseline.datasets import (
    load_dga_sample,
    load_network_flow_sample,
    load_auth_log_sample
)

def test_load_dga_sample():
    df = load_dga_sample()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert 'domain' in df.columns
    assert 'is_dga' in df.columns
    assert 'entropy' in df.columns
    google_ent = df.loc[df['domain'] == 'google.com', 'entropy'].values[0]
    assert google_ent < 3.5

def test_load_network_flow_sample():
    df = load_network_flow_sample()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 300
    assert 'node_id' in df.columns
    assert 'bytes_transferred' in df.columns
    
def test_load_auth_log_sample():
    df = load_auth_log_sample()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'username' in df.columns
    assert 'department' in df.columns
    assert 'role' in df.columns
    assert 'failed_logins' in df.columns
