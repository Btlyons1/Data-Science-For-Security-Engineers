import os
import pandas as pd
from detection_baseline.behavioral import shannon_entropy

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def load_dga_sample() -> pd.DataFrame:
    """
    Load a sample dataset containing benign and malicious (DGA) domain names.
    
    Includes precomputed Shannon entropy for each domain name.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ['domain', 'is_dga', 'entropy'].
    """
    path = os.path.join(DATA_DIR, 'dga_sample.csv')
    df = pd.read_csv(path)
    df['entropy'] = df['domain'].apply(shannon_entropy)
    return df

def load_network_flow_sample() -> pd.DataFrame:
    """
    Load a sample network flow dataset of daily bytes transferred.
    
    Contains:
    - Benign nodes whose bytes transferred follow a log-normal distribution
      (conforming to Benford's Law).
    - An anomalous exfiltration node transferring fixed or repeating payload sizes
      (deviating from Benford's Law).
      
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ['node_id', 'bytes_transferred'].
    """
    path = os.path.join(DATA_DIR, 'network_flow_sample.csv')
    return pd.read_csv(path)

def load_auth_log_sample() -> pd.DataFrame:
    """
    Load a sample authentication log dataset with peer cohorts.
    
    Includes usernames, department roles, hours, weekend flags, and failed login counts.
    Contains compromised and exfiltration anomalous profiles.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ['username', 'department', 'role', 'hour', 'is_weekend', 'action', 'failed_logins'].
    """
    path = os.path.join(DATA_DIR, 'auth_log_sample.csv')
    return pd.read_csv(path)
