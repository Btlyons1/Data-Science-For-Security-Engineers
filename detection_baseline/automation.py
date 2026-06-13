import numpy as np
from collections.abc import Sequence
from typing import Any, Union


def analyze_timing_regularity(inter_event_times: Union[np.ndarray, Sequence[float]]) -> dict[str, Any]:
    """
    Detect automation through timing analysis.
    
    Human behavior: high variance, variable timing
    Automated behavior: low variance, consistent timing
    
    Key metric: Coefficient of Variation (CV) = std / mean
    - CV < 0.2 is suspiciously regular (likely automation)
    - CV > 0.5 is typical for humans
    
    Parameters:
    -----------
    inter_event_times : array-like
        The intervals between sequential events.
        
    Returns:
    --------
    dict
        Dictionary containing timing regularity metrics and assessment.
    """
    times = np.array(inter_event_times)
    
    if len(times) < 2:
        return {'error': 'Need at least 2 events'}
    
    mean_val = np.mean(times)
    std_val = np.std(times)
    
    # Coefficient of Variation
    cv = std_val / mean_val if mean_val > 0 else 0.0
    
    # Check for suspiciously round intervals
    round_intervals = sum(1 for t in times if abs(t - round(t)) < 0.01)
    round_ratio = round_intervals / len(times)
    
    # Check sub-second precision clustering
    fractional = times % 1.0
    fractional_std = np.std(fractional)
    
    # Determine if likely automated
    automated_signals = []
    if cv < 0.2:
        automated_signals.append(f'Low CV ({cv:.3f})')
    if round_ratio > 0.3:
        automated_signals.append(f'Many round intervals ({round_ratio:.0%})')
    if fractional_std < 0.1:
        automated_signals.append('Tight fractional clustering')
    
    return {
        'cv': cv,
        'mean_interval': mean_val,
        'std_interval': std_val,
        'round_ratio': round_ratio,
        'fractional_std': fractional_std,
        'likely_automated': len(automated_signals) >= 2,
        'signals': automated_signals
    }

def analyze_session_burstiness(event_timestamps: Union[np.ndarray, Sequence[float]], window_seconds: int = 60) -> dict[str, Any]:
    """
    Analyze session for human vs automated patterns.
    
    Uses the Fano factor (variance/mean of event counts per window):
    - Fano > 1: Bursty (human-like)
    - Fano ≈ 1: Random (Poisson process)
    - Fano < 1: Regular (automation-like)
    
    Parameters:
    -----------
    event_timestamps : array-like
        The timestamps of events (numeric or epoch format).
    window_seconds : int
        The bucket window size in seconds (default 60).
        
    Returns:
    --------
    dict
        Dictionary containing burstiness analysis.
    """
    timestamps = np.array(sorted(event_timestamps))
    
    if len(timestamps) < 10:
        return {'error': 'Need at least 10 events'}
    
    # Calculate duration and number of windows
    duration = timestamps[-1] - timestamps[0]
    n_windows = max(1, int(duration / window_seconds))
    
    # Count events per window
    counts = np.zeros(n_windows)
    for t in timestamps:
        window_idx = min(int((t - timestamps[0]) / window_seconds), n_windows - 1)
        counts[window_idx] += 1
    
    # Fano factor
    mean_count = np.mean(counts)
    var_count = np.var(counts)
    fano = var_count / mean_count if mean_count > 0 else 0.0
    
    # Count idle windows (no events)
    idle_windows = np.sum(counts == 0)
    idle_ratio = idle_windows / n_windows
    
    # Sustained activity ratio
    active_windows = np.sum(counts > 0)
    sustained_ratio = active_windows / n_windows
    
    # Interpretation
    if fano < 0.5 and sustained_ratio > 0.9:
        interpretation = 'LIKELY AUTOMATED - steady activity, no idle time'
        is_automated = True
    elif fano > 1.5 and idle_ratio > 0.1:
        interpretation = 'LIKELY HUMAN - bursty with idle periods'
        is_automated = False
    else:
        interpretation = 'UNCERTAIN - could be either'
        is_automated = None
    
    return {
        'fano_factor': fano,
        'idle_ratio': idle_ratio,
        'sustained_ratio': sustained_ratio,
        'events_per_window_mean': mean_count,
        'events_per_window_std': np.sqrt(var_count),
        'likely_automated': is_automated,
        'interpretation': interpretation
    }

def fano_factor(data: Union[np.ndarray, Sequence[float]]) -> float:
    """
    Calculate the Fano Factor (Variance / Mean) of a dataset.
    
    Parameters:
    -----------
    data : array-like
        Input values.
        
    Returns:
    --------
    float
        The Fano Factor.
    """
    arr = np.array(data)
    mean_val = np.mean(arr)
    if mean_val == 0:
        return 0.0
    return float(np.var(arr) / mean_val)

def analyze_ai_agent_autonomy(events: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Analyze AI agent telemetry for signs of autonomous behavior.
    
    Looks for patterns suggesting the agent is acting without
    appropriate user oversight.
    
    Parameters:
    -----------
    events : list of dicts
        List of event objects, each containing at least 'decision_source'.
        
    Returns:
    --------
    dict
        Autonomy summary and risk flags.
    """
    # Categorize decision sources
    user_sources = {'user_prompt', 'user_input', 'user_command', 'user_approval'}
    auto_sources = {'config', 'hook', 'scheduled', 'automated', 'system'}
    
    # Extract decision sources
    sources = [e.get('decision_source', 'unknown') for e in events]
    
    # Count by category
    user_count = sum(1 for s in sources if s in user_sources)
    auto_count = sum(1 for s in sources if s in auto_sources)
    total = len(sources)
    
    # Find last user-initiated action
    last_user_idx = -1
    for i, src in enumerate(sources):
        if src in user_sources:
            last_user_idx = i
    
    # Count autonomous actions after last user input
    autonomous_after_user = 0
    if last_user_idx >= 0:
        for src in sources[last_user_idx + 1:]:
            if src in auto_sources:
                autonomous_after_user += 1
    
    # Calculate autonomy drift over session
    window_size = max(1, len(sources) // 5)
    autonomy_by_window = []
    for i in range(0, len(sources), window_size):
        window = sources[i:i+window_size]
        if not window:
            continue
        auto_ratio = sum(1 for s in window if s in auto_sources) / len(window)
        autonomy_by_window.append(auto_ratio)
    
    autonomy_trend = 0.0
    if len(autonomy_by_window) > 1:
        autonomy_trend = autonomy_by_window[-1] - autonomy_by_window[0]
    
    # Risk assessment
    risk_signals = []
    if autonomous_after_user > 10:
        risk_signals.append(f'{autonomous_after_user} autonomous actions after last user input')
    if autonomy_trend > 0.3:
        risk_signals.append(f'Autonomy increasing over session (trend: +{autonomy_trend:.0%})')
    if total > 0 and auto_count / total > 0.8 and total > 20:
        risk_signals.append(f'Session is {auto_count/total:.0%} autonomous')
    
    return {
        'total_events': total,
        'user_initiated': user_count,
        'autonomous': auto_count,
        'autonomy_ratio': auto_count / total if total > 0 else 0.0,
        'autonomous_after_last_user': autonomous_after_user,
        'autonomy_trend': autonomy_trend,
        'risk_signals': risk_signals,
        'alert': len(risk_signals) > 0
    }
