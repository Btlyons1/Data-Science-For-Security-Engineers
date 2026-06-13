import numpy as np
from detection_baseline.automation import (
    analyze_timing_regularity,
    analyze_session_burstiness,
    fano_factor,
    analyze_ai_agent_autonomy
)

def test_analyze_timing_regularity():
    # Regular bot times (std dev ~0)
    bot_times = [1.0, 1.0, 1.0, 1.0, 1.0]
    res = analyze_timing_regularity(bot_times)
    assert res['cv'] == 0.0
    assert res['likely_automated'] == True
    
    # Variable human times (non-integer to avoid roundness heuristics)
    human_times = [1.23, 4.56, 1.12, 2.89, 2.01]
    res_human = analyze_timing_regularity(human_times)
    assert res_human['cv'] > 0.4
    assert res_human['likely_automated'] == False

def test_analyze_session_burstiness():
    # Regular timestamps (steady activity)
    bot_timestamps = [i * 2.0 for i in range(50)]
    res = analyze_session_burstiness(bot_timestamps, window_seconds=10)
    # Counts per 10s window will be steady, leading to low Fano Factor (<0.5)
    assert res['fano_factor'] < 0.5
    assert res['likely_automated'] == True

def test_fano_factor():
    data = [2, 2, 2, 2]
    assert fano_factor(data) == 0.0

def test_analyze_ai_agent_autonomy():
    events = [
        {'decision_source': 'user_prompt'},
        {'decision_source': 'hook'},
        {'decision_source': 'hook'},
        {'decision_source': 'hook'},
    ]
    res = analyze_ai_agent_autonomy(events)
    assert res['total_events'] == 4
    assert res['user_initiated'] == 1
    assert res['autonomous'] == 3
    # 3 hook events after the last user prompt
    assert res['autonomous_after_last_user'] == 3
