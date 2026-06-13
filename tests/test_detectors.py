from detection_baseline.detectors import FirstSeenDetector

def test_first_seen_detector():
    historical = ['192.168.1.1', '192.168.1.2', '10.0.0.1']
    detector = FirstSeenDetector()
    detector.train(historical)
    
    # Check known IP
    res_known = detector.evaluate('192.168.1.1')
    assert res_known['is_novel'] == False
    assert res_known['risk'] == 'low'
    
    # Check novel IP without context
    res_novel_no_ctx = detector.evaluate('8.8.8.8')
    assert res_novel_no_ctx['is_novel'] == True
    assert res_novel_no_ctx['risk'] == 'low'
    
    # Check novel IP with suspicious context
    context = {
        'hour': 2,               # off-hours
        'action': 'export',      # sensitive action
        'user_role': 'admin'     # privileged user
    }
    res_suspicious = detector.evaluate('8.8.8.8', context)
    assert res_suspicious['is_novel'] == True
    assert len(res_suspicious['risk_factors']) == 3
    assert res_suspicious['risk'] == 'high'
