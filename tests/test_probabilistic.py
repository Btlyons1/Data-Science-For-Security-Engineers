from detection_baseline.probabilistic import (
    calculate_detection_effectiveness,
    required_fpr_for_precision,
    calculate_precision,
    fishers_combined_p,
    optimize_detection_threshold
)

def test_calculate_detection_effectiveness():
    res = calculate_detection_effectiveness(
        base_rate=0.0001,
        detection_rate=0.99,
        false_positive_rate=0.01
    )
    assert abs(res['precision'] - 0.0098) < 0.001
    assert abs(res['expected_tp'] - 0.99) < 1e-5

def test_required_fpr_for_precision():
    fpr = required_fpr_for_precision(
        base_rate=0.0001,
        detection_rate=0.95,
        target_precision=0.50
    )
    expected = (0.95 * 0.0001 * 0.5) / (0.5 * 0.9999)
    assert abs(fpr - expected) < 1e-8

def test_calculate_precision():
    precision = calculate_precision(base_rate=0.01, tpr=0.99, fpr=0.01)
    assert abs(precision - 0.5) < 1e-5

def test_fishers_combined_p():
    # Test with multiple high P-values (not significant)
    p_vals_normal = [0.8, 0.7, 0.9]
    res_normal = fishers_combined_p(p_vals_normal)
    assert res_normal['is_significant'] == False
    
    # Test with low P-values (very significant)
    p_vals_attack = [0.01, 0.05, 0.02]
    res_attack = fishers_combined_p(p_vals_attack)
    assert res_attack['is_significant'] == True

def test_optimize_detection_threshold():
    # Simple dataset with scores and labels
    # Benign: [1, 2, 3, 4]
    # Attack: [5, 6, 7]
    scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    labels = [False, False, False, False, True, True, True]
    
    # Cost model: FP = 1, FN = 10
    # A threshold of 5.0 splits them perfectly (FP = 0, FN = 0, Cost = 0)
    res = optimize_detection_threshold(scores, labels, cost_fp=1.0, cost_fn=10.0)
    assert 'error' not in res
    assert res['optimal_threshold'] == 5.0
    assert res['minimized_cost'] == 0.0
    assert res['true_positives'] == 3
    assert res['false_positives'] == 0
