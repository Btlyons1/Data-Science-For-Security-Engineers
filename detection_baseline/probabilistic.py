import numpy as np
from scipy.stats import chi2
from collections.abc import Sequence
from typing import Union, Any

def calculate_detection_effectiveness(
    base_rate: float,           # P(attack) - how often attacks occur
    detection_rate: float,      # P(alert|attack) - true positive rate
    false_positive_rate: float  # P(alert|no attack) - false positive rate
) -> dict[str, float]:
    """
    Calculate the real-world effectiveness of a detection rule.
    
    This uses Bayes' Theorem to answer: 
    "When this alert fires, what's the probability it's a real attack?"
    
    This is the PRECISION of your detection.
    
    Parameters:
    -----------
    base_rate : float
        P(attack) - prior probability of attack per event.
    detection_rate : float
        P(alert|attack) - True Positive Rate (tpr).
    false_positive_rate : float
        P(alert|no attack) - False Positive Rate (fpr).
        
    Returns:
    --------
    dict
        Dictionary containing precision, expected true/false positives, and total alerts.
    """
    no_attack_rate = 1.0 - base_rate
    p_alert = (detection_rate * base_rate) + (false_positive_rate * no_attack_rate)
    
    if p_alert == 0:
        p_attack_given_alert = 0.0
    else:
        p_attack_given_alert = (detection_rate * base_rate) / p_alert
    
    events = 10000
    expected_attacks = base_rate * events
    expected_true_positives = detection_rate * base_rate * events
    expected_false_positives = false_positive_rate * no_attack_rate * events
    total_alerts = expected_true_positives + expected_false_positives
    
    print('Detection Effectiveness Analysis')
    print('=' * 50)
    print(f'Base rate (attack frequency):  {base_rate:.6f} ({base_rate*100:.4f}%)')
    print(f'Detection rate (TPR):          {detection_rate:.2f} ({detection_rate*100:.0f}%)')
    print(f'False positive rate (FPR):     {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)\n')
    print(f'Per {events:,} events:')
    print(f'  Expected attacks:         {expected_attacks:.1f}')
    print(f'  True positives (caught):  {expected_true_positives:.1f}')
    print(f'  False positives:          {expected_false_positives:.1f}')
    print(f'  Total alerts:             {total_alerts:.1f}\n')
    print(f'📊 PRECISION: {p_attack_given_alert*100:.1f}% of alerts are true attacks')
    
    if p_attack_given_alert < 0.5:
        print(f'\n⚠️  WARNING: Most of your alerts are FALSE POSITIVES!')
        ratio = (1.0 - p_attack_given_alert) / p_attack_given_alert if p_attack_given_alert > 0 else float('inf')
        print(f'   For every true attack, you have {ratio:.1f} false alerts\n')
    
    return {
        'precision': p_attack_given_alert,
        'expected_tp': expected_true_positives,
        'expected_fp': expected_false_positives,
        'total_alerts': total_alerts
    }

def required_fpr_for_precision(base_rate: float, detection_rate: float, target_precision: float) -> float:
    """
    Calculate what false positive rate you need to achieve a target precision.
    
    This helps you understand how tight your detection needs to be.
    
    Parameters:
    -----------
    base_rate : float
        P(attack) - prior probability of attack per event.
    detection_rate : float
        P(alert|attack) - True Positive Rate (tpr).
    target_precision : float
        The target precision (P(Attack | Alert)) desired.
        
    Returns:
    --------
    float
        Required False Positive Rate.
    """
    if target_precision <= 0 or target_precision >= 1:
        raise ValueError("target_precision must be strictly between 0 and 1.")
        
    numerator = detection_rate * base_rate * (1.0 - target_precision)
    denominator = target_precision * (1.0 - base_rate)
    
    required_fpr = numerator / denominator
    
    print(f'To achieve {target_precision*100:.0f}% precision:')
    print(f'  Base rate: {base_rate:.6f}')
    print(f'  Required FP rate: {required_fpr:.6f} ({required_fpr*100:.4f}%)')
    
    if required_fpr > 0:
        print(f'  That\'s 1 false positive per {1/required_fpr:,.0f} benign events\n')
    else:
        print('  Required FP rate is 0 (impossible to satisfy with non-zero target precision unless base_rate or detection_rate is 0)\n')
        
    return required_fpr

def calculate_precision(base_rate: float, tpr: float, fpr: float) -> float:
    """
    Calculate the precision given the base rate, true positive rate, and false positive rate.
    
    Parameters:
    -----------
    base_rate : float
        P(attack) - prior probability of attack.
    tpr : float
        True Positive Rate (recall).
    fpr : float
        False Positive Rate.
        
    Returns:
    --------
    float
        Precision (P(Attack | Alert)).
    """
    prob_alert_given_attack = tpr
    prob_attack = base_rate
    prob_alert_given_benign = fpr
    prob_benign = 1.0 - base_rate
    
    true_positives = prob_alert_given_attack * prob_attack
    false_positives = prob_alert_given_benign * prob_benign
    
    total = true_positives + false_positives
    if total == 0:
        return 0.0
    return true_positives / total

def fishers_combined_p(p_values: list[float]) -> dict[str, Any]:
    """
    Combine multiple independent P-values using Fisher's combined probability test.
    
    This combines weak indicators (P-values) into a single unified significance score.
    
    Formula:
    X^2 = -2 * sum(ln(p_i))
    Degrees of Freedom = 2 * k
    
    Parameters:
    -----------
    p_values : list of floats
        List of P-values to combine.
        
    Returns:
    --------
    dict
        Fisher statistic, degrees of freedom, combined p-value, and alert status.
    """
    valid_ps = [max(1e-15, min(1.0, p)) for p in p_values if p is not None]
    k = len(valid_ps)
    if k == 0:
        return {'error': 'No valid P-values provided.'}
        
    stat = -2.0 * sum(np.log(valid_ps))
    dof = 2 * k
    combined_p = chi2.sf(stat, dof)
    
    return {
        'fisher_stat': float(stat),
        'dof': int(dof),
        'combined_p_value': float(combined_p),
        'is_significant': bool(combined_p < 0.05)
    }

def optimize_detection_threshold(
    scores: Union[np.ndarray, Sequence[float]],
    labels: Union[np.ndarray, Sequence[bool]],
    cost_fp: float = 1.0,
    cost_fn: float = 10.0
) -> dict[str, Any]:
    """
    Select the mathematically optimal detection threshold based on operational cost.
    
    Minimizes the cost function:
    Cost = (False Positives * cost_fp) + (False Negatives * cost_fn)
    
    Parameters:
    -----------
    scores : array-like
        Anomaly scores (e.g. Z-scores, probabilities) for the evaluation dataset.
    labels : array-like
        True binary labels (True = attack, False = benign).
    cost_fp : float
        Cost of triaging a single false positive.
    cost_fn : float
        Cost of a missed threat / false negative.
        
    Returns:
    --------
    dict
        Optimal threshold, minimized total cost, and rates at that threshold.
    """
    arr_scores = np.array(scores, dtype=float)
    arr_labels = np.array(labels, dtype=bool)
    
    if len(arr_scores) != len(arr_labels):
        raise ValueError("scores and labels must have the same length.")
        
    if len(arr_scores) == 0:
        return {'error': 'Dataset is empty.'}
        
    thresholds = np.unique(arr_scores)
    thresholds = np.append(thresholds, thresholds[-1] + 1e-5) if len(thresholds) > 0 else np.array([0.0])
    
    best_threshold = thresholds[0]
    min_cost = float('inf')
    best_tp = 0
    best_fp = 0
    best_fn = 0
    best_tn = 0
    
    total_pos = int(np.sum(arr_labels))
    total_neg = len(arr_labels) - total_pos
    
    for t in thresholds:
        preds = arr_scores >= t
        
        tp = int(np.sum(preds & arr_labels))
        fp = int(np.sum(preds & ~arr_labels))
        fn = total_pos - tp
        tn = total_neg - fp
        
        cost = (fp * cost_fp) + (fn * cost_fn)
        
        if cost < min_cost:
              min_cost = cost
              best_threshold = t
              best_tp = tp
              best_fp = fp
              best_fn = fn
              best_tn = tn
              
    return {
        'optimal_threshold': float(best_threshold),
        'minimized_cost': float(min_cost),
        'true_positives': best_tp,
        'false_positives': best_fp,
        'false_negatives': best_fn,
        'true_negatives': best_tn,
        'total_events': len(arr_scores),
        'tpr': best_tp / total_pos if total_pos > 0 else 0.0,
        'fpr': best_fp / total_neg if total_neg > 0 else 0.0
    }
