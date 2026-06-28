import nltk
import math
import numpy as np
from collections.abc import Iterable
from typing import Any, Union


def sequence_anomaly_score(sequence: list[Any], known_good_bigrams: Iterable[tuple[Any, Any]]) -> float:
    """
    Measure percentage of unknown transitions (bigrams) in a sequence.
    
    Parameters:
    -----------
    sequence : list
        The list of actions/events in the sequence.
    known_good_bigrams : set or list
        The set of transitions that are known to be benign.
        
    Returns:
    --------
    float
        The anomaly score (fraction of unknown bigrams).
    """
    test_bigrams = list(nltk.bigrams(sequence))
    if not test_bigrams:
        return 0.0
        
    unknown_transitions = 0
    known_set = set(known_good_bigrams)
    
    for bg in test_bigrams:
        if bg not in known_set:
            unknown_transitions += 1
            
    return unknown_transitions / len(test_bigrams)

def jaccard(seq1: Iterable[Any], seq2: Iterable[Any]) -> float:
    """
    Calculate the Jaccard similarity (Intersection over Union) of two sets/sequences.
    
    Parameters:
    -----------
    seq1 : iterable
        First sequence/set.
    seq2 : iterable
        Second sequence/set.
        
    Returns:
    --------
    float
        The Jaccard similarity score between 0.0 and 1.0.
    """
    set1 = set(seq1)
    set2 = set(seq2)
    union_len = len(set1.union(set2))
    if union_len == 0:
        return 0.0
    return len(set1.intersection(set2)) / union_len

def shannon_entropy(data_string: str) -> float:
    """
    Calculate the Shannon Entropy (randomness) of a text string.
    
    Parameters:
    -----------
    data_string : str
        The input string (e.g., domain name, command line).
        
    Returns:
    --------
    float
        The Shannon entropy in bits (usually between 0.0 and 8.0).
    """
    if not data_string:
        return 0.0
    entropy = 0.0
    length = len(data_string)
    for x in set(data_string):
        p_x = float(data_string.count(x)) / length
        entropy += - p_x * math.log(p_x, 2)
    return entropy


def detect_sequence_anomalies(train_sequences: list[list[Any]], test_sequence: list[Any], threshold: float = 0.01) -> dict:
    """
    Score sequence transition anomalies using a first-order Markov Chain.
    """
    from collections import defaultdict
    transitions = defaultdict(lambda: defaultdict(int))
    state_counts = defaultdict(int)
    
    for seq in train_sequences:
        for i in range(len(seq) - 1):
            state = seq[i]
            next_state = seq[i + 1]
            transitions[state][next_state] += 1
            state_counts[state] += 1
            
    probs = defaultdict(dict)
    for state, next_states in transitions.items():
        tot = state_counts[state]
        for next_state, count in next_states.items():
            probs[state][next_state] = count / tot
            
    anomalies = []
    probabilities = []
    for i in range(len(test_sequence) - 1):
        state = test_sequence[i]
        next_state = test_sequence[i + 1]
        p = probs.get(state, {}).get(next_state, 0.0)
        probabilities.append(p)
        if p < threshold:
            anomalies.append({
                'transition': (state, next_state),
                'probability': p,
                'index': i
            })
            
    mean_prob = np.mean(probabilities) if probabilities else 1.0
    return {
        'mean_transition_probability': float(mean_prob),
        'anomalies': anomalies,
        'is_anomaly': len(anomalies) > 0
    }


def detect_hmm_anomalies(
    observations: list[Any],
    states: list[Any],
    start_prob: dict[Any, float],
    trans_prob: dict[Any, dict[Any, float]],
    emit_prob: dict[Any, dict[Any, float]],
    threshold: float = -20.0
) -> dict:
    """
    Calculate log-likelihood of observations using the Hidden Markov Model (HMM) forward algorithm.
    """
    n = len(observations)
    if n == 0:
        return {'log_likelihood': 0.0, 'is_anomaly': False}
        
    forward = []
    o_0 = observations[0]
    f_0 = {}
    for s in states:
        f_0[s] = start_prob.get(s, 0.0) * emit_prob.get(s, {}).get(o_0, 1e-9)
    forward.append(f_0)
    
    for t in range(1, n):
        o_t = observations[t]
        f_t = {}
        for s in states:
            prob_sum = sum(forward[t - 1][prev_s] * trans_prob.get(prev_s, {}).get(s, 0.0) for prev_s in states)
            f_t[s] = prob_sum * emit_prob.get(s, {}).get(o_t, 1e-9)
        forward.append(f_t)
        
    total_prob = sum(forward[-1][s] for s in states)
    log_likelihood = math.log(total_prob) if total_prob > 0 else float('-inf')
    
    return {
        'log_likelihood': log_likelihood,
        'is_anomaly': log_likelihood < threshold
    }


def detect_pagerank_spikes(edges: list[tuple[Any, Any]], threshold: float = 3.5) -> dict:
    """
    Detect network centrality anomalies using PageRank centrality spikes.
    """
    import networkx as nx
    from detection_baseline.statistics import calculate_modified_z
    
    G = nx.Graph()
    G.add_edges_from(edges)
    
    if len(G) == 0:
        return {'anomalies': [], 'pageranks': {}}
        
    pageranks = nx.pagerank(G)
    scores = list(pageranks.values())
    
    median = np.median(scores)
    mad = np.median([abs(x - median) for x in scores])
    
    anomalies = []
    for node, score in pageranks.items():
        z = calculate_modified_z(score, median, mad)
        if z > threshold:
            anomalies.append({
                'node': node,
                'pagerank': float(score),
                'z_score': float(z)
            })
            
    return {
        'pageranks': {str(k): float(v) for k, v in pageranks.items()},
        'anomalies': anomalies,
        'is_anomaly': len(anomalies) > 0
    }


def find_association_rules(transactions: list[list[Any]], min_support: float = 0.1, min_confidence: float = 0.5) -> list[dict]:
    """
    Identify file or resource access guidelines using Association Rule Mining (Apriori).
    """
    from collections import defaultdict
    import itertools
    
    num_transactions = len(transactions)
    if num_transactions == 0:
        return []
        
    item_counts = defaultdict(int)
    for trans in transactions:
        for item in set(trans):
            item_counts[item] += 1
            
    frequent_items = {}
    for item, count in item_counts.items():
        support = count / num_transactions
        if support >= min_support:
            frequent_items[item] = support
            
    pair_counts = defaultdict(int)
    for trans in transactions:
        unique_items = sorted(list(set(trans)))
        frequent_trans_items = [item for item in unique_items if item in frequent_items]
        for pair in itertools.combinations(frequent_trans_items, 2):
            pair_counts[pair] += 1
            
    rules = []
    for pair, count in pair_counts.items():
        support_pair = count / num_transactions
        if support_pair >= min_support:
            support_1 = frequent_items[pair[0]]
            confidence_1_to_2 = support_pair / support_1
            if confidence_1_to_2 >= min_confidence:
                rules.append({
                    'antecedent': pair[0],
                    'consequent': pair[1],
                    'support': float(support_pair),
                    'confidence': float(confidence_1_to_2)
                })
                
            support_2 = frequent_items[pair[1]]
            confidence_2_to_1 = support_pair / support_2
            if confidence_2_to_1 >= min_confidence:
                rules.append({
                    'antecedent': pair[1],
                    'consequent': pair[0],
                    'support': float(support_pair),
                    'confidence': float(confidence_2_to_1)
                })
                
    return rules
