import nltk
import math
from collections.abc import Iterable
from typing import Any


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
