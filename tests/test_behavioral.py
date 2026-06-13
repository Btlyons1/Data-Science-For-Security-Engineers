from detection_baseline.behavioral import sequence_anomaly_score, jaccard, shannon_entropy
import nltk

def test_sequence_anomaly_score():
    # Make sure nltk resources are downloaded or mock them if needed, but let's test normally
    admin_seq = ['login', 'read', 'logout']
    attacker_seq = ['login', 'admin', 'logout']
    
    known_bigrams = list(nltk.bigrams(admin_seq))
    score = sequence_anomaly_score(attacker_seq, known_bigrams)
    # attacker_seq has ('login', 'admin') and ('admin', 'logout')
    # neither is in known_bigrams ('login', 'read'), ('read', 'logout')
    # Score should be 1.0 (all 2 transitions are unknown)
    assert score == 1.0

def test_jaccard():
    seq1 = ['a', 'b', 'c']
    seq2 = ['b', 'c', 'd']
    # intersection: {'b', 'c'} (size 2)
    # union: {'a', 'b', 'c', 'd'} (size 4)
    # similarity: 0.5
    assert jaccard(seq1, seq2) == 0.5

def test_shannon_entropy():
    # Repetitive string: 0
    assert shannon_entropy('aaaa') == 0.0
    
    # 2 unique characters with equal probability: 1.0
    assert shannon_entropy('abab') == 1.0
    
    # 4 unique characters with equal probability: 2.0
    assert shannon_entropy('abcd') == 2.0
