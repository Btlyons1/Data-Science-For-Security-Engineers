from detection_baseline.behavioral import (
    sequence_anomaly_score,
    jaccard,
    shannon_entropy,
    detect_sequence_anomalies,
    detect_hmm_anomalies,
    detect_pagerank_spikes,
    find_association_rules
)
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


def test_detect_sequence_anomalies():
    train = [
        ['login', 'read', 'logout'],
        ['login', 'read', 'write', 'logout'],
        ['login', 'write', 'logout']
    ]
    test_normal = ['login', 'read', 'logout']
    test_anom = ['login', 'admin', 'logout']
    
    res_normal = detect_sequence_anomalies(train, test_normal, threshold=0.01)
    assert res_normal['is_anomaly'] == False
    
    res_anom = detect_sequence_anomalies(train, test_anom, threshold=0.01)
    assert res_anom['is_anomaly'] == True
    assert len(res_anom['anomalies']) > 0


def test_detect_hmm_anomalies():
    states = ['Normal', 'Compromised']
    start_prob = {'Normal': 0.9, 'Compromised': 0.1}
    trans_prob = {
        'Normal': {'Normal': 0.95, 'Compromised': 0.05},
        'Compromised': {'Normal': 0.1, 'Compromised': 0.9}
    }
    emit_prob = {
        'Normal': {'login': 0.7, 'read': 0.2999, 'malware': 0.0001},
        'Compromised': {'login': 0.8, 'read': 0.1999, 'malware': 0.0001}
    }
    
    # Highly normal observations
    obs_normal = ['login', 'read', 'read']
    res_normal = detect_hmm_anomalies(obs_normal, states, start_prob, trans_prob, emit_prob, threshold=-10.0)
    assert res_normal['is_anomaly'] == False
    
    # Highly anomalous observations
    obs_anom = ['malware', 'malware', 'malware']
    res_anom = detect_hmm_anomalies(obs_anom, states, start_prob, trans_prob, emit_prob, threshold=-10.0)
    assert res_anom['is_anomaly'] == True


def test_detect_pagerank_spikes():
    # Simple hub-and-spoke graph: node 0 is the center hub, nodes 1-10 connect to it
    edges = [(0, i) for i in range(1, 11)]
    res = detect_pagerank_spikes(edges, threshold=2.0)
    assert res['is_anomaly'] == True
    # Node 0 has high centrality spike
    assert any(anom['node'] == 0 for anom in res['anomalies'])


def test_find_association_rules():
    transactions = [
        ['ShareA', 'ShareB', 'ShareC'],
        ['ShareA', 'ShareB', 'ShareC'],
        ['ShareA', 'ShareB'],
        ['ShareA', 'ShareC'],
        ['ShareB', 'ShareC']
    ]
    rules = find_association_rules(transactions, min_support=0.3, min_confidence=0.6)
    assert len(rules) > 0
    # Rule should be: ShareA -> ShareB (since support_pair is 3/5=0.6, support_A is 4/5=0.8, confidence is 0.75)
    # Antecedent can be ShareA, consequent ShareB
    has_rule = False
    for r in rules:
        if r['antecedent'] == 'ShareA' and r['consequent'] == 'ShareB':
            has_rule = True
            assert abs(r['confidence'] - 0.75) < 1e-5
    assert has_rule == True
