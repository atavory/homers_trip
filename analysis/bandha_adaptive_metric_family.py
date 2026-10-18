"""Verify a metric-general projection family with freely chosen paid review."""

import argparse
from collections import Counter
import csv
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path


def loss(e):
    return 1 - F(1, 2) ** e


def cost(k):
    return F(1, 2) * (1 - F(1, 2) ** k) + F(k, 8)


@lru_cache(None)
def blind(k, e, n):
    if not e:
        return F(0)
    p = F(e, n)
    query = cost(k + 1) - cost(k) + p * blind(k + 1, e - 1, n - 1)
    if e < n:
        query += (1 - p) * blind(k + 1, e, n - 1)
    return min(loss(e), query)


def encode(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    return value


def measures(labels, scores):
    positive = [s for row, ys in zip(scores, labels) for s, y in zip(row, ys) if y]
    negative = [s for row, ys in zip(scores, labels) for s, y in zip(row, ys) if not y]
    auc = sum((F(p > n) + F(p == n, 2) for p in positive for n in negative), F()) / (len(positive) * len(negative))
    brier = sum(((s - y) ** 2 for row, ys in zip(scores, labels) for s, y in zip(row, ys)), F()) / sum(map(len, scores))
    best_f1 = F(0)
    ap = F(0)
    tp = fp = 0
    for level in sorted(set(positive + negative), reverse=True):
        added = sum(p == level for p in positive)
        tp += added
        fp += sum(n == level for n in negative)
        ap += F(added * tp, len(positive) * (tp + fp))
        best_f1 = max(best_f1, F(2 * tp, len(positive) + tp + fp))
    return {'auc': auc, 'ap': ap, 'best_f1': best_f1, 'brier': brier}


def family(m):
    a, z, b = F(9, 10), F(1, 2), F(1, 10)
    labels = ((1,) * (m - 1) + (0,), (1,) + (0,) * (m - 1))
    scores = ((z,) * (m - 1) + (b,), (a,) + (z,) * (m - 1))
    means = tuple(sum(row) / m for row in scores)
    shifts = (a - means[0], b - means[1])
    projected = ((a,) * m, (b,) * m)
    law = Counter((s, y) for row, ys in zip(scores, labels) for s, y in zip(row, ys))
    shifted_law = Counter((s, y) for row, ys in zip(projected, labels) for s, y in zip(row, ys))
    expected = Counter({(z, 1): m - 1, (z, 0): m - 1, (a, 1): 1, (b, 0): 1})
    expected_shifted = Counter({(a, 1): m - 1, (b, 0): m - 1, (b, 1): 1, (a, 0): 1})
    assert law == expected and shifted_law == expected_shifted
    original_metrics = measures(labels, scores)
    projected_metrics = measures(labels, projected)
    assert projected_metrics['auc'] > original_metrics['auc']
    assert projected_metrics['ap'] > original_metrics['ap']
    assert projected_metrics['best_f1'] > original_metrics['best_f1']
    assert projected_metrics['brier'] < original_metrics['brier']
    known = [min(loss(e - k) + cost(k) for k in range(e + 1)) for e in (m - 1, 1)]
    coarse = [blind(0, e, m) for e in (m - 1, 1)]
    assert known == [loss(m - 1), cost(1)]
    assert coarse == [loss(m - 1), loss(1)]
    original_risk = sum(known) / 2
    projected_risk = sum(coarse) / 2
    assert projected_risk - original_risk == F(1, 16)
    for stop_after in range(1, m + 1):
        hit_probability = F(stop_after, m)
        expected_queries = F(stop_after * (2 * m - stop_after + 1), 2 * m)
        assert expected_queries / hit_probability >= F(m + 1, 2)
        assert expected_queries / 8 > loss(1) * hit_probability
    return {'length': m, 'labels': labels, 'scores': scores, 'offsets': shifts,
            'shifted_projected_scores': projected, 'original_metrics': original_metrics,
            'shifted_projected_metrics': projected_metrics,
            'original_optimal_risk': original_risk,
            'shifted_projected_optimal_risk': projected_risk,
            'risk_gap': projected_risk - original_risk,
            'single_error_minimum_queries_per_success': F(m + 1, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    records = [family(m) for m in range(8, 25)]
    result = {'all_checks_passed': True, 'records': records,
              'loss': '1-2^(-e)', 'cost': '(1-2^(-k))/2+k/8',
              'metric_assumptions': 'Strict preference for balanced separated labels over balanced tied scores, and finite-support mixture continuity at those laws.',
              'generic_laws': ['mu_s=(1-1/m)*mu_tied+(1/m)*mu_separated',
                              'mu_OcTs=(1-1/m)*mu_separated+(1/m)*mu_flipped'],
              'scope': 'Known initial counts, anonymous uniformly permuted units, paid perfect verification, equal score-observation cost, no quota or common cutoff.',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'verification.json').write_text(json.dumps(encode(result), indent=2) + '\n')
    with (args.out / 'family.csv').open('w', newline='') as stream:
        fields = ['length', 'original_auc', 'projected_auc', 'original_best_f1', 'projected_best_f1',
                  'original_risk', 'projected_risk', 'gap']
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for r in records:
            writer.writerow({'length': r['length'], 'original_auc': str(r['original_metrics']['auc']),
                             'projected_auc': str(r['shifted_projected_metrics']['auc']),
                             'original_best_f1': str(r['original_metrics']['best_f1']),
                             'projected_best_f1': str(r['shifted_projected_metrics']['best_f1']),
                             'original_risk': str(r['original_optimal_risk']),
                             'projected_risk': str(r['shifted_projected_optimal_risk']),
                             'gap': str(r['risk_gap'])})
    print(json.dumps({'checked_lengths': [8, 24], 'risk_gap': '1/16', 'checks': True}))


if __name__ == '__main__':
    main()
