"""Standalone exact affine-loss/affine-work review reversal verification."""
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import argparse
import hashlib
import itertools
import json


@lru_cache(None)
def blind(e, n):
    if not e:
        return F(0)
    p = F(e, n)
    query = F(1, 2) + p * blind(e - 1, n - 1)
    if e < n:
        query += (1 - p) * blind(e, n - 1)
    return min(F(e), query)


@lru_cache(None)
def stopping_tree_returns(e, n):
    if not e:
        return frozenset((F(0),))
    p = F(e, n)
    positive = stopping_tree_returns(e - 1, n - 1)
    negative = stopping_tree_returns(e, n - 1) if e < n else (F(0),)
    returns = {F(e)}
    returns.update(F(1, 2) + p * first + (1 - p) * second
                   for first, second in itertools.product(positive, negative))
    return frozenset(returns)


def auc(labels, scores):
    positive = [s for ys, row in zip(labels, scores) for y, s in zip(ys, row) if y]
    negative = [s for ys, row in zip(labels, scores) for y, s in zip(ys, row) if not y]
    return sum(F(p > n) + F(p == n, 2) for p in positive for n in negative) / (len(positive) * len(negative))


def check():
    records = []
    for e in (1, 3):
        assert blind(e, 4) == min(stopping_tree_returns(e, 4))
    for m in range(4, 65):
        # Enumerate single-error cutoff returns independently of the recurrence.
        cutoffs = [F(1)] + [1 - F(t, m) + F(t * (2 * m - t + 1), 4 * m)
                            for t in range(1, m + 1)]
        assert blind(1, m) == min(cutoffs) == F(1)
        assert blind(m - 1, m) == F(m, 2) - F(1, 2 * m)
        # Known positions: enumerate every possible successful repair count.
        known_local = [min(F(e - k) + F(k, 2) for k in range(e + 1))
                       for e in (m - 1, 1)]
        known = sum(known_local) / 2
        coarse = (blind(m - 1, m) + blind(1, m)) / 2
        assert known == F(m, 4)
        assert coarse - known == F(1, 2) - F(1, 4 * m)
        labels = ((1,) * (m - 1) + (0,), (1,) + (0,) * (m - 1))
        a, z, b = F(9, 10), F(1, 2), F(1, 10)
        scores = ((z,) * (m - 1) + (b,), (a,) + (z,) * (m - 1))
        shifts = (a - sum(scores[0]) / m, b - sum(scores[1]) / m)
        shifted_projection = tuple((sum(row) / m + c,) * m for row, c in zip(scores, shifts))
        original_auc = auc(labels, scores)
        projected_auc = auc(labels, shifted_projection)
        assert original_auc == F(1, 2) + F(1, m) - F(1, 2 * m * m)
        assert projected_auc == 1 - F(1, m)
        assert projected_auc > original_auc
        records.append({'m': m, 'known_risk': str(known), 'projected_risk': str(coarse),
                        'risk_gap': str(coarse - known), 'original_auc': str(original_auc),
                        'projected_auc': str(projected_auc)})
    # Explicit hidden-placement enumeration verifies the high-error full search.
    placements = tuple(itertools.combinations(range(9), 8))
    placement_average = sum(F(max(a) + 1, 2) for a in placements) / len(placements)
    assert placement_average == blind(8, 9) == F(40, 9)
    return {'checks': True, 'utility': 'U(e)=-e', 'loss': 'L(e)=e', 'work': 'C(k)=k/2',
            'assumptions': ['Two responses with m-1 and 1 errors, each length m.',
                            'Independent uniform hidden label-score pair permutations.',
                            'Common exact initial counts and known model; no other locator.',
                            'Paid perfect unit verification and repair; unchanged unchecked units.',
                            'Uniform response weights; separable work and loss.'],
            'verification': ['Exact independent Bellman recursion, m=4..64.',
                             'Deterministic stopping-tree return enumeration, m=4.',
                             'Single-error deterministic cutoff enumeration, m=4..64.',
                             'Explicit high-error hidden-placement enumeration, m=9.',
                             'Direct pooled AUROC pair enumeration, m=4..64.'],
            'analytic_formulas': {'known_risk': 'm/4',
                                  'projected_risk': 'm/4+1/2-1/(4m)',
                                  'risk_gap': '1/2-1/(4m)',
                                  'original_auc': '1/2+1/m-1/(2m^2)',
                                  'projected_auc': '1-1/m'},
            'records': records,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = check()
    output = args.out / 'verification.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'checks': True, 'lengths': [4, 64],
                      'm9': next(r for r in result['records'] if r['m'] == 9),
                      'output': str(output)}))
