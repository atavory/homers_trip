"""Check the paper's finite constructions using exact arithmetic."""
import argparse
from collections import Counter
from fractions import Fraction as F
import itertools
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    checks = []
    for R in range(2, 8):
        positives = [F(2*r+1, 2*(R+1)) for r in range(1, R+1)]
        negatives = [F(r, R+1) for r in range(1, R+1)]
        reference = Counter([(s, 1) for s in positives] + [(s, 0) for s in negatives])
        successes = set()
        for order in itertools.permutations(range(R)):
            shuffled = [negatives[i] for i in order]
            assert Counter([(s, 1) for s in positives] + [(s, 0) for s in shuffled]) == reference
            successes.add(sum(p > n for p, n in zip(positives, shuffled)))
        assert successes == set(range(1, R+1))
        checks.append({'responses': R, 'pooled_measure_invariant': True,
                       'minimum_one_token_recovery': str(F(min(successes), R)),
                       'maximum_one_token_recovery': str(F(max(successes), R))})
    m = 10
    A = [[(F(1, 2), 1)]*(m-1) + [(F(0), 0)], [(F(1), 1)] + [(F(1, 2), 0)]*(m-1)]
    B = [[(F(1), y) for _, y in A[0]], [(F(0), y) for _, y in A[1]]]
    def loss(rows):
        return sum((s-y)**2 for row in rows for s, y in row) / (2*m)
    def accuracy(rows):
        return F(sum((s >= F(1, 2)) == bool(y) for row in rows for s, y in row), 2*m)
    assert loss(A) == F(9, 40) and loss(B) == F(1, 10)
    assert accuracy(A) == F(11, 20) and accuracy(B) == F(9, 10)
    for r in range(2):
        P = sum(y for _, y in A[r])
        for k in range(1, m):
            assert min(k, P) > F(k*P, m)
    result = {'status': 'PASS', 'rearrangement_checks': checks,
              'affine_example': {'squared_error_A': str(loss(A)), 'squared_error_B': str(loss(B)),
                                 'accuracy_A': str(accuracy(A)), 'accuracy_B': str(accuracy(B))}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
