"""Exact pair-count checks for the response-size pooling construction."""

import argparse
from fractions import Fraction
from itertools import product
import json
from pathlib import Path


def auc(positive, negative):
    wins = sum(p > n for p in positive for n in negative)
    ties = sum(p == n for p in positive for n in negative)
    return Fraction(2 * wins + ties, 2 * len(positive) * len(negative))


def check(counts):
    p1, n1, p2, n2 = counts
    denominator = (p1 + p2) * (n1 + n2)
    full = auc([1] * p1 + [3] * p2, [0] * n1 + [2] * n2)
    constant = auc([2] * p1 + [1] * p2, [2] * n1 + [1] * n2)
    assert full == 1 - Fraction(p1 * n2, denominator)
    assert constant == Fraction(2 * p1 * n2 + p1 * n1 + p2 * n2, 2 * denominator)
    assert auc([1] * p1, [0] * n1) == auc([3] * p2, [2] * n2) == 1
    assert auc([2] * p1, [2] * n1) == auc([1] * p2, [1] * n2) == Fraction(1, 2)
    return full, constant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    general_cases = 0
    for counts in product(range(1, 5), repeat=4):
        check(counts)
        general_cases += 1
    symmetric = []
    for m in range(2, 101):
        full, constant = check((m - 1, 1, 1, m - 1))
        assert full == Fraction(2 * m - 1, m * m)
        assert constant == Fraction(m - 1, m)
        assert (constant > full) == (m >= 3)
        assert (full < Fraction(1, 2)) == (m >= 4)
        symmetric.append({'m': m, 'total_items': 2 * m,
                          'pooled_full': str(full), 'pooled_constant': str(constant),
                          'within_pair_fraction': str(Fraction(2 * (m - 1), m * m))})
    example = next(row for row in symmetric if row['m'] == 10)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'all_checks_passed': True, 'general_cases': general_cases,
              'symmetric_cases': len(symmetric), 'example': example,
              'symmetric_results': symmetric,
              'scope': 'Exact analytic pair counts with half credit for ties; no fitted detector or correction outcome.'}
    (args.out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = [r'\newcommand{\ParametricPoolingExampleSize}{10}',
             r'\newcommand{\ParametricPoolingExampleA}{' + f"{float(Fraction(example['pooled_full'])):.2f}" + '}',
             r'\newcommand{\ParametricPoolingExampleB}{' + f"{float(Fraction(example['pooled_constant'])):.2f}" + '}']
    (args.out / 'values.tex').write_text('\n'.join(lines) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'symmetric_results'}))


if __name__ == '__main__':
    main()
