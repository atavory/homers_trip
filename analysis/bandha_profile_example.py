"""Exact analytic outcomes and profile reconstruction for the paper example."""
import argparse
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import hashlib
import json


def profiles(rows):
    errors, work = defaultdict(Fraction), defaultdict(Fraction)
    weight = Fraction(1, len(rows))
    for row in rows:
        errors[row['errors']] += weight
        work[row['checks']] += weight
    return dict(errors), dict(work)


def pgf(hist, x):
    return sum(weight * x ** value for value, weight in hist.items())


def mean(hist):
    return sum(value * weight for value, weight in hist.items())


def value(rows, a, b, shared, continuing):
    return sum(a ** r['errors'] - shared * (1 - b ** r['checks'])
               - continuing * r['checks'] for r in rows) / len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    labels = [(1, 1, 0, 0), (1, 1, 0, 0)]
    selections = {'A': [(1, 1, 0, 0), (0, 0, 0, 0)],
                  'B': [(1, 0, 0, 0), (1, 0, 0, 0)]}
    methods = {}
    for name, flags in selections.items():
        rows = [{'errors': sum(y * (1 - d) for y, d in zip(ys, ds)),
                 'checks': sum(ds)} for ys, ds in zip(labels, flags)]
        methods[name] = {'rows': rows, 'profiles': profiles(rows)}
    ea, ka = methods['A']['profiles']
    eb, kb = methods['B']['profiles']
    assert mean(ea) == mean(eb) == mean(ka) == mean(kb) == 1
    coefficients = {i: ea.get(i, 0) - eb.get(i, 0) for i in range(3)}
    assert coefficients == {0: Fraction(1, 2), 1: -1, 2: Fraction(1, 2)}
    assert {i: ka.get(i, 0) - kb.get(i, 0) for i in range(3)} == coefficients
    tested = 0
    for a in [Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)]:
        for b in [Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)]:
            for shared in [Fraction(0), Fraction(1, 2), Fraction(2)]:
                for continuing in [Fraction(0), Fraction(1, 8), Fraction(1)]:
                    computed = {}
                    for name, data in methods.items():
                        errors, work = data['profiles']
                        calculated = pgf(errors, a) - shared * (1 - pgf(work, b)) - continuing * mean(work)
                        assert calculated == value(data['rows'], a, b, shared, continuing)
                        computed[name] = calculated
                    expected = ((1 - a) ** 2 + shared * (1 - b) ** 2) / 2
                    assert computed['A'] - computed['B'] == expected > 0
                    tested += 1
    same_endpoints_a = {0: Fraction(1, 2), 2: Fraction(1, 2)}
    same_endpoints_b = {0: Fraction(1, 2), 1: Fraction(1, 4), 3: Fraction(1, 4)}
    assert pgf(same_endpoints_a, 0) == pgf(same_endpoints_b, 0)
    assert mean(same_endpoints_a) == mean(same_endpoints_b)
    assert pgf(same_endpoints_a, Fraction(1, 2)) != pgf(same_endpoints_b, Fraction(1, 2))
    report = {'kind': 'exact_analytic_fixture', 'all_checks_passed': True,
              'parameter_cases': tested, 'rows': {name: d['rows'] for name, d in methods.items()},
              'difference_coefficients': {str(k): str(v) for k, v in coefficients.items()},
              'difference_formula': '((1-a)^2+F*(1-b)^2)/2',
              'same_mean_and_zero_endpoint_do_not_determine_profile': True,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    values = []
    for name, data in methods.items():
        for suffix, field in [('Errors', 'errors'), ('Checks', 'checks')]:
            vals = ','.join(str(r[field]) for r in data['rows'])
            values.append(r'\newcommand{\ProfileExample' + name + suffix + '}{(' + vals + ')}')
    (args.out / 'values.tex').write_text('\n'.join(values) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
