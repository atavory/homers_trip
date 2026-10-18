"""Exact detection-count example for the introduction's fixed flag-check policy."""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def assess(truth, flags):
    pairs = [(y, p) for ys, ps in zip(truth, flags) for y, p in zip(ys, ps)]
    tp = sum(y and p for y, p in pairs)
    fp = sum(not y and p for y, p in pairs)
    fn = sum(y and not p for y, p in pairs)
    tn = sum(not y and not p for y, p in pairs)
    residuals = [sum(y and not p for y, p in zip(ys, ps)) for ys, ps in zip(truth, flags)]
    checks = [sum(ps) for ps in flags]
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': Fraction(tp, tp + fp), 'recall': Fraction(tp, tp + fn),
            'f1': Fraction(2 * tp, 2 * tp + fp + fn),
            'residual_errors': residuals, 'checks': checks,
            'completed_responses': sum(e == 0 for e in residuals),
            'responses_checked': sum(k > 0 for k in checks),
            'total_checks': sum(checks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    truth = ((1, 1, 0, 0), (1, 1, 0, 0))
    flags = {'A': ((1, 1, 0, 0), (0, 0, 0, 0)),
             'B': ((1, 0, 0, 0), (1, 0, 0, 0))}
    results = {name: assess(truth, selected) for name, selected in flags.items()}
    for key in ('tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'f1', 'total_checks'):
        assert results['A'][key] == results['B'][key]
    assert (results['A']['tp'], results['A']['fn'], results['A']['f1']) == (2, 2, Fraction(2, 3))
    assert results['A']['completed_responses'] == 1 and results['B']['completed_responses'] == 0
    assert results['A']['responses_checked'] == 1 and results['B']['responses_checked'] == 2
    values = {'tp': results['A']['tp'], 'fn': results['A']['fn'],
              'completed_A': results['A']['completed_responses'],
              'completed_B': results['B']['completed_responses']}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'values.csv').open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['quantity', 'value'])
        writer.writerows(values.items())
    tex = [r'\newcommand{\SleekIntroValue}[1]{\csname si@#1\endcsname}']
    tex += [r'\expandafter\def\csname si@' + k + r'\endcsname{' + str(v) + '}' for k, v in values.items()]
    (args.out / 'values.tex').write_text('\n'.join(tex) + '\n')
    encoded = {name: {k: str(v) if isinstance(v, Fraction) else v for k, v in item.items()}
               for name, item in results.items()}
    record = {'inputs': {'truth': truth, 'flags': flags}, 'results': encoded,
              'all_checks_passed': True,
              'scope': 'Verify every flag and correct its marked unit perfectly. No regeneration, free policy optimization, or claim of a strict score-ordering reversal.',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'command': 'python3 analysis/sleek_detection_credit_example.py --out analysis/sleek_detection_credit_example_results',
              'outputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [args.out / 'values.csv', args.out / 'values.tex']}}
    (args.out / 'verification.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'all_checks_passed': True, 'same_pooled_f1': str(results['A']['f1']),
                      'completed_responses': [results['A']['completed_responses'], results['B']['completed_responses']]}))


if __name__ == '__main__':
    main()
