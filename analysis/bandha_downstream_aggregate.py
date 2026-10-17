"""Generate manuscript values from the checked downstream-loss fixtures."""

import argparse
import csv
import hashlib
import json
from fractions import Fraction
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.input.read_text())
    assert record['all_checks_passed']
    fixture = next(x for x in record['fixtures'] if x['horizon'] == 5 and x['marked_count'] == 4)
    values = {'fixture.horizon': Fraction(fixture['horizon']), 'fixture.marked': Fraction(fixture['marked_count'])}
    for name, policy in fixture['policies'].items():
        for key in ['tp', 'fp', 'f1', 'auc', 'ap']:
            values['fixture.' + name + '.' + key] = Fraction(policy['frozen'][key])
        for mode in ['feedback', 'feedback_disabled']:
            for key, value in policy[mode]['summary'].items():
                values['fixture.' + name + '.' + mode + '.' + key] = Fraction(value)
    calibrated = record['calibrated_checkpoint']
    for name, metrics in calibrated['metrics'].items():
        for key in ['f1', 'accuracy', 'auc', 'brier']:
            values['calibrated.' + name + '.' + key] = Fraction(metrics[key])
    comparison = next(x for x in calibrated['continuation_gains'] if x['horizon'] == 20 and Fraction(x['override_cost']) == 0)
    for name, value in comparison['net_improvement_over_keep'].items():
        values['calibrated.' + name + '.control_gain'] = Fraction(value)
    assert values['calibrated.refined.f1'] > values['calibrated.coarse.f1']
    assert values['calibrated.refined.brier'] < values['calibrated.coarse.brier']
    assert values['calibrated.refined.control_gain'] < 0 < values['calibrated.coarse.control_gain']
    assert values['fixture.offset.feedback.errors'] > values['fixture.detector.feedback.errors']
    assert values['fixture.offset.feedback.edits'] == values['fixture.detector.feedback.edits']
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'values.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['key', 'exact', 'decimal'])
        writer.writerows((key, str(value), float(value)) for key, value in values.items())
    lines = [r'\providecommand{\HomerDownstreamValue}[1]{\csname hd@#1\endcsname}']
    for key, value in values.items():
        tex = str(value.numerator) if value.denominator == 1 else rf'\frac{{{value.numerator}}}{{{value.denominator}}}'
        lines.append(rf'\expandafter\def\csname hd@{key}\endcsname{{{tex}}}')
    (args.out / 'values.tex').write_text('\n'.join(lines) + '\n')
    result = {
        'all_checks_passed': True, 'kind': 'checked_fixture_value_binding',
        'input': str(args.input), 'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'values': {key: str(value) for key, value in values.items()},
        'fixture_scope': record['scope'],
        'calibrated_scope': calibrated['scope'],
    }
    (args.out / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'all_checks_passed': True, 'value_count': len(values), 'control_gain': {name: str(values['calibrated.' + name + '.control_gain']) for name in ['coarse', 'refined']}}))


if __name__ == '__main__':
    main()
