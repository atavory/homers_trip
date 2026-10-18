"""Exact common-threshold classification improvement with worse grouped risk."""

import argparse
import csv
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path


LABELS = ((1, 1, 1, 0), (1, 1, 1, 0))
SCORES = (tuple(F(i, 10) for i in (9, 8, 7, 6)),
          tuple(F(i, 10) for i in (4, 3, 2, 1)))
OFFSETS = (F(-1, 5), F(3, 10))


def encode(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {key: encode(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


def readout(scores, threshold):
    selected = [[score >= threshold for score in row] for row in scores]
    recovered = [sum(y for y, flag in zip(labels, flags) if flag)
                 for labels, flags in zip(LABELS, selected)]
    residual = [sum(labels) - count for labels, count in zip(LABELS, recovered)]
    tp = sum(recovered)
    fp = sum(sum(flags) for flags in selected) - tp
    positives = [score for row, labels in zip(scores, LABELS)
                 for score, y in zip(row, labels) if y]
    negatives = [score for row, labels in zip(scores, LABELS)
                 for score, y in zip(row, labels) if not y]
    auc = sum((F(p > n) + F(p == n, 2) for p in positives for n in negatives), F()) / 12
    levels = sorted(set(positives + negatives), reverse=True)
    cumulative_p = cumulative_n = 0
    ap = F()
    for level in levels:
        mass = sum(score == level for score in positives)
        cumulative_p += mass
        cumulative_n += sum(score == level for score in negatives)
        ap += mass * F(cumulative_p, cumulative_p + cumulative_n) / 6
    brier = sum(((score - y) ** 2 for row, labels in zip(scores, LABELS)
                 for score, y in zip(row, labels)), F()) / 8
    return {
        'scores': scores, 'selected': selected, 'recovered_by_response': recovered,
        'residual_by_response': residual, 'tp': tp, 'fp': fp, 'fn': 6 - tp,
        'tn': 2 - fp, 'inspection_count': tp + fp,
        'reviewed_responses': sum(any(flags) for flags in selected),
        'precision': F(tp, tp + fp), 'recall': F(tp, 6),
        'f1': F(2 * tp, 6 + tp + fp), 'accuracy': F(tp + 2 - fp, 8),
        'auc': auc, 'ap': ap, 'brier': brier,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    shifted = tuple(tuple(score + offset for score in row)
                    for row, offset in zip(SCORES, OFFSETS))
    cases = {'original': readout(SCORES, F(11, 20)),
             'offset': readout(shifted, F(11, 20))}
    a, b = cases['original'], cases['offset']
    assert a['residual_by_response'] == [0, 3]
    assert b['residual_by_response'] == [1, 1]
    assert (a['tp'], a['fp'], b['tp'], b['fp']) == (3, 1, 4, 0)
    assert a['inspection_count'] == b['inspection_count'] == 4
    assert (a['reviewed_responses'], b['reviewed_responses']) == (1, 2)
    assert (a['auc'], b['auc']) == (F(3, 4), F(1))
    for metric in ('precision', 'recall', 'f1', 'accuracy', 'auc', 'ap'):
        assert b[metric] > a[metric]
    assert b['brier'] < a['brier']
    for threshold in (F(i, 100) for i in range(51, 60)):
        assert readout(SCORES, threshold)['selected'] == a['selected']
        assert readout(shifted, threshold)['selected'] == b['selected']
    for original, changed in zip(SCORES, shifted):
        assert all((original[i] - original[j]) == (changed[i] - changed[j])
                   for i in range(4) for j in range(4))

    risk_cases = []
    for saturation in (F(i, 20) for i in range(1, 20)):
        for setup in (F(0), F(1, 10), F(1), F(10)):
            for marginal in (F(0), F(1, 10), F(1)):
                values = {}
                for name, record in cases.items():
                    loss = sum((1 - saturation ** e for e in record['residual_by_response']), F()) / 2
                    cost = F(setup * record['reviewed_responses'] +
                             marginal * record['inspection_count'], 2)
                    values[name] = {'loss': loss, 'cost': cost, 'risk': loss + cost}
                difference = values['offset']['risk'] - values['original']['risk']
                expected = (1 - 2 * saturation + saturation ** 3 + setup) / 2
                assert difference == expected
                if saturation <= F(1, 2):
                    assert difference > 0
                risk_cases.append({'a': saturation, 'setup': setup, 'marginal': marginal,
                                   'models': values, 'difference': difference})
    result = {
        'all_checks_passed': True, 'labels': LABELS, 'offsets': OFFSETS,
        'threshold': F(11, 20), 'valid_threshold_interval': '(1/2,3/5)',
        'models': cases, 'risk_cases': risk_cases, 'checked_risk_cases': len(risk_cases),
        'risk_difference': '(2*L(1)-L(3)+F)/2',
        'smooth_risk_difference': '(1-2*a+a^3+F)/2',
        'assumptions': ['Selected erroneous units are correctly repaired; selected correct units are preserved.',
                        'Unselected units remain unchanged; error count is the stipulated response-loss input.',
                        'The common threshold invokes review on each response with any selected unit.',
                        'Setup and marginal inspection prices are common across the two equal-length responses.'],
        'scope': 'Exact constructed policy comparison; no fitted utility curve or real repair-system outcome.',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'results.json').write_text(json.dumps(encode(result), indent=2) + '\n')
    with (args.out / 'classification.csv').open('w', newline='') as stream:
        fields = ['model', 'tp', 'fp', 'fn', 'tn', 'f1', 'accuracy', 'auc', 'ap',
                  'brier', 'inspection_count', 'reviewed_responses']
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for name, record in cases.items():
            writer.writerow({'model': name, **{key: encode(record[key]) for key in fields[1:]}})
    print(json.dumps({'checks': True, 'risk_cases': len(risk_cases),
                      'models': {name: {key: encode(row[key]) for key in ('tp', 'fp', 'residual_by_response', 'f1')}
                                 for name, row in cases.items()}}))


if __name__ == '__main__':
    main()
