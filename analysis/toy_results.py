"""Compute the constructed examples and render their vector diagram."""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def auc(rows):
    pos = [score for row in rows for score, label in row if label]
    neg = [score for row in rows for score, label in row if not label]
    return Fraction(sum(2 * (a > b) + (a == b) for a in pos for b in neg), 2 * len(pos) * len(neg))


def compute(config):
    count = config['tokens_per_response']
    a, b = [], []
    for positives, base in zip(config['positive_counts'], config['baseline_scores']):
        assert 0 < positives < count
        a.append([(base + config['positive_offset'], 1)] * positives + [(base, 0)] * (count - positives))
        b.append([(config['constant_score_ceiling'] - base, label) for _, label in a[-1]])
    def mean(rows, label):
        vals = [s for row in rows for s, y in row if y == label]
        return Fraction(sum(vals), len(vals))
    values = {'responses': len(a), 'tokens_per_response': count, 'pooled_auc_A': float(auc(a)), 'pooled_auc_B': float(auc(b)), 'within_auc_A': 1.0, 'within_auc_B': 0.5}
    fractions = {f'{label}_{name}': str(mean(rows, y)) for name, rows in [('A', a), ('B', b)] for label, y in [('positive_mean', 1), ('negative_mean', 0)]}
    f = config['affine_example'];m = f['tokens_per_response'];lo, mid, hi = [Fraction(str(f[k])) for k in ['low', 'middle', 'high']]
    a2 = [[(mid, 1)] * (m - 1) + [(lo, 0)], [(hi, 1)] + [(mid, 0)] * (m - 1)]
    b2 = [[(hi, y) for _, y in a2[0]], [(lo, y) for _, y in a2[1]]]
    for name, rows in [('A', a2), ('B', b2)]:
        values[f'affine_loss_{name}'] = float(sum((s - y) ** 2 for row in rows for s, y in row) / (2 * m))
        values[f'affine_accuracy_{name}'] = float(Fraction(sum((s >= mid) == bool(y) for row in rows for s, y in row), 2 * m))
    return values, fractions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'toy_cases.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args();args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text());values, fractions = compute(config)
    assert len(config['positive_counts']) == 6, 'This diagram uses a six-response layout.'
    template = (Path(__file__).parent / 'simpson_layout.tex').read_text()
    substitutions = {'ROWS': ','.join(f'{i}/{p}/{base}' for i, (p, base) in enumerate(zip(config['positive_counts'], config['baseline_scores']), 1)), 'POS_A': fractions['positive_mean_A'], 'NEG_A': fractions['negative_mean_A'], 'POS_B': fractions['positive_mean_B'], 'NEG_B': fractions['negative_mean_B'], 'CENTER_A': str((Fraction(fractions['positive_mean_A']) + Fraction(fractions['negative_mean_A'])) / 2), 'CENTER_B': str((Fraction(fractions['positive_mean_B']) + Fraction(fractions['negative_mean_B'])) / 2)}
    for key, value in substitutions.items():template = template.replace('@@' + key + '@@', value)
    assert '@@' not in template
    (args.out / 'simpson_reversal.tex').write_text(template)
    payload = {'values': values, 'exact_means': fractions, 'config': config, 'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out / 'results.json').write_text(json.dumps(payload, indent=2) + '\n')
    with (args.out / 'results.csv').open('w', newline='') as stream:
        writer = csv.writer(stream);writer.writerow(['quantity', 'value']);writer.writerows(values.items())
    print(json.dumps(values))


if __name__ == '__main__':main()
