"""Evaluate preserved token predictions, including eligibility-mask controls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import expanded as core

REQUIRED = {'scorer', 'response_id', 'source_id', 'token_index', 'label', 'score', 'eligible'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with args.predictions.open(newline='') as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f'Missing CSV columns: {sorted(missing)}')
        groups = {}
        for row in reader:
            key = (row['scorer'], row['response_id'], row['source_id'])
            label, eligible = int(row['label']), int(row['eligible'])
            if label not in [0, 1] or eligible not in [0, 1]:
                raise SystemExit('Labels and eligibility must be binary.')
            groups.setdefault(key, []).append((int(row['token_index']), label, float(row['score']), eligible))
    by_model = {}
    for (model, response, source), tokens in groups.items():
        tokens.sort()
        if len({t[0] for t in tokens}) != len(tokens):
            raise SystemExit(f'Duplicate token indices: {model}, {response}')
        selected = [t for t in tokens if t[3]]
        if not selected:
            continue
        y = np.array([t[1] for t in selected], dtype=np.int8)
        score = np.array([t[2] for t in selected], dtype=float)
        if not np.isfinite(score).all():
            raise SystemExit('Scores must be finite.')
        by_model.setdefault(model, []).append((response, source, y, score, float(np.mean([t[2] for t in tokens]))))
    result = {'input_sha256': hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'tie_rule': 'half credit for pair ties; response means over explicit eligible or all-scored rows', 'models': {}}
    rows = []
    for model, records in sorted(by_model.items()):
        output = {}
        for transformation in ['original', 'eligible_mean', 'all_scored_mean']:
            scores = [s if transformation == 'original' else np.full(len(s), np.mean(s) if transformation == 'eligible_mean' else all_mean)
                      for _, _, _, s, all_mean in records]
            local = [core.rank_metrics(y, s) for (_, _, y, _, _), s in zip(records, scores)]
            mixed = [a for a in local if a['P'] and a['N']]
            y = np.concatenate([a[2] for a in records]);flat = np.concatenate(scores)
            pooled = core.rank_metrics(y, flat)
            W = sum(a['P'] * a['N'] for a in local)
            within_wins = sum(a['wins'] for a in mixed)
            C = pooled['P'] * pooled['N'] - W
            values = {'responses': len(records), 'sources': len({a[1] for a in records}), 'mixed': len(mixed),
                      'positives': pooled['P'], 'negatives': pooled['N'], 'within_pairs': W, 'cross_pairs': C,
                      'within_pair_weight': W / (W + C) if W + C else None,
                      'pooled_auc': pooled['auc'], 'pooled_ap': pooled['ap'],
                      'within_pair_auc': within_wins / W if W else None,
                      'within_macro_auc': np.mean([a['auc'] for a in mixed]).item() if mixed else None,
                      'cross_auc': (pooled['wins'] - within_wins) / C if C else None}
            output[transformation] = values
            rows.extend([model, transformation, key, value] for key, value in values.items())
        result['models'][model] = output
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.out / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream);writer.writerow(['scorer', 'transformation', 'metric', 'value']);writer.writerows(rows)
    print(json.dumps({'models': list(by_model), 'output': str(args.out)}, indent=2))


if __name__ == '__main__':
    main()
