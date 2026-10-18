"""Grouped annotation-loss and review-volume readouts at frozen thresholds."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import expanded as core


SATURATIONS = (0.25, 0.5, 0.75)
METRICS = ('remaining_words', 'review_rate', 'context_words', 'selected_words',
           'loss_0.25', 'loss_0.5', 'loss_0.75')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ragtruth', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--source-ids', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    sources = json.loads(args.source_ids.read_text())
    weights = np.load(args.weights)
    assert sha(args.ragtruth / 'dataset/response.jsonl') == baseline['response_sha256']
    assert sha(args.ragtruth / 'dataset/source_info.jsonl') == baseline['source_sha256']
    assert hashlib.sha256(weights.tobytes()).hexdigest() == baseline['bootstrap']['weights_sha256']
    rows = core.parse(args.ragtruth)
    score, fit = core.fit_models(rows)
    assert fit == baseline['fit']
    test = [r for r in rows if r['split'] == 'test' and r['quality'] == 'good' and len(r['y'])]
    assert sorted({r['source_id'] for r in test}) == sources
    source_index = {source: i for i, source in enumerate(sources)}
    counts = np.zeros(len(sources), dtype=int)
    for row in test:
        counts[source_index[row['source_id']]] += 1
    denominator = weights @ counts
    assert np.all(denominator > 0)
    summaries, per_response, estimates = {}, [], {}
    for model in core.MODELS:
        threshold = baseline['models'][model]['threshold']
        arrays = [score(row, model) for row in test]
        pooled = core.rank_metrics(np.concatenate([r['y'] for r in test]), np.concatenate(arrays))
        assert abs(pooled['auc'] - baseline['models'][model]['pooled_auc']) < 1e-12
        atoms = np.zeros((len(sources), len(METRICS)))
        tp = fp = 0
        for row, scores in zip(test, arrays):
            selected = scores >= threshold
            found = int(row['y'][selected].sum())
            residual = int(row['y'].sum()) - found
            inspected = int(selected.sum())
            visited = int(inspected > 0)
            n = len(row['y'])
            values = [residual, visited, n * visited, inspected]
            values.extend(1 - saturation ** residual for saturation in SATURATIONS)
            atoms[source_index[row['source_id']]] += values
            tp += found; fp += inspected - found
            per_response.append({'model': model, 'response_id': row['id'],
                                 'source_id': row['source_id'], 'words': n,
                                 'positive_words': int(row['y'].sum()),
                                 **dict(zip(METRICS, values))})
        f1 = 2 * tp / (sum(int(row['y'].sum()) for row in test) + tp + fp)
        assert abs(f1 - baseline['models'][model]['pooled_f1']) < 1e-12
        point = atoms.sum(axis=0) / len(test)
        boot = (weights @ atoms) / denominator[:, None]
        summaries[model] = {'threshold': threshold, 'pooled_f1': f1,
                            'pooled_auc': pooled['auc'],
                            'point': dict(zip(METRICS, map(float, point))),
                            'confidence95': {key: core.ci(boot[:, i]) for i, key in enumerate(METRICS)}}
        estimates[model] = boot
    contrasts = {}
    for first, second, name in [('lexical_metadata', 'lexical', 'offset'),
                                ('task_generator', 'lexical', 'metadata_minus_lexical')]:
        difference = estimates[first] - estimates[second]
        contrasts[name] = {
            key: {'point': summaries[first]['point'][key] - summaries[second]['point'][key],
                  'confidence95': core.ci(difference[:, i])}
            for i, key in enumerate(METRICS)}
    result = {
        'models': summaries, 'paired_contrasts': contrasts, 'responses': len(test),
        'sources': len(sources), 'saturation_grid': SATURATIONS,
        'bootstrap': {'draws': len(weights), 'weights_sha256': hashlib.sha256(weights.tobytes()).hexdigest(),
                      'fitting_thresholds_and_actions_fixed': True, 'response_weighting': 'uniform'},
        'input_sha256': {str(path): sha(path) for path in [args.baseline, args.weights, args.source_ids,
                         args.ragtruth / 'dataset/response.jsonl', args.ragtruth / 'dataset/source_info.jsonl']},
        'script_sha256': sha(Path(__file__)), 'scorer_sha256': sha(Path(core.__file__)),
        'scope': 'Missed annotated-word loss under illustrative curves and context/verification volume; not measured task utility, time or repaired-output quality.',
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.out / 'per_response.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_response[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(per_response)
    np.savez_compressed(args.out / 'bootstrap.npz',
                        **{name + '__' + key: value[:, i] for name, value in estimates.items()
                           for i, key in enumerate(METRICS)})
    print(json.dumps({'responses': len(test), 'models': {name: row['point'] for name, row in summaries.items()}}))


if __name__ == '__main__':
    main()
