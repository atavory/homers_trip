"""Measure pooled, response-average and conditional score relationships."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

import expanded as core


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atoms(rows, scores):
    values = []
    for row, score in zip(rows, scores):
        labels = row['y'].astype(float)
        pi = float(labels.mean())
        constant = bool(np.all(score == score[0]))
        mean = float(score[0] if constant else score.mean())
        centered = score - mean
        probabilities = expit(score)
        mean_probability = float(probabilities[0] if constant else probabilities.mean())
        covariance = float(np.mean(centered * (labels - pi))) if not constant else 0.0
        values.append([
            len(labels), pi, mean,
            float(np.mean(centered ** 2)) if not constant else 0.0,
            covariance, mean_probability,
            float(np.mean((probabilities - labels) ** 2)),
            (mean_probability - pi) ** 2,
            float(np.mean(((probabilities - mean_probability) - (labels - pi)) ** 2)),
        ])
    return np.asarray(values, dtype=float)


def summaries(values, q):
    n, pi, mean, variance, covariance, mean_probability, loss, mean_loss, centered_loss = values.T
    pooled_mean = q @ mean
    pooled_prevalence = q @ pi
    within = q @ covariance
    between = q @ (mean * pi) - pooled_mean * pooled_prevalence
    variance_means = q @ (mean ** 2) - pooled_mean ** 2
    if np.all(mean == mean[0]):
        between = np.zeros_like(pooled_mean)
        variance_means = np.zeros_like(pooled_mean)
    variance_prevalences = q @ (pi ** 2) - pooled_prevalence ** 2
    total_variance = q @ variance + variance_means
    label_variance = pooled_prevalence * (1 - pooled_prevalence)
    with np.errstate(divide='ignore', invalid='ignore'):
        pooled_correlation = (within + between) / np.sqrt(total_variance * label_variance)
        response_correlation = between / np.sqrt(variance_means * variance_prevalences)
    return {
        'pooled_covariance': within + between,
        'within_covariance': within,
        'between_covariance': between,
        'pooled_correlation': pooled_correlation,
        'response_mean_prevalence_correlation': response_correlation,
        'probability_squared_error': q @ loss,
        'response_mean_probability_squared_error': q @ mean_loss,
        'centered_probability_squared_error': q @ centered_loss,
        'mean_predicted_probability': q @ mean_probability,
        'observed_prevalence': pooled_prevalence,
    }


def scalar(value):
    return float(value) if np.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ragtruth', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    for filename, key in [('response.jsonl', 'response_sha256'), ('source_info.jsonl', 'source_sha256')]:
        assert digest(args.ragtruth / 'dataset' / filename) == baseline[key]
    rows = core.parse(args.ragtruth)
    score, fit = core.fit_models(rows)
    assert fit == baseline['fit']
    test = [r for r in rows if r['split'] == 'test' and r['quality'] == 'good' and len(r['y'])]
    source_ids = sorted({r['source_id'] for r in test})
    source_index = {source: i for i, source in enumerate(source_ids)}
    indices = np.array([source_index[r['source_id']] for r in test])
    lengths = np.array([len(r['y']) for r in test])
    bootstrap = baseline['bootstrap']
    rng = np.random.default_rng(bootstrap['seed'])
    weights = rng.multinomial(len(source_ids), np.full(len(source_ids), 1 / len(source_ids)), size=bootstrap['draws']).astype(np.int16)
    assert hashlib.sha256(weights.tobytes()).hexdigest() == bootstrap['weights_sha256']
    multiplicity = weights[:, indices]
    q_boot = {}
    q_point = {'token': lengths / lengths.sum(), 'response': np.full(len(test), 1 / len(test))}
    for mode in q_point:
        q = multiplicity.astype(float)
        if mode == 'token':
            q *= lengths
        q /= q.sum(axis=1, keepdims=True)
        q_boot[mode] = q

    result = {
        'dataset_revision': baseline['revision'],
        'response_sha256': baseline['response_sha256'],
        'source_sha256': baseline['source_sha256'],
        'baseline_sha256': digest(args.baseline),
        'script_sha256': digest(Path(__file__)),
        'helper_sha256': digest(Path(core.__file__)),
        'fit': fit, 'numpy_version': np.__version__,
        'bootstrap': bootstrap,
        'population': {'responses': len(test), 'sources': len(source_ids), 'words': int(lengths.sum()), 'positive_words': int(sum(r['y'].sum() for r in test))},
        'score_scale': 'original fixed logit scores; probability diagnostics use expit at token level before averaging',
        'models': {}, 'checks': {},
    }
    result_rows, response_rows = [], []
    for model in core.MODELS:
        scores = [score(r, model) for r in test]
        values = atoms(test, scores)
        pooled = core.rank_metrics(np.concatenate([r['y'] for r in test]), np.concatenate(scores))
        assert abs(pooled['auc'] - baseline['models'][model]['pooled_auc']) < 1e-12
        result['models'][model] = {}
        for mode, q in q_point.items():
            point = summaries(values, q)
            estimates = summaries(values, q_boot[mode])
            assert np.allclose(estimates['probability_squared_error'], estimates['response_mean_probability_squared_error'] + estimates['centered_probability_squared_error'], rtol=1e-11, atol=1e-12)
            assert abs(point['probability_squared_error'] - point['response_mean_probability_squared_error'] - point['centered_probability_squared_error']) < 1e-12
            if mode == 'token':
                flat_s = np.concatenate(scores)
                flat_y = np.concatenate([r['y'] for r in test]).astype(float)
                direct = float(np.mean((flat_s - flat_s.mean()) * (flat_y - flat_y.mean())))
                assert abs(direct - point['pooled_covariance']) < 1e-12
            for key, value in point.items():
                samples = estimates[key]
                samples = samples[np.isfinite(samples)]
                ci = np.quantile(samples, [.025, .975]).tolist() if len(samples) and np.isfinite(value) else None
                record = {'point': scalar(value), 'ci95': ci, 'valid_draws': len(samples)}
                result['models'][model].setdefault(mode, {})[key] = record
                result_rows.append([model, mode, key, record['point'], *(ci or [None, None]), len(samples)])
        for row, value in zip(test, values):
            response_rows.append([model, row['id'], row['source_id'], *value.tolist()])

    for mode in q_point:
        lexical = result['models']['lexical'][mode]['within_covariance']['point']
        combined = result['models']['lexical_metadata'][mode]['within_covariance']['point']
        assert abs(lexical - combined) < 1e-12
    for model in ['global', 'task', 'generator', 'task_generator', 'lexical_mean', 'lexical_metadata_mean']:
        for mode in q_point:
            assert result['models'][model][mode]['within_covariance']['point'] == 0
    result['checks'] = {'baseline_auc_reproduced': True, 'source_weights_identical': True, 'all_covariance_and_loss_identities_passed': True, 'constant_scores_zero_conditional_covariance': True, 'offset_preserves_conditional_covariance': True}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (args.out / 'relationships.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model', 'weighting', 'quantity', 'point', 'ci95_low', 'ci95_high', 'valid_draws'])
        writer.writerows(result_rows)
    with (args.out / 'response_atoms.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model', 'response_id', 'source_id', 'words', 'prevalence', 'mean_logit', 'variance_logit', 'covariance_logit_label', 'mean_probability', 'probability_squared_error', 'mean_probability_error', 'centered_probability_error'])
        writer.writerows(response_rows)
    tex = [r'\providecommand{\HomerDecisionValue}[1]{\csname hd@#1\endcsname}']
    for model, modes in result['models'].items():
        for mode, quantities in modes.items():
            for key, record in quantities.items():
                if record['point'] is not None:
                    name = '.'.join([model, mode, key])
                    tex.append(r'\expandafter\def\csname hd@' + name + r'\endcsname{' + format(record['point'], '.5f') + '}')
    (args.out / 'values.tex').write_text('\n'.join(tex) + '\n')
    lines = ['# Score relationship results', '', 'Generated by `analysis/decision_relationships.py` from the frozen RAGTruth scorers.', '', 'Covariances use the original logit score scale. They are association components, not predictive information fractions. Probability diagnostics apply the logistic function to every token score before averaging. Intervals resample the same complete source groups with fitting fixed.', '']
    columns = ['pooled_covariance', 'within_covariance', 'between_covariance', 'response_mean_prevalence_correlation']
    for mode in ['token', 'response']:
        lines.extend(['## ' + mode.capitalize() + ' weighting', '', '| Scorer | Pooled covariance | Within covariance | Between covariance | Mean-score/prevalence correlation |', '|---|---|---|---|---|'])
        for model in core.MODELS:
            quantities = result['models'][model][mode]
            cells = []
            for key in columns:
                record = quantities[key]
                if record['point'] is None:
                    cells.append('Undefined')
                else:
                    lo, hi = record['ci95']
                    cells.append(f"{record['point']:.5f} [{lo:.5f} to {hi:.5f}]")
            lines.append('| ' + ' | '.join([model, *cells]) + ' |')
        lines.append('')
    lines.extend(['## Checks', '', 'Baseline AUROCs, frozen fitting, source weights, covariance identities, squared-error identities, zero conditional covariance of response-constant scores, and offset invariance all passed.', ''])
    (args.out / 'summary.md').write_text('\n'.join(lines))
    print(json.dumps({'population': result['population'], 'checks': result['checks']}))


if __name__ == '__main__':
    main()
