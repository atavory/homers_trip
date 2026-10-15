"""Aggregate paired four-condition operator comparisons from frozen scores."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, digits=4):
    text = format(float(value), f'.{digits}f')
    if text.startswith('0.'):
        return text[1:]
    if text.startswith('-0.'):
        return '-' + text[2:]
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expanded', type=Path, required=True)
    parser.add_argument('--bootstrap', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    parent = json.loads(args.expanded.read_text())
    arrays = np.load(args.bootstrap)
    arms = ['lexical', 'lexical_mean', 'lexical_metadata', 'lexical_metadata_mean']
    metrics = ['pooled_auc', 'pooled_ap', 'pooled_f1']
    labels = {'pooled_auc': 'Pooled AUROC', 'pooled_ap': 'Pooled AP',
              'pooled_f1': 'Validation-threshold F1'}
    result = {
        'expanded_sha256': digest(args.expanded),
        'bootstrap_sha256': digest(args.bootstrap),
        'script_sha256': digest(Path(__file__)),
        'dataset_revision': parent['revision'],
        'bootstrap': parent['bootstrap'],
        'four_conditions': dict(zip(['original', 'mean', 'offset', 'offset_mean'], arms)),
        'comparison_scope': 'AUROC and AP use the same score-label functional; F1 uses separate frozen validation-selected thresholds',
        'metrics': {},
    }
    values, sources = {}, []
    table = [r'\begin{tabular}{@{}lccc@{}}', r'\toprule',
             r'Metric & \shortstack{Deviation gain\\at original levels} & \shortstack{Deviation gain\\after offset} & Interaction \\',
             r'\midrule']
    f1_table = list(table)
    for metric in metrics:
        point = np.array([parent['models'][arm][metric] for arm in arms], dtype=float)
        samples = np.array([arrays[arm + '__' + metric] for arm in arms], dtype=float)
        assert samples.shape == (4, parent['bootstrap']['draws'])
        assert np.all(np.isfinite(samples))
        for index, arm in enumerate(arms):
            prior_ci = parent['models'][arm]['confidence95'][metric]
            assert np.allclose(np.quantile(samples[index], [.025, .975]),
                               [prior_ci['low'], prior_ci['high']], rtol=0, atol=1e-12)
        contrasts = {
            'g0': (point[0] - point[1], samples[0] - samples[1]),
            'gc': (point[2] - point[3], samples[2] - samples[3]),
            'offset_full': (point[2] - point[0], samples[2] - samples[0]),
            'offset_mean': (point[3] - point[1], samples[3] - samples[1]),
            'interaction': (point[2] - point[3] - point[0] + point[1],
                            samples[2] - samples[3] - samples[0] + samples[1]),
        }
        assert abs(contrasts['interaction'][0] -
                   contrasts['offset_full'][0] + contrasts['offset_mean'][0]) < 1e-12
        short = metric.removeprefix('pooled_')
        result['metrics'][metric] = {}
        displays = {}
        for name, (estimate, vector) in contrasts.items():
            lo, hi = np.quantile(vector, [.025, .975])
            result['metrics'][metric][name] = {
                'point': float(estimate), 'ci95': [float(lo), float(hi)],
                'valid_draws': len(vector),
            }
            key = short + '.' + name
            values[key] = number(estimate) + ' [' + number(lo) + ', ' + number(hi) + ']'
            values[key + '.point'] = number(estimate)
            displays[name] = r'\shortstack{' + number(estimate) + r'\\{}[' + number(lo) + ', ' + number(hi) + ']}'
            sources.append([key, metric, name, repr(float(estimate)), repr(float(lo)), repr(float(hi)), values[key]])
        row = labels[metric] + ' & ' + ' & '.join(displays[k] for k in ['g0', 'gc', 'interaction']) + r' \\'
        (f1_table if metric == 'pooled_f1' else table).append(row)
    for lines in [table, f1_table]:
        lines.extend([r'\bottomrule', r'\end{tabular}'])
    models = [r'\begin{tabular}{@{}llcc@{}}', r'\toprule',
              r'Condition & Score structure & \shortstack{Pooled\\AUROC} & \shortstack{Conditional\\macro AUROC} \\',
              r'\midrule']
    rows = [
        ('Metadata prior', 'Response constant', 'task_generator'),
        (r'Original $\hat y$', '$m+d$', 'lexical'),
        (r'Mean $T\hat y$', '$m$', 'lexical_mean'),
        (r'Offset $O_{\mathbf c}\hat y$', r'$m+\mathbf c+d$', 'lexical_metadata'),
        (r'Offset mean $O_{\mathbf c}T\hat y$', r'$m+\mathbf c$', 'lexical_metadata_mean'),
    ]
    for name, structure, arm in rows:
        cells = []
        for metric in ['pooled_auc', 'within_macro_auc']:
            raw = parent['models'][arm][metric]
            ci = parent['models'][arm]['confidence95'][metric]
            display = r'\shortstack{' + number(raw, 3) + r'\\{}[' + number(ci['low'], 3) + ', ' + number(ci['high'], 3) + ']}'
            cells.append(display)
            sources.append(['models.' + arm + '.' + metric, metric, arm, repr(raw),
                            repr(ci['low']), repr(ci['high']), display])
        models.append(name + ' & ' + structure + ' & ' + ' & '.join(cells) + r' \\')
    models.extend([r'\bottomrule', r'\end{tabular}'])
    assert np.max(np.abs(arrays['lexical__within_macro_auc'] -
                         arrays['lexical_metadata__within_macro_auc'])) < 1e-12
    result['checks'] = {'all_four_arm_draws_valid': True,
                        'paired_difference_identity': True,
                        'conditional_offset_invariance': True}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, lines in [('interaction', table), ('f1_interaction', f1_table), ('models', models)]:
        (args.out / (name + '.tex')).write_text('\n'.join(lines) + '\n')
    tex = [r'\providecommand{\OperatorValue}[1]{\csname OperatorValue@#1\endcsname}']
    tex.extend(r'\expandafter\def\csname OperatorValue@' + key + r'\endcsname{' + value + '}'
               for key, value in values.items())
    (args.out / 'values.tex').write_text('\n'.join(tex) + '\n')
    with (args.out / 'sources.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['quantity', 'metric', 'contrast_or_model', 'unrounded', 'ci_low', 'ci_high', 'display'])
        writer.writerows(sources)
    print(json.dumps({'checks': result['checks'], 'contrasts': result['metrics']}))


if __name__ == '__main__':
    main()
