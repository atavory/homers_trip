"""Generate shadow-manuscript tables from recorded experiment results."""

import argparse
import csv
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, digits=3):
    text = format(float(value), f'.{digits}f')
    if text.startswith('0.'):
        return text[1:]
    if text.startswith('-0.'):
        return '-' + text[2:]
    return text


def fraction(value):
    value = F(value)
    body = str(value.numerator) if value.denominator == 1 else (
        r'\tfrac{' + str(value.numerator) + '}{' + str(value.denominator) + '}'
    )
    return r'\ensuremath{' + body + '}'


def conditional_auc(response):
    positive = [F(s) for s, y in zip(response['scores'], response['labels']) if y]
    negative = [F(s) for s, y in zip(response['scores'], response['labels']) if not y]
    assert positive and negative
    wins = sum(F(1) if p > n else F(1, 2) if p == n else F(0)
               for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expanded', type=Path, required=True)
    parser.add_argument('--alternatives', type=Path, required=True)
    parser.add_argument('--relationships', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in
             ['expanded', 'alternatives', 'relationships', 'fixtures']}
    inputs = {name: json.loads(path.read_text()) for name, path in paths.items()}
    expanded, alternatives, relationships, fixtures = (
        inputs[name] for name in ['expanded', 'alternatives', 'relationships', 'fixtures']
    )
    for record in [alternatives, relationships]:
        assert record['response_sha256'] == expanded['response_sha256']
        assert record['source_sha256'] == expanded['source_sha256']
        assert record['fit'] == expanded['fit']
    assert relationships['bootstrap']['weights_sha256'] == expanded['bootstrap']['weights_sha256']
    assert alternatives['bootstrap']['weight_sha256'] == expanded['bootstrap']['weights_sha256']
    assert fixtures['all_checks_passed']
    cells, values = [], {}

    def value(name, raw, source, display=None):
        display = display if display is not None else number(raw)
        values[name] = display
        cells.append([name, source, str(raw), display])
        return display

    def table_point(name, raw, source, digits=3):
        return value(name, raw, source, number(raw, digits))

    def ranking_cell(model, key, name):
        record = expanded['models'][model]
        raw = record[key]
        ci = record['confidence95'].get(key)
        point = table_point(name + '.point', raw, f'expanded.models.{model}.{key}')
        if ci is None:
            return point
        low = table_point(name + '.low', ci['low'], f'expanded.models.{model}.confidence95.{key}.low')
        high = table_point(name + '.high', ci['high'], f'expanded.models.{model}.confidence95.{key}.high')
        return r'\shortstack{' + point + r'\\{}[' + low + ', ' + high + ']}'

    labels = {'global': 'Global prior', 'task_generator': 'Metadata prior',
              'lexical': 'Lexical', 'lexical_metadata': 'Lexical + metadata'}
    models = [r'\begin{tabular}{@{}lccc@{}}', r'\toprule',
              r'Scorer & \shortstack{Pooled\\AUROC} & \shortstack{After response\\means} & \shortstack{Conditional\\macro AUROC} \\',
              r'\midrule']
    for model, label in labels.items():
        parent = model + '_mean' if model in ['lexical', 'lexical_metadata'] else model
        if model in ['lexical', 'lexical_metadata']:
            assert abs(expanded['models'][model]['within_macro_auc'] -
                       alternatives['models'][model]['point']['within_macro_auc']) < 1e-12
        models.append(label + ' & ' + ' & '.join([
            ranking_cell(model, 'pooled_auc', f'models.{model}.pooled'),
            ranking_cell(parent, 'pooled_auc', f'models.{model}.mean'),
            ranking_cell(model, 'within_macro_auc', f'models.{model}.conditional'),
        ]) + r' \\')
    models.extend([r'\bottomrule', r'\end{tabular}'])

    covariance = [r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
                  r'Scorer & Pooled & Within & Between & \shortstack{Mean/prevalence\\correlation} \\',
                  r'\midrule']
    prediction_error = [r'\begin{tabular}{@{}lrrr@{}}', r'\toprule',
                        r'Scorer & \shortstack{Probability\\error} & \shortstack{Response-mean\\error} & \shortstack{Centered\\error} \\',
                        r'\midrule']
    for model in ['task_generator', 'lexical', 'lexical_metadata']:
        quantities = relationships['models'][model]['token']
        alias = {'task_generator': 'metadata', 'lexical': 'lexical', 'lexical_metadata': 'combined'}[model]
        keys = [('pooled_covariance', 'pooled'), ('within_covariance', 'within'),
                ('between_covariance', 'between'), ('response_mean_prevalence_correlation', 'correlation')]
        rendered = [table_point(f'cov.{alias}.{short}', quantities[key]['point'],
                                f'relationships.models.{model}.token.{key}.point', 5)
                    for key, short in keys]
        covariance.append(labels[model] + ' & ' + ' & '.join(rendered) + r' \\')
        loss_keys = ['probability_squared_error', 'response_mean_probability_squared_error',
                     'centered_probability_squared_error']
        rendered = [table_point(f'loss.{alias}.{key}', quantities[key]['point'],
                                f'relationships.models.{model}.token.{key}.point', 5)
                    for key in loss_keys]
        prediction_error.append(labels[model] + ' & ' + ' & '.join(rendered) + r' \\')
    covariance.extend([r'\bottomrule', r'\end{tabular}'])
    prediction_error.extend([r'\bottomrule', r'\end{tabular}'])

    inspection = [r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
                  r'Scorer & \multicolumn{2}{c}{Word recall} & \multicolumn{2}{c}{Complete-span recall} \\',
                  r' & Conditional & Global & Conditional & Global \\', r'\midrule']
    for model in ['task_generator', 'lexical', 'lexical_metadata']:
        record = alternatives['models'][model]['point']
        keys = ['conditional_word_recall_0.1', 'global_word_recall_0.1',
                'conditional_complete_span_recall_0.1', 'global_complete_span_recall_0.1']
        rendered = [table_point(f'inspection.{model}.{key}', record[key],
                                f'alternatives.models.{model}.point.{key}')
                    for key in keys]
        inspection.append(labels[model] + ' & ' + ' & '.join(rendered) + r' \\')
    inspection.extend([r'\bottomrule', r'\end{tabular}'])

    for model, alias in [('lexical', 'lexical'), ('lexical_metadata', 'combined')]:
        record = expanded['projection_contrasts'][model]['pooled_auc']
        raw = record['point']
        ci = record['ci95']
        display = number(raw, 4) + ' [' + number(ci['low'], 4) + ', ' + number(ci['high'], 4) + ']'
        value('projection.' + alias, raw,
              f'expanded.projection_contrasts.{model}.pooled_auc.point', display)
        cells.append(['projection.' + alias + '.low',
                      f'expanded.projection_contrasts.{model}.pooled_auc.ci95.low',
                      str(ci['low']), number(ci['low'], 4)])
        cells.append(['projection.' + alias + '.high',
                      f'expanded.projection_contrasts.{model}.pooled_auc.ci95.high',
                      str(ci['high']), number(ci['high'], 4)])

    joint = fixtures['fixtures']['joint_nonidentification']
    for side in ['A', 'B']:
        raw = sum(conditional_auc(row) for row in joint[side]) / len(joint[side])
        value('fixture.macro_' + side, raw,
              f'mean(conditional_auc(row) for row in fixtures.fixtures.joint_nonidentification.{side})',
              fraction(raw))
        raw = F(joint['utility_' + side])
        value('fixture.capture_' + side, raw,
              f'fixtures.fixtures.joint_nonidentification.utility_{side}', fraction(raw))
    for index in range(2):
        raw = F(joint['shared']['means'][index])
        value('fixture.mean' + str(index + 1), raw,
              f'fixtures.fixtures.joint_nonidentification.shared.means.{index}', fraction(raw))
    fixed = fixtures['fixtures']['conditional_auc_fixed_budget']
    for source, target in [('auc_A', 'budget_auc_A'), ('auc_B', 'budget_auc_B'),
                           ('capture_A', 'budget_capture_A'), ('capture_B', 'budget_capture_B')]:
        raw = F(fixed[source])
        value('fixture.' + target, raw,
              'fixtures.fixtures.conditional_auc_fixed_budget.' + source, fraction(raw))

    args.out.mkdir(parents=True, exist_ok=True)
    for name, lines in [('models', models), ('covariance', covariance),
                        ('prediction_error', prediction_error), ('inspection', inspection)]:
        (args.out / (name + '.tex')).write_text('\n'.join(lines) + '\n')
    tex = [r'\providecommand{\ShadowValue}[1]{\csname ShadowValue@#1\endcsname}']
    tex.extend(r'\expandafter\def\csname ShadowValue@' + name + r'\endcsname{' + display + '}'
               for name, display in sorted(values.items()))
    (args.out / 'values.tex').write_text('\n'.join(tex) + '\n')
    with (args.out / 'cell_sources.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['quantity', 'source_path', 'unrounded_value', 'display'])
        writer.writerows(cells)
    manifest = {
        'script_sha256': digest(Path(__file__)),
        'inputs': {name: {'path': str(path), 'sha256': digest(path)}
                   for name, path in paths.items()},
        'dataset_revision': expanded['revision'],
        'source_bootstrap_weight_sha256': expanded['bootstrap']['weights_sha256'],
        'checks': {'input_hashes_agree': True, 'fitting_agrees': True,
                   'bootstrap_weights_agree': True, 'conditional_points_agree': True,
                   'analytic_fixtures_passed': True},
        'outputs': {str(path.name): digest(path) for path in args.out.iterdir()
                    if path.is_file() and path.name != 'manifest.json'},
    }
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'generated_tables': 4, 'mapped_values': len(cells), 'checks': manifest['checks']}))


if __name__ == '__main__':
    main()
