"""Generate manuscript bindings for response-loss theory and frozen-score readouts."""

import argparse
import csv
import hashlib
import json
from fractions import Fraction
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--theory', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    theory = json.loads(args.theory.read_text())
    benchmark = json.loads(args.benchmark.read_text())
    assert theory['all_checks_passed']
    args.out.mkdir(parents=True, exist_ok=True)
    values, provenance = {}, []

    def bind(key, value, source, pointer, style=None):
        if style == 'fraction':
            q = Fraction(value)
            text = str(q.numerator) if q.denominator == 1 else rf'\tfrac{{{q.numerator}}}{{{q.denominator}}}'
        elif style == 'vector':
            text = ','.join(map(str, value))
        elif style == 'percent':
            text = f'{100 * value:.1f}'
        elif style == 'volume':
            text = f'{value:.1f}'
        elif isinstance(value, float):
            text = f'{value:.3f}'
        else:
            text = str(value)
        values[key] = text
        provenance.append({'key': key, 'unrounded': json.dumps(value), 'formatted_tex': text,
                           'source': str(source), 'pointer': pointer})

    for model, record in theory['models'].items():
        for metric in ('tp', 'fp', 'fn', 'tn', 'inspection_count', 'reviewed_responses'):
            bind(f'proof.{model}.{metric}', record[metric], args.theory,
                 f'models.{model}.{metric}')
        bind(f'proof.{model}.residual', record['residual_by_response'], args.theory,
             f'models.{model}.residual_by_response', 'vector')
        for metric in ('f1', 'accuracy', 'precision', 'recall', 'auc', 'ap', 'brier'):
            bind(f'proof.{model}.{metric}', record[metric], args.theory,
                 f'models.{model}.{metric}', 'fraction')
    bind('proof.risk_checks', theory['checked_risk_cases'], args.theory, 'checked_risk_cases')
    example = next(row for row in theory['risk_cases'] if row['a'] == '1/4'
                   and row['setup'] == '0' and row['marginal'] == '0')
    for model, record in example['models'].items():
        bind(f'proof.{model}.quarter_loss', record['loss'], args.theory,
             f'risk_cases[a=1/4,setup=0,marginal=0].models.{model}.loss', 'fraction')

    proof = [r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
             r'Scores & True detections & False alarms & Residual errors & Reviewed responses\\',
             r'\midrule']
    for name, label in [('original', '$s$'), ('offset', r'$O_{\mathbf c}s$')]:
        proof.append(label + ' & ' + ' & '.join([
            rf'\LossValue{{proof.{name}.tp}}', rf'\LossValue{{proof.{name}.fp}}',
            rf'$(\LossValue{{proof.{name}.residual}})$',
            rf'\LossValue{{proof.{name}.reviewed_responses}}']) + r'\\')
    proof.extend([r'\bottomrule', r'\end{tabular}'])
    (args.out / 'proof_table.tex').write_text('\n'.join(proof) + '\n')
    ranking = [r'\begin{tabular}{@{}lrrrrr@{}}', r'\toprule',
               r'Scores & F1 & Accuracy & Pooled AUROC & AP & Brier loss\\', r'\midrule']
    for name, label in [('original', '$s$'), ('offset', r'$O_{\mathbf c}s$')]:
        ranking.append(label + ' & ' + ' & '.join(
            rf'$\LossValue{{proof.{name}.{key}}}$'
            for key in ('f1', 'accuracy', 'auc', 'ap', 'brier')) + r'\\')
    ranking.extend([r'\bottomrule', r'\end{tabular}'])
    (args.out / 'proof_metrics.tex').write_text('\n'.join(ranking) + '\n')

    for name, row in benchmark['models'].items():
        for metric, point in row['point'].items():
            bind(f'data.{name}.{metric}', point, args.benchmark,
                 f'models.{name}.point.{metric}', 'percent' if metric == 'review_rate'
                 else 'volume' if metric in ('context_words', 'selected_words') else None)
    table = [r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
             r'Scores & Annotation loss & Review (\%) & Context words & Flagged words\\', r'\midrule']
    for name, label in [('task_generator', 'Metadata'), ('lexical', 'Lexical'),
                        ('lexical_metadata', 'Lexical + metadata'),
                        ('lexical_mean', 'Lexical mean'),
                        ('lexical_metadata_mean', 'Combined mean')]:
        table.append(label + ' & ' + ' & '.join(
            rf'\LossValue{{data.{name}.{key}}}'
            for key in ('loss_0.5', 'review_rate', 'context_words', 'selected_words')) + r'\\')
    table.extend([r'\bottomrule', r'\end{tabular}'])
    (args.out / 'benchmark_table.tex').write_text('\n'.join(table) + '\n')
    for name, contrasts in benchmark['paired_contrasts'].items():
        for metric, row in contrasts.items():
            precision = 1 if metric in ('context_words', 'selected_words') else 4
            text = f'{row["point"]:+.{precision}f} [{row["confidence95"]["low"]:+.{precision}f}, {row["confidence95"]["high"]:+.{precision}f}]'
            bind(f'contrast.{name}.{metric}', text, args.benchmark,
                 f'paired_contrasts.{name}.{metric}')
    definitions = [r'\newcommand{\LossValue}[1]{\csname LossValue@#1\endcsname}']
    definitions.extend(rf'\expandafter\def\csname LossValue@{key}\endcsname{{{value}}}'
                       for key, value in sorted(values.items()))
    (args.out / 'values.tex').write_text('\n'.join(definitions) + '\n')
    with (args.out / 'values.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(provenance[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(provenance)
    manifest = {'script_sha256': sha(Path(__file__)),
                'inputs': {str(p): sha(p) for p in (args.theory, args.benchmark)},
                'outputs': {p.name: sha(p) for p in args.out.iterdir() if p.name != 'manifest.json'},
                'value_count': len(values)}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'values': len(values), 'out': str(args.out)}))


if __name__ == '__main__':
    main()
