"""Bind the revised manuscript's new tables to saved evaluator and fixture outputs."""

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
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    controls = json.loads(args.control.read_text())
    fixture = json.loads(args.fixture.read_text())
    assert fixture['status'] == 'PASS'
    args.out.mkdir(parents=True, exist_ok=True)
    values = {}
    provenance = []

    def bind(key, value, source, pointer, style=None):
        if style == 'fraction':
            f = Fraction(str(value))
            text = str(f.numerator) if f.denominator == 1 else rf'\tfrac{{{f.numerator}}}{{{f.denominator}}}'
        elif style == 'integer':
            text = f'{value:,}'
        elif style == 'six':
            text = f'{value:.6f}'
        elif isinstance(value, float):
            text = f'{value:.3f}'
        else:
            text = str(value)
        values[key] = text
        provenance.append({'key': key, 'value': value, 'formatted_tex': text,
                           'input': str(source), 'pointer': pointer})

    selected = [r for r in controls['ragtruth_results'] if r['population'] == 'test_good']
    for population in controls['populations']:
        for metric in ('responses', 'sources', 'characters', 'responses_with_overlapping_gold'):
            bind(f'population.{population["population"]}.{metric}', population[metric],
                 args.control, f'populations[population={population["population"]}].{metric}', 'integer')
    for row in controls['ragtruth_results']:
        if row['population'] == 'test_all_qualities':
            bind('all_quality.' + row['control'] + '.f1', row['f1'], args.control,
                 f'ragtruth_results[population=test_all_qualities,control={row["control"]}].f1', 'six')
    bind('mu.language_files', len(controls['mu_inputs']), args.control, 'len(mu_inputs)', 'integer')
    names = {
        'mark_none': 'Mark no response', 'mark_all': 'Mark every response',
        'metadata_char': 'Metadata character prior',
        'metadata_any': 'Metadata any-error prior',
        'label_informed_whole_response_ceiling': 'Label-informed screening ceiling',
    }
    lines = [r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
             r'Whole-response control & Precision & Recall & F1 & Flagged\\', r'\midrule']
    for row in selected:
        name = row['control']
        for metric in ('precision', 'recall', 'f1', 'flagged_responses'):
            bind('character.' + name + '.' + metric, row[metric], args.control,
                 f'ragtruth_results[population=test_good,control={name}].{metric}',
                 'integer' if metric == 'flagged_responses' else None)
        lines.append(names[name] + ' & ' + ' & '.join(
            rf'\RevisionValue{{character.{name}.{m}}}'
            for m in ('precision', 'recall', 'f1', 'flagged_responses')) + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (args.out / 'character_controls.tex').write_text('\n'.join(lines) + '\n')

    a, b = fixture['records']['A'], fixture['records']['B']
    assert a['pooled_law'] == b['pooled_law']
    assert a['mean_recovery_count_by_k'] == b['mean_recovery_count_by_k']
    for detector, record in fixture['records'].items():
        for metric in ('macro_iou_k3', 'macro_f1_k3', 'macro_auc',
                       'macro_average_precision', 'complete_support_rate'):
            bind(f'fixture.{detector}.{metric}', record[metric], args.fixture,
                 f'records.{detector}.{metric}', 'fraction')
        for metric in ('inspection_work', 'successful_corrections'):
            bind(f'fixture.{detector}.{metric}', record[metric], args.fixture,
                 f'records.{detector}.{metric}', 'integer')
    bind('fixture.mean', a['rows'][0]['score_mean'], args.fixture,
         'records.A.rows[0].score_mean', 'fraction')
    bind('fixture.prevalence', a['rows'][0]['prevalence'], args.fixture,
         'records.A.rows[0].prevalence', 'fraction')
    recovery = ','.join((
        str(Fraction(v).numerator) if Fraction(v).denominator == 1 else
        rf'\tfrac{{{Fraction(v).numerator}}}{{{Fraction(v).denominator}}}')
        for v in a['mean_recovery_count_by_k'])
    bind('fixture.mean_curve', recovery, args.fixture, 'records.A.mean_recovery_count_by_k')
    rows = [r'\begin{tabular}{@{}lcccc@{}}', r'\toprule',
            r'Response & $A$ rank labels & $B$ rank labels & $H_r^A(3)$ & $H_r^B(3)$\\',
            r'\midrule']
    for i, (ra, rb) in enumerate(zip(a['rows'], b['rows']), 1):
        for d, row in [('A', ra), ('B', rb)]:
            bind(f'fixture.{d}.row{i}.labels', row['labels_in_descending_score_order'],
                 args.fixture, f'records.{d}.rows[{i - 1}].labels_in_descending_score_order')
            bind(f'fixture.{d}.row{i}.h3', row['recovery_counts_by_k'][3],
                 args.fixture, f'records.{d}.rows[{i - 1}].recovery_counts_by_k[3]')
        rows.append(str(i) + ' & ' + ' & '.join([
            rf'\texttt{{\RevisionValue{{fixture.A.row{i}.labels}}}}',
            rf'\texttt{{\RevisionValue{{fixture.B.row{i}.labels}}}}',
            rf'\RevisionValue{{fixture.A.row{i}.h3}}',
            rf'\RevisionValue{{fixture.B.row{i}.h3}}']) + r'\\')
    rows.extend([r'\bottomrule', r'\end{tabular}'])
    (args.out / 'fixed_summary_rows.tex').write_text('\n'.join(rows) + '\n')
    definitions = [r'\newcommand{\RevisionValue}[1]{\csname RevisionValue@#1\endcsname}']
    definitions.extend(rf'\expandafter\def\csname RevisionValue@{k}\endcsname{{{v}}}'
                       for k, v in sorted(values.items()))
    (args.out / 'values.tex').write_text('\n'.join(definitions) + '\n')
    with (args.out / 'values.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(provenance[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(provenance)
    manifest = {'script': str(Path(__file__)), 'script_sha256': sha(Path(__file__)),
                'inputs': {str(p): sha(p) for p in (args.control, args.fixture)},
                'outputs': {p.name: sha(p) for p in args.out.iterdir() if p.is_file() and p.name != 'manifest.json'},
                'value_count': len(values), 'checks': 'saved exact fixture and evaluator readouts'}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'value_count': len(values), 'out': str(args.out)}))


if __name__ == '__main__':
    main()
