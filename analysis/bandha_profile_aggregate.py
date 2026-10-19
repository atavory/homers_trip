"""Bind all manuscript numbers and tables to verified profile results."""
import argparse
import hashlib
import json
from pathlib import Path


LABELS = {'no_action': 'No checks', 'lexical': 'Lexical', 'lexical_metadata': 'Lexical + metadata',
          'task_generator': 'Task/generator', 'generator': 'Generator'}


def fmt(value, digits=3):
    return f'{value:.{digits}f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.results / 'results.json').read_text())
    verification = json.loads((args.results / 'verification.json').read_text()); assert verification['all_checks_passed']
    values = {'responses': str(report['population']['responses']), 'sources': str(report['population']['sources']),
              'draws': str(report['bootstrap']['draws']), 'validation.responses': str(report['validation_tuning_scope']['validation_responses']),
              'validation.sources': str(report['validation_tuning_scope']['validation_sources'])}
    pair = report['comparisons']['lexical_metadata__minus__lexical']
    for key, kind, pos, scale in [('mean_errors', 'errors', -1, 1), ('mean_checks', 'work', -1, 1),
                                 ('checked_rate', 'work', 0, 100), ('zero_errors', 'errors', 0, -100)]:
        data = pair[kind]
        triple = [data['point'][pos] * scale, data['simultaneous_low'][pos] * scale, data['simultaneous_high'][pos] * scale]
        if scale < 0: triple[1:] = triple[1:][::-1]
        for name, value in zip(['point', 'low', 'high'], triple):values['offset.' + key + '.' + name] = fmt(value)
    for kind in ['errors', 'work']:
        diag = report['root_diagnostics']['lexical_metadata__minus__lexical'][kind]
        values['root.' + kind + '.cross_pct'] = fmt(100 * diag['bootstrap_fraction_with_crossing'], 1)
        if diag['zero_compatible_grid_regions']:
            values['root.' + kind + '.low'] = fmt(diag['zero_compatible_grid_regions'][0][0])
            values['root.' + kind + '.high'] = fmt(diag['zero_compatible_grid_regions'][-1][1])
    for key, field in [('tuning', 'validation_tuning'), ('transport', 'ideal_policy_transport')]:
        rows = report[field]; positive = sum(r['simultaneous_confidence95'][0] > 0 for r in rows.values())
        negative = sum(r['simultaneous_confidence95'][1] < 0 for r in rows.values())
        values[key + '.total'] = str(len(rows)); values[key + '.positive'] = str(positive)
        values[key + '.negative'] = str(negative); values[key + '.uncertain'] = str(len(rows) - positive - negative)
    negative = 'lexical__objective_3__fix_0.75_harm_0.05_response_coupled'
    for tag, row in [('negative', report['validation_tuning'][negative]),
                     ('independent', report['validation_tuning']['lexical_metadata__objective_0__fix_0.9_harm_0.01_independent_units']),
                     ('transport', report['ideal_policy_transport']['lexical_metadata__objective_0__fix_0.9_harm_0.01_independent_units'])]:
        for name, val in zip(['point', 'low', 'high'], [row['contrast']['point'][0], *row['simultaneous_confidence95']]):values[tag + '.' + name] = fmt(val)
    for name in ['lexical', 'lexical_metadata']:
        h = report['models'][name]['error_histogram']; baseline = report['models']['no_action']['error_histogram'][0]
        values[name + '.new_zero_errors'] = str(h[0] - baseline)
    tex = [r'\makeatletter', r'\newcommand{\ProfileValue}[1]{\csname PV@#1\endcsname}']
    for key, value in values.items():tex.append(r'\expandafter\def\csname PV@' + key + r'\endcsname{' + value + '}')
    tex.append(r'\makeatother')
    (args.out / 'values.tex').write_text('\n'.join(tex) + '\n')
    table = [r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule', r'Scorer & Mean errors & Mean checks & Zero errors (\%) & Checked (\%)\\', r'\midrule']
    for name in LABELS:
        d = report['models'][name]['metrics']; cells = []
        for key in ['mean_errors', 'mean_checks']:
            v = d[key]; cells.append(r'\shortstack{' + fmt(v['point']) + r'\\{}[' + ', '.join(fmt(x) for x in v['confidence95']) + ']}')
        table.append(LABELS[name] + ' & ' + ' & '.join(cells) + ' & ' + fmt(100 * d['zero_errors']['point'], 1) + ' & ' + fmt(100 * (1 - d['zero_checks']['point']), 1) + r'\\')
    table += [r'\bottomrule', r'\end{tabular}']; (args.out / 'profile_readouts.tex').write_text('\n'.join(table) + '\n')
    table = [r'\begin{tabular}{@{}llrrr@{}}', r'\toprule', r'Scorer & Correction & F1-rule value & Value-rule value & Difference\\', r'\midrule']
    for name in ['lexical', 'lexical_metadata', 'task_generator']:
        for suffix, description in [('', 'Perfect'), ('__fix_0.9_harm_0.01_independent_units', '$p=.9,h=.01$')]:
            row = report['validation_tuning'][name + '__objective_0' + suffix]
            diff = r'\shortstack{' + fmt(row['contrast']['point'][0]) + r'\\{}[' + ', '.join(fmt(x) for x in row['simultaneous_confidence95']) + ']}'
            table.append(LABELS[name] + ' & ' + description + ' & ' + fmt(row['test_f1_value']) + ' & ' + fmt(row['test_tuned_value']) + ' & ' + diff + r'\\')
    table += [r'\bottomrule', r'\end{tabular}']; (args.out / 'tuning_table.tex').write_text('\n'.join(table) + '\n')
    rows = []
    for field in ['validation_tuning', 'ideal_policy_transport']:
        for name, data in report[field].items():rows.append({'kind': field, 'policy': name, 'model': data['model'],
                         'objective': data['objective'], 'fix_probability': data['fix_probability'], 'harm_probability': data['harm_probability'],
                         'dependence': data['dependence'], 'difference': data['contrast']['point'][0],
                         'simultaneous_confidence95': data['simultaneous_confidence95']})
    (args.out / 'all_policy_comparisons.json').write_text(json.dumps(rows, indent=2) + '\n')
    record = {'results_sha256': hashlib.sha256((args.results / 'results.json').read_bytes()).hexdigest(),
              'verification_sha256': hashlib.sha256((args.results / 'verification.json').read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'macro_values': values,
              'outputs': ['values.tex', 'profile_readouts.tex', 'tuning_table.tex', 'all_policy_comparisons.json']}
    (args.out / 'aggregation.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'macro_values': len(values), 'policy_comparisons': len(rows)}))


if __name__ == '__main__':
    main()
