"""Bind common-weight recovery comparisons to manuscript tables and values."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    replay = json.loads(args.replay.read_text())
    assert replay['all_checks_passed']
    assert replay['maximum_original_point_error'] == 0
    assert replay['maximum_original_bootstrap_error'] == 0
    definitions = [r'\providecommand{\JointValue}[1]{\csname hj@#1\endcsname}']
    bindings = []
    table = [r'\begin{tabular}{@{}llrrr@{}}', r'\toprule',
             r'Budget & Span weights & Touched & Mean fraction & Complete\\', r'\midrule']

    def bind(key, value, source, display):
        definitions.append(r'\expandafter\def\csname hj@' + key + r'\endcsname{' + display + '}')
        bindings.append((key, source, value, display))

    for key in ('original_point_metric_count', 'original_bootstrap_scalar_count'):
        bind(key, replay[key], key, f'{replay[key]:,}')

    for budget in ('0.05', '0.1', '0.2'):
        for weights in ('uniform_span', 'length_weighted_span'):
            result = replay['comparisons'][budget][weights]
            source = 'comparisons.' + budget + '.' + weights
            prefix = budget + '.' + weights
            cells = []
            for field in ('touched_difference', 'partial_difference', 'complete_difference'):
                value = result[field]
                display = f'{value:+.4f}'
                bind(prefix + '.' + field, value, source + '.' + field, display)
                cells.append('$' + display + '$')
            roots = result['positive_alpha_root_brackets']
            for i, entry in enumerate(roots):
                bind(prefix + '.root' + str(i + 1), entry['root'], source + '.positive_alpha_root_brackets.' + str(i) + '.root', f"{entry['root']:.3f}")
            label = 'Uniform' if weights == 'uniform_span' else 'Length'
            table.append(f'{int(float(budget) * 100)}\\% & {label} & ' + ' & '.join(cells) + r'\\')
    table.extend([r'\bottomrule', r'\end{tabular}'])
    endpoint = replay['paired_cross_unit_endpoints']['global_length_weighted_complete_span_recall_0.05']
    for family in ('confidence95', 'three_budget_bonferroni_percentile'):
        value = endpoint[family]
        display = '[' + f"{value['low']:+.4f}, {value['high']:+.4f}" + ']'
        bind('length_complete_0.05.' + family, value, 'paired_cross_unit_endpoints.global_length_weighted_complete_span_recall_0.05.' + family, display)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'values.tex').write_text('\n'.join(definitions) + '\n')
    (args.out / 'endpoints.tex').write_text('\n'.join(table) + '\n')
    with (args.out / 'sources.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['key', 'replay_json_key', 'unrounded_value', 'display'])
        writer.writerows((k, source, json.dumps(value), display) for k, source, value, display in bindings)
    manifest = {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'input': str(args.replay), 'input_sha256': hashlib.sha256(args.replay.read_bytes()).hexdigest(),
                'output_sha256': {name: hashlib.sha256((args.out / name).read_bytes()).hexdigest()
                                  for name in ('values.tex', 'endpoints.tex', 'sources.csv')},
                'bindings': len(bindings), 'scope': 'Fixed-weight point comparisons; point roots carry no simultaneous population inference.'}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'bindings': len(bindings), 'output': str(args.out)}))


if __name__ == '__main__':
    main()
