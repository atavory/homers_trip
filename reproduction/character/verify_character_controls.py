"""Replay the current character controls without neural model dependencies."""

import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'analysis'))
import bandha_span_evaluator_control as core


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected, pointer='', errors=None):
    errors = [] if errors is None else errors
    if isinstance(expected, dict):
        for key, value in expected.items():
            compare(actual[key], value, pointer + '.' + key, errors)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('Reference length disagreement: ' + pointer)
        for index, value in enumerate(expected):
            compare(actual[index], value, f'{pointer}[{index}]', errors)
    elif isinstance(expected, (int, float)):
        error = abs(actual - expected)
        if error > 1e-12:
            raise ValueError(f'Reference disagreement at {pointer}: {actual} versus {expected}')
        errors.append(float(error))
    elif actual != expected:
        raise ValueError('Reference disagreement: ' + pointer)
    return errors


def direct_counts(raw, selected):
    gold = sum(label['end'] - label['start'] for row in raw for label in row['labels'])
    predicted = sum(len(row['response']) for row in raw if row['id'] in selected)
    covered = sum(label['end'] - label['start'] for row in raw if row['id'] in selected
                  for label in row['labels'])
    return {'gold_span_length_sum': gold, 'predicted_characters': predicted,
            'covered_gold_characters': covered,
            'precision': Fraction(covered, predicted) if predicted else Fraction(0),
            'recall': Fraction(covered, gold) if gold else Fraction(0),
            'f1': Fraction(2 * covered, gold + predicted) if gold + predicted else Fraction(0)}


def ceiling_certificate(raw, selected, optimum):
    gold = sum(label['end'] - label['start'] for row in raw for label in row['labels'])
    advantages = []
    for row in raw:
        mass = sum(label['end'] - label['start'] for label in row['labels'])
        advantage = 2 * mass - optimum * len(row['response'])
        if row['id'] in selected and advantage < 0:
            raise ValueError('Ceiling selects a negative-advantage response.')
        if row['id'] not in selected and advantage > 0:
            raise ValueError('Ceiling omits a positive-advantage response.')
        advantages.append(advantage)
    upper_bound = sum((max(Fraction(0), value) for value in advantages), Fraction(0))
    if upper_bound != optimum * gold:
        raise ValueError('Exact global subset ceiling certificate failed.')
    return {'all_response_subset_upper_bound_verified': True,
            'exact_optimal_f1': str(optimum), 'responses_checked': len(raw),
            'maximum_subset_advantage': str(upper_bound),
            'optimal_f1_times_total_gold': str(optimum * gold),
            'proof': 'For any subset S, 2*G(S)-f*N(S) <= sum_r max(0,2*g_r-f*n_r) = f*G_total; hence its F1 cannot exceed f.'}


def encode(value):
    if isinstance(value, Fraction):
        return {'exact': str(value), 'decimal': float(value)}
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [encode(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ragtruth', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not __debug__:
        raise ValueError('Use Python without -O; the original public functions use assertions.')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Output directory must be empty.')
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    pin = json.loads((ROOT / 'datasets.json').read_text())['RAGTruth']
    if {name: sha(args.ragtruth / name) for name in pin['files']} != pin['files']:
        raise ValueError('RAGTruth files disagree with their pinned hashes.')
    manifest = json.loads((ROOT / 'vendor/source_manifest.json').read_text())
    extract = ROOT / 'vendor/lettuce_character_evaluator.py'
    if sha(extract) != manifest['extract_sha256']:
        raise ValueError('Vendored evaluator extract changed.')
    if sha(ROOT / 'vendor/LICENSE_LettuceDetect') != manifest['license_sha256']:
        raise ValueError('Vendored license changed.')
    if sha(Path(core.__file__)) != manifest['public_control_sha256']:
        raise ValueError('Original public control implementation changed.')
    evaluator, ast_hash = core.released_function(extract, manifest['function_name'])
    if ast_hash != manifest['function_ast_sha256']:
        raise ValueError('Extracted function AST disagrees with the pinned original.')

    rows, inputs = core.load_ragtruth(args.ragtruth)
    raw_rows = [json.loads(line) for line in (args.ragtruth / 'dataset/response.jsonl').read_text().splitlines()]
    for row in raw_rows:
        if any(row['response'][label['start']:label['end']] != label['text'] for label in row['labels']):
            raise ValueError('Raw gold span text disagrees with its boundaries.')
    splits = {name: {row['source_id'] for row in rows if row['split'] == name}
              for name in ['train', 'validation', 'test']}
    if any(splits[first] & splits[second] for first, second in
           [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise ValueError('Sources cross fitting, validation or test partitions.')
    rates, fit = core.train_rates(rows)
    validation = [row for row in rows if row['split'] == 'validation' and row['quality'] == 'good' and row['n']]
    fallback = {'metadata_char': fit['char_prevalence'], 'metadata_any': fit['response_any_rate']}
    scores = {name: lambda row, name=name: rates.get(row['key'], fallback)[name]
              for name in fallback}
    thresholds = {name: core.best_subset(validation, score)['threshold'] for name, score in scores.items()}
    result_rows, populations, arithmetic, certificates, direct_errors = [], [], [], [], []
    for population in ('test_good', 'test_all_qualities'):
        test = [row for row in rows if row['split'] == 'test' and row['n'] and
                (population == 'test_all_qualities' or row['quality'] == 'good')]
        raw = [row for row in raw_rows if row['split'] == 'test' and row['response'] and
               (population == 'test_all_qualities' or row['quality'] == 'good')]
        if {row['id'] for row in raw} != {row['id'] for row in test}:
            raise ValueError('Independent raw panel disagrees with original loader.')
        populations.append({'population': population, 'responses': len(test),
                            'sources': len({row['source_id'] for row in test}),
                            'characters': sum(row['n'] for row in test),
                            'gold_span_length_sum': sum(row['g'] for row in test),
                            'gold_union_characters': sum(row['gold_union_characters'] for row in test),
                            'responses_with_overlapping_gold': sum(row['g'] != row['gold_union_characters'] for row in test)})
        controls = {'mark_none': set(), 'mark_all': {row['id'] for row in test}}
        controls.update({name: {row['id'] for row in test if score(row) >= thresholds[name]}
                         for name, score in scores.items()})
        ceiling = core.best_subset(test, lambda row: Fraction(row['g'], row['n']))
        controls['label_informed_whole_response_ceiling'] = ceiling['ids']
        for name, selected in controls.items():
            metrics = core.score_selected(test, selected, evaluator)
            direct = direct_counts(raw, selected)
            for key in ['precision', 'recall', 'f1', 'covered_gold_characters']:
                error = abs(metrics[key] - direct[key])
                direct_errors.append(float(error))
                if error > 1e-12:
                    raise ValueError('Independent raw arithmetic disagrees with released evaluator.')
            result_rows.append({'population': population, 'control': name, **metrics})
            arithmetic.append({'population': population, 'control': name,
                               'direct_raw_arithmetic': encode(direct),
                               'selected_ids': sorted(selected)})
        certificates.append({'population': population,
                             **ceiling_certificate(raw, ceiling['ids'], ceiling['f1'])})

    report = {'input_sha256': inputs, 'fit': fit, 'validation_responses': len(validation),
              'validation_selected_thresholds': thresholds, 'populations': populations,
              'ragtruth_results': result_rows, 'released_function_sources': manifest,
              'independent_raw_arithmetic': arithmetic, 'global_ceiling_certificates': certificates,
              'source_disjoint': True, 'model_inference': False,
              'scope': 'Whole-response marking on the frozen RAGTruth panel. Raw gold-span lengths and pairwise intersections follow the released evaluator. The held-out-label ceiling is an upper bound for this strategy, not a deployable baseline, floor or additive attribution.'}
    reference = json.loads((ROOT / 'published_character_reference.json').read_text())
    errors = compare(report, reference)
    report['verification'] = {'all_checks_passed': True, 'numeric_comparisons': len(errors),
                              'maximum_absolute_error': max(errors),
                              'independent_arithmetic_comparisons': len(direct_errors),
                              'independent_arithmetic_maximum_absolute_error': max(direct_errors),
                              'elapsed_seconds': time.monotonic() - started,
                              'helper_sha256': sha(Path(__file__)),
                              'reference_sha256': sha(ROOT / 'published_character_reference.json')}
    (args.out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.out / 'ragtruth_controls.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result_rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(result_rows)
    main_rows = [row for row in result_rows if row['population'] == 'test_good']
    definitions = [r'\providecommand{\RevisionValue}[1]{\csname RevisionValue@#1\endcsname}']
    for row in main_rows:
        for key in ['precision', 'recall', 'f1', 'flagged_responses']:
            value = f'{row[key]:,}' if key == 'flagged_responses' else f'{row[key]:.3f}'
            definitions.append(r'\expandafter\def\csname RevisionValue@character.' +
                               row['control'] + '.' + key + r'\endcsname{' + value + '}')
    (args.out / 'character_values.tex').write_text('\n'.join(definitions) + '\n')
    print(json.dumps({'verification': report['verification'], 'test_good': main_rows,
                      'ceiling_certificates': certificates}, indent=2))


if __name__ == '__main__':
    main()
