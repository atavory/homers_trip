"""Reproduce the short paper's tables from explicit public source inputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reproduction/short"
sys.path.insert(0, str(ROOT / 'analysis'))
import expanded as core


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def compare(actual, expected, pointer='', findings=None):
    findings = [] if findings is None else findings
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f'Missing reference quantity: {pointer}.{key}')
            compare(actual[key], value, pointer + '.' + key, findings)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f'Length disagreement: {pointer}')
        for i, value in enumerate(expected):
            compare(actual[i], value, f'{pointer}[{i}]', findings)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        error = abs(actual - expected)
        if not np.isfinite(actual) or error > 1e-12:
            raise ValueError(f'Numeric disagreement: {pointer}: {actual} versus {expected}')
        findings.append({'quantity': pointer, 'absolute_error': float(error)})
    elif actual != expected:
        raise ValueError(f'Reference disagreement: {pointer}: {actual} versus {expected}')
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ragtruth', type=Path, required=True,
                        help='Cache root containing dataset/response.jsonl and source_info.jsonl')
    parser.add_argument('--fred-html', type=Path, required=True,
                        help='Primary FRED v2 HTML; see package_sources.json for its pinned hash')
    parser.add_argument('--out', type=Path, required=True, help='New, empty output directory')
    args = parser.parse_args()
    if not __debug__:
        raise ValueError('Run Python without -O: the original analytic checkers use assertions.')
    data, fred, out = args.ragtruth.resolve(), args.fred_html.resolve(), args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty to prevent stale-result reuse.')
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    pin = json.loads((ROOT / 'datasets.json').read_text())['RAGTruth']
    sources = json.loads((REFERENCE / 'package_sources.json').read_text())
    input_hashes = {rel: sha(data / rel) for rel in pin['files']}
    if input_hashes != pin['files']:
        raise ValueError('RAGTruth inputs disagree with the pinned file hashes.')
    accepted_fred = sources['fred'].get('accepted_input_sha256',
                                        [sources['fred']['raw_sha256']])
    if sha(fred) not in accepted_fred:
        raise ValueError('FRED HTML disagrees with the retained primary snapshot hash.')
    for name, expected_hash in sources['code_sha256'].items():
        if sha(ROOT / 'analysis' / name) != expected_hash:
            raise ValueError(f'Public analysis module was changed: {name}')

    rows = core.parse(data)
    score, fit = core.fit_models(rows)
    eligible = lambda row: row['quality'] == 'good' and len(row['y'])
    test = [row for row in rows if row['split'] == 'test' and eligible(row)]
    validation = [row for row in rows if row['split'] == 'validation' and eligible(row)]
    ids = sorted({row['source_id'] for row in test})
    splits = {split: {row['source_id'] for row in rows if row['split'] == split}
              for split in ('train', 'validation', 'test')}
    if any(splits[a] & splits[b] for a, b in [('train', 'validation'),
                                           ('train', 'test'), ('validation', 'test')]):
        raise ValueError('Sources cross a fitting, validation or test partition.')
    weights = np.random.default_rng(20261014).multinomial(
        len(ids), np.full(len(ids), 1 / len(ids)), size=2000).astype(np.int16)
    weights_dir = out / 'source_bootstrap'
    weights_dir.mkdir()
    np.save(weights_dir / 'source_weights.npy', weights)
    save(weights_dir / 'source_ids.json', ids)
    result = {'fit': fit, 'revision': pin['revision'],
              'response_sha256': input_hashes['dataset/response.jsonl'],
              'source_sha256': input_hashes['dataset/source_info.jsonl'],
              'code_sha256': sha(ROOT / 'analysis/expanded.py'),
              'bootstrap': {'seed': 20261014, 'draws': 2000, 'source_count': len(ids),
                            'weights_sha256': hashlib.sha256(weights.tobytes()).hexdigest(),
                            'conditional_on_fitted_models': True},
              'models': {}}
    boots = {}
    for model in core.MODELS:
        print('START', model, round(time.monotonic() - started, 1), flush=True)
        arrays = [score(row, model) for row in test]
        val_scores = [score(row, model) for row in validation]
        threshold = core.rank_metrics(np.concatenate([row['y'] for row in validation]),
                                      np.concatenate(val_scores))['threshold']
        point, local = core.summarize(test, arrays, threshold)
        boot = core.source_bootstrap(test, arrays, local, threshold, weights, ids)
        point['confidence95'] = {key: core.ci(value) for key, value in boot.items()}
        result['models'][model], boots[model] = point, boot
        print('DONE', model, 'AUROC', point['pooled_auc'], flush=True)

    def contrast(first, second):
        return {key: {'point': result['models'][first][key] - result['models'][second][key],
                      'ci95': core.ci(boots[first][key] - boots[second][key])}
                for key in ('pooled_auc', 'pooled_ap', 'within_pair_auc',
                            'within_macro_auc', 'pooled_f1')}
    result['primary_contrast'] = contrast('lexical_metadata', 'lexical')
    result['projection_contrasts'] = {model: contrast(model + '_mean', model)
                                      for model in ('lexical', 'lexical_metadata')}
    expanded = out / 'expanded'
    save(expanded / 'expanded_results.json', result)
    np.savez_compressed(expanded / 'bootstrap_estimates.npz',
                        **{model + '__' + key: value for model, boot in boots.items()
                           for key, value in boot.items()})
    save(out / 'data_counts.json', {'test_words': result['models']['lexical']['P'] +
                                   result['models']['lexical']['N'],
                                   'test_positives': result['models']['lexical']['P'],
                                   'train_responses': fit['train_responses'],
                                   'train_sources': fit['train_sources'],
                                   'validation_responses': len(validation),
                                   'validation_sources': len({row['source_id'] for row in validation}),
                                   'source_disjoint': True})

    runtime = out / 'runtime'
    (runtime / 'analysis').mkdir(parents=True)
    for name in sources['code_sha256']:
        shutil.copy2(ROOT / 'analysis' / name, runtime / 'analysis' / name)
    commands = []
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONPATH='',
               PYTHONDONTWRITEBYTECODE='1', PYTHONOPTIMIZE='0')
    def run(name, *arguments):
        command = [sys.executable, str(runtime / 'analysis' / name), *map(str, arguments)]
        print('RUN', name, flush=True)
        subprocess.run(command, cwd=runtime, env=env, check=True)
        commands.append({'script': name, 'arguments': list(map(str, arguments))})

    run('operator_comparisons.py', '--expanded', expanded / 'expanded_results.json',
        '--bootstrap', expanded / 'bootstrap_estimates.npz', '--out', out / 'operators')
    run('bandha_response_loss_check.py', '--out', out / 'response_loss_exact')
    run('bandha_response_loss_benchmark.py', '--ragtruth', data,
        '--baseline', expanded / 'expanded_results.json',
        '--weights', weights_dir / 'source_weights.npy',
        '--source-ids', weights_dir / 'source_ids.json', '--out', out / 'work')
    run('bandha_response_loss_aggregate.py', '--theory', out / 'response_loss_exact/results.json',
        '--benchmark', out / 'work/results.json', '--out', out / 'response_loss_values')
    run('bandha_marginal_response_check.py', '--out', runtime / 'analysis/bandha_marginal_response_results')
    run('bandha_adaptive_review_check.py', '--out', out / 'adaptive')
    run('bandha_adaptive_metric_family.py', '--out', out / 'adaptive_metric_family')
    run('bandha_affine_review_check.py', '--out', out / 'affine_review')
    run('bandha_source_case_values.py', '--records', REFERENCE / 'rarr_comparisons.csv', '--out', out / 'rarr_values')
    run('evaluate_operations.py', '--outcomes', REFERENCE / 'operation_fixture.csv', '--reference', 'A',
        '--policy-description', 'Analytic matched fixture, not a repair experiment',
        '--cost-description', 'Fixture work priced in utility units', '--out', out / 'operation_fixture')
    run('bandha_published_case_values.py', '--fred-html', fred,
        '--adequacy-csv', REFERENCE / 'adequacy.csv', '--out', out / 'published_cases')

    character_command = [sys.executable, '-I', '-B', str(ROOT / 'reproduction/character/verify_character_controls.py'),
                         '--ragtruth', str(data), '--out', str(out / 'character')]
    subprocess.run(character_command, env=env, check=True)
    commands.append({'script': 'reproduction/character/verify_character_controls.py',
                     'arguments': ['--ragtruth', str(data), '--out', str(out / 'character')]})

    actual = {**result, 'work': json.loads((out / 'work/results.json').read_text()),
              'adaptive': json.loads((out / 'adaptive/verification.json').read_text()),
              'character': json.loads((out / 'character/results.json').read_text()),
              'operation_fixture': json.loads((out / 'operation_fixture/operation_evaluation.json').read_text()),
              'affine': json.loads((out / 'affine_review/verification.json').read_text()),
              'fred': json.loads((out / 'published_cases/verification.json').read_text())['values']}
    reference = json.loads((REFERENCE / 'reviewer_reference.json').read_text())
    table_hashes = reference.pop('table_sha256')
    comparisons = compare(actual, reference)
    generated_hashes = {rel: sha(out / rel) for rel in table_hashes}
    compare(generated_hashes, table_hashes)
    elapsed = time.monotonic() - started
    report = {'all_checks_passed': True, 'numeric_comparisons': len(comparisons),
              'maximum_absolute_error': max(row['absolute_error'] for row in comparisons),
              'generated_table_files_identical': generated_hashes,
              'bootstrap_draws': 2000, 'source_count': len(ids),
              'elapsed_seconds': elapsed, 'comparisons': comparisons}
    save(out / 'verification.json', report)
    save(out / 'reproduction_manifest.json', {
        'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
        'commands': commands, 'input_sha256': {**input_hashes, 'fred_v2_html': sha(fred)},
        'package_sha256': {str(path.relative_to(ROOT)): sha(path)
                           for path in [Path(__file__), ROOT / 'datasets.json',
                                        *sorted(REFERENCE.iterdir()),
                                        *(ROOT / 'analysis' / n for n in sources['code_sha256'])]
                           if path.is_file()},
        'result_sha256': {str(path.relative_to(out)): sha(path)
                          for path in sorted(out.rglob('*')) if path.is_file()},
        'elapsed_seconds': elapsed, 'network_access': False, 'model_inference': False,
        'scope': 'Current short-paper numerical tables and analytic checks; no legacy campaigns or figures.'})
    print(json.dumps({key: value for key, value in report.items() if key != 'comparisons'}, indent=2))


if __name__ == '__main__':
    main()
