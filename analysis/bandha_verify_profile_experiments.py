"""Independent checks of source inference, correction moments and selected policies."""
import argparse
import csv
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--ragtruth', type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('checked_profiles', Path(__file__).with_name('bandha_profile_experiments.py'))
    api = importlib.util.module_from_spec(spec); spec.loader.exec_module(api)
    result = json.loads((args.results / 'results.json').read_text())
    original = np.load(args.inputs / 'source_weights.npy')
    weights_path = args.results / 'source_weights.npy'
    if weights_path.exists():
        weights = np.load(weights_path)
    else:
        with np.load(args.results / 'source_weights.npz') as stored:
            weights = stored['weights']
    assert np.array_equal(weights[:2000], original)
    expected = np.random.default_rng(result['bootstrap']['additional_seed']).multinomial(450, np.full(450, 1 / 450), len(weights) - 2000)
    assert np.array_equal(weights[2000:], expected)
    assert hashlib.sha256(weights.tobytes()).hexdigest() == result['bootstrap']['weights_sha256']
    maxima = np.load(args.results / 'joint_max_statistics.npy')
    critical = np.quantile(maxima, .95, method='higher')
    assert critical == result['bootstrap']['critical_max_t']
    records = list(csv.DictReader((args.inputs / 'records.csv').open()))
    grouped = {}
    for row in records: grouped.setdefault(row['model'], {})[row['response_id']] = row
    ids = sorted(grouped['lexical'], key=int)
    sources = json.loads((args.inputs / 'source_ids.json').read_text()); lookup = {s: i for i, s in enumerate(sources)}
    index = np.array([lookup[grouped['lexical'][i]['source_id']] for i in ids])
    response_weights = weights[:, index]
    denominator = response_weights.sum(axis=1)
    checked = 0
    for name, pair in result['comparisons'].items():
        for field, label, caps in [('remaining_words', 'errors', [1, 2]), ('selected_words', 'work', [1, 14, 15])]:
            a = np.array([int(grouped[pair['first']][i][field]) for i in ids])
            b = np.array([int(grouped[pair['second']][i][field]) for i in ids])
            summary = pair[label]
            for cap in sorted(set([min(x, len(summary['point'])) for x in caps] + [len(summary['point'])])):
                delta = np.minimum(a, cap) - np.minimum(b, cap)
                draws = (response_weights @ delta) / denominator
                assert abs(delta.mean() - summary['point'][cap - 1]) < 1e-12
                assert abs(draws.std(ddof=1) - summary['standard_error'][cap - 1]) < 1e-12
                assert np.max(np.abs(np.quantile(draws, [.025, .975]) - np.array([summary['pointwise_low'][cap - 1], summary['pointwise_high'][cap - 1]]))) < 1e-12
                checked += 3
    fix, harm, parameter = Fraction(4, 5), Fraction(1, 10), Fraction(2, 3)
    missed, hits, false = 1, 2, 3
    independent = sum(Fraction(math.comb(hits, j) * math.comb(false, k)) *
                      (1 - fix) ** j * fix ** (hits - j) * harm ** k * (1 - harm) ** (false - k) *
                      parameter ** (missed + j + k) for j in range(hits + 1) for k in range(false + 1))
    coupled = sum(p * q * parameter ** (missed + j + k)
                  for j, p in [(0, fix), (hits, 1 - fix)]
                  for k, q in [(0, 1 - harm), (false, harm)])
    for dependence, expected in [('independent_units', independent), ('response_coupled', coupled)]:
        calculated = api.modeled_quality(np.array([missed]), np.array([hits]), np.array([false]), float(parameter), float(fix), float(harm), dependence)[0]
        assert abs(calculated - float(expected)) < 1e-12
        checked += 1
    assert coupled > independent
    pair = result['comparisons']['lexical_metadata__minus__lexical']
    c = grouped['lexical_metadata']; l = grouped['lexical']
    ec = np.array([int(c[i]['remaining_words']) for i in ids]); el = np.array([int(l[i]['remaining_words']) for i in ids])
    kc = np.array([int(c[i]['selected_words']) for i in ids]); kl = np.array([int(l[i]['selected_words']) for i in ids])
    covered = maxima <= critical
    for a in (.25, .5, .75):
        for b in (.25, .5, .75):
            for F in (0, .1):
                for v in (0, .01):
                    values = a ** ec - a ** el + F * (b ** kc - b ** kl) - v * (kc - kl)
                    point, low, high = api.value_band(pair['errors'], pair['work'], a, b, F, v, critical)
                    assert abs(values.mean() - point) < 1e-12
                    draws = (response_weights @ values) / denominator
                    assert np.all(draws[covered] >= low - 1e-12) and np.all(draws[covered] <= high + 1e-12)
                    checked += 2
    policy_checks = []
    if args.ragtruth:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import expanded as core
        all_rows = core.parse(args.ragtruth); score, fit = core.fit_models(all_rows)
        reference = json.loads((args.inputs / 'reference.json').read_text()); assert fit == reference['fit']
        validation = [r for r in all_rows if r['split'] == 'validation' and r['quality'] == 'good' and len(r['y'])]
        test_map = {r['id']: r for r in all_rows if r['split'] == 'test' and r['quality'] == 'good' and len(r['y'])}
        for name in ['lexical__objective_0', 'lexical_metadata__objective_0__fix_0.9_harm_0.01_independent_units']:
            row = result['validation_tuning'][name]; model = row['model']; a, b, F, v = row['objective']
            fix, harm, dependence = row['fix_probability'], row['harm_probability'], row['dependence']
            val_scores = [score(r, model) for r in validation]
            candidates = np.unique(np.r_[np.quantile(np.concatenate(val_scores), np.linspace(0, 1, 101)), row['f1_threshold'], np.inf])
            total = np.zeros(len(candidates)); checks = np.zeros(len(candidates))
            for r, s in zip(validation, val_scores):
                mask = s[:, None] >= candidates[None, :]
                k = mask.sum(axis=0); tp = ((r['y'] == 1)[:, None] & mask).sum(axis=0)
                e = int(r['y'].sum()) - tp; fp = k - tp
                total += api.modeled_quality(e, tp, fp, a, fix, harm, dependence) - F * (1 - b ** k) - v * k
                checks += k
            scores = total / len(validation)
            assert abs(scores.max() - row['validation_value']) < 1e-12
            threshold = np.inf if row['value_threshold'] is None else row['value_threshold']
            selected = np.flatnonzero(candidates == threshold)[0]
            assert abs(scores[selected] - scores.max()) < 1e-12
            e, k, quality = [], [], []
            for response in ids:
                r = test_map[response]; mask = score(r, model) >= threshold
                ki = int(mask.sum()); hi = int(r['y'][mask].sum()); ei = int(r['y'].sum()) - hi
                e.append(ei); k.append(ki)
                quality.append(float(api.modeled_quality(np.array([ei]), np.array([hi]), np.array([ki - hi]), a, fix, harm, dependence)[0]) - F * (1 - b ** ki) - v * ki)
            assert abs(np.mean(e) - row['test_tuned_errors']) < 1e-12
            assert abs(np.mean(k) - row['test_tuned_checks']) < 1e-12
            assert abs(np.mean(quality) - row['test_tuned_value']) < 1e-12
            policy_checks.append(name)
    report = {'all_checks_passed': True, 'numeric_checks': checked, 'source_draws_verified': len(weights),
              'independently_enumerated_correction_quality': {'independent': str(independent), 'coupled': str(coupled)},
              'direct_mask_validation_and_test_checks': policy_checks,
              'family_bound_checks': 36, 'joint_bootstrap_fraction_covered': float(covered.mean())}
    (args.results / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
