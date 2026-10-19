"""Paired-source uncertainty and correction sensitivity for response profiles."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom

PAIRS = [('lexical_metadata', 'lexical'), ('task_generator', 'lexical'),
         ('generator', 'lexical_metadata'), ('lexical_mean', 'lexical'),
         ('lexical_metadata_mean', 'lexical_metadata')]
GRID = np.linspace(0, 1, 501)
FIX_GRID = (0, .5, .75, .9, 1)
HARM_GRID = (0, .005, .01, .05, .1, .2)
SENSITIVITY_CASES = ((.9, .01), (.75, .05))
OBJECTIVES = ((.5, .5, .01, .001), (.5, .5, .05, .001),
              (.5, .5, .01, .005), (.9, .5, .01, .001))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intervals(boot):
    return np.quantile(boot, [.025, .975], axis=0)


def max_standardized(boot, point, se):
    valid = se > 1e-13
    if not np.any(valid):
        return np.zeros(len(boot))
    result = np.zeros(len(boot))
    for start in range(0, len(boot), 500):
        result[start:start + 500] = np.max(
            np.abs((boot[start:start + 500, valid] - point[valid]) / se[valid]), axis=1)
    return result


def curve_summary(point, boot):
    se = np.std(boot, axis=0, ddof=1)
    ci = intervals(boot)
    return {'point': point.tolist(), 'standard_error': se.tolist(),
            'pointwise_low': ci[0].tolist(), 'pointwise_high': ci[1].tolist()}, max_standardized(boot, point, se)


def source_sum(values, index, sources):
    result = np.zeros((sources,) + values.shape[1:])
    np.add.at(result, index, values)
    return result


def geometric_coefficients(parameter, support):
    if not 0 <= parameter <= 1:
        raise ValueError('The generating-function parameter must lie in [0,1].')
    if support == 0:
        return np.zeros(0)
    return np.r_[(1 - parameter) ** 2 * parameter ** np.arange(support - 1),
                 (1 - parameter) * parameter ** (support - 1)]


def propagated_band(summary, parameter, critical):
    point = np.asarray(summary['point'])
    se = np.asarray(summary['standard_error'])
    coefficients = geometric_coefficients(parameter, len(point))
    return (-float(coefficients @ point),
            -float(coefficients @ (point + critical * se)),
            -float(coefficients @ (point - critical * se)))


def value_band(error_summary, work_summary, a, b, shared, continuing, critical):
    if shared < 0 or continuing < 0:
        raise ValueError('Cost prices must be nonnegative.')
    e = np.asarray(error_summary['point']); k = np.asarray(work_summary['point'])
    es = np.asarray(error_summary['standard_error']); ks = np.asarray(work_summary['standard_error'])
    ew = geometric_coefficients(a, len(e)); kw = shared * geometric_coefficients(b, len(k))
    kw[-1] += continuing
    point = -float(ew @ e + kw @ k)
    low = -float(ew @ (e + critical * es) + kw @ (k + critical * ks))
    high = -float(ew @ (e - critical * es) + kw @ (k - critical * ks))
    return point, low, high


def root_diagnostics(point, boot, low, high):
    signs = np.sign(boot[:, :-1]); signs[np.abs(boot[:, :-1]) < 1e-12] = 0
    counts = np.zeros(len(boot), dtype=int)
    previous = np.zeros(len(boot))
    for column in signs.T:
        nonzero = column != 0
        counts += nonzero & (previous != 0) & (column != previous)
        previous = np.where(nonzero, column, previous)
    compatible = (np.asarray(low)[:-1] <= 0) & (np.asarray(high)[:-1] >= 0)
    regions = []
    start = None
    for i, flag in enumerate(np.r_[compatible, False]):
        if flag and start is None: start = i
        if not flag and start is not None:
            regions.append([float(GRID[start]), float(GRID[i - 1])]); start = None
    crossings = []
    previous = None
    for i, v in enumerate(point[:-1]):
        if abs(v) <= 1e-12: continue
        if previous is not None and previous[1] * v < 0:
            crossings.append([float(GRID[previous[0]]), float(GRID[i])])
        previous = (i, v)
    return {'point_sign_change_brackets': crossings,
            'bootstrap_grid_crossing_counts': {str(i): int(np.sum(counts == i)) for i in np.unique(counts)},
            'bootstrap_fraction_with_crossing': float(np.mean(counts > 0)),
            'zero_compatible_grid_regions': regions,
            'scope': 'Grid diagnostics, not a percentile CI conditional on a selected unique root. Common endpoint 1 excluded.'}


def modeled_quality(e, tp, fp, parameter, fix, harm, dependence):
    if dependence == 'independent_units':
        return parameter ** e * (fix + (1 - fix) * parameter) ** tp * (1 - harm + harm * parameter) ** fp
    return parameter ** e * (fix + (1 - fix) * parameter ** tp) * (1 - harm + harm * parameter ** fp)


def modeled_histograms(e, tp, fp, fix, harm, dependence):
    support = int(np.max(e + tp + fp))
    probabilities = np.zeros((len(e), support + 1))
    cache = {}
    for i, (missed, hit, false) in enumerate(zip(e, tp, fp)):
        key = (int(hit), int(false))
        if key not in cache:
            if dependence == 'independent_units':
                a = binom.pmf(np.arange(hit + 1), hit, 1 - fix)
                b = binom.pmf(np.arange(false + 1), false, harm)
            else:
                a = np.zeros(hit + 1); a[0] += fix; a[hit] += 1 - fix
                b = np.zeros(false + 1); b[0] += 1 - harm; b[false] += harm
            cache[key] = np.convolve(a, b)
        probabilities[i, missed:missed + len(cache[key])] = cache[key]
    assert np.max(np.abs(probabilities.sum(axis=1) - 1)) < 1e-12
    means = probabilities @ np.arange(support + 1)
    assert np.max(np.abs(means - (e + (1 - fix) * tp + harm * fp))) < 1e-10
    for parameter in (0, .25, .5, .75, 1):
        direct = probabilities @ (parameter ** np.arange(support + 1))
        analytic = modeled_quality(e, tp, fp, parameter, fix, harm, dependence)
        assert np.max(np.abs(direct - analytic)) < 1e-12
    return probabilities


def expected_caps(probabilities):
    tails = np.cumsum(probabilities[:, ::-1], axis=1)[:, ::-1][:, 1:]
    return np.cumsum(tails, axis=1)


def load_inputs(folder, draws, seed):
    rows = list(csv.DictReader((folder / 'records.csv').open()))
    reference = json.loads((folder / 'reference.json').read_text())
    source_ids = json.loads((folder / 'source_ids.json').read_text())
    original = np.load(folder / 'source_weights.npy')
    assert hashlib.sha256(original.tobytes()).hexdigest() == reference['bootstrap']['weights_sha256']
    assert original.shape == (2000, len(source_ids)) and draws >= 2000
    rng = np.random.default_rng(seed)
    added = rng.multinomial(len(source_ids), np.full(len(source_ids), 1 / len(source_ids)), draws - 2000)
    weights = np.vstack([original, added]).astype(np.int16)
    assert np.all(weights.sum(axis=1) == len(source_ids))
    grouped = {}
    for row in rows: grouped.setdefault(row['model'], {})[row['response_id']] = row
    first = next(iter(grouped.values())); ids = sorted(first, key=lambda x: int(x))
    source_index = {source: i for i, source in enumerate(source_ids)}
    index = np.array([source_index[first[i]['source_id']] for i in ids])
    identity = {i: (first[i]['source_id'], first[i]['words'], first[i]['positive_words']) for i in ids}
    models = {}
    for name, records in grouped.items():
        assert set(records) == set(ids)
        assert {i: (records[i]['source_id'], records[i]['words'], records[i]['positive_words']) for i in ids} == identity
        data = {key: np.array([int(records[i][field]) for i in ids])
                for key, field in [('e', 'remaining_words'), ('k', 'selected_words'), ('p', 'positive_words'), ('n', 'words')]}
        data['tp'] = data['p'] - data['e']; data['fp'] = data['k'] - data['tp']
        assert np.all(data['e'] >= 0) and np.all(data['e'] <= data['p'])
        assert np.all(data['k'] >= 0) and np.all(data['k'] <= data['n']) and np.all(data['fp'] >= 0)
        models[name] = data
    original_counts = next(iter(models.values()))
    models['no_action'] = {'e': original_counts['p'], 'k': np.zeros(len(ids), dtype=int),
                          'p': original_counts['p'], 'n': original_counts['n'],
                          'tp': np.zeros(len(ids), dtype=int), 'fp': np.zeros(len(ids), dtype=int)}
    assert len(ids) == reference['responses'] == 2675 and len(source_ids) == reference['sources'] == 450
    denominator = weights @ np.bincount(index, minlength=len(source_ids))
    assert np.all(denominator > 0)
    return models, ids, index, source_ids, weights, denominator, reference


def tuned_policies(repo, models, ids, index, source_ids, weights, denominator, reference):
    previous_path = list(sys.path)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import expanded as core
    finally:
        sys.path[:] = previous_path
    assert sha(repo / 'dataset/response.jsonl') == reference['response_sha256']
    assert sha(repo / 'dataset/source_info.jsonl') == reference['source_sha256']
    rows = core.parse(repo); scorer, fit = core.fit_models(rows)
    assert fit == reference['fit']
    validation = [r for r in rows if r['split'] == 'validation' and r['quality'] == 'good' and len(r['y'])]
    test_map = {r['id']: r for r in rows if r['split'] == 'test' and r['quality'] == 'good' and len(r['y'])}
    assert set(test_map) == set(ids)
    assert not {r['source_id'] for r in validation} & set(source_ids)
    test = [test_map[i] for i in ids]
    outputs, maxima, logs, transport = {}, [], [], {}
    for model in ('lexical', 'lexical_metadata', 'task_generator'):
        prepared = {}
        for split, data in [('validation', validation), ('test', test)]:
            pairs = []
            for row in data:
                scores = scorer(row, model); order = np.argsort(scores)
                pairs.append((scores[order], np.r_[0, np.cumsum(row['y'][order])], int(row['y'].sum())))
            prepared[split] = pairs
        scores = np.concatenate([x[0] for x in prepared['validation']])
        old = reference['models'][model]['threshold']
        candidates = np.unique(np.r_[np.quantile(scores, np.linspace(0, 1, 101)), old, np.inf])
        def counts(split, thresholds):
            es, ks = [], []
            for sorted_scores, cumulative, positive in prepared[split]:
                cuts = np.searchsorted(sorted_scores, thresholds, side='left')
                es.append(cumulative[cuts]); ks.append(len(sorted_scores) - cuts)
            return np.asarray(es), np.asarray(ks)
        ve, vk = counts('validation', candidates)
        te, tk = counts('test', np.array([old])); te = te[:, 0]; tk = tk[:, 0]
        assert np.array_equal(te, models[model]['e']) and np.array_equal(tk, models[model]['k'])
        positives_validation = np.array([x[2] for x in prepared['validation']])[:, None]
        positives_test = np.array([x[2] for x in prepared['test']])
        settings = [(1, 0, 'independent_units')]
        settings += [(fix, harm, dependence) for fix, harm in SENSITIVITY_CASES
                     for dependence in ('independent_units', 'response_coupled')]
        ideal_actions = {}
        for fix, harm, dependence in settings:
            for j, (a, b, shared, continuing) in enumerate(OBJECTIVES):
                vp = positives_validation - ve; vf = vk - vp
                quality = modeled_quality(ve, vp, vf, a, fix, harm, dependence)
                vv = np.mean(quality - shared * (1 - b ** vk) - continuing * vk, axis=0)
                winners = np.flatnonzero(np.isclose(vv, vv.max(), atol=1e-12, rtol=0))
                winner = min(winners, key=lambda x: (float(vk[:, x].mean()), -candidates[x]))
                chosen = candidates[winner]
                re, rk = counts('test', np.array([chosen])); re = re[:, 0]; rk = rk[:, 0]
                tp_old = positives_test - te; fp_old = tk - tp_old
                tp_new = positives_test - re; fp_new = rk - tp_new
                baseline = modeled_quality(te, tp_old, fp_old, a, fix, harm, dependence) - shared * (1 - b ** tk) - continuing * tk
                revised = modeled_quality(re, tp_new, fp_new, a, fix, harm, dependence) - shared * (1 - b ** rk) - continuing * rk
                difference = revised - baseline
                point = np.array([difference.mean()])
                boot = (weights @ source_sum(difference, index, len(source_ids))) / denominator
                summary, maximum = curve_summary(point, boot[:, None]); maxima.append(maximum)
                name = model + '__objective_' + str(j)
                if fix != 1 or harm != 0:
                    name += f'__fix_{fix}_harm_{harm}_{dependence}'
                outputs[name] = {'model': model, 'objective': [a, b, shared, continuing],
                                 'fix_probability': fix, 'harm_probability': harm, 'dependence': dependence,
                                 'f1_threshold': old, 'value_threshold': None if np.isinf(chosen) else float(chosen),
                                 'no_action_selected': bool(np.isinf(chosen)),
                                 'validation_value': float(vv[winner]),
                                 'f1_validation_value': float(vv[np.flatnonzero(candidates == old)[0]]),
                                 'test_f1_value': float(baseline.mean()), 'test_tuned_value': float(revised.mean()),
                                 'test_f1_checks': float(tk.mean()), 'test_tuned_checks': float(rk.mean()),
                                 'test_f1_errors': float(te.mean()), 'test_tuned_errors': float(re.mean()),
                                 'test_f1_expected_errors': float(np.mean(te + (1 - fix) * tp_old + harm * fp_old)),
                                 'test_tuned_expected_errors': float(np.mean(re + (1 - fix) * tp_new + harm * fp_new)),
                                 'contrast': summary}
                for i, response in enumerate(ids):
                    logs.append({'policy': name, 'response_id': response, 'source_id': source_ids[index[i]],
                                 'annotation_misses': int(re[i]), 'flags': int(rk[i]),
                                 'positive_words': int(positives_test[i]), 'flagged_errors': int(tp_new[i]),
                                 'flagged_correct_units': int(fp_new[i]), 'fix_probability': fix,
                                 'harm_probability': harm, 'dependence': dependence,
                                 'expected_remaining_errors': float(re[i] + (1-fix)*tp_new[i] + harm*fp_new[i])})
                if fix == 1 and harm == 0:
                    ideal_actions[j] = (re, rk, chosen)
                else:
                    ie, ik, ith = ideal_actions[j]
                    itp = positives_test - ie; ifp = ik - itp
                    transported = modeled_quality(ie, itp, ifp, a, fix, harm, dependence) - shared * (1 - b ** ik) - continuing * ik
                    diff = transported - baseline
                    summary, maximum = curve_summary(np.array([diff.mean()]), ((weights @ source_sum(diff, index, len(source_ids))) / denominator)[:, None])
                    maxima.append(maximum)
                    transport[name] = {'model': model, 'objective': [a, b, shared, continuing],
                                       'fix_probability': fix, 'harm_probability': harm, 'dependence': dependence,
                                       'ideal_selected_threshold': None if np.isinf(ith) else float(ith),
                                       'test_transported_value': float(transported.mean()),
                                       'test_f1_value': float(baseline.mean()), 'contrast': summary}
    return outputs, maxima, logs, {'validation_responses': len(validation),
                                  'validation_sources': len({r['source_id'] for r in validation}),
                                  'selection_uses_test_labels': False, 'fit_identical': True}, transport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ragtruth', type=Path)
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261019)
    args = parser.parse_args(); started = time.time()
    models, ids, index, source_ids, weights, denominator, reference = load_inputs(args.inputs, args.draws, args.seed)
    sources, n = len(source_ids), len(ids)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / 'source_weights.npy', weights)
    raw = {}; source_ge = {}; source_gk = {}; all_maxima = []
    metrics = ('mean_errors', 'mean_checks', 'zero_errors', 'zero_checks', 'true_positives', 'false_positives', 'context_words')
    for name, data in models.items():
        values = np.column_stack([data['e'], data['k'], data['e'] == 0, data['k'] == 0,
                                  data['tp'], data['fp'], data['n'] * (data['k'] > 0)])
        atoms = source_sum(values, index, sources); point = values.mean(axis=0)
        boot = (weights @ atoms) / denominator[:, None]; ci = intervals(boot)
        if name != 'no_action':
            assert abs(point[0] - reference['models'][name]['point']['remaining_words']) < 1e-12
            assert abs(point[1] - reference['models'][name]['point']['selected_words']) < 1e-12
            for a in (.25, .5, .75):
                assert abs(np.mean(1 - a ** data['e']) - reference['models'][name]['point'][f'loss_{a}']) < 1e-12
        raw[name] = {'metrics': {metric: {'point': float(point[i]), 'confidence95': [float(ci[0, i]), float(ci[1, i])]}
                                  for i, metric in enumerate(metrics)},
                     'error_histogram': np.bincount(data['e']).tolist(), 'work_histogram': np.bincount(data['k']).tolist()}
        source_ge[name] = source_sum(GRID[None, :] ** data['e'][:, None], index, sources)
        source_gk[name] = source_sum(GRID[None, :] ** data['k'][:, None], index, sources)
    print(json.dumps({'stage': 'fixed_policy_readouts', 'models': len(models), 'draws': len(weights)}), flush=True)
    comparisons = {}
    source_cap_cache = {}
    def cap_source(name, field, support):
        key = (name, field, support)
        if key not in source_cap_cache:
            values = np.minimum(models[name][field][:, None], np.arange(1, support + 1)[None, :])
            source_cap_cache[key] = source_sum(values, index, sources)
        return source_cap_cache[key]
    for first, second in PAIRS:
        name = first + '__minus__' + second; pair = {'first': first, 'second': second}
        for field, label in [('e', 'errors'), ('k', 'work')]:
            support = int(max(models[first][field].max(), models[second][field].max()))
            atoms = cap_source(first, field, support) - cap_source(second, field, support)
            point = atoms.sum(axis=0) / n; boot = (weights @ atoms) / denominator[:, None]
            summary, maximum = curve_summary(point, boot); all_maxima.append(maximum)
            pair[label] = summary
        comparisons[name] = pair
    sensitivity = {}; sensitivity_quality_grid = (0, .05, .1, .25, .5, .75, .9, .95, .99, 1)
    for fix in FIX_GRID:
        for harm in HARM_GRID:
            for dependence in ('independent_units', 'response_coupled'):
                name = f'fix_{fix}_harm_{harm}_{dependence}'
                sensitivity[name] = {'fix_probability': fix, 'harm_probability': harm, 'dependence': dependence, 'models': {}}
                for model in models:
                    data = models[model]
                    sensitivity[name]['models'][model] = {
                        'mean_errors': float(np.mean(data['e'] + (1 - fix) * data['tp'] + harm * data['fp'])),
                        'quality': {str(a): float(np.mean(modeled_quality(data['e'], data['tp'], data['fp'], a, fix, harm, dependence))) for a in sensitivity_quality_grid}}
    sensitivity_bands = {}
    for fix, harm in SENSITIVITY_CASES:
        for dependence in ('independent_units', 'response_coupled'):
            name = f'fix_{fix}_harm_{harm}_{dependence}'
            probabilities = {}
            for model in ('lexical_metadata', 'lexical'):
                d = models[model]; probabilities[model] = modeled_histograms(d['e'], d['tp'], d['fp'], fix, harm, dependence)
            support = max(x.shape[1] for x in probabilities.values()) - 1
            values = {}
            for model, p in probabilities.items():
                padded = np.pad(p, ((0, 0), (0, support + 1 - p.shape[1])))
                values[model] = expected_caps(padded)
            atoms = source_sum(values['lexical_metadata'] - values['lexical'], index, sources)
            point = atoms.sum(axis=0) / n; boot = (weights @ atoms) / denominator[:, None]
            summary, maximum = curve_summary(point, boot); all_maxima.append(maximum)
            sensitivity_bands[name] = {'errors': summary, 'work': comparisons['lexical_metadata__minus__lexical']['work'],
                                       'fix_probability': fix, 'harm_probability': harm, 'dependence': dependence}
    print(json.dumps({'stage': 'correction_sensitivity', 'scenarios': len(sensitivity), 'inference_scenarios': len(sensitivity_bands)}), flush=True)
    tuning, tuning_maxima, logs, tuning_scope, transport = ({}, [], [], {}, {})
    if args.ragtruth:
        tuning, tuning_maxima, logs, tuning_scope, transport = tuned_policies(args.ragtruth, models, ids, index, source_ids, weights, denominator, reference)
        all_maxima.extend(tuning_maxima)
    maximum = np.maximum.reduce(all_maxima)
    critical = float(np.quantile(maximum, .95, method='higher'))
    np.save(args.out / 'joint_max_statistics.npy', maximum)
    for pair in list(comparisons.values()) + list(sensitivity_bands.values()):
        for label in ('errors', 'work'):
            summary = pair[label]; point = np.asarray(summary['point']); se = np.asarray(summary['standard_error'])
            summary['simultaneous_low'] = (point - critical * se).tolist()
            summary['simultaneous_high'] = (point + critical * se).tolist()
        curves = {}
        for label in ('errors', 'work'):
            bands = np.array([propagated_band(pair[label], x, critical) for x in GRID])
            curves[label] = {'parameter': GRID.tolist(), 'delta': bands[:, 0].tolist(),
                             'simultaneous_low': bands[:, 1].tolist(), 'simultaneous_high': bands[:, 2].tolist()}
        pair['generating_function_contrasts'] = curves
    roots = {}
    for name, pair in comparisons.items():
        roots[name] = {}
        for label, cached in [('errors', source_ge), ('work', source_gk)]:
            atoms = cached[pair['first']] - cached[pair['second']]
            point = atoms.sum(axis=0) / n; boot = (weights @ atoms) / denominator[:, None]
            point[-1] = 0; boot[:, -1] = 0
            curve = pair['generating_function_contrasts'][label]
            assert np.max(np.abs(point - np.asarray(curve['delta']))) < 1e-12
            roots[name][label] = root_diagnostics(point, boot, curve['simultaneous_low'], curve['simultaneous_high'])
            if name == 'lexical_metadata__minus__lexical' and label == 'errors':
                for a in (.25, .5, .75):
                    column = int(round(a * 500)); existing = reference['paired_contrasts']['offset'][f'loss_{a}']['confidence95']
                    ci = intervals(-boot[:2000, column])
                    assert abs(ci[0] - existing['low']) < 1e-12 and abs(ci[1] - existing['high']) < 1e-12
    for row in list(tuning.values()) + list(transport.values()):
        s = row['contrast']; point = s['point'][0]; se = s['standard_error'][0]
        row['simultaneous_confidence95'] = [point - critical * se, point + critical * se]
    surfaces = []
    pair = comparisons['lexical_metadata__minus__lexical']
    for a in (.05, .1, .25, .5, .75, .9, .99):
        for b in (.1, .25, .5, .75, .9, .95, .99):
            for shared in (0, .01, .05, .1, .25, .5, 1):
                for continuing in (0, .001, .005, .01, .025, .05, .1):
                    point, low, high = value_band(pair['errors'], pair['work'], a, b, shared, continuing, critical)
                    surfaces.append({'a': a, 'b': b, 'F': shared, 'v': continuing, 'difference': point,
                                     'simultaneous_low': low, 'simultaneous_high': high})
    report = {'population': {'responses': n, 'sources': sources, 'initially_erroneous': int(np.sum(models['no_action']['e'] > 0))},
              'bootstrap': {'draws': len(weights), 'preserved_original_draws': 2000, 'additional_seed': args.seed,
                            'weights_sha256': hashlib.sha256(weights.tobytes()).hexdigest(), 'critical_max_t': critical,
                            'joint_family': f'All five fixed-policy capped-profile contrasts, four modeled-correction contrasts, {len(tuning)} validation-tuning contrasts and {len(transport)} ideal-policy transport contrasts.',
                            'propagation': 'Nonnegative discrete concave-basis coefficients give simultaneous bounds for every common geometric error/cost parameter and nonnegative price, on the declared finite count support.',
                            'conditioning': 'Fitting, original thresholds, modeled repair probabilities and validation-selected rules held fixed; complete source groups resampled together.'},
              'models': raw, 'comparisons': comparisons, 'root_diagnostics': roots,
              'correction_sensitivity': sensitivity, 'correction_sensitivity_bands': sensitivity_bands,
              'validation_tuning': tuning, 'validation_tuning_scope': tuning_scope, 'ideal_policy_transport': transport,
              'input_sha256': {p.name: sha(p) for p in args.inputs.iterdir() if p.is_file()},
              'script_sha256': sha(Path(__file__)), 'elapsed_seconds': time.time() - started,
              'scope': 'Annotation misses and flagged-unit work under stated ideal/model-based correction. No human or neural repaired-output experiment; bootstrap does not estimate uncertainty in assumed repair probabilities.',
              'all_checks_passed': True}
    (args.out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for filename, records in [('value_surface.csv', surfaces), ('tuned_policy_counts.csv', logs)]:
        if records:
            with (args.out / filename).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(records)
    print(json.dumps({'stage': 'complete', 'critical_max_t': critical, 'elapsed_seconds': report['elapsed_seconds'],
                      'tuning': {name: {'delta': row['contrast']['point'][0], 'confidence95': row['simultaneous_confidence95'], 'no_action': row['no_action_selected']} for name, row in tuning.items() if row['fix_probability'] == 1 and row['harm_probability'] == 0}, 'tuning_comparisons': len(tuning), 'transport_comparisons': len(transport)}), flush=True)


if __name__ == '__main__':
    main()
