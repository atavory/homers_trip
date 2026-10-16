"""Reproduce fixed frequency scorers and evaluate consistent span utility families."""

import argparse
import collections
from fractions import Fraction
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
from scipy import sparse
from scipy.optimize import brentq

import alternatives as alt
import expanded as core


BUDGETS = (0.05, 0.10, 0.20)
MODELS = ('lexical', 'lexical_metadata')
DEFAULT_DATA = Path('.')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recovery_law(above, tied, size, tie_population, selected_ties):
    lower = max(0, tied - (tie_population - selected_ties))
    upper = min(tied, selected_ties)
    denominator = math.comb(tie_population, tied)
    pmf = [
        (Fraction(above + count, size),
         Fraction(math.comb(selected_ties, count) *
                  math.comb(tie_population - selected_ties, tied - count), denominator))
        for count in range(lower, upper + 1)
    ]
    assert sum(probability for _, probability in pmf) == 1
    mean = sum(fraction * probability for fraction, probability in pmf)
    assert mean == (Fraction(above, size) +
                    Fraction(selected_ties * tied, tie_population * size))
    return pmf


def point_distributions(rows, scores, selected_count):
    flat = np.concatenate(scores)
    threshold, q, tie_n = alt.cutoff(flat, selected_count)
    positives = sum(int(row['y'].sum()) for row in rows)
    span_count = sum(len(row['spans']) for row in rows)
    weighted = {'uniform_span': collections.defaultdict(list),
                'length_weighted_span': collections.defaultdict(list)}
    span_records = []
    for row, score in zip(rows, scores):
        for index, indices in enumerate(row['spans']):
            local = score[indices]
            above = int(np.sum(local > threshold))
            tied = int(np.sum(local == threshold))
            size = len(indices)
            pmf = recovery_law(above, tied, size, tie_n, q)
            for fraction, probability in pmf:
                weighted['uniform_span'][fraction].append(float(probability) / span_count)
                weighted['length_weighted_span'][fraction].append(float(probability) * size / positives)
            span_records.append({
                'response_id': row['id'], 'source_id': row['source_id'], 'span_index': index,
                'span_words': size, 'above_cutoff_words': above, 'cutoff_tie_words': tied,
                'law': [{'numerator': fraction.numerator, 'denominator': fraction.denominator,
                         'probability': float(probability)} for fraction, probability in pmf],
            })
    laws = {name: {fraction: math.fsum(masses) for fraction, masses in masses_by_fraction.items()}
            for name, masses_by_fraction in weighted.items()}
    assert all(abs(math.fsum(law.values()) - 1) < 1e-12 for law in laws.values())
    return laws, span_records, {'threshold': float(threshold), 'selected_words': selected_count,
                               'cutoff_tie_population': tie_n, 'selected_cutoff_ties': q}


def utility(law, alpha):
    if alpha == 0:
        return math.fsum(mass for fraction, mass in law.items() if fraction > 0)
    if math.isinf(alpha):
        return law.get(Fraction(1), 0.0)
    return math.fsum(mass * float(fraction) ** alpha
                     for fraction, mass in law.items() if fraction > 0)


def distribution_comparison(a, b, alphas):
    def difference(alpha):
        return utility(a, alpha) - utility(b, alpha)
    values = [{'alpha': alpha, 'lexical_metadata_minus_lexical': difference(alpha)}
              for alpha in alphas]
    brackets = []
    for left, right in zip(values, values[1:]):
        if abs(left['lexical_metadata_minus_lexical']) <= 1e-12 or abs(right['lexical_metadata_minus_lexical']) <= 1e-12:
            continue
        if left['lexical_metadata_minus_lexical'] * right['lexical_metadata_minus_lexical'] < 0:
            root = brentq(difference, left['alpha'], right['alpha'])
            brackets.append({'left': left['alpha'], 'right': right['alpha'], 'root': root})
    support = sorted(set(a) | set(b))
    cumulative_a = cumulative_b = 0.0
    differences = []
    for fraction in support:
        cumulative_a += a.get(fraction, 0.0)
        cumulative_b += b.get(fraction, 0.0)
        differences.append(cumulative_a - cumulative_b)
    return {
        'power_curve': values, 'positive_alpha_root_brackets': brackets,
        'touched_difference': difference(0), 'partial_difference': difference(1),
        'complete_difference': difference(math.inf),
        'minimum_cdf_metadata_minus_lexical': min(differences),
        'maximum_cdf_metadata_minus_lexical': max(differences),
        'cdf_crosses': min(differences) < -1e-12 and max(differences) > 1e-12,
        'lexical_metadata_fosd': max(differences) <= 1e-12,
        'lexical_fosd': min(differences) >= -1e-12,
        'root_scan_limit': 'Brackets establish point-curve roots; absence of a bracket is not a proof of no roots.',
    }


def selection_inputs(rows, scores, source_ids):
    lookup = {source: index for index, source in enumerate(source_ids)}
    flat = np.concatenate(scores)
    _, bins = np.unique(flat, return_inverse=True)
    sources = np.concatenate([np.full(len(row['y']), lookup[row['source_id']], dtype=np.int64)
                              for row in rows])
    uniform_partial_word_weight = np.concatenate([
        np.zeros(len(row['y']), dtype=float) for row in rows])
    span_sources, span_sizes = [], []
    budgets = np.zeros((len(source_ids), len(BUDGETS)))
    offset = 0
    for row in rows:
        source = lookup[row['source_id']]
        for i, budget in enumerate(BUDGETS):
            budgets[source, i] += math.ceil(budget * len(row['y']))
        for indices in row['spans']:
            uniform_partial_word_weight[offset + indices] += 1.0 / len(indices)
            span_sources.append(source)
            span_sizes.append(len(indices))
        offset += len(row['y'])
    assert abs(uniform_partial_word_weight.sum() - len(span_sizes)) < 1e-9
    shape = (len(source_ids), int(bins.max()) + 1)
    partial_histogram = sparse.csr_matrix(
        (uniform_partial_word_weight, (sources, bins)), shape=shape, dtype=np.float64)
    histograms, spans = alt.score_histograms(rows, scores, source_ids)
    assert np.array_equal(spans[0], np.asarray(span_sources))
    return histograms[:2], partial_histogram, spans, np.asarray(span_sizes), budgets


def bootstrap_endpoints(rows, scores, source_ids, weights, saved_boot, model):
    histograms, partial_histogram, spans, sizes, budgets = selection_inputs(rows, scores, source_ids)
    span_source, span_lo, _, span_nlo, _ = spans
    output = collections.defaultdict(list)
    maximum_original_error = 0.0
    for start in range(0, len(weights), 100):
        w = weights[start:start + 100].astype(float)
        pc, nc = [histogram.T.dot(w.T).T for histogram in histograms]
        partial = partial_histogram.T.dot(w.T).T
        total = pc + nc
        cumulative_total, cumulative_p, cumulative_partial = [
            np.cumsum(array, axis=1) for array in (total, pc, partial)]
        reverse_total = np.cumsum(total[:, ::-1], axis=1)
        positive_count = pc.sum(axis=1)
        span_count = w[:, span_source].sum(axis=1)
        row_ids = np.arange(len(w))
        selected_counts = w @ budgets
        for i, budget in enumerate(BUDGETS):
            selected = selected_counts[:, i]
            cuts = total.shape[1] - 1 - np.argmax(reverse_total >= selected[:, None], axis=1)
            tie_n = total[row_ids, cuts].astype(np.int64)
            q = (selected - (total.sum(axis=1) - cumulative_total[row_ids, cuts])).astype(np.int64)
            assert np.all((q > 0) & (q <= tie_n))
            partial_recall = (partial.sum(axis=1) - cumulative_partial[row_ids, cuts] +
                              q / tie_n * partial[row_ids, cuts]) / span_count
            word_recall = (positive_count - cumulative_p[row_ids, cuts] +
                           q / tie_n * pc[row_ids, cuts]) / positive_count
            complete_uniform, complete_length = [], []
            for j, cut in enumerate(cuts):
                multiplicity = w[j, span_source]
                eq = span_lo == cut
                probability = alt.all_selected_probability(int(tie_n[j]), int(q[j]), span_nlo[eq])
                uniform = multiplicity[span_lo > cut].sum() + np.dot(multiplicity[eq], probability)
                length = (multiplicity[span_lo > cut] * sizes[span_lo > cut]).sum()
                length += np.dot(multiplicity[eq] * sizes[eq], probability)
                complete_uniform.append(uniform / span_count[j])
                complete_length.append(length / positive_count[j])
            complete_uniform, complete_length = np.asarray(complete_uniform), np.asarray(complete_length)
            for key, values in ((f'global_word_recall_{budget}', word_recall),
                                (f'global_complete_span_recall_{budget}', complete_uniform)):
                expected = saved_boot[model + '__' + key][start:start + len(w)]
                error = float(np.max(np.abs(values - expected)))
                assert error < 1e-12, (model, key, error)
                maximum_original_error = max(maximum_original_error, error)
            output[f'global_uniform_span_partial_recall_{budget}'].extend(partial_recall.tolist())
            output[f'global_length_weighted_complete_span_recall_{budget}'].extend(complete_length.tolist())
    return {key: np.asarray(values) for key, values in output.items()}, maximum_original_error


def confidence(values, family_size=1):
    finite = values[np.isfinite(values)]
    low, high = np.quantile(finite, [0.025 / family_size, 1 - 0.025 / family_size])
    return {'low': float(low), 'high': float(high), 'valid_draws': len(finite),
            'invalid_draws': len(values) - len(finite), 'family_size': family_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-project', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--ragtruth', type=Path, default=Path('data/RAGTruth'))
    parser.add_argument('--saved-results', type=Path)
    parser.add_argument('--source-groups', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    base = args.data_project
    saved = args.saved_results or base / 'analysis/alternatives_results'
    groups = args.source_groups or saved
    out = args.out or base / 'analysis/bandha_nonadditive_replay_results'
    out.mkdir(parents=True, exist_ok=True)
    results = json.loads((saved / 'results.json').read_text())
    assert sha256(Path(alt.__file__)) == results['code_sha256']
    assert sha256(Path(core.__file__)) == results['helper_sha256']
    for filename, key in (('response.jsonl', 'response_sha256'), ('source_info.jsonl', 'source_sha256')):
        assert sha256(args.ragtruth / 'dataset' / filename) == results[key]
    rows = core.parse(args.ragtruth)
    score_function, fit = core.fit_models(rows)
    assert fit == results['fit']
    test = [row for row in rows if row['split'] == 'test' and row['quality'] == 'good' and len(row['y'])]
    annotated = alt.attach_annotations(test, args.ragtruth / 'dataset/response.jsonl')
    assert annotated == results['annotations']
    sources = sorted({row['source_id'] for row in test})
    source_ids_path = groups / 'source_ids.json'
    if source_ids_path.exists():
        assert sources == json.loads(source_ids_path.read_text())
    assert len(sources) == results['bootstrap']['source_count']
    (out / 'source_ids.json').write_text(json.dumps(sources, indent=2) + '\n')
    weights = np.load(groups / 'source_weights.npy')
    assert hashlib.sha256(weights.tobytes()).hexdigest() == results['bootstrap']['weight_sha256']
    saved_boot = np.load(saved / 'bootstrap_estimates.npz')
    fitted = inspect.getclosurevars(score_function).nonlocals
    rates = {model: [{'key': list(key) if isinstance(key, tuple) else key, 'probability': value}
                     for key, value in sorted(table.items())]
             for model, table in fitted['rates'].items()}
    tables = {'fit': fit, 'pi': fitted['pi'], 'lexical_rates': dict(sorted(fitted['lex_rates'].items())),
              'metadata_rates': rates,
              'recipe': 'Unmodified expanded.py parse and fit_models; original training-only split, eligibility and smoothing.'}
    (out / 'bandha_fixed_frequency_tables.json').write_text(json.dumps(tables, indent=2) + '\n')
    scores_by_model = {model: [score_function(row, model) for row in test] for model in MODELS}
    original_point = {}
    maximum_original_point_error = 0.0
    for model, scores in scores_by_model.items():
        atoms, _ = alt.point_atoms(test, scores, results['models'][model]['threshold'], sources)
        histograms, span_hist = alt.score_histograms(test, scores, sources)
        point = {key: float(value[0]) for key, value in alt.evaluate_draws(
            atoms, histograms, span_hist, np.ones((1, len(sources)), dtype=np.int16)).items()}
        for key, value in point.items():
            error = abs(value - results['models'][model]['point'][key])
            assert error < 1e-12, (model, key, error)
            maximum_original_point_error = max(maximum_original_point_error, error)
        original_point[model] = point
    row_offsets = np.r_[0, np.cumsum([len(row['y']) for row in test])]
    np.savez_compressed(out / 'bandha_fixed_score_vectors.npz',
                        lexical=np.concatenate(scores_by_model['lexical']),
                        lexical_metadata=np.concatenate(scores_by_model['lexical_metadata']),
                        labels=np.concatenate([row['y'] for row in test]), response_offsets=row_offsets)
    (out / 'bandha_response_index.json').write_text(json.dumps([
        {'id': row['id'], 'source_id': row['source_id'], 'word_count': len(row['y']),
         'spans': [indices.tolist() for indices in row['spans']]}
        for row in test], separators=(',', ':')) + '\n')
    alphas = sorted(set([0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0] +
                        np.geomspace(0.0001, 512, 500).tolist()))
    distributions, cutoffs, point_endpoints = {}, {}, {}
    for budget in BUDGETS:
        selected_count = sum(math.ceil(budget * len(row['y'])) for row in test)
        distributions[str(budget)], cutoffs[str(budget)], point_endpoints[str(budget)] = {}, {}, {}
        for model, scores in scores_by_model.items():
            laws, spans, cutoff = point_distributions(test, scores, selected_count)
            distributions[str(budget)][model] = laws
            cutoffs[str(budget)][model] = cutoff
            endpoints = {family: {name: utility(law, alpha) for name, alpha in
                                 (('touched', 0), ('partial', 1), ('complete', math.inf))}
                         for family, law in laws.items()}
            point_endpoints[str(budget)][model] = endpoints
            old = original_point[model]
            assert abs(endpoints['length_weighted_span']['partial'] - old[f'global_word_recall_{budget}']) < 1e-12
            assert abs(endpoints['uniform_span']['complete'] - old[f'global_complete_span_recall_{budget}']) < 1e-12
            assert abs(endpoints['uniform_span']['touched'] - old[f'global_touched_span_recall_{budget}']) < 1e-12
            (out / f'bandha_{model}_span_laws_{budget}.json').write_text(json.dumps({
                'cutoff': cutoff, 'spans': spans}, separators=(',', ':')) + '\n')
    comparisons = {budget: {
        family: distribution_comparison(models['lexical_metadata'][family], models['lexical'][family], alphas)
        for family in ('uniform_span', 'length_weighted_span')}
        for budget, models in distributions.items()}
    boot, maximum_original_boot_error = {}, 0.0
    for model, scores in scores_by_model.items():
        print('BOOTSTRAP', model, flush=True)
        estimates, error = bootstrap_endpoints(test, scores, sources, weights, saved_boot, model)
        boot[model] = estimates
        maximum_original_boot_error = max(maximum_original_boot_error, error)
    np.savez_compressed(out / 'bandha_endpoint_bootstrap.npz',
                        **{model + '__' + key: values for model, estimates in boot.items()
                           for key, values in estimates.items()})
    paired = {}
    for budget in BUDGETS:
        for key, family, endpoint in (
            (f'global_uniform_span_partial_recall_{budget}', 'uniform_span', 'partial'),
            (f'global_length_weighted_complete_span_recall_{budget}', 'length_weighted_span', 'complete'),
        ):
            delta = boot['lexical_metadata'][key] - boot['lexical'][key]
            point = (point_endpoints[str(budget)]['lexical_metadata'][family][endpoint] -
                     point_endpoints[str(budget)]['lexical'][family][endpoint])
            paired[key] = {'point': point, 'confidence95': confidence(delta),
                           'three_budget_bonferroni_percentile': confidence(delta, 3),
                           'six_endpoint_bonferroni_percentile': confidence(delta, 6)}
    serialized_laws = {budget: {model: {family: [
        {'numerator': fraction.numerator, 'denominator': fraction.denominator, 'mass': mass}
        for fraction, mass in sorted(law.items())] for family, law in families.items()}
        for model, families in models.items()} for budget, models in distributions.items()}
    inputs = [saved / 'results.json', saved / 'bootstrap_estimates.npz', groups / 'source_weights.npy',
              Path(alt.__file__), Path(core.__file__),
              args.ragtruth / 'dataset/response.jsonl', args.ragtruth / 'dataset/source_info.jsonl']
    if source_ids_path.exists():
        inputs.append(source_ids_path)
    report = {
        'kind': 'exact_fixed_scorer_reproduction_and_joint_span_law',
        'command': ' '.join(sys.argv), 'script_sha256': sha256(Path(__file__)),
        'python_executable': sys.executable,
        'input_sha256': {str(path): sha256(path) for path in inputs},
        'dataset_revision': results['dataset_revision'], 'fit': fit, 'bootstrap': results['bootstrap'],
        'prediction_unit': 'whitespace-delimited words with any annotated-character overlap',
        'utility_unit': 'merged annotation spans',
        'weights': {'uniform_span': '1 / number of annotation spans',
                    'length_weighted_span': 'span words / total hallucinated words'},
        'new_tuning': False, 'llm_inference': False, 'tie_sampling': False,
        'tie_law': '(above_count + Hypergeometric(tie_population, selected_ties, span_ties)) / span_words',
        'original_point_metric_count': len(original_point['lexical']) * len(MODELS),
        'original_bootstrap_scalar_count': len(MODELS) * len(BUDGETS) * 2 * len(weights),
        'maximum_original_point_error': maximum_original_point_error,
        'maximum_original_bootstrap_error': maximum_original_boot_error,
        'cutoffs': cutoffs, 'point_endpoints': point_endpoints,
        'weighted_recovery_laws': serialized_laws, 'comparisons': comparisons,
        'paired_cross_unit_endpoints': paired, 'all_checks_passed': True,
        'elapsed_seconds': time.monotonic() - started,
    }
    (out / 'bandha_replay.json').write_text(json.dumps(report, indent=2) + '\n')
    curve_csv = ['budget,weighting,alpha,lexical,lexical_metadata,offset_contrast']
    for budget, models in distributions.items():
        for family in ('uniform_span', 'length_weighted_span'):
            for alpha in alphas + [math.inf]:
                a = utility(models['lexical'][family], alpha)
                b = utility(models['lexical_metadata'][family], alpha)
                curve_csv.append(f'{budget},{family},{alpha},{a:.17g},{b:.17g},{b-a:.17g}')
    (out / 'bandha_power_curves.csv').write_text('\n'.join(curve_csv) + '\n')
    endpoint_csv = ['budget,weighting,touched_delta,partial_delta,complete_delta,point_curve_roots']
    for budget, families in comparisons.items():
        for family, contrast in families.items():
            roots = ';'.join(str(item['root']) for item in contrast['positive_alpha_root_brackets'])
            endpoint_csv.append(f"{budget},{family},{contrast['touched_difference']:.17g},{contrast['partial_difference']:.17g},{contrast['complete_difference']:.17g},{roots}")
    (out / 'bandha_consistent_endpoints.csv').write_text('\n'.join(endpoint_csv) + '\n')
    print(json.dumps({
        'all_checks_passed': True, 'maximum_original_point_error': maximum_original_point_error,
        'maximum_original_bootstrap_error': maximum_original_boot_error,
        'consistent_endpoint_contrasts': {budget: {family: {
            key: value for key, value in contrast.items() if key in (
                'touched_difference', 'partial_difference', 'complete_difference',
                'positive_alpha_root_brackets', 'cdf_crosses', 'lexical_metadata_fosd', 'lexical_fosd')}
            for family, contrast in families.items()} for budget, families in comparisons.items()},
        'paired_cross_unit_endpoints': paired, 'output': str(out),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
