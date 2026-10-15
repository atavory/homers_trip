"""Evaluate response-conditional alternatives on fixed RAGTruth scores."""
import argparse
import bisect
import collections
import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.dont_write_bytecode = True
import numpy as np
from scipy import sparse
import expanded as core

BUDGETS = (0.05, 0.10, 0.20)
ATOM_NAMES = [
    'responses', 'words', 'positives', 'negatives', 'mixed', 'mixed_words',
    'unmarked_responses', 'unmarked_words', 'marked_responses', 'marked_words',
    'within_wins', 'within_pairs', 'auc_sum', 'ap_sum', 'nap_sum',
    'prevalence_sum', 'random_ap_sum', 'threshold_tp', 'threshold_fp',
    'unmarked_alarm_responses', 'unmarked_flagged_words', 'spans', 'characters',
]
for b in BUDGETS:
    ATOM_NAMES.extend(f'{name}_{b}' for name in [
        'budget', 'conditional_tp', 'conditional_complete', 'conditional_touch',
        'conditional_characters', 'conditional_unmarked_cost', 'lift_sum',
    ])
IX = {name: i for i, name in enumerate(ATOM_NAMES)}


def all_selected_probability(n, q, sizes):
    """Probability that all specified items lie in a uniform q-subset."""
    sizes = np.asarray(sizes, dtype=np.int64)
    out = np.zeros(sizes.shape, dtype=float)
    if q == n:
        return np.ones(sizes.shape, dtype=float)
    out[sizes == 0] = 1.0
    valid = (sizes > 0) & (sizes <= q)
    if np.any(valid):
        maximum = int(sizes[valid].max())
        offsets = np.arange(maximum)
        probabilities = np.r_[1.0, np.exp(np.cumsum(np.log((q - offsets) / (n - offsets))))]
        out[valid] = probabilities[sizes[valid]]
    return out


def cutoff(scores, k):
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = scores > threshold
    tied = scores == threshold
    return threshold, int(k - above.sum()), int(tied.sum())


def span_events(scores, spans, threshold, q, n):
    minimum, maximum, min_sizes, max_sizes = [], [], [], []
    for indices in spans:
        selected = scores[indices]
        lo, hi = selected.min(), selected.max()
        minimum.append(lo)
        maximum.append(hi)
        min_sizes.append(int((selected == lo).sum()))
        max_sizes.append(int((selected == hi).sum()))
    minimum, maximum = np.asarray(minimum), np.asarray(maximum)
    complete = (minimum > threshold).astype(float)
    touched = (maximum > threshold).astype(float)
    eq = minimum == threshold
    complete[eq] = all_selected_probability(n, q, np.asarray(min_sizes)[eq])
    eq = maximum == threshold
    touched[eq] = 1.0 - all_selected_probability(n, n - q, np.asarray(max_sizes)[eq])
    return complete, touched


def attach_annotations(rows, response_file):
    wanted = {r['id']: r for r in rows}
    span_count = excluded = 0
    for record in map(json.loads, response_file.read_text().splitlines()):
        if record['id'] not in wanted:
            continue
        row = wanted[record['id']]
        matches = list(re.finditer(r'\S+', record['response']))
        starts, ends = [m.start() for m in matches], [m.end() for m in matches]
        intervals = []
        for a, b in sorted((v['start'], v['end']) for v in record['labels']):
            if intervals and a <= intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], max(b, intervals[-1][1]))
            else:
                intervals.append((a, b))
        row['spans'] = []
        row['characters'] = np.zeros(len(matches), dtype=np.int64)
        covered = set()
        for a, b in intervals:
            left, right = bisect.bisect_right(ends, a), bisect.bisect_left(starts, b)
            indices = np.arange(left, right, dtype=np.int32)
            if not len(indices):
                excluded += 1
                continue
            row['spans'].append(indices)
            covered.update(map(int, indices))
            for i in indices:
                row['characters'][i] += min(b, ends[i]) - max(a, starts[i])
            span_count += 1
        assert covered == set(map(int, np.flatnonzero(row['y']))), row['id']
        assert len(matches) == len(row['y'])
    return {'eligible_merged_spans': span_count, 'spans_without_eligible_words': excluded}


def point_atoms(rows, scores, threshold, source_ids):
    source_index = {v: i for i, v in enumerate(source_ids)}
    atoms = np.zeros((len(source_ids), len(ATOM_NAMES)))
    per_response = []
    maximum_length = max(len(r['y']) for r in rows)
    harmonic = np.r_[0.0, np.cumsum(1.0 / np.arange(1, maximum_length + 1))]
    for row, score in zip(rows, scores):
        z = core.rank_metrics(row['y'], score)
        n, p, neg = len(score), z['P'], z['N']
        mixed = bool(p and neg)
        ap = z['ap'] if mixed else 0.0
        prevalence = p / n
        nap = (ap - prevalence) / (1 - prevalence) if mixed else 0.0
        if mixed and np.all(score == score[0]):
            assert abs(ap - prevalence) < 1e-12
            nap = 0.0
        random_ap = (p - 1) / (n - 1) + neg * harmonic[n] / (n * (n - 1)) if mixed else 0.0
        flagged = score >= threshold
        values = {
            'responses': 1, 'words': n, 'positives': p, 'negatives': neg,
            'mixed': int(mixed), 'mixed_words': n if mixed else 0,
            'unmarked_responses': int(p == 0), 'unmarked_words': n if p == 0 else 0,
            'marked_responses': int(neg == 0), 'marked_words': n if neg == 0 else 0,
            'within_wins': z['wins'] if mixed else 0, 'within_pairs': p * neg,
            'auc_sum': z['auc'] if mixed else 0, 'ap_sum': ap, 'nap_sum': nap,
            'prevalence_sum': prevalence if mixed else 0, 'random_ap_sum': random_ap,
            'threshold_tp': int(row['y'][flagged].sum()),
            'threshold_fp': int((1 - row['y'][flagged]).sum()),
            'unmarked_alarm_responses': int(p == 0 and np.any(flagged)),
            'unmarked_flagged_words': int(flagged.sum()) if p == 0 else 0,
            'spans': len(row['spans']), 'characters': int(row['characters'].sum()),
        }
        for b in BUDGETS:
            k = math.ceil(b * n)
            thr, q, tie_n = cutoff(score, k)
            tp = core.budget_counts(row['y'], score, k)
            complete, touched = span_events(score, row['spans'], thr, q, tie_n)
            assert np.all(complete <= touched + 1e-12)
            values.update({
                f'budget_{b}': k, f'conditional_tp_{b}': tp,
                f'conditional_complete_{b}': float(complete.sum()),
                f'conditional_touch_{b}': float(touched.sum()),
                f'conditional_characters_{b}': float(row['characters'][score > thr].sum() + q / tie_n * row['characters'][score == thr].sum()),
                f'conditional_unmarked_cost_{b}': k if p == 0 else 0,
                f'lift_sum_{b}': (tp / k) / prevalence if mixed else 0,
            })
        atom = np.array([values[name] for name in ATOM_NAMES])
        atoms[source_index[row['source_id']]] += atom
        per_response.append({'id': row['id'], 'source_id': row['source_id'], 'n': n, 'n_positive': p,
                             'mixed': mixed, 'within_auc': z['auc'], 'within_ap': z['ap'],
                             'within_nap': nap if mixed else None, 'values': values})
    return atoms, per_response


def ratios(atom):
    def at(name): return atom[:, IX[name]]
    def divide(a, b):
        return np.divide(a, b, out=np.full(np.shape(a), np.nan), where=b != 0)
    out = {
        'within_pair_auc': divide(at('within_wins'), at('within_pairs')),
        'within_macro_auc': divide(at('auc_sum'), at('mixed')),
        'within_macro_ap': divide(at('ap_sum'), at('mixed')),
        'within_macro_nap': divide(at('nap_sum'), at('mixed')),
        'mixed_macro_prevalence': divide(at('prevalence_sum'), at('mixed')),
        'mixed_macro_random_ap_expectation': divide(at('random_ap_sum'), at('mixed')),
        'pooled_f1': divide(2 * at('threshold_tp'), at('positives') + at('threshold_tp') + at('threshold_fp')),
        'unmarked_response_alarm_rate': divide(at('unmarked_alarm_responses'), at('unmarked_responses')),
        'unmarked_word_false_positive_rate': divide(at('unmarked_flagged_words'), at('unmarked_words')),
    }
    for b in BUDGETS:
        out.update({
            f'conditional_word_recall_{b}': divide(at(f'conditional_tp_{b}'), at('positives')),
            f'conditional_word_precision_{b}': divide(at(f'conditional_tp_{b}'), at(f'budget_{b}')),
            f'conditional_mixed_precision_lift_{b}': divide(at(f'lift_sum_{b}'), at('mixed')),
            f'conditional_complete_span_recall_{b}': divide(at(f'conditional_complete_{b}'), at('spans')),
            f'conditional_touched_span_recall_{b}': divide(at(f'conditional_touch_{b}'), at('spans')),
            f'conditional_character_coverage_{b}': divide(at(f'conditional_characters_{b}'), at('characters')),
            f'conditional_unmarked_cost_fraction_{b}': divide(at(f'conditional_unmarked_cost_{b}'), at(f'budget_{b}')),
        })
    return out


def score_histograms(rows, scores, source_ids):
    lookup = {s: i for i, s in enumerate(source_ids)}
    y = np.concatenate([r['y'] for r in rows])
    values, bins = np.unique(np.concatenate(scores), return_inverse=True)
    sources = np.concatenate([np.full(len(r['y']), lookup[r['source_id']], dtype=np.int32) for r in rows])
    shape = len(source_ids), len(values)
    def matrix(weights): return sparse.csr_matrix((weights, (sources, bins)), shape=shape, dtype=np.float64)
    pure = np.concatenate([np.full(len(r['y']), int(r['y'].sum() == 0)) for r in rows])
    histograms = [matrix(y), matrix(1 - y), matrix(np.concatenate([r['characters'] for r in rows])), matrix(pure)]
    span_source, span_lo, span_hi, span_nlo, span_nhi = [], [], [], [], []
    offset = 0
    for row in rows:
        local_bins = bins[offset:offset + len(row['y'])]
        for indices in row['spans']:
            selected = local_bins[indices]
            lo, hi = selected.min(), selected.max()
            span_source.append(lookup[row['source_id']]); span_lo.append(lo); span_hi.append(hi)
            span_nlo.append(int((selected == lo).sum())); span_nhi.append(int((selected == hi).sum()))
        offset += len(row['y'])
    spans = [np.array(v, dtype=np.int64) for v in [span_source, span_lo, span_hi, span_nlo, span_nhi]]
    return histograms, spans


def evaluate_draws(atoms, histograms, spans, weights):
    output = collections.defaultdict(list)
    span_source, span_lo, span_hi, span_nlo, span_nhi = spans
    for start in range(0, len(weights), 100):
        w = weights[start:start + 100].astype(float)
        atom = w @ atoms
        values = ratios(atom)
        pc, nc, cc, clean = [h.T.dot(w.T).T for h in histograms]
        cum_p, cum_n, cum_c, cum_clean = [np.cumsum(v, axis=1) for v in [pc, nc, cc, clean]]
        P, N = pc.sum(1), nc.sum(1)
        values['pooled_auc'] = np.sum(pc * (cum_n - 0.5 * nc), axis=1) / (P * N)
        reverse_p = np.cumsum(pc[:, ::-1], axis=1)
        reverse_total = np.cumsum((pc + nc)[:, ::-1], axis=1)
        values['pooled_ap'] = np.sum(pc[:, ::-1] * reverse_p / np.maximum(reverse_total, 1), axis=1) / P
        for b in BUDGETS:
            B = atom[:, IX[f'budget_{b}']]
            cuts = pc.shape[1] - 1 - np.argmax(reverse_total >= B[:, None], axis=1)
            row_ids = np.arange(len(w))
            tie_n = (pc + nc)[row_ids, cuts].astype(np.int64)
            q = (B - (P + N - (cum_p + cum_n)[row_ids, cuts])).astype(np.int64)
            assert np.all((q > 0) & (q <= tie_n))
            tp = P - cum_p[row_ids, cuts] + q / tie_n * pc[row_ids, cuts]
            chars = cc.sum(1) - cum_c[row_ids, cuts] + q / tie_n * cc[row_ids, cuts]
            pure_cost = clean.sum(1) - cum_clean[row_ids, cuts] + q / tie_n * clean[row_ids, cuts]
            complete, touched = [], []
            for i, cut in enumerate(cuts):
                multiplicity = w[i, span_source]
                eq_lo, eq_hi = span_lo == cut, span_hi == cut
                a = float(multiplicity[span_lo > cut].sum())
                a += float(np.dot(multiplicity[eq_lo], all_selected_probability(int(tie_n[i]), int(q[i]), span_nlo[eq_lo])))
                z = float(multiplicity[span_hi > cut].sum())
                z += float(np.dot(multiplicity[eq_hi], 1 - all_selected_probability(int(tie_n[i]), int(tie_n[i] - q[i]), span_nhi[eq_hi])))
                complete.append(a); touched.append(z)
            complete, touched = np.array(complete), np.array(touched)
            assert np.all(complete <= touched + 1e-9)
            values.update({
                f'global_word_recall_{b}': tp / P,
                f'global_word_precision_{b}': tp / B,
                f'global_complete_span_recall_{b}': complete / atom[:, IX['spans']],
                f'global_touched_span_recall_{b}': touched / atom[:, IX['spans']],
                f'global_character_coverage_{b}': chars / atom[:, IX['characters']],
                f'global_unmarked_cost_fraction_{b}': pure_cost / B,
            })
        for key, value in values.items(): output[key].extend(value.tolist())
    return {k: np.asarray(v) for k, v in output.items()}


def permutation_control(rows, scores, seed):
    y = np.concatenate([r['y'] for r in rows])
    flat = np.concatenate(scores)
    original = flat.copy()
    rng = np.random.default_rng(seed)
    for label in [0, 1]:
        indices = np.flatnonzero(y == label)
        flat[indices] = rng.permutation(flat[indices])
        assert np.array_equal(np.sort(flat[indices]), np.sort(original[indices]))
    output = []; offset = 0
    for row in rows:
        output.append(flat[offset:offset + len(row['y'])].copy()); offset += len(row['y'])
    return output


def verify_small_cases():
    comparisons = 0
    for n in range(1, 8):
        for q in range(n + 1):
            choices = list(itertools.combinations(range(n), q))
            for size in range(1, n + 1):
                target = set(range(size))
                complete = sum(target <= set(choice) for choice in choices) / len(choices)
                touched = sum(bool(target & set(choice)) for choice in choices) / len(choices)
                assert abs(all_selected_probability(n, q, [size])[0] - complete) < 1e-12
                assert abs(1 - all_selected_probability(n, n - q, [size])[0] - touched) < 1e-12
                comparisons += 1
    assert abs(sum(1 / k for k in [1, 2]) / 2 - 0.75) < 1e-12
    return {'enumerated_tie_probability_cases': comparisons, 'all_passed': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ragtruth', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    checks = verify_small_cases()
    if args.verify:
        print(json.dumps(checks)); return
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    baseline_path = args.baseline or Path(__file__).parent / 'expanded_results/expanded_results.json'
    baseline = json.loads(baseline_path.read_text())
    for name, key in [('response.jsonl', 'response_sha256'), ('source_info.jsonl', 'source_sha256')]:
        assert hashlib.sha256((args.ragtruth / 'dataset' / name).read_bytes()).hexdigest() == baseline[key]
    rows = core.parse(args.ragtruth)
    score, fit = core.fit_models(rows)
    eligible = lambda r: r['quality'] == 'good' and len(r['y'])
    test = [r for r in rows if r['split'] == 'test' and eligible(r)]
    validation = [r for r in rows if r['split'] == 'validation' and eligible(r)]
    annotations = attach_annotations(test, args.ragtruth / 'dataset/response.jsonl')
    source_ids = sorted({r['source_id'] for r in test})
    rng = np.random.default_rng(20261014)
    weights = rng.multinomial(len(source_ids), np.full(len(source_ids), 1 / len(source_ids)), size=args.draws).astype(np.int16)
    weight_hash = hashlib.sha256(weights.tobytes()).hexdigest()
    if args.draws == 2000: assert weight_hash == baseline['bootstrap']['weights_sha256']
    result = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'dataset_revision': baseline['revision'], 'response_sha256': baseline['response_sha256'],
              'source_sha256': baseline['source_sha256'], 'fit': fit, 'annotations': annotations,
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'helper_sha256': hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
              'numpy_version': np.__version__, 'small_case_verification': checks,
              'bootstrap': {'seed': 20261014, 'draws': args.draws, 'source_count': len(source_ids),
                            'weight_sha256': weight_hash, 'fitting_and_thresholds_held_fixed': True},
              'models': {}, 'paired_contrasts': {}, 'checks': {}}
    models = list(core.MODELS) + ['task_generator_permuted', 'lexical_permuted']
    estimates, saved_atoms, scores_by_name = {}, {}, {}
    for model in models:
        parent = model.removesuffix('_permuted')
        scores = [score(r, parent) for r in test]
        if model.endswith('_permuted'):
            seed = 20261015 if parent == 'task_generator' else 20261016
            scores = permutation_control(test, scores, seed)
        val_scores = [score(r, parent) for r in validation]
        threshold = core.rank_metrics(np.concatenate([r['y'] for r in validation]), np.concatenate(val_scores))['threshold']
        atoms, responses = point_atoms(test, scores, threshold, source_ids)
        histograms, spans = score_histograms(test, scores, source_ids)
        point = {k: float(v[0]) for k, v in evaluate_draws(atoms, histograms, spans, np.ones((1, len(source_ids)), dtype=np.int16)).items()}
        print('POINT', model, 'within', point['within_macro_auc'], 'AP', point['within_macro_ap'], 'conditional span', point['conditional_complete_span_recall_0.1'], 'global span', point['global_complete_span_recall_0.1'], flush=True)
        boot = evaluate_draws(atoms, histograms, spans, weights)
        summary = {'threshold': threshold, 'point': point, 'confidence95': {k: core.ci(v) for k, v in boot.items()},
                   'counts': {name: float(atoms[:, IX[name]].sum()) for name in ['responses', 'words', 'positives', 'mixed', 'mixed_words', 'unmarked_responses', 'unmarked_words', 'marked_responses', 'marked_words', 'spans', 'characters', 'unmarked_alarm_responses', 'unmarked_flagged_words']}}
        summary['budgets'] = {str(b): {'selected_words': int(atoms[:, IX[f'budget_{b}']].sum()), 'conditional_unmarked_words': int(atoms[:, IX[f'conditional_unmarked_cost_{b}']].sum())} for b in BUDGETS}
        if model.endswith('_permuted'): summary['retrospective_permutation_seed'] = seed
        else:
            old = baseline['models'][model]
            for new, prior in [('pooled_auc', 'pooled_auc'), ('within_pair_auc', 'within_pair_auc'), ('within_macro_auc', 'within_macro_auc'), ('pooled_f1', 'pooled_f1')]:
                assert abs(point[new] - old[prior]) < 1e-12, (model, new, point[new], old[prior])
            for b in BUDGETS:
                for new, prior in [('conditional_word_recall', 'within_recall'), ('global_word_recall', 'global_recall')]:
                    assert abs(point[f'{new}_{b}'] - old['budget'][str(b)][prior]) < 1e-12
        result['models'][model] = summary; estimates[model] = boot; saved_atoms[model] = atoms; scores_by_name[model] = scores
        (args.out / f'{model}_responses.json').write_text(json.dumps(responses, separators=(',', ':')) + '\n')
        (args.out / 'results.partial.json').write_text(json.dumps(result, indent=2) + '\n')
        print('DONE', model, round(time.monotonic() - start, 2), flush=True)
    conditional_columns = [i for name, i in IX.items() if not name.startswith(('threshold_', 'unmarked_alarm', 'unmarked_flagged'))]
    assert np.array_equal(saved_atoms['lexical'][:, conditional_columns], saved_atoms['lexical_metadata'][:, conditional_columns])
    result['checks']['offset_preserves_all_conditional_source_atoms_exactly'] = True
    for parent in ['task_generator', 'lexical']:
        alternate = parent + '_permuted'
        for key in ['pooled_auc', 'pooled_ap', 'pooled_f1'] + [f'global_word_recall_{b}' for b in BUDGETS]:
            assert abs(result['models'][parent]['point'][key] - result['models'][alternate]['point'][key]) < 1e-12
    result['checks']['label_permutations_preserve_full_test_pooled_metrics'] = True
    for name, a, b in [('offset', 'lexical_metadata', 'lexical'), ('metadata_minus_lexical', 'task_generator', 'lexical'), ('combined_mean', 'lexical_metadata_mean', 'lexical_metadata'), ('lexical_mean', 'lexical_mean', 'lexical'), ('metadata_permutation', 'task_generator_permuted', 'task_generator'), ('lexical_permutation', 'lexical_permuted', 'lexical')]:
        result['paired_contrasts'][name] = {k: {'point': result['models'][a]['point'][k] - result['models'][b]['point'][k], 'confidence95': core.ci(estimates[a][k] - estimates[b][k])} for k in estimates[a]}
    np.savez_compressed(args.out / 'bootstrap_estimates.npz', **{m + '__' + k: v for m, d in estimates.items() for k, v in d.items()})
    np.save(args.out / 'source_weights.npy', weights)
    result['elapsed_seconds'] = time.monotonic() - start
    (args.out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print('FINISHED', round(result['elapsed_seconds'], 2), flush=True)


if __name__ == '__main__':
    main()
