"""Check span expectations and bootstrap results against explicit selections."""
import argparse
import itertools
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
import alternatives as alt
import expanded as core


def verify_span_selection():
    scores = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    spans = [np.array(v) for v in [[0], [1, 2], [3, 4], [2, 3], [1, 2, 3], [0, 1, 2, 3, 4, 5]]]
    tests = 0
    for k in range(1, len(scores) + 1):
        threshold, q, n = alt.cutoff(scores, k)
        above = set(map(int, np.flatnonzero(scores > threshold)))
        tie = list(map(int, np.flatnonzero(scores == threshold)))
        outcomes = [above | set(v) for v in itertools.combinations(tie, q)]
        complete = [sum(set(v) <= choice for choice in outcomes) / len(outcomes) for v in spans]
        touched = [sum(bool(set(v) & choice) for choice in outcomes) / len(outcomes) for v in spans]
        actual_complete, actual_touched = alt.span_events(scores, spans, threshold, q, n)
        assert np.max(np.abs(actual_complete - complete)) < 1e-12
        assert np.max(np.abs(actual_touched - touched)) < 1e-12
        tests += len(spans)
    return tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ragtruth', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    report = {'enumerated_span_selection_cases': verify_span_selection()}
    labels = np.array([1, 0])
    ap = [core.rank_metrics(labels, np.array(s))['ap'] for s in [(1, 0), (0, 1)]]
    assert np.mean(ap) == 0.75
    assert core.rank_metrics(labels, np.array([0.0, 0.0]))['ap'] == 0.5
    report['two_word_random_ap_expectation'] = 0.75
    report['two_word_tied_ap_baseline'] = 0.5
    result = json.loads((args.results / 'results.json').read_text())
    estimates = np.load(args.results / 'bootstrap_estimates.npz')
    weights = np.load(args.results / 'source_weights.npy')[0]
    rows = core.parse(args.ragtruth)
    score, _ = core.fit_models(rows)
    test = [r for r in rows if r['split'] == 'test' and r['quality'] == 'good' and len(r['y'])]
    alt.attach_annotations(test, args.ragtruth / 'dataset/response.jsonl')
    sources = sorted({r['source_id'] for r in test})
    source_index = {s: i for i, s in enumerate(sources)}
    copied = [(dict(r, instance_id=(r['id'], k)), r) for r in test for k in range(int(weights[source_index[r['source_id']]]))]
    copied_rows = [v[0] for v in copied]
    assert len({r['instance_id'] for r in copied_rows}) == len(copied_rows)
    report['first_draw_response_instances'] = len(copied_rows)
    report['bootstrap_checks'] = {}
    for model in ['task_generator', 'lexical']:
        scores = [score(original, model) for _, original in copied]
        threshold = result['models'][model]['threshold']
        direct, _ = core.summarize(copied_rows, scores, threshold)
        values = {'pooled_auc': direct['pooled_auc'], 'pooled_ap': direct['pooled_ap'],
                  'within_pair_auc': direct['within_pair_auc'], 'within_macro_auc': direct['within_macro_auc'],
                  'within_macro_nap': direct['within_macro_nAP']}
        y = np.concatenate([r['y'] for r in copied_rows]); flat = np.concatenate(scores)
        pure_indices = [i for i, r in enumerate(copied_rows) if not r['y'].sum()]
        values['unmarked_response_alarm_rate'] = sum(np.any(scores[i] >= threshold) for i in pure_indices) / len(pure_indices)
        values['unmarked_word_false_positive_rate'] = sum(int((scores[i] >= threshold).sum()) for i in pure_indices) / sum(len(scores[i]) for i in pure_indices)
        for b in alt.BUDGETS:
            values[f'global_word_recall_{b}'] = direct['budget'][str(b)]['global_recall']
            values[f'conditional_word_recall_{b}'] = direct['budget'][str(b)]['within_recall']
            budget = sum(math.ceil(b * len(r['y'])) for r in copied_rows)
            thr, q, tie_n = alt.cutoff(flat, budget)
            all_complete, all_touched, characters, total_char, span_count = 0.0, 0.0, 0.0, 0, 0
            for r, s in zip(copied_rows, scores):
                complete, touched = alt.span_events(s, r['spans'], thr, q, tie_n)
                all_complete += float(complete.sum()); all_touched += float(touched.sum()); span_count += len(r['spans'])
                characters += float(r['characters'][s > thr].sum() + q / tie_n * r['characters'][s == thr].sum())
                total_char += int(r['characters'].sum())
            values[f'global_complete_span_recall_{b}'] = all_complete / span_count
            values[f'global_touched_span_recall_{b}'] = all_touched / span_count
            values[f'global_character_coverage_{b}'] = characters / total_char
        differences = {key: abs(value - float(estimates[model + '__' + key][0])) for key, value in values.items()}
        assert max(differences.values()) < 1e-12, (model, differences)
        report['bootstrap_checks'][model] = {'checked_metrics': len(values), 'maximum_absolute_error': max(differences.values())}
    report['all_passed'] = True
    (args.results / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
