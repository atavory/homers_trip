"""Reproduce aggregation counterexamples and audit public RAGTruth annotations.
Only standard-library dependencies. Word units use Unicode whitespace splitting.
"""
from collections import defaultdict, Counter
from pathlib import Path
from provenance import dataset_revision
import argparse, csv, hashlib, json, re, subprocess

def metrics(rows):
    # Rows are (positive count, negative count, score).
    bins = defaultdict(lambda: [0, 0])
    for p, n, s in rows:
        bins[s][0] += p
        bins[s][1] += n
    P = sum(v[0] for v in bins.values())
    N = sum(v[1] for v in bins.values())
    wins, below = 0.0, 0
    for s in sorted(bins):
        p, n = bins[s]
        wins += p * (below + n / 2)
        below += n
    tp = fp = 0
    ap = 0.0
    thresholds = []
    for s in sorted(bins, reverse=True):
        p, n = bins[s]
        tp += p
        fp += n
        precision = tp / (tp + fp)
        if P:
            ap += p / P * precision
        f1 = 2 * tp / (P + tp + fp) if P + tp + fp else 0
        thresholds.append((f1, s))
    return {'auc': wins / (P * N) if P and N else None,
            'ap': ap if P else None, 'P': P, 'N': N,
            'wins': wins, 'best_threshold': max(thresholds)[1] if thresholds else None}

def f1_at(rows, threshold):
    tp = sum(p for p, n, s in rows if s >= threshold)
    fp = sum(n for p, n, s in rows if s >= threshold)
    P = sum(p for p, n, s in rows)
    return 2 * tp / (P + tp + fp) if P + tp + fp else None

def toy():
    local = [(9, 0, 1), (0, 1, 0), (1, 0, 3), (0, 9, 2)]
    constant = [(9, 1, 2), (1, 9, 1)]
    a, b = metrics(local), metrics(constant)
    # Independent exact values: 18 within wins + 1 cross win; 81 cross wins + 9 ties.
    assert abs(a['auc'] - 19 / 100) < 1e-12
    assert abs(b['auc'] - 90 / 100) < 1e-12
    assert abs(a['ap'] - (0.1 + 0.9 * 10 / 19)) < 1e-12
    assert abs(b['ap'] - 0.86) < 1e-12
    return {'local': a, 'constant': b, 'W': 18, 'C': 82,
            'constant_f1_at_2': f1_at(constant, 2),
            'local_within_auc': 1.0, 'constant_within_auc': 0.5}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def audit(repo):
    response_file = repo / 'dataset/response.jsonl'
    source_file = repo / 'dataset/source_info.jsonl'
    sources = {r['source_id']: r for r in map(json.loads, source_file.read_text().splitlines())}
    rows = []
    span_text_mismatches = 0
    source_splits = defaultdict(set)
    qualities = Counter()
    for r in map(json.loads, response_file.read_text().splitlines()):
        source_splits[r['source_id']].add(r['split'])
        qualities[r['quality']] += 1
        text = r['response']
        spans = [(a['start'], a['end']) for a in r['labels']]
        for a in r['labels']:
            assert 0 <= a['start'] < a['end'] <= len(text)
            span_text_mismatches += text[a['start']:a['end']] != a['text']
        units = list(re.finditer(r'\S+', text))
        P = sum(any(m.start() < end and m.end() > start for start, end in spans) for m in units)
        source = sources[r['source_id']]
        dev = int(hashlib.sha256(('homers-dream-dev-v1:' + r['source_id']).encode()).hexdigest()[:8], 16) % 5 == 0
        split = 'test' if r['split'] == 'test' else 'validation' if dev else 'train'
        rows.append({'id': r['id'], 'source_id': r['source_id'], 'split': split,
                     'official_split': r['split'], 'quality': r['quality'],
                     'task': source['task_type'], 'source': source['source'],
                     'model': r['model'], 'P': P, 'N': len(units) - P})
    assert all(len(s) == 1 for s in source_splits.values()), 'Official source leakage'
    good = [r for r in rows if r['quality'] == 'good' and r['P'] + r['N'] > 0]
    fit = [r for r in good if r['split'] == 'train']
    global_rate = sum(r['P'] for r in fit) / sum(r['P'] + r['N'] for r in fit)
    counts = defaultdict(lambda: [0, 0])
    for r in fit:
        counts[(r['task'], r['model'])][0] += r['P']
        counts[(r['task'], r['model'])][1] += r['P'] + r['N']
    # Fixed 1000-word pseudocount; no test labels or test-dependent orientation.
    priors = {k: (p + 1000 * global_rate) / (n + 1000) for k, (p, n) in counts.items()}
    score = lambda r: priors.get((r['task'], r['model']), global_rate)
    triples = lambda rs: [(r['P'], r['N'], score(r)) for r in rs]
    validation = [r for r in good if r['split'] == 'validation']
    threshold = metrics(triples(validation))['best_threshold']
    def summarize(rs):
        original = metrics(triples(rs))
        P, N = original['P'], original['N']
        W = sum(r['P'] * r['N'] for r in rs)
        mixed = [r for r in rs if r['P'] and r['N']]
        same_wins = W / 2
        C = P * N - W
        return {'responses': len(rs), 'sources': len({r['source_id'] for r in rs}),
                'mixed': len(mixed), 'all_negative': sum(r['P'] == 0 for r in rs),
                'all_positive': sum(r['N'] == 0 for r in rs), 'P': P, 'N': N,
                'prevalence': P / (P + N), 'W': W, 'C': C,
                'pair_fraction': W / (P * N), 'pooled_auc': original['auc'],
                'cross_auc': (original['wins'] - same_wins) / C,
                'within_pair_auc': 0.5 if W else None,
                'within_macro_auc': 0.5 if mixed else None,
                'pooled_ap': original['ap'], 'pooled_f1': f1_at(triples(rs), threshold),
                'within_macro_ap_mixed': sum(r['P'] / (r['P'] + r['N']) for r in mixed) / len(mixed)}
    test = [r for r in good if r['split'] == 'test']
    summaries = {'all': summarize(test)}
    for source in sorted({r['source'] for r in test}):
        summaries[source] = summarize([r for r in test if r['source'] == source])
    # Sensitivity: apply the same fitted predictor and threshold to all nonempty test responses.
    summaries['all_quality_sensitivity'] = summarize([r for r in rows if r['split'] == 'test' and r['P'] + r['N'] > 0])
    return {'dataset': 'RAGTruth', 'git_revision': dataset_revision(repo),
            'response_sha256': sha(response_file), 'source_sha256': sha(source_file),
            'units': 'Unicode whitespace-delimited, nonempty response units (regex \\S+)',
            'positive_rule': 'Any character overlap with any supplied hallucination span, end-exclusive',
            'quality_rule': 'Primary: good; sensitivity: all qualities',
            'span_text_mismatches': span_text_mismatches, 'qualities': dict(qualities),
            'train_responses': len(fit), 'validation_responses': len(validation),
            'train_sources': len({r['source_id'] for r in fit}), 'validation_sources': len({r['source_id'] for r in validation}),
            'predictor': 'Training-only task/generator word prevalence, 1000-word global-rate smoothing',
            'validation_selected_threshold': threshold, 'summaries': summaries}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ragtruth', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'toy': toy()}
    if args.ragtruth:
        result['ragtruth'] = audit(args.ragtruth)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
