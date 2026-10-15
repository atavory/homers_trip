"""Replay saved token predictions using preserved source bootstrap draws."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
import expanded as core


def load_predictions(path):
    groups = collections.defaultdict(list)
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            groups[(row['scorer'], row['source_id'], row['response_id'])].append(row)
    models = collections.defaultdict(dict)
    for (model, source, response), rows in groups.items():
        rows.sort(key=lambda r: int(r['token_index']))
        assert len({r['token_index'] for r in rows}) == len(rows)
        mask = np.array([int(r['eligible']) == 1 for r in rows])
        if not mask.any():continue
        all_scores = np.array([float(r['score']) for r in rows]);assert np.isfinite(all_scores).all()
        y = np.array([int(r['label']) for r, use in zip(rows, mask) if use], dtype=np.int8)
        assert np.isin(y, [0, 1]).all()
        models[model].setdefault(source, []).append({'response_id': response, 'y': y, 'original': all_scores[mask], 'eligible_mean': np.full(mask.sum(), all_scores[mask].mean()), 'all_scored_mean': np.full(mask.sum(), all_scores.mean())})
    return models


def metrics(records, mode, threshold):
    for r in records:
        cache = r.setdefault('_rank_cache', {})
        if mode not in cache:cache[mode] = core.rank_metrics(r['y'], r[mode])
    local = [r['_rank_cache'][mode] for r in records]
    mixed = [v for v in local if v['P'] and v['N']]
    y = np.concatenate([r['y'] for r in records]);scores = np.concatenate([r[mode] for r in records])
    pooled = core.rank_metrics(y, scores);W = sum(v['P'] * v['N'] for v in local)
    wins = sum(v['wins'] for v in mixed);C = pooled['P'] * pooled['N'] - W
    output = {'pooled_auc': pooled['auc'], 'pooled_ap': pooled['ap'], 'within_pair_auc': wins / W if W else None, 'within_macro_auc': float(np.mean([v['auc'] for v in mixed])) if mixed else None, 'cross_auc': (pooled['wins'] - wins) / C if C else None, 'within_macro_nap': float(np.mean([(v['ap'] - v['P'] / (v['P'] + v['N'])) / (1 - v['P'] / (v['P'] + v['N'])) for v in mixed])) if mixed else None}
    flagged = scores >= threshold
    tp = int(y[flagged].sum());fp = int((1 - y[flagged]).sum())
    output['fixed_threshold_f1'] = 2 * tp / (pooled['P'] + tp + fp) if pooled['P'] + tp + fp else None
    for fraction in [.1, .2]:
        ks = [math.ceil(fraction * len(r['y'])) for r in records]
        selected = []
        for r, k in zip(records, ks):
            cache = r.setdefault('_budget_cache', {})
            key = (mode, fraction)
            if key not in cache:cache[key] = core.budget_counts(r['y'], r[mode], k)
            selected.append(cache[key])
        budget = sum(ks)
        output[f'conditional_recall_{fraction}'] = sum(selected) / pooled['P'] if pooled['P'] else None
        output[f'conditional_precision_{fraction}'] = sum(selected) / budget
        output[f'global_recall_{fraction}'] = core.budget_counts(y, scores, budget) / pooled['P'] if pooled['P'] else None
        output[f'all_negative_inspection_cost_{fraction}'] = sum(k for r, k in zip(records, ks) if r['y'].sum() == 0)
        output[f'conditional_mixed_precision_{fraction}'] = float(np.mean([tp/k for tp,k,v in zip(selected,ks,local) if v['P'] and v['N']])) if mixed else None
    return output


def interval(values):
    a = np.array([float(v) for v in values if v is not None]);finite = a[np.isfinite(a)]
    return {'low': float(np.quantile(finite, .025)), 'high': float(np.quantile(finite, .975)), 'valid_draws': len(finite), 'invalid_draws': len(values) - len(finite)} if len(finite) else {'low':None,'high':None,'valid_draws':0,'invalid_draws':len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--bootstrap', type=Path, required=True)
    parser.add_argument('--thresholds', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args();args.out.mkdir(parents=True, exist_ok=True)
    models = load_predictions(args.predictions);draws = json.loads(args.bootstrap.read_text());thresholds = json.loads(args.thresholds.read_text())
    sources = draws['groups'];assert len(sources) == len(set(sources))
    if 'seed' in draws:
        regenerated = np.random.default_rng(draws['seed']).integers(len(sources), size=(len(draws['draw_indices']), len(sources))).tolist()
        assert regenerated == draws['draw_indices']
    result = {'input_sha256': {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in [('predictions',args.predictions),('bootstrap',args.bootstrap),('thresholds',args.thresholds)]},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed':draws.get('seed'),'draws':len(draws['draw_indices']),'models':{},'contrasts':{}}
    estimates = {}
    for model, groups in sorted(models.items()):
        assert set(groups) == set(sources)
        records = [r for source in sources for r in groups[source]]
        result['models'][model] = {}
        for mode in ['original', 'eligible_mean', 'all_scored_mean']:
            point = metrics(records, mode, thresholds[model]);boot=[]
            for draw in draws['draw_indices']:
                copies = [r for index in draw for r in groups[sources[index]]]
                boot.append(metrics(copies, mode, thresholds[model]))
            result['models'][model][mode] = {'point':point,'confidence95':{key:interval([v[key] for v in boot]) for key in point}}
            estimates[(model,mode)] = boot
    names = sorted(models)
    for a in names:
        for b in names:
            if a <= b:continue
            point_a=result['models'][a]['original']['point'];point_b=result['models'][b]['original']['point']
            result['contrasts'][a+'__minus__'+b] = {key:{'point':point_a[key]-point_b[key],'confidence95':interval([x[key]-y[key] if x[key] is not None and y[key] is not None else None for x,y in zip(estimates[(a,'original')],estimates[(b,'original')])])} for key in point_a if point_a[key] is not None and point_b[key] is not None}
    result['between_model_spreads'] = {key:max(result['models'][m]['original']['point'][key] for m in names)-min(result['models'][m]['original']['point'][key] for m in names) for key in ['pooled_auc','within_macro_auc','within_pair_auc']}
    (args.out/'bootstrap_results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (args.out/'bootstrap_metrics.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['scorer','transformation','metric','point','ci_low','ci_high'])
        for model,parts in result['models'].items():
            for mode,d in parts.items():
                for key,value in d['point'].items():writer.writerow([model,mode,key,value,d['confidence95'][key]['low'],d['confidence95'][key]['high']])
    print(json.dumps({'seed':result['seed'],'draws':result['draws'],'models':names}))


if __name__ == '__main__':main()
