"""Verify trained-detector curves against rational cutoff selection laws."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def rational_curve(scores,labels,offsets,response_weights,budget):
    weights=np.repeat(response_weights,np.diff(offsets))
    order=np.argsort(-scores,kind='stable');cum=np.cumsum(weights[order])
    k=round(budget*int(cum[-1]));threshold=scores[order[np.searchsorted(cum,k,side='left')]]
    above=scores>threshold;tied=scores==threshold
    population=int(weights[tied].sum());chosen=k-int(weights[above].sum())
    states=Counter();maximum=0
    for r,(lo,hi) in enumerate(zip(offsets[:-1],offsets[1:])):
        y=labels[lo:hi];s=scores[lo:hi]
        below=int(((y==1)&(s<threshold)).sum());p=int(((y==1)&(s==threshold)).sum())
        states[below,p]+=int(response_weights[r]);maximum=max(maximum,int(y.sum()))
    h=[Fraction() for _ in range(maximum+1)]
    for (below,p),mass in states.items():
        if not mass:continue
        denominator=math.comb(population,p)
        for selected in range(max(0,p-(population-chosen)),min(p,chosen)+1):
            prob=Fraction(math.comb(chosen,selected)*math.comb(population-chosen,p-selected),denominator)
            h[below+p-selected]+=mass*prob
    total=int(response_weights.sum());assert sum(h)==total
    return np.array([float(sum(m*min(e,t) for e,m in enumerate(h))/total) for t in range(1,maximum+1)])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--results',type=Path,required=True);ap.add_argument('--smoke',type=Path,required=True)
    a=ap.parse_args();index=json.loads((a.inputs/'response_index.json').read_text())
    report=json.loads((a.results/'results.json').read_text());points=np.load(a.results/'curves.npz')['point']
    names=report['methods'];checked=0;max_error=0
    with np.load(a.inputs/'trained_scores.npz') as z:
        for name in ['lettuce','lettuce_mean']:
            direct=rational_curve(z[name],z['labels'],z['response_offsets'],np.ones(len(index),dtype=int),.2)
            error=float(np.max(np.abs(direct-points[names.index(name)])));max_error=max(max_error,error)
            assert error<1e-9,(name,error);checked+=len(direct)
        smoke=np.load(a.smoke);batch_error=0
        for i,(lo,hi) in zip(smoke['response_indices'],zip(smoke['offsets'][:-1],smoke['offsets'][1:])):
            start,end=z['response_offsets'][i:i+2]
            error=float(np.max(np.abs(z['lettuce'][start:end]-smoke['scores'][lo:hi])))
            batch_error=max(batch_error,error);checked+=1
        assert batch_error<1e-4,batch_error
        np.testing.assert_array_equal(z['response_offsets'],np.r_[0,np.cumsum([r['word_count'] for r in index])])
        assert np.isfinite(z['lettuce']).all() and ((z['lettuce']>=0)&(z['lettuce']<=1)).all()
    best=points.min(axis=0)
    for j,n in enumerate(names):
        regret=float((points[j]-best).max())
        assert np.isclose(regret,report['summaries'][n]['regret'],atol=1e-12);checked+=1
    result={'all_checks_passed':True,'checked_values':checked,'maximum_rational_curve_error':max_error,
            'maximum_batch_score_difference':batch_error,'trained_responses':len(index),
            'comparison_set':names,'scores_sha256':hashlib.sha256((a.inputs/'trained_scores.npz').read_bytes()).hexdigest()}
    (a.results/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
