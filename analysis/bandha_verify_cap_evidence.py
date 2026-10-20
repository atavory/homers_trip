"""Check allocation expectations by exhaustive selections and saved exact results."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from bandha_cap_evidence import Allocation,regrets


def exhaustive(scores,labels,offsets,sources,weights,budget):
    copies=[r for r in range(len(sources)) for _ in range(weights[sources[r]])]
    ss=np.concatenate([scores[offsets[r]:offsets[r+1]] for r in copies])
    yy=np.concatenate([labels[offsets[r]:offsets[r+1]] for r in copies])
    oo=np.r_[0,np.cumsum([offsets[r+1]-offsets[r] for r in copies])]
    k=round(budget*len(ss)); threshold=np.sort(ss)[-k]
    base=ss>threshold; tie=np.flatnonzero(ss==threshold)
    picked=k-int(base.sum()); curves=[]
    maximum=int(max(labels[offsets[r]:offsets[r+1]].sum() for r in copies))
    for combination in itertools.combinations(tie,picked):
        selected=base.copy();selected[list(combination)]=True
        remaining=[yy[lo:hi][~selected[lo:hi]].sum() for lo,hi in zip(oo[:-1],oo[1:])]
        curves.append([np.minimum(remaining,t).mean() for t in range(1,maximum+1)])
    return np.mean(curves,axis=0)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--saved-exact',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();checks=0;largest_error=0
    labels=np.array([1,1,1,0,0,1,1,0,1,0,1,0]);offsets=np.arange(0,13,3)
    sources=np.array([0,0,1,1]);rng=np.random.default_rng(49283)
    for scores in [np.ones(12),np.repeat([4,2,3,1],3),rng.integers(0,3,12)]:
        allocation=Allocation(scores,labels,offsets,sources)
        for weights in [np.array([1,1]),np.array([0,2]),np.array([2,0])]:
            for budget in [.2,.5]:
                measured=allocation.evaluate(weights,budget)
                direct=exhaustive(scores,labels,offsets,sources,weights,budget)
                assert np.allclose(measured[:len(direct)],direct,atol=1e-12), (scores,weights,budget,measured,direct)
                checks+=1
    index=json.loads((a.inputs/'response_index.json').read_text())
    ids=sorted({r['source_id'] for r in index});lookup={s:i for i,s in enumerate(ids)}
    rs=np.array([lookup[r['source_id']] for r in index]);weights=np.ones(len(ids),dtype=int)
    reference=json.loads(a.saved_exact.read_text())
    curves=[];names=['generator','lexical','lexical_metadata']
    with np.load(a.inputs/'baseline_scores.npz') as z:
        for n in names:
            allocation=Allocation(z[n],z['labels'],z['response_offsets'],rs)
            curve,metadata,h=allocation.evaluate(weights,.2,return_histogram=True)
            old=np.array(reference['curves']['expected_ties']['all'][n])
            error=float(np.max(np.abs(curve-old)));largest_error=max(largest_error,error)
            assert error<1e-9,(n,error)
            curves.append(curve);checks+=len(curve)
            for _ in range(25):
                increments=np.sort(rng.random(len(curve)))[::-1];increments/=increments[0]
                loss=np.r_[0,np.cumsum(increments)]
                alpha=np.r_[increments[:-1]-increments[1:],increments[-1]]
                assert np.isclose(alpha.sum(),1)
                assert np.isclose(np.dot(h,loss),np.dot(alpha,curve),atol=1e-10)
                checks+=1
    curves=np.array(curves);r=regrets(curves)
    for j,n in enumerate(names):
        best=np.min(curves,axis=0);t=int(np.argmax(curves[j]-best))
        exact=reference['summaries']['expected_ties']['all']['three_method_regret']['by_method'][n]['value']
        assert np.isclose(r[j],exact,atol=1e-9),(n,r[j],exact)
        assert np.isclose(r[j],curves[j,t]-np.min(curves[:,t]));checks+=1
        for _ in range(50):
            alpha=rng.dirichlet(np.ones(curves.shape[1]))
            losses=curves@alpha
            assert losses[j]-losses.min()<=r[j]+1e-10;checks+=1
    a.out.write_text(json.dumps({'all_checks_passed':True,'checks':checks,
                                'maximum_saved_exact_error':largest_error,
                                'exhaustive_selection_fixtures':18},indent=2)+'\n')
    print(a.out.read_text())

if __name__=='__main__':main()
