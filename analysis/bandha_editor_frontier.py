"""Evaluate independent editing and conservative marginal-bound frontiers."""
import json,time
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import binom

class OutcomeModel:
 def __init__(self,triples):
  counts=Counter(map(tuple,triples));self.states=np.array(list(counts),dtype=int)
  self.weights=np.array(list(counts.values()),dtype=float)/len(triples)
  self.fn,self.tp,self.fp=self.states.T
  self.maximum=max(1,int((self.fn+self.tp+self.fp).max()))
  self.x=np.arange(self.maximum+1)
  self.valid=self.x[None,:]<=(self.tp+self.fp)[:,None]
  self.indices=(self.fn[:,None]+self.x[None,:])[self.valid]
  self.fft=1<<(self.maximum+1).bit_length()
  baseline=np.bincount(self.fn+self.tp,weights=self.weights,minlength=self.maximum+1)
  self.reference=np.cumsum(1-np.cumsum(baseline[:-1]))
 def evaluate(self,p,q):
  failed=binom.pmf(self.x[None,:],self.tp[:,None],1-p)
  harmed=binom.pmf(self.x[None,:],self.fp[:,None],q)
  conv=np.fft.irfft(np.fft.rfft(failed,n=self.fft)*np.fft.rfft(harmed,n=self.fft),n=self.fft)[:,:self.maximum+1]
  conv=np.maximum(conv,0);conv[~self.valid]=0;conv/=conv.sum(axis=1,keepdims=True)
  hist=np.bincount(self.indices,weights=(conv*self.weights[:,None])[self.valid],minlength=self.maximum+1)
  curve=np.cumsum(1-np.cumsum(hist[:-1]))
  mean=float(np.dot(self.weights,self.fn+(1-p)*self.tp+q*self.fp))
  first=1-float(np.sum(self.weights*(self.fn==0)*p**self.tp*(1-q)**self.fp))
  assert abs(curve[-1]-mean)<1e-9
  assert abs(curve[0]-first)<1e-10
  difference=curve-self.reference
  self.last_curve=curve
  return {'B':float(difference.max()),'cap':int(difference.argmax()+1),'first':float(difference[0]),'mean_change':float(curve[-1]-self.reference[-1])}
 def frontier(self,p):
  lo,hi=0.,1.
  if self.evaluate(p,1.)['B']<=1e-11:return 1.,1.
  for _ in range(24):
   mid=(lo+hi)/2
   if self.evaluate(p,mid)['B']<=1e-11:lo=mid
   else:hi=mid
  return lo,hi


def upper_curve(model,p,q):
    means=model.fn+(1-p)*model.tp+q*model.fp
    caps=np.arange(1,model.maximum+1)
    return np.sum(model.weights[:,None]*np.minimum(means[:,None],caps[None,:]),axis=0)


def sufficient_frontier(model,p):
    def excess(q):
        return float((upper_curve(model,p,q)-model.reference).max())
    if excess(1.)<=1e-11:
        return 1.,1.
    lo,hi=0.,1.
    for _ in range(32):
        mid=(lo+hi)/2
        if excess(mid)<=1e-11:
            lo=mid
        else:
            hi=mid
    return lo,hi


def main():
    import argparse,hashlib
    from scipy.signal import convolve
    parser=argparse.ArgumentParser()
    parser.add_argument('--scores',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    saved=np.load(args.scores);labels=saved['labels'].astype(bool)
    scores=saved['lettuce'];offsets=saved['response_offsets']
    selected=round(.2*len(labels));threshold=np.partition(scores,-selected)[-selected]
    assert (scores>=threshold).sum()==selected
    policies={'budget_20pct':scores>=threshold,'word_score_0.5':scores>=.5}
    reports={};all_triplets={};checks=0
    for triple in [(0,0,1),(1,1,0),(2,3,4),(0,2,0)]:
        model=OutcomeModel([triple]);f,t,a=triple
        for p in [0.,.5,1.]:
            for q in [0.,.01,1.]:
                model.evaluate(p,q)
                direct=convolve(binom.pmf(np.arange(t+1),t,1-p),binom.pmf(np.arange(a+1),a,q))
                direct=np.pad(direct,(f,0));caps=np.arange(1,model.maximum+1)
                expected=np.sum(direct[:,None]*np.minimum(np.arange(len(direct))[:,None],caps[None,:]),axis=0)
                assert np.allclose(model.last_curve,expected,atol=1e-11,rtol=0);checks+=1
    toy=OutcomeModel([(0,0,1),(1,1,0)])
    for p in [.2,.9,1.]:
        for q in [0.,.01,.1]:
            x=toy.evaluate(p,q)
            assert abs(x['first']-q/2)<1e-12;checks+=1
            assert abs(x['mean_change']-(-p+q)/2)<1e-12;checks+=1
    for name,flags in policies.items():
        triples=[]
        for a,b in zip(offsets[:-1],offsets[1:]):
            y=labels[a:b];z=flags[a:b]
            triples.append((int((y&~z).sum()),int((y&z).sum()),int((~y&z).sum())))
        all_triplets[name]=np.array(triples)
        model=OutcomeModel(triples);totals=np.array(triples).sum(axis=0)
        fn,tp,fp=map(int,totals)
        row={'FN_TP_FP':[fn,tp,fp],'flags':int(flags.sum()),'precision':float(tp/(tp+fp)),
             'maximum_residual_support':model.maximum,'point_p09_q001':model.evaluate(.9,.01),
             'upper_disadvantage_p09_q001':float((upper_curve(model,.9,.01)-model.reference).max()),
             'linear_q_at_p09':float(min(1,.9*tp/fp)),'frontier':{}}
        for p in np.linspace(0,1,11):
            lo,hi=model.frontier(float(p));a,b=sufficient_frontier(model,float(p))
            assert a<=lo+1e-7;checks+=1
            assert lo<=min(1,float(p)*tp/fp)+1e-7;checks+=1
            row['frontier'][f'{p:.1f}']={'independent_lower':lo,'independent_upper':hi,
                                       'sufficient_lower':a,'sufficient_upper':b}
            for q in [0.,.001,.01,.1,1.]:
                model.evaluate(float(p),q)
                assert np.all(model.last_curve<=upper_curve(model,float(p),q)+1e-9);checks+=1
        for field in ['independent_lower','sufficient_lower']:
            assert np.all(np.diff([v[field] for v in row['frontier'].values()])>=-1e-8);checks+=1
        reports[name]=row
        print(name,json.dumps({k:v for k,v in row.items() if k!='frontier'}),flush=True)
    result={'reference':'Original unedited responses, not the paper mean control',
            'model':'Item-local correction; unselected-item correctness preserved; homogeneous repair p and harm q. Independent-binomial curve is one model. Jensen sufficient bounds do not require independence when the stated marginal repair/harm bounds hold per flagged item.',
            'policies':reports,'verification_checks':checks,
            'uncertainty':'Numerical fixed-benchmark frontier estimates, not population intervals or measured editor performance. Bisection brackets are numerical crossing brackets, not confidence intervals.',
            'scores_sha256':hashlib.sha256(args.scores.read_bytes()).hexdigest()}
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(args.out/'triplets.npz',**all_triplets)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    figure,axes=plt.subplots(1,2,figsize=(8,3.2))
    for ax,(name,row) in zip(axes,reports.items()):
        p=np.array([float(x) for x in row['frontier'] if float(x)>0])
        front=[row['frontier'][f'{x:.1f}'] for x in p]
        ax.plot(p,[x['independent_lower'] for x in front],marker='o',ms=3,label='Independent-edit model')
        ax.plot(p,[x['sufficient_lower'] for x in front],marker='s',ms=3,label='Sufficient marginal-bound test')
        tp,fp=row['FN_TP_FP'][1:]
        ax.plot(p,np.minimum(1,p*tp/fp),ls=':',label='Mean-error bound')
        ax.set_yscale('log');ax.set_xlabel('Repair probability p')
        ax.set_ylabel('Tolerated corruption probability q')
        ax.set_title('20% word checks' if name=='budget_20pct' else 'Word-score cutoff 0.5')
        ax.spines[['top','right']].set_visible(False)
    figure.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=7)
    figure.tight_layout(rect=(0,.12,1,1))
    figure.savefig(args.out/'frontiers.pdf',bbox_inches='tight')
    figure.savefig(args.out/'frontiers.png',dpi=180,bbox_inches='tight')
    plt.close(figure)
    print(json.dumps({'verification_checks':checks,'written':str(args.out)}),flush=True)


if __name__=='__main__':
    main()
