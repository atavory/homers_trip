"""Expanded, source-disjoint RAGTruth evaluation. No LLM inference.
Dependencies: numpy, scipy, matplotlib. Outputs accompany the paper's data project.
"""
import argparse, collections, datetime, hashlib, json, math, re, subprocess, time
from pathlib import Path
from provenance import dataset_revision
import numpy as np
from scipy import sparse

MODELS = ['global', 'task', 'generator', 'task_generator', 'lexical', 'lexical_metadata', 'lexical_mean', 'lexical_metadata_mean']
BUDGETS = [.05, .10, .20]

def logit(x):
    x = np.clip(x, 1e-8, 1 - 1e-8)
    return np.log(x / (1 - x))

def safe(a, b):
    return float(a / b) if b else None

def rank_metrics(y, scores):
    P = int(y.sum()); N = len(y) - P
    values, inv = np.unique(scores, return_inverse=True)
    pos = np.bincount(inv, weights=y, minlength=len(values))
    neg = np.bincount(inv, weights=1-y, minlength=len(values))
    wins = float(np.dot(pos, np.cumsum(neg) - .5*neg))
    p, n = pos[::-1], neg[::-1]
    cum_p = np.cumsum(p); cum_n = np.cumsum(n)
    ap = float(np.sum(p * cum_p / (cum_p+cum_n)) / P) if P else None
    f1 = 2*cum_p / (P+cum_p+cum_n)
    j = int(np.argmax(f1)) if len(f1) else None
    return {'P':P, 'N':N, 'wins':wins, 'auc':safe(wins,P*N), 'ap':ap,
            'threshold':float(values[::-1][j]) if j is not None else None,
            'best_f1':float(f1[j]) if j is not None else None}

def budget_counts(y, scores, k):
    # Expected selected positives with a uniform score-independent cutoff tie.
    k = min(int(k), len(y))
    if not k:return 0.0
    threshold = np.partition(scores, len(scores)-k)[len(scores)-k]
    above = scores > threshold; tied = scores == threshold
    return float(y[above].sum() + (k-int(above.sum()))*y[tied].mean())

def lexical_key(text):
    return re.sub(r'^\W+|\W+$', '', text.casefold()) or '<punctuation>'

def parse(repo, unit='whitespace', overlap='any'):
    response_file = repo/'dataset/response.jsonl'; source_file=repo/'dataset/source_info.jsonl'
    sources={r['source_id']:r for r in map(json.loads,source_file.read_text().splitlines())}
    rows=[]; mismatches=0; source_splits=collections.defaultdict(set)
    for r in map(json.loads,response_file.read_text().splitlines()):
        source_splits[r['source_id']].add(r['split'])
        text=r['response']; spans=sorted((a['start'],a['end']) for a in r['labels'])
        merged=[]
        for a,b in spans:
            assert 0<=a<b<=len(text)
            if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
            else:merged.append((a,b))
        mismatches += sum(text[a['start']:a['end']]!=a['text'] for a in r['labels'])
        matches=list(re.finditer(r'\S+' if unit=='whitespace' else r'\w+',text))
        labels=[]; j=0
        for m in matches:
            while j<len(merged) and merged[j][1]<=m.start():j+=1
            intersection=0; z=j
            while z<len(merged) and merged[z][0]<m.end():
                intersection+=max(0,min(m.end(),merged[z][1])-max(m.start(),merged[z][0]));z+=1
            labels.append(intersection>0 if overlap=='any' else intersection*2>=m.end()-m.start())
        src=sources[r['source_id']]
        dev=int(hashlib.sha256(('homers-dream-dev-v1:'+r['source_id']).encode()).hexdigest()[:8],16)%5==0
        split='test' if r['split']=='test' else 'validation' if dev else 'train'
        rows.append({'id':r['id'],'source_id':r['source_id'],'split':split,'quality':r['quality'],
                     'task':src['task_type'],'source':src['source'],'model':r['model'],
                     'y':np.array(labels,dtype=np.int8),
                     'words':[lexical_key(m.group()) for m in matches]})
    assert all(len(s)==1 for s in source_splits.values())
    assert mismatches==0
    return rows

def fit_models(rows, smooth=1000):
    train=[r for r in rows if r['split']=='train' and r['quality']=='good' and len(r['y'])]
    P=sum(int(r['y'].sum()) for r in train); n=sum(len(r['y']) for r in train); pi=P/n
    metadata={m:collections.defaultdict(lambda:[0,0]) for m in ['task','generator','task_generator']}
    lexical=collections.defaultdict(lambda:[0,0])
    def key(r,m):return r['task'] if m=='task' else r['model'] if m=='generator' else (r['task'],r['model'])
    for r in train:
        pos=int(r['y'].sum());length=len(r['y'])
        for m,c in metadata.items():c[key(r,m)][0]+=pos;c[key(r,m)][1]+=length
        for word,y in zip(r['words'],r['y']):lexical[word][0]+=int(y);lexical[word][1]+=1
    rates={m:{k:(p+smooth*pi)/(n+smooth) for k,(p,n) in c.items()} for m,c in metadata.items()}
    lex_rates={w:(p+20*pi)/(n+20) for w,(p,n) in lexical.items() if n>=20}
    def score(r,m):
        if m=='global':return np.full(len(r['y']),float(logit(pi)))
        if m in rates:return np.full(len(r['y']),float(logit(rates[m].get(key(r,m),pi))))
        local=logit(np.array([lex_rates.get(w,pi) for w in r['words']]))
        base=local if m in ['lexical','lexical_mean'] else local+float(logit(rates['task_generator'].get(key(r,'task_generator'),pi))-logit(pi))
        return np.full(len(base),float(np.mean(base))) if m.endswith('_mean') else base
    return score, {'pi':pi,'train_responses':len(train),'train_sources':len({r['source_id'] for r in train}),
                   'metadata_smoothing':smooth,'lexical_smoothing':20,'lexical_min_count':20,'lexical_vocabulary':len(lex_rates)}

def summarize(rs, scores, threshold):
    ys=np.concatenate([r['y'] for r in rs]);ss=np.concatenate(scores)
    pooled=rank_metrics(ys,ss);P=pooled['P'];N=pooled['N']
    local=[rank_metrics(r['y'],s) for r,s in zip(rs,scores)]
    mixed=[z for z in local if z['P'] and z['N']]
    W=sum(z['P']*z['N'] for z in local);C=P*N-W;within_wins=sum(z['wins'] for z in mixed)
    tp=int(ys[ss>=threshold].sum());fp=int((1-ys[ss>=threshold]).sum())
    nAP=[(z['ap']-z['P']/(z['P']+z['N']))/(1-z['P']/(z['P']+z['N'])) for z in mixed]
    result={'responses':len(rs),'sources':len({r['source_id'] for r in rs}),'mixed':len(mixed),
            'P':P,'N':N,'W':W,'C':C,'w':safe(W,P*N),'pooled_auc':pooled['auc'],'pooled_ap':pooled['ap'],
            'within_pair_auc':safe(within_wins,W),'within_macro_auc':float(np.mean([z['auc'] for z in mixed])) if mixed else None,
            'within_macro_nAP':float(np.mean(nAP)) if mixed else None,
            'cross_auc':safe(pooled['wins']-within_wins,C),
            'threshold':threshold,'pooled_f1':safe(2*tp,P+tp+fp),'all_negative':sum(z['P']==0 for z in local),'all_positive':sum(z['N']==0 for z in local)}
    result['budget']={}
    for b in BUDGETS:
        ks=[math.ceil(b*len(r['y'])) for r in rs];tps=[budget_counts(r['y'],s,k) for r,s,k in zip(rs,scores,ks)]
        B=sum(ks);TP=sum(tps);mixed_ids=[i for i,z in enumerate(local) if z['P'] and z['N']]
        lift=[(tps[i]/ks[i])/(local[i]['P']/(local[i]['P']+local[i]['N'])) for i in mixed_ids]
        global_tp=budget_counts(ys,ss,B)
        result['budget'][str(b)]={'selected_words':B,'within_selected_TP':TP,'within_precision':TP/B,'within_recall':TP/P,
                                'within_mixed_precision_lift':float(np.mean(lift)),
                                'global_selected_TP':global_tp,'global_precision':global_tp/B,'global_recall':global_tp/P}
    assert abs(result['pooled_auc']-(result['w']*result['within_pair_auc']+(1-result['w'])*result['cross_auc']))<1e-12
    return result,local

def source_bootstrap(rs,scores,local,threshold,weights,source_ids):
    source_index={s:i for i,s in enumerate(source_ids)}
    ys=np.concatenate([r['y'] for r in rs]);ss=np.concatenate(scores)
    _,inv=np.unique(ss,return_inverse=True);bins=int(inv.max())+1
    token_sources=np.concatenate([np.full(len(r['y']),source_index[r['source_id']],dtype=np.int32) for r in rs])
    # Sparse source-by-score counts avoid independently resampling dependent tokens.
    pos=sparse.csr_matrix((ys,(token_sources,inv)),shape=(len(source_ids),bins),dtype=np.float64)
    neg=sparse.csr_matrix((1-ys,(token_sources,inv)),shape=pos.shape,dtype=np.float64)
    atoms=np.zeros((len(source_ids),8))
    budget_atoms={b:np.zeros((len(source_ids),4)) for b in BUDGETS}
    for r,s,z in zip(rs,scores,local):
        i=source_index[r['source_id']];mixed=bool(z['P'] and z['N'])
        atoms[i]+=[z['wins'] if mixed else 0,z['P']*z['N'],z['auc'] if mixed else 0,int(mixed),z['P'],int(r['y'][s>=threshold].sum()),int((1-r['y'][s>=threshold]).sum()),len(r['y'])]
        for b in BUDGETS:
            k=math.ceil(b*len(r['y']));tp=budget_counts(r['y'],s,k)
            lift=(tp/k)/(z['P']/len(r['y'])) if mixed else 0
            budget_atoms[b][i]+=[tp,k,lift,int(mixed)]
    out=collections.defaultdict(list)
    for start in range(0,len(weights),100):
        w=weights[start:start+100].astype(float)
        pc=pos.T.dot(w.T).T;nc=neg.T.dot(w.T).T
        P=pc.sum(1);N=nc.sum(1);valid=(P>0)&(N>0)
        wins=np.sum(pc*(np.cumsum(nc,axis=1)-.5*nc),axis=1)
        ap=np.sum(pc[:,::-1]*np.cumsum(pc[:,::-1],axis=1)/np.maximum(np.cumsum(pc[:,::-1]+nc[:,::-1],axis=1),1),axis=1)/P
        atom=w@atoms
        out['pooled_auc'].extend(np.where(valid,wins/(P*N),np.nan).tolist())
        out['pooled_ap'].extend(ap.tolist())
        out['within_pair_auc'].extend((atom[:,0]/atom[:,1]).tolist())
        out['within_macro_auc'].extend((atom[:,2]/atom[:,3]).tolist())
        out['pooled_f1'].extend((2*atom[:,5]/(atom[:,4]+atom[:,5]+atom[:,6])).tolist())
        for b in BUDGETS:
            a=w@budget_atoms[b]
            out['within_recall_'+str(b)].extend((a[:,0]/P).tolist())
            out['within_mixed_lift_'+str(b)].extend((a[:,2]/a[:,3]).tolist())
    return {k:np.array(v) for k,v in out.items()}

def ci(v):
    good=v[np.isfinite(v)]
    return {'low':float(np.quantile(good,.025)),'high':float(np.quantile(good,.975)),'valid_draws':len(good),'invalid_draws':len(v)-len(good)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ragtruth',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(exist_ok=True,parents=True);t0=time.monotonic()
    rows=parse(args.ragtruth);score,fit=fit_models(rows)
    eligible=lambda r: r['quality']=='good' and len(r['y'])
    test=[r for r in rows if r['split']=='test' and eligible(r)]
    validation=[r for r in rows if r['split']=='validation' and eligible(r)]
    source_ids=sorted({r['source_id'] for r in test})
    rng=np.random.default_rng(20261014);weights=rng.multinomial(len(source_ids),np.full(len(source_ids),1/len(source_ids)),size=2000).astype(np.int16)
    result={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fit':fit,
            'revision':dataset_revision(args.ragtruth),
            'response_sha256':hashlib.sha256((args.ragtruth/'dataset/response.jsonl').read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256((args.ragtruth/'dataset/source_info.jsonl').read_bytes()).hexdigest(),
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'numpy_version':np.__version__,'bootstrap':{'seed':20261014,'draws':2000,'source_count':len(source_ids),'weights_sha256':hashlib.sha256(weights.tobytes()).hexdigest(),'conditional_on_fitted_models':True},'models':{},'sensitivities':{}}
    boots={};saved={}
    for m in MODELS:
        print('START',m,round(time.monotonic()-t0,1),flush=True)
        scores=[score(r,m) for r in test];val_scores=[score(r,m) for r in validation]
        threshold=rank_metrics(np.concatenate([r['y'] for r in validation]),np.concatenate(val_scores))['threshold']
        summary,local=summarize(test,scores,threshold);saved[m]=(scores,local)
        summary['strata']={}
        for source in sorted({r['source'] for r in test}):
            indices=[i for i,r in enumerate(test) if r['source']==source]
            summary['strata'][source]=summarize([test[i] for i in indices],[scores[i] for i in indices],threshold)[0]
        boot=source_bootstrap(test,scores,local,threshold,weights,source_ids);boots[m]=boot
        summary['confidence95']={k:ci(v) for k,v in boot.items()}
        result['models'][m]=summary
        (args.out/'expanded_results.partial.json').write_text(json.dumps(result,indent=2)+'\n')
        print('DONE',m,'AUC',summary['pooled_auc'],'within',summary['within_macro_auc'],round(time.monotonic()-t0,1),flush=True)
    for r,a,b in zip(test,saved['lexical'][1],saved['lexical_metadata'][1]):
        if a['auc'] is not None:assert abs(a['auc']-b['auc'])<1e-12,(r['id'],a,b)
    result['primary_contrast']={}
    def contrast(a,b):
        output={}
        for k in ['pooled_auc','pooled_ap','within_pair_auc','within_macro_auc','pooled_f1']:
            output[k]={'point':result['models'][a][k]-result['models'][b][k],'ci95':ci(boots[a][k]-boots[b][k])}
        return output
    result['primary_contrast']=contrast('lexical_metadata','lexical')
    result['secondary_prior_minus_lexical']=contrast('task_generator','lexical')
    result['projection_contrasts']={m:contrast(m+'_mean',m) for m in ['lexical','lexical_metadata']}
    np.savez_compressed(args.out/'bootstrap_estimates.npz',**{m+'__'+k:v for m,b in boots.items() for k,v in b.items()})
    for unit,overlap,quality in [('whitespace','any','all'),('word','any','good'),('whitespace','majority','good')]:
        rs=rows if unit=='whitespace' and overlap=='any' else parse(args.ragtruth,unit,overlap)
        sc,fit_s=fit_models(rs);ts=[r for r in rs if r['split']=='test' and len(r['y']) and (quality=='all' or r['quality']=='good')]
        vs=[r for r in rs if r['split']=='validation' and r['quality']=='good' and len(r['y'])]
        th=rank_metrics(np.concatenate([r['y'] for r in vs]),np.concatenate([sc(r,'task_generator') for r in vs]))['threshold']
        result['sensitivities'][unit+'_'+overlap+'_'+quality]=summarize(ts,[sc(r,'task_generator') for r in ts],th)[0]
        print('SENSITIVITY',unit,overlap,quality,flush=True)
    for smoothing in [0,100,1000,10000]:
        sc,_=fit_models(rows,smooth=smoothing)
        th=rank_metrics(np.concatenate([r['y'] for r in validation]),np.concatenate([sc(r,'task_generator') for r in validation]))['threshold']
        result['sensitivities']['metadata_smoothing_'+str(smoothing)]=summarize(test,[sc(r,'task_generator') for r in test],th)[0]
    result['elapsed_seconds']=time.monotonic()-t0
    (args.out/'expanded_results.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    names=['Global','Task','Generator','Task + generator','Lexical','Lexical + metadata','Lexical mean','Lexical + metadata mean']
    fig,ax=plt.subplots(figsize=(9.2,3.4));x=np.arange(len(MODELS));width=.34
    for offset,k,label,color in [(-width/2,'pooled_auc','Pooled AUROC','#3f6a9d'),(width/2,'within_macro_auc','Within-response macro AUROC','#d58239')]:
        vals=np.array([result['models'][m][k] for m in MODELS]);low=np.array([result['models'][m]['confidence95'][k]['low'] for m in MODELS]);high=np.array([result['models'][m]['confidence95'][k]['high'] for m in MODELS])
        ax.bar(x+offset,vals,width,label=label,color=color,yerr=[np.maximum(vals-low,0),np.maximum(high-vals,0)],capsize=2)
    ax.set_xticks(x,names,rotation=20,ha='right');ax.set_ylabel('AUROC');ax.set_ylim(.4,1);ax.axhline(.5,color='gray',linewidth=.7,linestyle='--');ax.legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(args.out/'evaluation_axes.pdf');fig.savefig(args.out/'evaluation_axes.png',dpi=160)
    print('FINISHED',round(result['elapsed_seconds'],1),flush=True)

if __name__=='__main__':main()
