"""Freeze shared RAGTruth baseline scores and optionally merge trained predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import expanded as core


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ragtruth',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--shards',type=Path,nargs='*',default=[])
    ap.add_argument('--reference-scores',type=Path)
    a=ap.parse_args()
    rows=core.parse(a.ragtruth);score,fit=core.fit_models(rows)
    test=[r for r in rows if r['split']=='test' and r['quality']=='good' and len(r['y'])]
    a.out.mkdir(parents=True,exist_ok=True)
    index=[{'id':r['id'],'source_id':r['source_id'],'word_count':len(r['y'])} for r in test]
    offsets=np.r_[0,np.cumsum([len(r['y']) for r in test])]
    labels=np.concatenate([r['y'] for r in test])
    scores={n:np.concatenate([score(r,n) for r in test]) for n in ['generator','lexical','lexical_metadata']}
    if a.reference_scores:
        with np.load(a.reference_scores,allow_pickle=False) as z:
            for n in ['lexical','lexical_metadata']:assert np.array_equal(scores[n],z[n]),n
            assert np.array_equal(labels,z['labels'])
            assert np.array_equal(offsets,z['response_offsets'])
    shards=[]
    if a.shards:
        outputs={}
        for path in a.shards:
            report=json.loads(path.with_suffix('.json').read_text())
            assert hashlib.sha256(path.read_bytes()).hexdigest()==report['scores_sha256']
            shards.append(report)
            with np.load(path,allow_pickle=False) as z:
                for i,(lo,hi) in zip(z['response_indices'],zip(z['offsets'][:-1],z['offsets'][1:])):
                    assert int(i) not in outputs
                    outputs[int(i)]=z['scores'][lo:hi].copy()
        assert sorted(outputs)==list(range(len(test)))
        assert all(len(outputs[i])==len(test[i]['y']) for i in outputs)
        scores['lettuce']=np.concatenate([outputs[i] for i in range(len(test))]).astype(np.float64)
        assert np.isfinite(scores['lettuce']).all() and ((scores['lettuce']>=0)&(scores['lettuce']<=1)).all()
        scores['lettuce_mean']=np.concatenate([np.full(len(outputs[i]),np.mean(outputs[i],dtype=np.float64)) for i in range(len(test))])
    filename='trained_scores.npz' if a.shards else 'baseline_scores.npz'
    np.savez_compressed(a.out/filename,labels=labels,response_offsets=offsets,**scores)
    (a.out/'response_index.json').write_text(json.dumps(index,separators=(',',':'))+'\n')
    record={'fit':fit,'responses':len(test),'words':len(labels),'annotated_words':int(labels.sum()),
            'methods':list(scores),'inference_shards':shards,
            'raw_sha256':{f:hashlib.sha256((a.ragtruth/'dataset'/f).read_bytes()).hexdigest() for f in ['response.jsonl','source_info.jsonl']},
            'scores_sha256':hashlib.sha256((a.out/filename).read_bytes()).hexdigest()}
    (a.out/'input_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'responses':len(test),'words':len(labels),'methods':list(scores)}))

if __name__=='__main__':main()
