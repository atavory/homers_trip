"""Render archived detector tables from replayed, unrounded metrics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

FAMILIES=[('source_summary_mlp','Source-summary MLP'),('layered_graph','Layered graph'),('parameter_matched_mlp','Matched local MLP')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics',type=Path,required=True)
    parser.add_argument('--bootstrap',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--provenance',type=Path)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True);tables=args.out/'tables';tables.mkdir(exist_ok=True)
    metrics=json.loads(args.metrics.read_text());bootstrap=json.loads(args.bootstrap.read_text())
    definitions=[r'\providecommand{\HomerArchiveValue}[1]{\csname HomerArchiveValue@#1\endcsname}'];records=[]
    def define(key,value,digits=4,source='metrics',path=''):
        display=f'{value:.{digits}f}'
        if display.startswith('0.'):display=display[1:]
        definitions.append(r'\expandafter\def\csname HomerArchiveValue@'+key+r'\endcsname{'+display+'}')
        records.append([key,source,path,value,display])
    for family,parts in metrics['models'].items():
        for mode,values in parts.items():
            for key,value in values.items():
                if isinstance(value,(int,float)) and value is not None:
                    define(f'{family}.{mode}.{key}',value,path=f'models.{family}.{mode}.{key}')
                    define(f'{family}.{mode}.{key}.3',value,3,path=f'models.{family}.{mode}.{key}')
    for key,value in bootstrap['between_model_spreads'].items():define('spread.'+key,value,source='bootstrap',path='between_model_spreads.'+key)
    for name, result in bootstrap['contrasts'].items():
        for key, item in result.items():
            ci=item['confidence95'];display=f"{item['point']:+.4f} [{ci['low']:+.4f}, {ci['high']:+.4f}]"
            definitions.append(r'\expandafter\def\csname HomerArchiveValue@contrast.'+name+'.'+key+r'\endcsname{'+display+'}')

    for key,value in [('bootstrap.seed',bootstrap['seed']),('bootstrap.draws',bootstrap['draws']),('cost.assessed_negative_10',bootstrap['models']['source_summary_mlp']['original']['point']['all_negative_inspection_cost_0.1'])]:
        definitions.append(r'\expandafter\def\csname HomerArchiveValue@'+key+r'\endcsname{'+str(int(value))+'}')
    first=metrics['models']['source_summary_mlp']['original']
    for key in ['responses','sources','mixed','positives','negatives','within_pairs','cross_pairs']:
        definitions.append(r'\expandafter\def\csname HomerArchiveValue@counts.'+key+r'\endcsname{'+f'{int(first[key]):,}'+'}')
    for key, value in [('within_pairs',first['within_pairs']),('total_pairs',first['within_pairs']+first['cross_pairs'])]:
        definitions.append(r'\expandafter\def\csname HomerArchiveValue@counts.'+key+r'.math\endcsname{'+f'{int(value):,}'.replace(',', '{,}')+'}')
    if args.provenance:
        provenance=json.loads(args.provenance.read_text())
        all_rows=provenance['prepared']['test']['rows_per_family']
        for key,value in [('all_scored_rows',all_rows),('masked_rows',all_rows-first['positives']-first['negatives'])]:
            definitions.append(r'\expandafter\def\csname HomerArchiveValue@counts.'+key+r'\endcsname{'+f'{int(value):,}'+'}')
        for key, value in provenance['split_counts'].items():
            definitions.append(r'\expandafter\def\csname HomerArchiveValue@counts.'+key+r'_groups\endcsname{'+str(value)+'}')
    define('weight',first['within_pair_weight'],6,path='models.source_summary_mlp.original.within_pair_weight')
    rows=[]
    for family,label in FAMILIES:
        v=metrics['models'][family]['original'];rows.append([label,*[f'{v[key]:.4f}' for key in ['pooled_auc','cross_auc','within_pair_auc','within_macro_auc']]])
    for mode,label in [('eligible_mean','Eligible-row mean'),('all_scored_mean','All-scored-row mean')]:
        v=metrics['models']['source_summary_mlp'][mode];rows.append([label,*[f'{v[key]:.4f}' for key in ['pooled_auc','cross_auc','within_pair_auc','within_macro_auc']]])
    tex=['\\begin{tabular}{lrrrr}','\\toprule',r'Experimental family & Pooled & Cross-response & Within, pair & Within, macro\\','\\midrule',*[' & '.join(row)+r'\\' for row in rows],'\\bottomrule','\\end{tabular}','']
    (tables/'archive_models.tex').write_text('\n'.join(tex))
    with (args.out/'archive_values.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['key','source','json_path','unrounded_value','formatted_value']);writer.writerows(records)
    (args.out/'archive_values.tex').write_text('\n'.join(definitions)+'\n')
    manifest={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'inputs':{'metrics':hashlib.sha256(args.metrics.read_bytes()).hexdigest(),'bootstrap':hashlib.sha256(args.bootstrap.read_bytes()).hexdigest()},'outputs':{str(p.relative_to(args.out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [tables/'archive_models.tex',args.out/'archive_values.csv',args.out/'archive_values.tex']},'rounding':'Differences and spreads are calculated before rounding.'}
    (args.out/'archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(bootstrap['between_model_spreads']))


if __name__=='__main__':main()
