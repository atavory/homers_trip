"""Install generated table bodies and numerical values into a manuscript."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

TABLES=[('sections/experiments.tex','tab:models',None,'models'),('sections/experiments.tex','tab:budget',None,'budget'),('sections/experiments.tex','tab:false-costs',None,'false_costs'),('sections/supplementary_experiments.tex','tab:datasets',None,'strata'),('sections/supplementary_experiments.tex',None,'Full budget comparison.','full_budget'),('sections/supplementary_experiments.tex',None,'Standard pooled classification outcomes','pooled_classification'),('sections/supplementary_experiments.tex',None,'Point estimates and 95','alarm_intervals'),('sections/archived_counterexample.tex','tab:ragtruth',None,'archive_models')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper',type=Path,required=True)
    parser.add_argument('--aggregates',type=Path,required=True)
    parser.add_argument('--toy',type=Path,required=True)
    parser.add_argument('--bindings',type=Path)
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args();generated=args.paper/'generated';(generated/'tables').mkdir(parents=True,exist_ok=True)
    changes=[]
    for filename,label,caption,name in TABLES:
        p=args.paper/filename;text=p.read_text()
        candidates=list(re.finditer(r'\\begin\{table\}(?:\[[^]]*\])?[\s\S]*?\\end\{table\}',text))
        found=[m for m in candidates if (r'\label{'+label+'}') in m.group() ] if label else [m for m in candidates if caption in m.group()]
        if len(found)!=1:raise ValueError(f'Expected one table: {filename}, {name}; found {len(found)}')
        block=found[0].group();include=r'\input{generated/tables/'+name+'}'
        if include not in block:
            body=re.search(r'\\begin\{tabular\}[\s\S]*?\\end\{tabular\}',block)
            if not body:body=re.search(r'\\input\{tables/[^}]+\}',block)
            if not body:raise ValueError(f'Missing generated table body {name}')
            block=block[:body.start()]+include+block[body.end():]
            updated=text[:found[0].start()]+block+text[found[0].end():]
            p.write_text(updated);changes.append(filename)
        shutil.copyfile(args.aggregates/'tables'/f'{name}.tex',generated/'tables'/f'{name}.tex')
    if args.bindings:
        with args.bindings.open(newline='') as stream:bindings=list(csv.DictReader(stream))
        for b in bindings:
            p=args.paper/b['file'];text=p.read_text();before=b['before'];after=b['after'];expected=int(b.get('count') or 1)
            count=text.count(before)
            if count==expected:p.write_text(text.replace(before,after));changes.append(b['file'])
            elif text.count(after)!=expected:raise ValueError(f"Binding mismatch: {b['file']}: {before[:80]}")
    for name in ['paper_values','archive_values']:
        source=args.aggregates/(name+'.tex')
        if source.exists():shutil.copyfile(source,generated/(name+'.tex'))
    main=args.paper/'main.tex';text=main.read_text();includes='\\input{generated/paper_values}\n\\input{generated/archive_values}\n'
    if '\\input{generated/paper_values}' not in text:main.write_text(text.replace('\\begin{document}',includes+'\\begin{document}'));changes.append('main.tex')
    shutil.copyfile(args.toy/'simpson_reversal.tex',args.paper/'figures/simpson_reversal.tex')
    if (args.aggregates/'figures/inspection_alternatives.pdf').exists():shutil.copyfile(args.aggregates/'figures/inspection_alternatives.pdf',args.paper/'figures/inspection_alternatives.pdf')
    args.manifest.parent.mkdir(parents=True,exist_ok=True)
    payload={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'changed_files':sorted(set(changes)),'generated_outputs':{str(p.relative_to(args.paper)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(generated.rglob('*')) if p.is_file()},'bindings_sha256':hashlib.sha256(args.bindings.read_bytes()).hexdigest() if args.bindings else None}
    args.manifest.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({'installed_tables':len(TABLES),'changed_files':sorted(set(changes))}))


if __name__=='__main__':main()
