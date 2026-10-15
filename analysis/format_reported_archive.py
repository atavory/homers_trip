"""Format preserved archival summaries without claiming a raw-score replay."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.summary.open(newline='')))
    names = ['pooled_auc', 'cross_auc', 'within_pair_auc', 'within_macro_auc']
    for row in rows:
        for name in names:
            if row[name] != 'unavailable' and not 0 <= float(row[name]) <= 1:
                raise ValueError((row['label'], name, row[name]))
    args.out.mkdir(parents=True, exist_ok=True)
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Experimental family & Pooled & Cross-response & Within, pair & Within, macro\\', r'\midrule']
    lines.extend(' & '.join([row['label'], *[row[name] for name in names]]) + r'\\' for row in rows)
    lines.extend([r'\bottomrule', r'\end{tabular}', ''])
    (args.out/'archive_table.tex').write_text('\n'.join(lines))
    manifest = {'status':'reported_summary_only', 'raw_score_replayed':False,
                'summary_sha256':hashlib.sha256(args.summary.read_bytes()).hexdigest(),
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file() and p.name!='manifest.json'}}
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'rows':len(rows), 'status':manifest['status']}))


if __name__=='__main__':
    main()
