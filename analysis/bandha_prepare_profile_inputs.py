"""Prepare portable frozen inputs without changing their numeric content."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['records', 'reference', 'fit', 'weights', 'source_ids', 'out']:
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    reference = json.loads(args.reference.read_text()); fit = json.loads(args.fit.read_text())
    selected = {k: reference[k] for k in ['models', 'paired_contrasts', 'responses', 'sources', 'saturation_grid', 'bootstrap']}
    selected.update({k: fit[k] for k in ['fit', 'response_sha256', 'source_sha256']})
    selected['scorer_sha256'] = reference['scorer_sha256']
    for name, source in [('records.csv', args.records), ('source_weights.npy', args.weights), ('source_ids.json', args.source_ids)]:
        shutil.copyfile(source, args.out / name)
    (args.out / 'reference.json').write_text(json.dumps(selected, indent=2) + '\n')
    provenance = {'source_sha256': {name: sha(getattr(args, name)) for name in ['records', 'reference', 'fit', 'weights', 'source_ids']},
                  'output_sha256': {p.name: sha(p) for p in args.out.iterdir() if p.is_file() and p.name != 'input_provenance.json'},
                  'transformation': 'Counts/draws copied byte-for-byte. Reference metadata filtered to numeric/scoping fields and source hashes; no numeric reaggregation.'}
    (args.out / 'input_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'responses': selected['responses'], 'sources': selected['sources']}))


if __name__ == '__main__':
    main()
