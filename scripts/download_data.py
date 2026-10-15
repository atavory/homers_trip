"""Download and hash-check the pinned public RAGTruth input files."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('data/RAGTruth'))
    args = parser.parse_args()
    spec = json.loads((Path(__file__).resolve().parents[1] / 'datasets.json').read_text())['RAGTruth']
    args.out.mkdir(parents=True, exist_ok=True)
    verified = {}
    for name, expected in spec['files'].items():
        target = args.out / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            verified[name] = expected
            continue
        url = f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{spec['revision']}/{name}"
        with urllib.request.urlopen(url, timeout=90) as response:
            content = response.read()
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise SystemExit(f'Input hash mismatch: {name}')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.download')
        temporary.write_bytes(content)
        temporary.replace(target)
        verified[name] = actual
    manifest = {'dataset': 'RAGTruth', 'revision': spec['revision'], 'input_sha256': verified}
    (args.out / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
