"""Optional source acquisition; the numerical runner itself is offline."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reproduction/short"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    pin = json.loads((ROOT / 'datasets.json').read_text())['RAGTruth']
    source = json.loads((REFERENCE / 'package_sources.json').read_text())['fred']
    files = [(f'https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{pin["revision"]}/{name}',
              args.out / 'RAGTruth' / name, digest)
             for name, digest in pin['files'].items()]
    files.append((source['url'], args.out / 'fred_v2.html', source['raw_sha256']))
    records = []
    for url, destination, expected in files:
        destination.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'short-paper-reproduction/1.0'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    raw = response.read()
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise ValueError(f'Source hash changed at {url}: {actual}; expected {expected}. '
                             'Use the retained source snapshot for exact replay.')
        destination.write_bytes(raw)
        records.append({'url': url, 'path': str(destination.relative_to(args.out)),
                        'bytes': len(raw), 'sha256': actual})
        print('VERIFIED', url, len(raw), actual, flush=True)
    (args.out / 'acquisition_manifest.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
