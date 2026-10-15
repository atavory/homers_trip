"""Run the public experiment and paper-aggregation pipeline."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/RAGTruth'))
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--skip-download', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data, out = args.data.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    commands = []
    if not args.skip_download:
        commands.append(['scripts/download_data.py', '--out', str(data)])
    commands.extend([
        ['analysis/check_theory.py', '--out', str(out/'theory.json')],
        ['analysis/decision_framework_check.py', '--output', str(out/'decision_framework')],
        ['analysis/toy_results.py', '--out', str(out/'toy')],
        ['analysis/evaluate.py', '--ragtruth', str(data), '--output', str(out/'initial_audit.json')],
        ['analysis/expanded.py', '--ragtruth', str(data), '--out', str(out/'expanded')],
        ['analysis/decision_relationships.py', '--ragtruth', str(data), '--baseline', str(out/'expanded/expanded_results.json'), '--out', str(out/'decision_relationships')],
        ['analysis/alternatives.py', '--ragtruth', str(data), '--baseline', str(out/'expanded/expanded_results.json'), '--out', str(out/'alternatives')],
        ['analysis/verify_alternatives.py', '--ragtruth', str(data), '--results', str(out/'alternatives')],
        ['analysis/plot_alternatives.py', '--results', str(out/'alternatives/results.json'), '--out', str(out/'alternatives/inspection_alternatives.pdf')],
        ['analysis/aggregate_paper.py', '--expanded', str(out/'expanded/expanded_results.json'), '--alternatives', str(out/'alternatives/results.json'), '--out', str(out/'paper_aggregates')],
        ['analysis/paper_values.py', '--expanded', str(out/'expanded/expanded_results.json'), '--alternatives', str(out/'alternatives/results.json'), '--audit', str(out/'initial_audit.json'), '--toy', str(out/'toy/results.json'), '--out', str(out/'paper_values')],
    ])
    for command in commands:
        print('RUN', command, flush=True)
        subprocess.run([sys.executable, *command], cwd=root, check=True)
    manifest = {'commands': commands, 'python': sys.version,
                'code_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for folder in ['analysis', 'scripts'] for p in sorted((root/folder).glob('*.py'))}}
    (out/'reproduction_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
