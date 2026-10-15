"""Insert generated table inputs into a manuscript and record copied hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, required=True)
    parser.add_argument('--aggregates', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    plans = {
        'sections/experiments.tex': ['models', 'budget', 'false_costs'],
        'sections/supplementary_experiments.tex': ['full_budget', 'pooled_classification', 'alarm_intervals', 'strata'],
    }
    receipt = {'aggregation_manifest_sha256': hashlib.sha256((args.aggregates/'manifest.json').read_bytes()).hexdigest(), 'tables': {}}
    destination = args.paper/'tables';destination.mkdir(exist_ok=True)
    for filename, names in plans.items():
        path = args.paper/filename
        text = path.read_text()
        matches = list(re.finditer(r'\\begin\{table\}.*?\\end\{table\}', text, re.S))
        if len(matches) != len(names):
            raise SystemExit(f'Unexpected table count in {filename}: {len(matches)}')
        for match, name in reversed(list(zip(matches, names))):
            original = match.group(0)
            old = re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}', original, re.S)
            inclusion = r'\input{tables/' + name + '}'
            if old:
                replacement = original[:old.start()] + inclusion + original[old.end():]
            elif inclusion in original:
                replacement = original
            else:
                raise SystemExit(f'Unexpected table body: {filename}, {name}')
            text = text[:match.start()] + replacement + text[match.end():]
            source = args.aggregates/'tables'/f'{name}.tex'
            shutil.copyfile(source, destination/source.name)
            receipt['tables'][str((destination/source.name).relative_to(args.paper))] = {
                'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'section': filename}
        path.write_text(text)
    source = args.aggregates/'tables/headline_stats.tex'
    shutil.copyfile(source, destination/source.name)
    main = args.paper/'main.tex'
    text = main.read_text()
    if r'\input{tables/headline_stats}' not in text:
        text = text.replace(r'\begin{document}', r'\input{tables/headline_stats}' + '\n' + r'\begin{document}', 1)
        main.write_text(text)
    substitutions = {
        r'0.0376\%': r'\HTWithinPairPercent{}',
        '.633': r'\HTLexicalPooledAUROC{}',
        '.674': r'\HTMetadataPooledAUROC{}',
        '.736': r'\HTLexicalMetadataPooledAUROC{}',
        '.686': r'\HTLexicalWithinMacro{}',
        '2,675 responses': r'\HTTestResponses{} responses',
        '450 sources': r'\HTTestSources{} sources',
        '904 mixed responses': r'\HTMixedResponses{} mixed responses',
        r'22.9\%': r'\HTLexicalConditionalRecallPercent{}',
        r'10.3\%': r'\HTMetadataConditionalRecallPercent{}',
    }
    for filename in ['sections/abstract.tex', 'sections/introduction.tex']:
        path = args.paper/filename
        text = path.read_text()
        for old, new in substitutions.items():
            text = text.replace(old, new)
        path.write_text(text)
    receipt['headline_statistics'] = {'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'inserted_tables': len(receipt['tables'])}, indent=2))


if __name__ == '__main__':
    main()
