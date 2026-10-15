"""Generate inline TeX values with a CSV record of their numerical sources."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def number(value, digits=3, signed=False, percent=False):
    if value is None or not math.isfinite(float(value)):
        raise ValueError('A manuscript value must be finite.')
    value = float(value) * (100 if percent else 1)
    text = f'{value:+.{digits}f}' if signed else f'{value:.{digits}f}'
    if text.startswith('0.'):
        return text[1:]
    return text.replace('+0.', '+.').replace('-0.', '-.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expanded', type=Path, required=True)
    parser.add_argument('--alternatives', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--toy', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args();args.out.mkdir(parents=True, exist_ok=True)
    inputs = {'expanded': args.expanded, 'alternatives': args.alternatives, 'audit': args.audit, 'toy': args.toy}
    payload = {name: json.loads(path.read_text()) for name, path in inputs.items()}
    values, records = {}, []
    def put(key, value, display, source, locator, lower=None, upper=None):
        if key in values:raise ValueError(f'Duplicate macro {key}')
        values[key] = display
        records.append([key, source, locator, value, lower, upper, display])
    def scalar(key, value, source, locator, ci=None):
        for suffix, digits, signed, percent in [('', 3, False, False), ('.4', 4, False, False), ('.pct', 1, False, True), ('.pct2', 2, False, True)]:
            put(key + suffix, value, number(value, digits, signed, percent), source, locator)
        if ci:
            for suffix, digits in [('.ci', 3), ('.ci4', 4)]:
                display = f"{number(value, digits)} [{number(ci['low'], digits)}, {number(ci['high'], digits)}]"
                put(key + suffix, value, display, source, locator, ci['low'], ci['high'])
    alt = payload['alternatives'];expanded = payload['expanded'];audit = payload['audit']['ragtruth'];toy = payload['toy']['values']
    for model, result in alt['models'].items():
        for metric, value in result['point'].items():
            scalar(f'm.{model}.{metric}', value, 'alternatives', f'models.{model}.point.{metric}', result['confidence95'][metric])
    for name, contrasts in alt['paired_contrasts'].items():
        for metric, result in contrasts.items():
            for suffix, digits in [('', 3), ('.4', 4)]:
                ci = result['confidence95'];value = result['point']
                display = f"{number(value, digits, signed=True)} [{number(ci['low'], digits)}, {number(ci['high'], digits)}]"
                put(f'c.{name}.{metric}' + suffix, value, display, 'alternatives', f'paired_contrasts.{name}.{metric}', ci['low'], ci['high'])
            put(f'c.{name}.{metric}.point', result['point'], number(result['point']), 'alternatives', f'paired_contrasts.{name}.{metric}.point')
    counts = alt['models']['global']['counts'];base = expanded['models']['global']
    fields = {key: (counts[key], 'alternatives', 'models.global.counts.' + key) for key in ['responses', 'words', 'positives', 'mixed', 'mixed_words', 'unmarked_responses', 'unmarked_words', 'marked_responses', 'marked_words', 'spans', 'characters']}
    fields['negatives'] = (counts['words'] - counts['positives'], 'alternatives', 'models.global.counts.words - models.global.counts.positives')
    for key, field in [('sources', 'sources'), ('within_pairs', 'W'), ('cross_pairs', 'C')]:
        fields[key] = (base[field], 'expanded', 'models.global.' + field)
    for key in ['train_responses', 'train_sources', 'validation_responses', 'validation_sources']:
        fields[key] = (audit[key], 'audit', 'ragtruth.' + key)
    for key, (value, source, locator) in fields.items():
        assert int(value) == value
        put('data.' + key, value, f'{int(value):,}', source, locator)
        put('data.' + key + '.math', value, f'{int(value):,}'.replace(',', '{,}'), source, locator)
    put('data.raw_responses', sum(audit['qualities'].values()), f"{sum(audit['qualities'].values()):,}", 'audit', 'sum(ragtruth.qualities)')
    for key, value, digits in [('pair_fraction', base['w'], 6), ('pair_fraction_pct', 100 * base['w'], 4), ('prevalence_pct', 100 * counts['positives'] / counts['words'], 2)]:
        put('data.' + key, value, f'{value:.{digits}f}', 'expanded', 'models.global counts/weight')
    for b, result in alt['models']['global']['budgets'].items():
        selected = result['selected_words'];pct = 100 * selected / counts['words']
        put('budget.' + b + '.words', selected, f'{selected:,}', 'alternatives', f'models.global.budgets.{b}.selected_words')
        put('budget.' + b + '.effective_pct', pct, f'{pct:.2f}', 'alternatives', f'models.global.budgets.{b}.selected_words / words')
    for name, result in expanded['sensitivities'].items():
        put('sensitivity.' + name, result['pooled_auc'], f"{result['pooled_auc']:.4f}", 'expanded', f'sensitivities.{name}.pooled_auc')
    for key, value in toy.items():
        if isinstance(value, (int, float)):
            display = str(value) if isinstance(value, int) else f'{value:.3f}'
            put('toy.' + key, value, display, 'toy', 'values.' + key)
    definition = [r'\providecommand{\HomerValue}[1]{\csname HomerValue@#1\endcsname}']
    definition.extend(r'\expandafter\def\csname HomerValue@' + key + r'\endcsname{' + value + '}' for key, value in sorted(values.items()))
    (args.out / 'paper_values.tex').write_text('\n'.join(definition) + '\n')
    with (args.out / 'paper_values.csv').open('w', newline='') as stream:
        writer = csv.writer(stream);writer.writerow(['key','source','json_path_or_expression','unrounded_value','ci_low','ci_high','formatted_value']);writer.writerows(records)
    manifest = {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'inputs': {name: {'filename': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in inputs.items()}, 'outputs': {name: hashlib.sha256((args.out / name).read_bytes()).hexdigest() for name in ['paper_values.tex','paper_values.csv']}, 'macro_count': len(values)}
    (args.out / 'values_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'values': len(values)}))


if __name__ == '__main__':main()
