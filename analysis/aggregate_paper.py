"""Generate manuscript tables and CSVs from recorded experiment outputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

MODELS = [
    ('global', 'Global prior'), ('task', 'Task prior'),
    ('generator', 'Generator prior'), ('task_generator', 'Task + generator prior'),
    ('lexical', 'Lexical prior'), ('lexical_metadata', 'Lexical + metadata'),
    ('lexical_mean', 'Lexical, response mean'),
    ('lexical_metadata_mean', 'Lexical + metadata, response mean'),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(path, header, rows):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def tabular(columns, header, rows):
    return '\n'.join([
        '\\begin{tabular}{' + columns + '}', '\\toprule',
        header + r'\\', '\\midrule',
        *[' & '.join(row) + r'\\' for row in rows],
        '\\bottomrule', '\\end{tabular}', '',
    ])


def interval(model, key, point, digits=3):
    limits = model['confidence95'][key]
    return f"{point:.{digits}f} [{limits['low']:.{digits}f}, {limits['high']:.{digits}f}]"


def flatten(value, path=''):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten(item, f'{path}.{key}' if path else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flatten(item, f'{path}[{index}]')
    else:
        yield path, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expanded', type=Path, required=True)
    parser.add_argument('--alternatives', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.expanded.read_text())
    alternatives = json.loads(args.alternatives.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    tables = args.out / 'tables'
    tables.mkdir(exist_ok=True)
    base, alt = baseline['models'], alternatives['models']
    for name, _ in MODELS:
        for key in ['pooled_auc', 'pooled_ap', 'pooled_f1', 'within_pair_auc', 'within_macro_auc']:
            assert abs(base[name][key] - alt[name]['point'][key]) < 1e-12, (name, key)
    inputs = {'expanded': digest(args.expanded), 'alternatives': digest(args.alternatives)}
    save_csv(args.out / 'all_statistics.csv', ['experiment', 'json_path', 'value'],
             [(label, key, value) for label, result in [('expanded', baseline), ('alternatives', alternatives)]
              for key, value in flatten(result)])
    mapping = []
    def metric(name, key, table, row, column, experiment='alternatives'):
        point = alt[name]['point'][key] if experiment == 'alternatives' else base[name][key]
        mapping.append([table, row, column, experiment, f'models.{name}.point.{key}' if experiment == 'alternatives' else f'models.{name}.{key}', point])
        return point
    rows, raw = [], []
    for name, label in MODELS:
        auc = metric(name, 'pooled_auc', 'models', name, 'pooled_auc', 'expanded')
        pair = metric(name, 'within_pair_auc', 'models', name, 'within_pair_auc')
        macro = metric(name, 'within_macro_auc', 'models', name, 'within_macro_auc')
        ap = metric(name, 'within_macro_ap', 'models', name, 'within_macro_ap')
        nap = metric(name, 'within_macro_nap', 'models', name, 'within_macro_nap')
        constant = name not in ['lexical', 'lexical_metadata']
        local = f'{macro:.3f} (exact)' if constant else interval(base[name], 'within_macro_auc', macro)
        rows.append([label, interval(base[name], 'pooled_auc', auc), f'{pair:.3f}', local, f'{ap:.3f}', f'{nap:.3f}'])
        raw.append([name, auc, pair, macro, ap, nap])
    save_csv(args.out / 'models.csv', ['scorer', 'pooled_auc', 'within_pair_auc', 'within_macro_auc', 'within_macro_ap', 'within_macro_nap'], raw)
    (tables / 'models.tex').write_text(tabular('lrrrrr', r'Scorer & Pooled AUROC [95\% CI] & Within pair & Within macro [95\% CI] & Macro AP & nAP', rows))
    budget_names = ['global', 'task_generator', 'lexical', 'lexical_metadata', 'lexical_metadata_mean']
    labels = dict(MODELS)
    keys = ['conditional_word_recall_0.1', 'conditional_complete_span_recall_0.1', 'global_word_recall_0.1', 'global_complete_span_recall_0.1']
    raw, rows = [], []
    for name in budget_names:
        vals = [metric(name, key, 'budget', name, key) for key in keys]
        raw.append([name, *vals]);rows.append([labels[name], *[f'{v:.3f}' for v in vals]])
    save_csv(args.out / 'inspection.csv', ['scorer', *keys], raw)
    header = r' & \multicolumn{2}{c}{Conditional allocation} & \multicolumn{2}{c}{Global allocation}\\' + '\n' + 'Scorer & Word recall & Span recall & Word recall & Span recall'
    (tables / 'budget.tex').write_text(tabular('lrrrr', header, rows))
    cost_names = ['task_generator', 'lexical', 'lexical_metadata']
    keys = ['unmarked_response_alarm_rate', 'unmarked_word_false_positive_rate', 'global_unmarked_cost_fraction_0.1']
    raw, rows, interval_rows = [], [], []
    for name in cost_names:
        vals = [metric(name, key, 'false_costs', name, key) for key in keys]
        raw.append([name, *vals]);rows.append([labels[name], *[f'{100*v:.1f}' for v in vals]])
        interval_rows.append([labels[name], *[interval(alt[name], key, v) for key, v in zip(keys, vals)]])
    save_csv(args.out / 'alarm_costs.csv', ['scorer', *keys], raw)
    (tables / 'false_costs.tex').write_text(tabular('lrrr', r'Scorer & Responses flagged (\%) & Words flagged (\%) & Global cost share (\%)', rows))
    (tables / 'alarm_intervals.tex').write_text(tabular('lrrr', r'Scorer & \shortstack{Unmarked-response\\alarm rate} & \shortstack{Unmarked-word\\false-positive rate} & \shortstack{Global unmarked\\cost share}', interval_rows))
    raw, rows = [], []
    for name in cost_names:
        for budget in [0.05, 0.1, 0.2]:
            keys = [f'conditional_word_recall_{budget}', f'conditional_mixed_precision_lift_{budget}', f'global_word_recall_{budget}']
            vals = [metric(name, key, 'full_budget', f'{name}:{budget}', key) for key in keys]
            raw.append([name, budget, *vals]);rows.append([labels[name], f'{100*budget:.0f}' + r'\%', *[f'{v:.3f}' for v in vals]])
    save_csv(args.out / 'full_budget.csv', ['scorer', 'budget', 'conditional_recall', 'precision_lift', 'global_recall'], raw)
    (tables / 'full_budget.tex').write_text(tabular('llrrr', 'Scorer & Budget & Conditional recall & Precision lift & Global recall', rows))
    rows = [[label, f"{base[name]['pooled_ap']:.3f}", f"{base[name]['pooled_f1']:.3f}"] for name, label in MODELS]
    (tables / 'pooled_classification.tex').write_text(tabular('lrr', 'Scorer & Pooled AP & Validation-threshold F1', rows))
    strata = base['task_generator']['strata']
    rows, raw = [], []
    for source, label in [('MS MARCO', 'MS MARCO'), ('Yelp', 'Yelp'), ('CNN/DM', 'CNN/DailyMail'), ('Recent News', 'Recent news'), (None, 'All strata')]:
        item = strata[source] if source else base['task_generator']
        prevalence = item['P'] / (item['P'] + item['N'])
        vals = [item['responses'], item['mixed'], item['w'], prevalence, item['pooled_auc'], item['pooled_ap'], item['pooled_f1']]
        raw.append([label, *vals]);rows.append([label, f'{vals[0]:,}', str(vals[1]), f'{100*vals[2]:.4f}', *[f'{v:.4f}' for v in vals[3:]]])
    save_csv(args.out / 'strata.csv', ['stratum', 'responses', 'mixed', 'within_pair_weight', 'prevalence', 'pooled_auc', 'pooled_ap', 'pooled_f1'], raw)
    (tables / 'strata.tex').write_text(tabular('lrrrrrrr', r'RAGTruth stratum & Responses & Mixed & $w$ (\%) & Prevalence & AUC & AP & F1', rows))
    headline = {
        'HTWithinPairPercent': f"{100*base['lexical']['w']:.4f}" + r'\%',
        'HTTestResponses': f"{base['lexical']['responses']:,}",
        'HTTestSources': str(base['lexical']['sources']),
        'HTMixedResponses': str(base['lexical']['mixed']),
        'HTLexicalPooledAUROC': f"{base['lexical']['pooled_auc']:.3f}"[1:],
        'HTMetadataPooledAUROC': f"{base['task_generator']['pooled_auc']:.3f}"[1:],
        'HTLexicalMetadataPooledAUROC': f"{base['lexical_metadata']['pooled_auc']:.3f}"[1:],
        'HTLexicalWithinMacro': f"{base['lexical']['within_macro_auc']:.3f}"[1:],
        'HTOffsetPooledGain': f"{base['lexical_metadata']['pooled_auc']-base['lexical']['pooled_auc']:.3f}"[1:],
        'HTLexicalConditionalRecallPercent': f"{100*alt['lexical']['point']['conditional_word_recall_0.1']:.1f}" + r'\%',
        'HTMetadataConditionalRecallPercent': f"{100*alt['task_generator']['point']['conditional_word_recall_0.1']:.1f}" + r'\%',
    }
    (tables / 'headline_stats.tex').write_text(''.join('\\newcommand{\\' + key + '}{' + value + '}\n' for key, value in headline.items()))
    for name, _ in MODELS:
        for key in ['pooled_ap', 'pooled_f1']:
            mapping.append(['pooled_classification', name, key, 'expanded', f'models.{name}.{key}', base[name][key]])
        for key in ['pooled_auc', 'within_macro_auc']:
            for bound in ['low', 'high']:
                mapping.append(['models', name, key+'_'+bound, 'expanded', f'models.{name}.confidence95.{key}.{bound}', base[name]['confidence95'][key][bound]])
    for name in cost_names:
        for key in ['unmarked_response_alarm_rate', 'unmarked_word_false_positive_rate', 'global_unmarked_cost_fraction_0.1']:
            mapping.append(['alarm_intervals', name, key, 'alternatives', f'models.{name}.point.{key}', alt[name]['point'][key]])
            for bound in ['low', 'high']:
                mapping.append(['alarm_intervals', name, key+'_'+bound, 'alternatives', f'models.{name}.confidence95.{key}.{bound}', alt[name]['confidence95'][key][bound]])
    for source in [*strata, None]:
        item = strata[source] if source else base['task_generator']
        prefix = f'models.task_generator.strata.{source}' if source else 'models.task_generator'
        for key in ['responses', 'mixed', 'w', 'pooled_auc', 'pooled_ap', 'pooled_f1']:
            mapping.append(['strata', source or 'All strata', key, 'expanded', prefix+'.'+key, item[key]])
        mapping.append(['strata', source or 'All strata', 'prevalence', 'expanded', prefix+'.P / ('+prefix+'.P + '+prefix+'.N)', item['P']/(item['P']+item['N'])])
    save_csv(args.out / 'table_sources.csv', ['table', 'row', 'column', 'experiment', 'json_path', 'unrounded_value'], mapping)
    generated = [args.out/name for name in ['all_statistics.csv', 'models.csv', 'inspection.csv', 'alarm_costs.csv', 'full_budget.csv', 'strata.csv', 'table_sources.csv']] + [tables/(name+'.tex') for name in ['models', 'budget', 'false_costs', 'alarm_intervals', 'full_budget', 'pooled_classification', 'strata', 'headline_stats']]
    outputs = {str(f.relative_to(args.out)): digest(f) for f in generated}
    manifest = {'script_sha256': digest(Path(__file__)), 'input_sha256': inputs, 'outputs': outputs,
                'command': ['aggregate_paper.py', '--expanded', str(args.expanded), '--alternatives', str(args.alternatives), '--out', str(args.out)],
                'scope': 'Formatting and aggregation of recorded experiment results; no refitting or new inference.'}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'tables': len(list(tables.glob('*.tex'))), 'outputs': len(outputs), 'input_sha256': inputs}, indent=2))


if __name__ == '__main__':
    main()
