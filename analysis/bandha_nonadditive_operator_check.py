"""Exact projection/offset witness for additive and conjunctive recovery."""

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import bandha_intervention_check as core


F = core.F
BASE = (
    core.row((1, 1, 1, 0), (F(9, 10), F(8, 10), F(7, 10), F(6, 10))),
    core.row((1, 1, 1, 0), (F(5, 10), F(4, 10), F(3, 10), F(2, 10))),
)
OFFSETS = (F(0), F(7, 20))
BUDGET = 4
SPAN_SIZE = 3


def project(rows):
    return tuple(core.row(r.labels, [core.mean(r.scores)] * len(r.labels)) for r in rows)


def offset(rows):
    return tuple(core.row(r.labels, [s + c for s in r.scores]) for r, c in zip(rows, OFFSETS))


def selections(rows, budget):
    items = [(s, (r, t)) for r, row in enumerate(rows) for t, s in enumerate(row.scores)]
    if budget == 0:
        return [(frozenset(), F(1))]
    threshold = sorted((s for s, _ in items), reverse=True)[budget - 1]
    above = {i for s, i in items if s > threshold}
    tied = [i for s, i in items if s == threshold]
    q = budget - len(above)
    probability = F(1, math.comb(len(tied), q))
    return [(frozenset(above | set(take)), probability) for take in itertools.combinations(tied, q)]


def recover(rows):
    utility = {'word_recall': F(0), 'span_mean': F(0), 'complete': F(0), 'touched': F(0)}
    law = {}
    actions = []
    for chosen, probability in selections(rows, BUDGET):
        rho = tuple(F(sum((r, t) in chosen for t in range(SPAN_SIZE)), SPAN_SIZE) for r in range(2))
        positive = sum(rows[r].labels[t] for r, t in chosen)
        utility['word_recall'] += probability * F(positive, 2 * SPAN_SIZE)
        utility['span_mean'] += probability * sum(rho) / 2
        utility['complete'] += probability * F(sum(x == 1 for x in rho), 2)
        utility['touched'] += probability * F(sum(x > 0 for x in rho), 2)
        for x in rho:
            law[x] = law.get(x, F(0)) + probability / 2
        actions.append({'chosen': sorted(chosen), 'probability': probability, 'rho': rho})
    assert utility['word_recall'] == utility['span_mean']
    return utility, law, actions


def survival(law, u):
    return sum(p for x, p in law.items() if x > u)


def serialize(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serialize(v) for v in value]
    if hasattr(value, 'labels') and hasattr(value, 'scores'):
        return {'labels': serialize(value.labels), 'scores': serialize(value.scores)}
    return value


def make_plots(output, conditions, outcomes, critical):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.1))
    labels = ['$D$', '$TD$', '$O_{\\mathbf{c}}D$', '$O_{\\mathbf{c}}TD$']
    for ax, ((name, rows), label) in zip(axes, zip(conditions.items(), labels)):
        ax.set(xlim=(-.2, 3.45), ylim=(-2.5, 5.2))
        ax.axis('off')
        chosen = set(map(tuple, outcomes[name]['actions'][0]['chosen']))
        ax.set_title(label, fontsize=12)
        for r, row in enumerate(rows):
            x = r * 1.8
            for t, (truth, score) in enumerate(zip(row.labels, row.scores)):
                y = 3 - t
                ax.add_patch(Rectangle((x, y), 1.0, .96, facecolor='black' if truth else 'white', edgecolor='#666666', linewidth=.5))
                ax.text(x + .5, y + .48, f'{float(score):.2f}', color='white' if truth else 'black', ha='center', va='center', fontsize=8)
                if (r, t) in chosen:
                    ax.add_patch(Rectangle((x, y), 1.0, .96, fill=False, edgecolor='#d57912', linewidth=1.5))
            ax.add_patch(Rectangle((x + 1.05, 0), .12, 4, facecolor='.25', edgecolor='#777777', linewidth=.4))
            ax.text(x + .5, -.45, '$r_' + str(r + 1) + '$', ha='center', fontsize=9)
        v = outcomes[name]['utility']
        ax.text(1.4, -1.65, f"Words: {float(v['word_recall']):.2f}\nComplete: {float(v['complete']):.2f}", ha='center', va='center', fontsize=8)
    fig.suptitle('One three-unit hallucination span per response; four inspected units', fontsize=10)
    fig.text(.5, .015, 'Black: hallucinated units. Orange outline: inspected units. Grey bars: prevalence.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .96))
    for suffix in ['pdf', 'png']:
        fig.savefig(output / ('operator_columns.' + suffix), dpi=180)
    plt.close(fig)

    alpha = [i / 100 for i in range(1, 801)]
    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    ax.axhline(.5, color='#215b85', label='$D$, $TD$, $O_{\\mathbf{c}}TD$')
    ax.plot(alpha, [(2 / 3) ** a for a in alpha], color='#a33a2c', label='$O_{\\mathbf{c}}D$')
    ax.axvline(1, linestyle=':', color='#888888', linewidth=.8)
    ax.axvline(critical, linestyle='--', color='#666666', linewidth=.8)
    ax.scatter([1, 1], [.5, 2 / 3], color=['#215b85', '#a33a2c'], s=20, zorder=3)
    ax.text(1.03, .12, 'Word recall', rotation=90, fontsize=8, color='#666666')
    ax.text(critical + .08, .72, f'Crossing: {critical:.3f}', fontsize=8)
    ax.set(xlabel='Severity exponent alpha', ylabel='Expected span utility', ylim=(-.02, 1.02), xlim=(0, 8))
    ax.set_title('Common weights and equal-length spans', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    for suffix in ['pdf', 'png']:
        fig.savefig(output / ('severity_curve.' + suffix), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    conditions = {'original': BASE, 'projection': project(BASE), 'offset': offset(BASE), 'offset_projection': offset(project(BASE))}
    assert project(project(BASE)) == project(BASE)
    assert project(offset(BASE)) == offset(project(BASE))
    outcomes, values = {}, {}
    for name, rows in conditions.items():
        assert all(0 <= s <= 1 for row in rows for s in row.scores)
        utility, law, actions = recover(rows)
        auc = core.auc(core.pooled_law(rows, [F(1, 2)] * 2))
        local = [core.local_auc(r) for r in rows]
        outcomes[name] = {'rows': rows, 'pooled_auc': auc, 'conditional_auc': local, 'utility': utility, 'rho_law': law, 'actions': actions}
        values[name + '.auc'] = auc
        for key, v in utility.items():
            values[name + '.' + key] = v
    assert values['original.auc'] == F(3, 4)
    assert values['offset.auc'] == 1
    assert values['projection.auc'] == values['offset_projection.auc'] == F(1, 2)
    assert outcomes['original']['conditional_auc'] == outcomes['offset']['conditional_auc'] == [F(1), F(1)]
    assert values['original.word_recall'] == F(1, 2)
    assert values['offset.word_recall'] == F(2, 3)
    assert values['original.complete'] == F(1, 2)
    assert values['offset.complete'] == 0
    assert values['original.touched'] == F(1, 2)
    assert values['offset.touched'] == 1
    for original, shifted in zip(BASE, conditions['offset']):
        assert tuple(s - core.mean(original.scores) for s in original.scores) == tuple(s - core.mean(shifted.scores) for s in shifted.scores)
        for k in range(len(original.labels) + 1):
            assert selections((original,), k) == selections((shifted,), k)
    for metric in ['auc', 'word_recall', 'complete', 'touched']:
        values['interaction.' + metric] = values['offset.' + metric] - values['offset_projection.' + metric] - values['original.' + metric] + values['projection.' + metric]
    assert values['interaction.word_recall'] == F(1, 6)
    assert values['interaction.complete'] == F(-1, 2)
    critical = math.log(.5) / math.log(2 / 3)
    assert 1 < critical < 2
    assert abs((2 / 3) ** critical - .5) < 1e-14
    original_law, offset_law = outcomes['original']['rho_law'], outcomes['offset']['rho_law']
    assert survival(offset_law, F(1, 2)) - survival(original_law, F(1, 2)) == F(1, 2)
    assert survival(offset_law, F(4, 5)) - survival(original_law, F(4, 5)) == F(-1, 2)
    power = [{'alpha': a, 'original': sum(p * x ** a for x, p in original_law.items()), 'offset': sum(p * x ** a for x, p in offset_law.items())} for a in range(1, 9)]
    assert power[0]['offset'] > power[0]['original']
    assert all(row['offset'] < row['original'] for row in power[1:])
    args.out.mkdir(parents=True, exist_ok=True)
    output = {
        'kind': 'exact_finite_operator_and_structured_recovery',
        'all_checks_passed': True,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'helper_sha256': hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        'budget': BUDGET, 'span_size': SPAN_SIZE, 'offsets': OFFSETS,
        'conditions': outcomes, 'values': values, 'integer_power_values': power,
        'critical_alpha': critical,
        'critical_alpha_expression': r'\log(1/2)/\log(2/3)',
        'scope': 'Two equal-length spans; exact fixed-score inspection; no autoregressive feedback or empirical LLM execution.',
    }
    (args.out / 'verification.json').write_text(json.dumps(serialize(output), indent=2) + '\n')
    with (args.out / 'values.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['key', 'exact', 'decimal'])
        writer.writerows((k, str(v), float(v)) for k, v in values.items())
    definitions = [r'\providecommand{\HomerNonadditiveValue}[1]{\csname hn@#1\endcsname}']
    for key, value in values.items():
        tex = str(value.numerator) if value.denominator == 1 else rf'\frac{{{value.numerator}}}{{{value.denominator}}}'
        definitions.append(rf'\expandafter\def\csname hn@{key}\endcsname{{{tex}}}')
    (args.out / 'values.tex').write_text('\n'.join(definitions) + '\n')
    if args.plots:
        make_plots(args.out, conditions, outcomes, critical)
    print(json.dumps({'all_checks_passed': True, 'critical_alpha': critical, 'values': serialize(values), 'script_sha256': output['script_sha256']}))


if __name__ == '__main__':
    main()
