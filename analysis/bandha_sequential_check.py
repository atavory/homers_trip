"""Exact autoregressive correction fixtures with accepted-prefix feedback."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import bandha_intervention_check as core


F = core.F
HORIZON = 20
PROMPTS = ('error_bearing', 'critical_clean', 'terminal_clean_1', 'terminal_clean_2')
THRESHOLD = F(1, 2)


def proposal(prompt, prefix, feedback=True):
    t = len(prefix)
    if prompt == 'error_bearing':
        return int(t >= HORIZON - 5)
    if prompt == 'critical_clean' and feedback and prefix:
        return int(prefix[0] == 1)
    return 0


def detector_score(prompt, prefix, candidate, variant='common_threshold'):
    t = len(prefix)
    if variant == 'f1_optimal':
        if prompt == 'error_bearing':
            if t == HORIZON - 5:
                return F(1, 5)
            return F(11, 20) if t > HORIZON - 5 else F(1, 20)
        if prompt == 'critical_clean':
            return F(1, 5) if t == 0 else F(1, 20)
        return F(4, 5) if t == HORIZON - 1 else F(2, 5)
    if prompt == 'error_bearing':
        if t == HORIZON - 5:
            return F(2, 5)
        return F(7, 10) if t > HORIZON - 5 else F(1, 10)
    if prompt == 'critical_clean':
        return F(2, 5) if t == 0 else F(1, 10)
    return F(3, 5) if t == HORIZON - 1 else F(1, 5)


def response_offset(prompt, variant='common_threshold'):
    magnitude = F(2, 5) if variant == 'f1_optimal' else F(1, 5)
    return magnitude if prompt in PROMPTS[:2] else -magnitude


def replacement(candidate):
    return 1 - candidate


def rollout(prompt, policy, feedback=True, variant='common_threshold', threshold=THRESHOLD):
    prefix, events = [], []
    for t in range(HORIZON):
        candidate = proposal(prompt, prefix, feedback)
        score = detector_score(prompt, prefix, candidate, variant)
        if policy == 'offset':
            score += response_offset(prompt, variant)
        selected = policy != 'none' and score >= threshold
        accepted = replacement(candidate) if selected else candidate
        events.append({
            'prompt': prompt, 't': t, 'prefix': tuple(prefix),
            'candidate': candidate, 'score': score, 'selected': selected,
            'accepted': accepted, 'proposed_error': candidate,
            'emitted_error': accepted,
        })
        prefix.append(accepted)
    return {'prompt': prompt, 'events': events, 'accepted': tuple(prefix)}


def frozen_rows(policy, variant='common_threshold'):
    rows = []
    for prompt in PROMPTS:
        run = rollout(prompt, 'none', variant=variant)
        scores = [e['score'] for e in run['events']]
        if policy == 'offset':
            scores = [s + response_offset(prompt, variant) for s in scores]
        rows.append(core.row([e['candidate'] for e in run['events']], scores))
    return rows


def average_precision(rows):
    groups = defaultdict(list)
    for r in rows:
        for score, label in zip(r.scores, r.labels):
            groups[score].append(label)
    total_positive = sum(sum(r.labels) for r in rows)
    seen, positives, result = 0, 0, F(0)
    for score in sorted(groups, reverse=True):
        labels = groups[score]
        seen += len(labels)
        positives += sum(labels)
        result += F(sum(labels), total_positive) * F(positives, seen)
    return result


def classification(rows):
    tp = fp = fn = tn = 0
    for r in rows:
        for score, label in zip(r.scores, r.labels):
            selected = score >= THRESHOLD
            tp += int(selected and label)
            fp += int(selected and not label)
            fn += int(not selected and label)
            tn += int(not selected and not label)
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': F(tp, tp + fp), 'recall': F(tp, tp + fn),
        'f1': F(2 * tp, 2 * tp + fp + fn),
        'accuracy': F(tp + tn, tp + fp + fn + tn),
        'auc': core.auc(core.pooled_law(rows, [F(1, 4)] * 4)),
        'ap': average_precision(rows),
        'conditional_auc_mixed': [core.local_auc(r) for r in rows if 0 < sum(r.labels) < len(r.labels)],
    }


def summarize(runs):
    events = [e for run in runs for e in run['events']]
    return {
        'errors': sum(e['emitted_error'] for e in events),
        'edits': sum(e['selected'] for e in events),
        'true_fixes': sum(e['selected'] and e['proposed_error'] for e in events),
        'false_edits': sum(e['selected'] and not e['proposed_error'] for e in events),
        'missed_proposed_errors': sum(not e['selected'] and e['proposed_error'] for e in events),
        'proposed_errors': sum(e['proposed_error'] for e in events),
        'emitted_tokens': len(events),
    }


def value_base(prompt, prefix, feedback=True):
    completed = list(prefix)
    while len(completed) < HORIZON:
        completed.append(proposal(prompt, completed, feedback))
    return -sum(completed)


def performance_difference(runs):
    total = 0
    for run in runs:
        prompt = run['prompt']
        for event in run['events']:
            h = event['prefix']
            a = event['accepted']
            total += value_base(prompt, h + (a,)) - value_base(prompt, h)
    return total


def absorbing_loss(k, epsilon):
    good, loss = F(1), F(0)
    for _ in range(k):
        good *= 1 - epsilon
        loss += 1 - good
    q = 1 - epsilon
    closed = F(k) - q * (1 - q ** k) / epsilon
    first_event = sum(epsilon * q ** (t - 1) * (k - t + 1) for t in range(1, k + 1))
    assert loss == closed == first_event
    return {
        'horizon': k, 'epsilon': epsilon, 'errors': loss,
        'actual_false_edits': 1 - q ** k,
        'original_state_false_alarm_total': k * epsilon,
        'linear_epsilon_coefficient': F(k * (k + 1), 2),
    }


def continuation_fixture(kind):
    def generate(prefix):
        if not prefix:
            return 1
        if kind == 'harmful_true_fix':
            return int(prefix[0] == 0)
        return int(prefix[0] == 1)
    base = []
    while len(base) < 3:
        base.append(generate(base))
    corrected = [0]
    while len(corrected) < 3:
        corrected.append(generate(corrected))
    return {'base': tuple(base), 'corrected': tuple(corrected), 'base_errors': sum(base), 'corrected_errors': sum(corrected)}


def occupancy_fixture():
    def next_token(h):
        return 0 if not h else h[0]
    def policy(h):
        return 1 if not h else 0
    def value(h):
        completed = list(h)
        while len(completed) < 2:
            completed.append(next_token(completed))
        return -sum(completed)
    h, actual_sum = (), 0
    while len(h) < 2:
        a = policy(h)
        actual_sum += value(h + (a,)) - value(h)
        h += (a,)
    nominal, old_sum = (), 0
    while len(nominal) < 2:
        old_sum += value(nominal + (policy(nominal),)) - value(nominal)
        nominal += (next_token(nominal),)
    true_difference = value(h) - value(())
    assert true_difference == actual_sum == -1
    assert old_sum == -2
    return {'controlled_trajectory': h, 'actual_value_difference': true_difference, 'controlled_occupancy_sum': actual_sum, 'original_occupancy_surrogate': old_sum}


def f1_at(rows, threshold):
    tp = fp = fn = 0
    for r in rows:
        for score, label in zip(r.scores, r.labels):
            selected = score >= threshold
            tp += int(selected and label)
            fp += int(selected and not label)
            fn += int(not selected and label)
    return F(2 * tp, 2 * tp + fp + fn)


def f1_optimal_fixture():
    policies, traces, values = {}, {}, {}
    for policy in ['detector', 'offset']:
        rows = frozen_rows(policy, 'f1_optimal')
        metrics = classification(rows)
        thresholds = sorted({s for r in rows for s in r.scores} | {F(1)})
        best_f1 = max(f1_at(rows, threshold) for threshold in thresholds)
        assert best_f1 == metrics['f1'] == f1_at(rows, THRESHOLD)
        policies[policy] = {'frozen': metrics, 'maximum_frozen_f1': best_f1, 'common_threshold_is_f1_optimal': True}
        for feedback in [False, True]:
            name = 'feedback' if feedback else 'fixed_suffix'
            runs = [rollout(p, policy, feedback, variant='f1_optimal') for p in PROMPTS]
            stats = summarize(runs)
            assert stats['edits'] == 6
            assert stats['errors'] == stats['proposed_errors'] - stats['true_fixes'] + stats['false_edits']
            policies[policy][name] = stats
            traces[policy + '.' + name] = runs
            values['optimal.' + policy + '.' + name + '.errors'] = F(stats['errors'])
        selection_on = [[(e['t'], e['candidate'], e['accepted']) for e in run['events'] if e['selected']] for run in traces[policy + '.feedback']]
        selection_off = [[(e['t'], e['candidate'], e['accepted']) for e in run['events'] if e['selected']] for run in traces[policy + '.fixed_suffix']]
        assert selection_on == selection_off
        gate_outcomes = {}
        for threshold in thresholds:
            stats = summarize([rollout(p, policy, variant='f1_optimal', threshold=threshold) for p in PROMPTS])
            gate_outcomes[str(threshold)] = stats
        policies[policy]['correction_threshold_grid'] = gate_outcomes
        policies[policy]['minimum_feedback_errors_over_thresholds'] = min(s['errors'] for s in gate_outcomes.values())
        for key in ['tp', 'fp', 'f1', 'auc', 'ap']:
            values['optimal.' + policy + '.' + key] = F(metrics[key])
        values['optimal.' + policy + '.edits'] = F(6)
    assert policies['detector']['frozen']['auc'] == F(653, 750)
    assert policies['detector']['frozen']['ap'] == F(383, 690)
    assert policies['offset']['frozen']['auc'] == F(749, 750)
    assert policies['offset']['frozen']['ap'] == F(29, 30)
    assert policies['detector']['feedback']['errors'] == 3
    assert policies['offset']['feedback']['errors'] == 20
    assert policies['offset']['fixed_suffix']['errors'] == 1
    assert policies['detector']['feedback']['proposed_errors'] == 5
    assert policies['offset']['feedback']['proposed_errors'] == 24
    assert policies['detector']['minimum_feedback_errors_over_thresholds'] == 3
    assert policies['offset']['minimum_feedback_errors_over_thresholds'] == 1
    return policies, traces, values


def same_frozen_law_fixture():
    def generate(h):
        return int(bool(h) and h[0] == 1 or len(h) in (HORIZON - 3, HORIZON - 2))
    def score(h, early):
        t = len(h)
        if t in (HORIZON - 3, HORIZON - 2):
            return F(7, 10)
        if t == (0 if early else HORIZON - 1):
            return F(3, 5)
        return F(1, 10)
    nominal = []
    while len(nominal) < HORIZON:
        nominal.append(generate(nominal))
    rows, outcomes = [], []
    for early in [True, False]:
        rows.append(core.row(nominal, [score(tuple(nominal[:t]), early) for t in range(HORIZON)]))
        accepted, edits = [], 0
        while len(accepted) < HORIZON:
            candidate = generate(accepted)
            selected = score(accepted, early) >= THRESHOLD
            accepted.append(replacement(candidate) if selected else candidate)
            edits += int(selected)
        outcomes.append({'accepted': tuple(accepted), 'errors': sum(accepted), 'edits': edits})
    assert core.pooled_law([rows[0]], [F(1)]) == core.pooled_law([rows[1]], [F(1)])
    assert core.local_auc(rows[0]) == core.local_auc(rows[1]) == F(1)
    assert [o['errors'] for o in outcomes] == [18, 1]
    assert [o['edits'] for o in outcomes] == [3, 3]
    return {'frozen_rows': rows, 'early_false_edit': outcomes[0], 'late_false_edit': outcomes[1], 'identical_complete_conditional_score_label_law': True}


def finite_tree_checks():
    from functools import lru_cache
    outputs = []
    for horizon in range(1, 8):
        def probability(h):
            return F(1 + (len(h) + sum(h)) % 3, 4)
        def gate(h, a):
            return F(1 + (len(h) + 2 * sum(h) + a) % 3, 5)
        def cost(h, a):
            return F(1 + len(h) + a, 10)
        def payoff(h):
            return sum(h) + 2 * int(h[0] == h[-1]) - 3 * sum(h[i] == h[i + 1] == 1 for i in range(horizon - 1))
        @lru_cache(None)
        def base(h):
            if len(h) == horizon:
                return F(payoff(h))
            p = probability(h)
            return (1 - p) * base(h + (0,)) + p * base(h + (1,))
        @lru_cache(None)
        def controlled(h):
            if len(h) == horizon:
                return F(payoff(h))
            p, value = probability(h), F(0)
            for a, mass in [(0, 1 - p), (1, p)]:
                g = gate(h, a)
                value += mass * ((1 - g) * controlled(h + (a,)) + g * (controlled(h + (1 - a,)) - cost(h, a)))
            return value
        def advantages(occupancy):
            total = F(0)
            for h, mass in occupancy.items():
                p = probability(h)
                for a, proposal_mass in [(0, 1 - p), (1, p)]:
                    total += mass * proposal_mass * gate(h, a) * (base(h + (1 - a,)) - base(h + (a,)) - cost(h, a))
            return total
        actual_d, nominal_d = {(): F(1)}, {(): F(1)}
        actual_sum, nominal_sum = F(0), F(0)
        for _ in range(horizon):
            actual_sum += advantages(actual_d)
            nominal_sum += advantages(nominal_d)
            new_actual, new_nominal = defaultdict(F), defaultdict(F)
            for h, mass in actual_d.items():
                p = probability(h)
                for a, proposal_mass in [(0, 1 - p), (1, p)]:
                    g = gate(h, a)
                    new_actual[h + (a,)] += mass * proposal_mass * (1 - g)
                    new_actual[h + (1 - a,)] += mass * proposal_mass * g
            for h, mass in nominal_d.items():
                p = probability(h)
                new_nominal[h + (0,)] += mass * (1 - p)
                new_nominal[h + (1,)] += mass * p
            actual_d, nominal_d = dict(new_actual), dict(new_nominal)
        difference = controlled(()) - base(())
        assert difference == actual_sum
        if horizon > 1:
            assert nominal_sum != difference
        outputs.append({'horizon': horizon, 'value_difference': difference, 'controlled_occupancy_sum': actual_sum, 'original_occupancy_surrogate': nominal_sum})
    return outputs


def plot_columns(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    groups = [
        ('Base generator', [rollout(p, 'none', variant='f1_optimal') for p in PROMPTS]),
        ('Detector controller', [rollout(p, 'detector', variant='f1_optimal') for p in PROMPTS]),
        ('Offset controller', [rollout(p, 'offset', variant='f1_optimal') for p in PROMPTS]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 4.8))
    for ax, (name, runs) in zip(axes, groups):
        ax.set(xlim=(-.55, 5.15), ylim=(-1.7, HORIZON + 1.4))
        ax.axis('off')
        total = summarize(runs)['errors']
        ax.set_title(name + '\n' + str(total) + ' final errors', fontsize=10)
        for i, run in enumerate(runs):
            x = i * 1.3
            errors = sum(run['accepted'])
            for t, event in enumerate(run['events']):
                y = HORIZON - 1 - t
                ax.add_patch(Rectangle((x, y), .8, .95, facecolor='black' if event['emitted_error'] else 'white', edgecolor='#666666', linewidth=.4))
                if event['selected']:
                    ax.add_patch(Rectangle((x, y), .8, .95, fill=False, edgecolor='#d57912', linewidth=1.2))
            ax.add_patch(Rectangle((x + .87, 0), .18, HORIZON, facecolor=str(1 - errors / HORIZON), edgecolor='#777777', linewidth=.4))
            ax.text(x + .4, -.8, '$r_' + str(i + 1) + '$', ha='center', fontsize=9)
    axes[0].annotate('', xy=(-.32, 1), xytext=(-.32, HORIZON - 1), arrowprops={'arrowstyle': '->', 'color': '#555555', 'lw': .7})
    fig.text(.5, .015, 'Black: erroneous output units. Orange outline: directly overridden units. Time runs downward.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for suffix in ['pdf', 'png']:
        fig.savefig(output / ('causal_columns.' + suffix), dpi=180)
    plt.close(fig)


def plot_results(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    epsilon = F(1, 100)
    lengths = list(range(1, 51))
    cascade = [float(absorbing_loss(k, epsilon)['errors']) for k in lengths]
    nominal = [float(k * epsilon) for k in lengths]
    ax.plot(lengths, cascade, color='#a33a2c', label='Errors after prefix feedback')
    ax.plot(lengths, nominal, color='#215b85', label='False alarms on unchanged prefixes')
    ax.set(xlabel='Generation horizon (units)', ylabel='Expected count', title='Constructed absorbing-prefix model, false-alarm probability 0.01')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    for suffix in ['pdf', 'png']:
        fig.savefig(output / ('compounding.' + suffix), dpi=180)
    plt.close(fig)

    from matplotlib.path import Path as PlotPath
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    ax.set(xlim=(0, 13.8), ylim=(0, 4.3))
    ax.axis('off')
    boxes = [
        (0.3, 2.0, 2.2, 'Context\naccepted prefix\n$h_t$'),
        (3.0, 2.0, 1.9, 'Same\ngenerator\nproposes $a_t$'),
        (5.4, 2.0, 2.5, 'Detector +\ncontroller\nkeep or override'),
        (8.4, 2.0, 2.0, 'Accepted\ntoken\n' + r'$\tilde a_t$'),
        (10.9, 2.0, 2.3, 'Updated\nhistory\n' + r'$h_{t+1}$'),
    ]
    for x, y, width, label in boxes:
        box = FancyBboxPatch((x, y), width, 1.0, boxstyle='round,pad=0.10', facecolor='#fff3df' if x == 8.4 else '#f1f5f8', edgecolor='#344654', linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + width / 2, y + .5, label, ha='center', va='center', fontsize=10)
    for a, b in [((2.6, 2.5), (2.9, 2.5)), ((5.0, 2.5), (5.3, 2.5)), ((8.0, 2.5), (8.3, 2.5)), ((10.5, 2.5), (10.8, 2.5))]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=12, linewidth=1.1, color='#344654'))
    path = PlotPath([(12.05, 1.9), (12.05, .9), (1.4, .9), (1.4, 1.9)], [PlotPath.MOVETO, PlotPath.LINETO, PlotPath.LINETO, PlotPath.LINETO])
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=12, linewidth=1.3, color='#9b5b1e'))
    ax.text(6.8, .45, 'Generate and rescore on the accepted prefix, including all earlier corrections', ha='center', fontsize=10, color='#714516')
    ax.text(6.9, 3.8, 'At termination: evaluate factuality, task quality, preservation, and cost', ha='center', fontsize=12)
    fig.tight_layout()
    for suffix in ['pdf', 'png']:
        fig.savefig(output / ('sequential_feedback.' + suffix), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    base_rows, offset_rows = frozen_rows('detector'), frozen_rows('offset')
    metrics = {'detector': classification(base_rows), 'offset': classification(offset_rows)}
    assert metrics['detector']['auc'] == F(149, 150)
    assert metrics['offset']['auc'] == F(749, 750)
    assert metrics['detector']['ap'] == F(37, 40)
    assert metrics['offset']['ap'] == F(29, 30)
    assert metrics['detector']['f1'] == F(8, 11)
    assert metrics['offset']['f1'] == F(10, 11)
    assert metrics['offset']['tp'] > metrics['detector']['tp']
    assert metrics['offset']['fp'] < metrics['detector']['fp']
    assert metrics['detector']['conditional_auc_mixed'] == metrics['offset']['conditional_auc_mixed'] == [F(1)]
    for r, shifted in zip(base_rows, offset_rows):
        assert all(0 <= s <= 1 for s in shifted.scores)
        assert tuple(s - core.mean(r.scores) for s in r.scores) == tuple(s - core.mean(shifted.scores) for s in shifted.scores)
    records, traces, values = {}, {}, {}
    baseline = summarize([rollout(p, 'none') for p in PROMPTS])
    assert baseline['errors'] == 5
    for policy in ['detector', 'offset']:
        records[policy] = {'frozen': metrics[policy]}
        for feedback in [False, True]:
            name = 'feedback' if feedback else 'fixed_suffix'
            runs = [rollout(p, policy, feedback) for p in PROMPTS]
            stats = summarize(runs)
            records[policy][name] = stats
            traces[policy + '.' + name] = runs
            values[policy + '.' + name + '.errors'] = F(stats['errors'])
            assert stats['edits'] == 6
            if feedback:
                assert performance_difference(runs) == baseline['errors'] - stats['errors']
        values[policy + '.tp'] = F(metrics[policy]['tp'])
        values[policy + '.fp'] = F(metrics[policy]['fp'])
        values[policy + '.f1'] = metrics[policy]['f1']
        values[policy + '.auc'] = metrics[policy]['auc']
        values[policy + '.ap'] = metrics[policy]['ap']
        values[policy + '.edits'] = F(records[policy]['feedback']['edits'])
    assert records['detector']['feedback']['errors'] == 3
    assert records['offset']['feedback']['errors'] == 20
    assert records['detector']['fixed_suffix']['errors'] == 3
    assert records['offset']['fixed_suffix']['errors'] == 1
    assert records['offset']['feedback']['missed_proposed_errors'] == 19
    beneficial, harmful = continuation_fixture('beneficial_true_fix'), continuation_fixture('harmful_true_fix')
    assert (harmful['base_errors'], harmful['corrected_errors']) == (1, 2)
    assert (beneficial['base_errors'], beneficial['corrected_errors']) == (3, 0)
    absorption = [absorbing_loss(k, epsilon) for k in [1, 2, 5, 20] for epsilon in [F(1, 100), F(1, 10)]]
    optimal_policies, optimal_traces, optimal_values = f1_optimal_fixture()
    values.update(optimal_values)
    traces['f1_optimal_variant'] = optimal_traces
    result = {
        'f1_optimal_variant': optimal_policies,
        'same_frozen_law_position_fixture': same_frozen_law_fixture(),
        'kind': 'constructed_autoregressive_correction', 'all_checks_passed': True,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'helper_sha256': hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        'inputs': 'finite generator, detector and repair functions embedded in script',
        'horizon': HORIZON, 'prompts': PROMPTS, 'threshold': THRESHOLD,
        'baseline': baseline, 'policies': records,
        'correct_local_edit_harms_continuation': harmful,
        'correct_local_edit_improves_continuation': beneficial,
        'absorbing_prefix_fixtures': absorption,
        'occupancy_fixture': occupancy_fixture(),
        'generic_finite_tree_policy_checks': finite_tree_checks(),
        'scope': 'Exact finite constructions, not measured LLM behavior; matched edit counts, not a real-model compute benchmark',
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'verification.json').write_text(json.dumps(core.serializable(result), indent=2) + '\n')
    (args.out / 'trajectories.json').write_text(json.dumps(core.serializable(traces), indent=2) + '\n')
    with (args.out / 'values.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['quantity', 'exact', 'decimal'])
        writer.writerows((k, str(v), format(float(v), '.12f')) for k, v in values.items())
    lines = [r'\providecommand{\HomerSequentialValue}[1]{\csname hs@#1\endcsname}']
    lines.extend(r'\expandafter\def\csname hs@' + k + r'\endcsname{' + core.tex_fraction(v) + '}' for k, v in values.items())
    (args.out / 'values.tex').write_text('\n'.join(lines) + '\n')
    if args.plots:
        plot_results(args.out)
        plot_columns(args.out)
    print(json.dumps({'all_checks_passed': True, 'baseline_errors': baseline['errors'], 'detector': core.serializable(records['detector']), 'offset': core.serializable(records['offset']), 'script_sha256': result['script_sha256']}))


if __name__ == '__main__':
    main()
