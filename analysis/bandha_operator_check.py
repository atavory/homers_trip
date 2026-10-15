"""Exact projection, response-offset, and four-arm metric constructions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import bandha_intervention_check as core


F = core.F


def project(rows):
    return [core.row(r.labels, [core.mean(r.scores)] * len(r.labels)) for r in rows]


def offset(rows, values):
    return [
        core.row(r.labels, [score + value for score in r.scores])
        for r, value in zip(rows, values)
    ]


def residual(rows):
    return [
        tuple(s - core.mean(r.scores) for s in r.scores)
        for r in rows
    ]


def covariance(values, rates, weights):
    mean_value = sum(q * value for q, value in zip(weights, values))
    mean_rate = sum(q * rate for q, rate in zip(weights, rates))
    return sum(
        q * (value - mean_value) * (rate - mean_rate)
        for q, value, rate in zip(weights, values, rates)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    original = [
        core.row([1, 1, 1, 1, 0], [16, 16, 16, 20, 12], 20),
        core.row([1, 0, 0, 0, 0], [14, 12, 12, 12, 0], 20),
    ]
    shifts = [F(-3, 20), F(2, 20)]
    weights = [F(1, 2), F(1, 2)]
    projected = project(original)
    shifted = offset(original, shifts)
    shifted_projected = offset(projected, shifts)
    assert project(projected) == projected
    assert project(shifted) == shifted_projected
    assert residual(original) == residual(shifted)
    assert residual(projected) == [tuple(F(0) for _ in r.labels) for r in original]
    for a, b in zip(original, shifted):
        assert all(
            x - y == u - v
            for x, u in zip(a.scores, b.scores)
            for y, v in zip(a.scores, b.scores)
        )
        assert all(core.capture(a, k) == core.capture(b, k) for k in range(6))
    arms = {
        'original': original, 'projected': projected,
        'offset': shifted, 'offset_projected': shifted_projected,
    }
    statistics = {}
    values = {}
    for name, rows in arms.items():
        assert all(0 <= score <= 1 for r in rows for score in r.scores)
        statistics[name] = core.decompositions(rows, weights)
        statistics[name]['conditional_aucs'] = [core.local_auc(r) for r in rows]
        statistics[name]['rows'] = rows
        values[name + '.auc'] = statistics[name]['pooled_auc']
        values[name + '.conditional_auc'] = statistics[name]['conditional_aucs'][0]
    auc = {name: statistics[name]['pooled_auc'] for name in arms}
    assert list(auc.values()) == [F(1), F(4, 5), F(16, 25), F(4, 5)]
    assert statistics['original']['conditional_aucs'] == [F(1), F(1)]
    assert statistics['offset']['conditional_aucs'] == [F(1), F(1)]
    assert statistics['projected']['conditional_aucs'] == [F(1, 2), F(1, 2)]
    assert statistics['offset_projected']['conditional_aucs'] == [F(1, 2), F(1, 2)]
    gains = {
        'deviation_original': auc['original'] - auc['projected'],
        'deviation_offset': auc['offset'] - auc['offset_projected'],
        'offset_with_deviation': auc['offset'] - auc['original'],
        'offset_without_deviation': auc['offset_projected'] - auc['projected'],
    }
    gains['interaction'] = gains['deviation_offset'] - gains['deviation_original']
    assert gains['deviation_original'] == F(1, 5)
    assert gains['deviation_offset'] == F(-4, 25)
    assert gains['interaction'] == F(-9, 25)
    values.update({'gain.' + name: value for name, value in gains.items()})
    before, after = statistics['original'], statistics['offset']
    rates = [core.prevalence(r) for r in original]
    assert before['within_covariance'] == after['within_covariance']
    assert after['pooled_covariance'] - before['pooled_covariance'] == covariance(shifts, rates, weights)
    loss_gain_before = statistics['projected']['squared_loss'] - before['squared_loss']
    loss_gain_after = statistics['offset_projected']['squared_loss'] - after['squared_loss']
    assert loss_gain_before == loss_gain_after
    brier_interaction = loss_gain_after - loss_gain_before
    assert brier_interaction == 0

    equal_a = [core.row([1, 1, 0], [1, 0, 0]), core.row([1, 0, 0], [1, 1, 0])]
    equal_b = [core.row([1, 1, 0], [1, 1, 1]), core.row([1, 0, 0], [0, 0, 0])]
    equal_shifts = [F(1) - core.mean(equal_a[0].scores), -core.mean(equal_a[1].scores)]
    assert offset(project(equal_a), equal_shifts) == equal_b
    assert core.pooled_law(equal_a, weights) == core.pooled_law(equal_b, weights)
    assert [core.local_auc(r) for r in equal_a] == [F(3, 4), F(3, 4)]
    assert [core.local_auc(r) for r in equal_b] == [F(1, 2), F(1, 2)]
    witnesses = {}
    for m in [10, 100]:
        a = [
            core.row([1] * (m - 1) + [0], [F(1, 2)] * (m - 1) + [F(0)]),
            core.row([1] + [0] * (m - 1), [F(1)] + [F(1, 2)] * (m - 1)),
        ]
        shifts_m = [F(1) - core.mean(a[0].scores), -core.mean(a[1].scores)]
        b = offset(project(a), shifts_m)
        assert all(s == F(1) for s in b[0].scores)
        assert all(s == F(0) for s in b[1].scores)
        assert core.auc(core.pooled_law(a, weights)) < core.auc(core.pooled_law(b, weights))
        assert all(core.capture(x, k) > core.capture(y, k) for x, y in zip(a, b) for k in range(1, m))
        witnesses[str(m)] = {'A': core.decompositions(a, weights), 'B': core.decompositions(b, weights), 'offsets': shifts_m, 'B_equals_offset_projection_A': True}

    result = {
        'kind': 'exact_operator_constructions', 'all_checks_passed': True,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'helper_sha256': hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        'weights': weights, 'offsets': shifts, 'four_arms': statistics,
        'pooled_auc_gains': gains, 'negative_squared_loss_interaction': brier_interaction,
        'same_pooled_law_operator_pair': {'A': equal_a, 'B': equal_b, 'offsets': equal_shifts},
        'general_reversal_witnesses': witnesses,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'verification.json').write_text(json.dumps(core.serializable(result), indent=2) + '\n')
    with (args.out / 'values.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['quantity', 'exact', 'decimal'])
        writer.writerows((name, str(value), format(float(value), '.12f')) for name, value in values.items())
    lines = [r'\providecommand{\HomerOperatorValue}[1]{\csname ho@#1\endcsname}']
    lines.extend(r'\expandafter\def\csname ho@' + name + r'\endcsname{' + core.tex_fraction(value) + '}' for name, value in values.items())
    (args.out / 'values.tex').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'all_checks_passed': True, 'pooled_aucs': core.serializable(auc), 'gains': core.serializable(gains), 'script_sha256': result['script_sha256']}))


if __name__ == '__main__':
    main()
