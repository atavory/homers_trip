"""Exact finite populations for decision-dependent hallucination evaluation."""

import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path


@dataclass(frozen=True)
class Row:
    labels: tuple
    scores: tuple

    def __post_init__(self):
        assert len(self.labels) == len(self.scores) > 0
        assert set(self.labels) <= {0, 1}


def row(labels, scores, denominator=1):
    return Row(tuple(labels), tuple(F(s, denominator) for s in scores))


def mean(values):
    return sum(values, F(0)) / len(values)


def prevalence(response):
    return mean(response.labels)


def compare(a, b):
    return F(1) if a > b else F(1, 2) if a == b else F(0)


def pooled_law(rows, weights):
    assert len(rows) == len(weights) and sum(weights) == 1
    law = defaultdict(F)
    for response, weight in zip(rows, weights):
        for score, label in zip(response.scores, response.labels):
            law[score, label] += weight / len(response.labels)
    return dict(law)


def auc(law):
    positive = [(s, p) for (s, y), p in law.items() if y == 1]
    negative = [(s, p) for (s, y), p in law.items() if y == 0]
    total_p = sum(p for _, p in positive)
    total_n = sum(p for _, p in negative)
    if not total_p or not total_n:
        return None
    return sum(
        p * n * compare(a, b) for a, p in positive for b, n in negative
    ) / (total_p * total_n)


def local_auc(response):
    return auc(pooled_law([response], [F(1)]))


def capture(response, budget):
    assert 0 <= budget <= len(response.labels)
    groups = defaultdict(list)
    for score, label in zip(response.scores, response.labels):
        groups[score].append(label)
    result = F(0)
    for score in sorted(groups, reverse=True):
        labels = groups[score]
        take = min(budget, len(labels))
        result += F(take, len(labels)) * sum(labels)
        budget -= take
    return result


def enumerated_capture(response, budget):
    if budget == 0:
        return F(0)
    cutoff = sorted(response.scores, reverse=True)[budget - 1]
    above = [i for i, s in enumerate(response.scores) if s > cutoff]
    tied = [i for i, s in enumerate(response.scores) if s == cutoff]
    subsets = list(itertools.combinations(tied, budget - len(above)))
    return mean([
        sum(response.labels[i] for i in above + list(subset))
        for subset in subsets
    ])


def utility(rows, weights, budgets):
    return sum(
        q * capture(r, k) for r, q, k in zip(rows, weights, budgets)
    )


def decompositions(rows, weights):
    law = pooled_law(rows, weights)
    means = [mean(r.scores) for r in rows]
    rates = [prevalence(r) for r in rows]
    score_mean = sum(q * m for q, m in zip(weights, means))
    label_mean = sum(q * p for q, p in zip(weights, rates))
    covariances = [
        mean([s * y for s, y in zip(r.scores, r.labels)]) - m * p
        for r, m, p in zip(rows, means, rates)
    ]
    within = sum(q * c for q, c in zip(weights, covariances))
    between = sum(
        q * (m - score_mean) * (p - label_mean)
        for q, m, p in zip(weights, means, rates)
    )
    pooled_covariance = sum(
        mass * (s - score_mean) * (y - label_mean)
        for (s, y), mass in law.items()
    )
    squared_loss = sum(mass * (s - y) ** 2 for (s, y), mass in law.items())
    mean_loss = sum(q * (m - p) ** 2 for q, m, p in zip(weights, means, rates))
    residual_loss = sum(
        q * mean([((s - m) - (y - p)) ** 2 for s, y in zip(r.scores, r.labels)])
        for r, q, m, p in zip(rows, weights, means, rates)
    )
    assert pooled_covariance == within + between
    assert squared_loss == mean_loss + residual_loss
    matrix_auc = F(0)
    within_weight = F(0)
    for a, qa, pa in zip(rows, weights, rates):
        for b, qb, pb in zip(rows, weights, rates):
            positive = [s for s, y in zip(a.scores, a.labels) if y == 1]
            negative = [s for s, y in zip(b.scores, b.labels) if y == 0]
            if not positive or not negative:
                continue
            coefficient = qa * pa * qb * (1 - pb) / (label_mean * (1 - label_mean))
            pair_auc = mean([compare(s, t) for s in positive for t in negative])
            matrix_auc += coefficient * pair_auc
        within_weight += qa ** 2 * pa * (1 - pa) / (label_mean * (1 - label_mean))
    assert matrix_auc == auc(law)
    return {
        'means': means, 'prevalences': rates, 'conditional_covariances': covariances,
        'pooled_covariance': pooled_covariance, 'within_covariance': within,
        'between_covariance': between, 'squared_loss': squared_loss,
        'response_mean_squared_loss': mean_loss, 'residual_squared_loss': residual_loss,
        'pooled_auc': matrix_auc, 'within_pair_weight': within_weight,
    }


def serializable(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, Row):
        return {'labels': value.labels, 'scores': serializable(value.scores)}
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def tex_fraction(value):
    return str(value.numerator) if value.denominator == 1 else (
        r'\frac{' + str(value.numerator) + '}{' + str(value.denominator) + '}'
    )


def column_diagram(a, b):
    lines = [r'\begin{tikzpicture}[x=1cm,y=1cm,font=\small]']
    for name, rows, shift in [('A', a, 0), ('B', b, 5.0)]:
        lines.append(r'\node at (' + str(shift + 2) + ',1.15) {Predictor $' + name + '$};')
        for i, response in enumerate(rows):
            x = shift + .75 + i * 2.4
            shade = 100 * prevalence(response)
            lines.append(r'\draw[draw=black!' + str(int(shade)) + ',line width=3pt] (' + str(x - .56) + ',-2.5) rectangle (' + str(x + .56) + ',.5);')
            lines.append(r'\node at (' + str(x) + ',.8) {$r=' + str(i + 1) + '$};')
            for t, (label, score) in enumerate(zip(response.labels, response.scores)):
                y = .12 - t * .68
                fill, foreground = ('black', 'white') if label else ('white', 'black')
                options = 'draw=black,fill=' + fill + ',text=' + foreground + ',minimum width=.9cm,minimum height=.5cm'
                if score == max(response.scores):
                    options += ',draw=orange!85!black,line width=1.4pt'
                lines.append(r'\node[' + options + '] at (' + str(x) + ',' + str(y) + ') {$' + str(score * 14) + '$};')
            lines.append(r'\node at (' + str(x) + r',-2.85) {$\pi_r=' + tex_fraction(prevalence(response)) + '$};')
            lines.append(r'\node at (' + str(x) + ',-3.25) {$m_r=' + tex_fraction(mean(response.scores)) + '$};')
    lines.append(r'\end{tikzpicture}')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    fixtures, values = {}, {}
    a = [row([1, 0, 0, 0], [5, 0, 12, 0], 14), row([1, 1, 0, 0], [7, 7, 6, 6], 14)]
    b = [row([1, 0, 0, 0], [5, 6, 6, 0], 14), row([1, 1, 0, 0], [7, 7, 0, 12], 14)]
    weights = [F(1, 2), F(1, 2)]
    da, db = decompositions(a, weights), decompositions(b, weights)
    assert pooled_law(a, weights) == pooled_law(b, weights)
    assert da == db
    ua, ub = utility(a, weights, [1, 1]), utility(b, weights, [1, 1])
    assert (ua, ub) == (F(1, 2), F(0))
    fixtures['joint_nonidentification'] = {'A': a, 'B': b, 'weights': weights, 'shared': da, 'utility_A': ua, 'utility_B': ub}
    values.update({'joint.local_A': ua, 'joint.local_B': ub, 'joint.pooled_auc': da['pooled_auc'], 'joint.mean1': da['means'][0], 'joint.mean2': da['means'][1]})
    for response_count in [2, 3, 8, 32]:
        original = [row([1, 0], [F(2 * r + 1, 2), F(r)], response_count + 1) for r in range(1, response_count + 1)]
        permuted = [row([1, 0], [F(2 * r + 1, 2), F(r + 1 if r < response_count else 1)], response_count + 1) for r in range(1, response_count + 1)]
        q = [F(1, response_count)] * response_count
        assert pooled_law(original, q) == pooled_law(permuted, q)
        v1, v2 = utility(original, q, [1] * response_count), utility(permuted, q, [1] * response_count)
        assert (v1, v2) == (F(1), F(1, response_count))
        fixtures['cyclic_' + str(response_count)] = {'responses': response_count, 'utility_A': v1, 'utility_B': v2, 'minimum_worst_error': (v1 - v2) / 2}
    for m in [3, 10, 100]:
        aa = [row([1] * (m - 1) + [0], [F(1, 2)] * (m - 1) + [0]), row([1] + [0] * (m - 1), [1] + [F(1, 2)] * (m - 1))]
        bb = [row(r.labels, [1 if i == 0 else 0] * m) for i, r in enumerate(aa)]
        assert all(capture(x, k) > capture(y, k) for x, y in zip(aa, bb) for k in range(1, m))
        sx, sy = decompositions(aa, weights), decompositions(bb, weights)
        if m >= 10:
            assert sx['pooled_auc'] < sy['pooled_auc']
            assert sx['squared_loss'] > sy['squared_loss']
        fixtures['local_dominance_' + str(m)] = {
            'A': sx, 'B': sy, 'all_local_budgets_strict': True,
            'pooled_auc_reversal': sx['pooled_auc'] < sy['pooled_auc'],
            'squared_loss_reversal': sx['squared_loss'] > sy['squared_loss'],
        }
    fixed_a = row([1, 1, 0, 0, 0], [3, 2, 4, 1, 0], 4)
    fixed_b = row([1, 1, 0, 0, 0], [4, 0, 3, 2, 1], 4)
    assert local_auc(fixed_a) > local_auc(fixed_b)
    assert capture(fixed_a, 1) < capture(fixed_b, 1)
    fixtures['conditional_auc_fixed_budget'] = {'A': fixed_a, 'B': fixed_b, 'auc_A': local_auc(fixed_a), 'auc_B': local_auc(fixed_b), 'capture_A': capture(fixed_a, 1), 'capture_B': capture(fixed_b, 1)}
    values.update({'budget.auc_A': local_auc(fixed_a), 'budget.auc_B': local_auc(fixed_b), 'budget.capture_A': capture(fixed_a, 1), 'budget.capture_B': capture(fixed_b, 1)})
    budget_cases = 0
    for n in range(2, 7):
        for labels in itertools.product([0, 1], repeat=n):
            p = sum(labels)
            if p in [0, n]:
                continue
            for scores in [list(range(n)), [i % 2 for i in range(n)], [0] * n]:
                response = row(labels, scores, n)
                curve = [capture(response, k) for k in range(1, n + 1)]
                assert all(v == enumerated_capture(response, k) for k, v in enumerate(curve, start=1))
                assert sum(curve) == F(p * (p + 1), 2) + p * (n - p) * local_auc(response)
                budget_cases += 1
    fixtures['budget_identity'] = {'exact_cases': budget_cases, 'lengths': [2, 3, 4, 5, 6], 'distinct_and_tied_scores': True, 'independent_subset_enumeration': True}
    for q in [[F(1, 2), F(1, 2)], [F(1, 3), F(2, 3)], [F(4, 7), F(3, 7)]]:
        samples = [row([1, 0, 1, 0], [1, 3, 4, 2], 5), row([0, 1, 0], [4, 2, 3], 5)]
        fixtures['weighted_identities_' + str(q[0])] = decompositions(samples, q)
    simpson = [row([1] * 9 + [0], [F(1, 3)] * 9 + [0]), row([1] + [0] * 9, [1] + [F(2, 3)] * 9)]
    ds = decompositions(simpson, weights)
    assert all(c > 0 for c in ds['conditional_covariances']) and ds['pooled_covariance'] < 0
    fixtures['simpson_association'] = ds
    calibration = [row([1, 0, 1, 1], [1, 1, 3, 3], 4), row([0, 0, 1, 0], [1, 1, 3, 3], 4)]
    law = pooled_law(calibration, weights)
    for score in {s for s, _ in law}:
        mass = sum(p for (s, _), p in law.items() if s == score)
        positives = sum(p for (s, y), p in law.items() if s == score and y)
        assert positives / mass == score
    assert [mean(r.scores) for r in calibration] != [prevalence(r) for r in calibration]
    fixtures['global_calibration'] = {'responses': calibration, 'global_calibrated': True, 'response_means': [mean(r.scores) for r in calibration], 'response_prevalences': [prevalence(r) for r in calibration]}
    dependency = {
        'perfect_dependence': {(0, 0): F(1, 2), (1, 1): F(1, 2)},
        'independence': {v: F(1, 4) for v in itertools.product([0, 1], repeat=2)},
        'exactly_one': {(0, 1): F(1, 2), (1, 0): F(1, 2)},
    }
    any_values = {}
    for name, joint in dependency.items():
        rate = sum(p * F(sum(y), 2) for y, p in joint.items())
        any_rate = sum(p for y, p in joint.items() if any(y))
        assert rate == F(1, 2)
        any_values[name] = {'expected_prevalence': rate, 'any_error_probability': any_rate}
    fixtures['response_any_error'] = any_values
    structured_a = row([1, 1, 1, 1, 0], [5, 4, 2, 1, 3], 5)
    structured_b = row([1, 1, 1, 1, 0], [5, 2, 4, 1, 3], 5)
    spans = [{0, 1}, {2, 3}]
    def complete(response):
        selected = set(sorted(range(5), key=lambda i: response.scores[i], reverse=True)[:2])
        return sum(span <= selected for span in spans)
    assert pooled_law([structured_a], [F(1)]) == pooled_law([structured_b], [F(1)])
    assert capture(structured_a, 2) == capture(structured_b, 2)
    assert (complete(structured_a), complete(structured_b)) == (1, 0)
    fixtures['structured_span'] = {'A': structured_a, 'B': structured_b, 'word_capture': capture(structured_a, 2), 'complete_spans_A': complete(structured_a), 'complete_spans_B': complete(structured_b)}
    shifted = [row(r.labels, [s + offset for s in r.scores]) for r, offset in zip(a, [F(1), F(-1)])]
    assert all(capture(x, k) == capture(y, k) for x, y in zip(a, shifted) for k in range(len(x.labels) + 1))
    selected_original = max(range(2), key=lambda i: mean(a[i].scores))
    selected_shifted = max(range(2), key=lambda i: mean(shifted[i].scores))
    assert selected_original != selected_shifted
    fixtures['two_stage_gate'] = {'selected_response_A': selected_original + 1, 'selected_response_shifted': selected_shifted + 1, 'utility_A': capture(a[selected_original], 1), 'utility_shifted': capture(shifted[selected_shifted], 1), 'all_fixed_conditional_actions_identical': True}
    tie_a = row([1, 0, 1], [1, 1, 0])
    tie_b = row([1, 0, 1], [0, 1, 1])
    assert pooled_law([tie_a], [F(1)]) == pooled_law([tie_b], [F(1)])
    index_a = tie_a.scores.index(max(tie_a.scores))
    index_b = tie_b.scores.index(max(tie_b.scores))
    assert (tie_a.labels[index_a], tie_b.labels[index_b]) == (1, 0)
    assert capture(tie_a, 1) == capture(tie_b, 1) == F(1, 2)
    fixtures['anonymous_tie_requirement'] = {
        'A': tie_a, 'B': tie_b,
        'lowest_index_capture_A': tie_a.labels[index_a],
        'lowest_index_capture_B': tie_b.labels[index_b],
        'uniform_capture_A_and_B': capture(tie_a, 1),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    result = {
        'kind': 'exact_analytic_fixtures', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs': 'finite populations embedded in this script', 'fixture_count': len(fixtures),
        'budget_identity_cases': budget_cases, 'all_checks_passed': True, 'fixtures': serializable(fixtures),
    }
    (args.out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.out / 'values.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['quantity', 'exact', 'decimal'])
        writer.writerows((key, str(value), format(float(value), '.12f')) for key, value in values.items())
    tex = [r'\providecommand{\HomerInterventionValue}[1]{\csname hi@#1\endcsname}']
    tex.extend(r'\expandafter\def\csname hi@' + key + r'\endcsname{' + tex_fraction(value) + '}' for key, value in values.items())
    (args.out / 'values.tex').write_text('\n'.join(tex) + '\n')
    diagram = column_diagram(a, b)
    (args.out / 'columns.tex').write_text(diagram)
    (args.out / 'columns_preview.tex').write_text(r'\documentclass[tikz,border=8pt]{standalone}' + '\n' + r'\begin{document}' + '\n' + diagram + r'\end{document}' + '\n')
    print(json.dumps({key: result[key] for key in ['kind', 'fixture_count', 'budget_identity_cases', 'all_checks_passed', 'script_sha256']}))


if __name__ == '__main__':
    main()
