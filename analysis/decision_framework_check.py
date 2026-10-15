"""Generate exact analytic fixtures for grouped score-to-action evaluation."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from fractions import Fraction as F
from pathlib import Path


def average(values, weights):
    return sum(v * w for v, w in zip(values, weights))


def moments(groups, response_weights=None):
    if response_weights is None:
        total = sum(len(group) for group in groups)
        response_weights = [F(len(group), total) for group in groups]
    assert sum(response_weights) == 1
    group_means = []
    group_prevalences = []
    conditional_covariances = []
    conditional_variances = []
    pooled_scores, pooled_labels, token_weights = [], [], []
    for group, q in zip(groups, response_weights):
        scores, labels = zip(*group)
        weights = [F(1, len(group))] * len(group)
        mean = average(scores, weights)
        prevalence = average(labels, weights)
        covariance = average([s * z for s, z in group], weights) - mean * prevalence
        variance = average([s * s for s in scores], weights) - mean * mean
        group_means.append(mean)
        group_prevalences.append(prevalence)
        conditional_covariances.append(covariance)
        conditional_variances.append(variance)
        pooled_scores.extend(scores)
        pooled_labels.extend(labels)
        token_weights.extend([q / len(group)] * len(group))
    mean = average(pooled_scores, token_weights)
    prevalence = average(pooled_labels, token_weights)
    covariance = average([s * z for s, z in zip(pooled_scores, pooled_labels)], token_weights) - mean * prevalence
    within = average(conditional_covariances, response_weights)
    between = average([m * p for m, p in zip(group_means, group_prevalences)], response_weights) - mean * prevalence
    assert covariance == within + between
    squared_error = average([(s - z) ** 2 for s, z in zip(pooled_scores, pooled_labels)], token_weights)
    mean_error = average([(m - p) ** 2 for m, p in zip(group_means, group_prevalences)], response_weights)
    centered_error = average([v + p * (1 - p) - 2 * c for v, p, c in zip(conditional_variances, group_prevalences, conditional_covariances)], response_weights)
    assert squared_error == mean_error + centered_error
    return {
        "weights": response_weights,
        "means": group_means,
        "prevalences": group_prevalences,
        "conditional_covariances": conditional_covariances,
        "pooled_covariance": covariance,
        "within_covariance": within,
        "between_covariance": between,
        "squared_error": squared_error,
        "response_mean_error": mean_error,
        "centered_error": centered_error,
    }


def pooled_law(groups):
    return Counter(row for group in groups for row in group)


def recovery(group, budget):
    ordered = sorted(group, reverse=True)
    if budget == 0:
        return F(0)
    cutoff = ordered[budget - 1][0]
    above = [z for s, z in group if s > cutoff]
    tied = [z for s, z in group if s == cutoff]
    return F(sum(above)) + F(budget - len(above), len(tied)) * sum(tied)


def conditional_recovery(groups, budget=1):
    return sum(recovery(group, budget) for group in groups) / len(groups)


def span_completion(group, spans, budget):
    scores = [s for s, _ in group]
    cutoff = sorted(scores, reverse=True)[budget - 1]
    assert sum(s == cutoff for s in scores) == 1
    selected = {i for i, s in enumerate(scores) if s >= cutoff}
    return sum(set(span) <= selected for span in spans)


def posterior_fixture():
    response_weights = [F(1, 3), F(2, 3)]
    posterior_groups = [[F(1, 4), F(3, 4)], [F(0), F(1, 2)]]
    prevalences = [sum(values) / len(values) for values in posterior_groups]
    mean = average(prevalences, response_weights)
    posterior_error = sum(q * sum(p * (1 - p) for p in values) / len(values) for q, values in zip(response_weights, posterior_groups))
    global_prior_error = mean * (1 - mean)
    between_gain = average([(p - mean) ** 2 for p in prevalences], response_weights)
    within_gain = sum(q * sum((p - pi) ** 2 for p in values) / len(values) for q, pi, values in zip(response_weights, prevalences, posterior_groups))
    assert global_prior_error - posterior_error == between_gain + within_gain
    return {
        "global_prior_error": global_prior_error,
        "posterior_error": posterior_error,
        "between_gain": between_gain,
        "within_gain": within_gain,
    }


def fixtures():
    results = {}
    classic = [[(F(1), 1)] * 9 + [(F(0), 0)], [(F(3), 1)] + [(F(2), 0)] * 9]
    results["simpson_association"] = moments(classic)
    assert results["simpson_association"]["within_covariance"] > 0
    assert results["simpson_association"]["pooled_covariance"] < 0

    a = [[(F(1, 2), 1)] * 9 + [(F(0), 0)], [(F(1), 1)] + [(F(1, 2), 0)] * 9]
    b = [[(F(1), 1)] * 9 + [(F(1), 0)], [(F(0), 1)] + [(F(0), 0)] * 9]
    results["strict_reversal_A"] = moments(a)
    results["strict_reversal_B"] = moments(b)
    assert results["strict_reversal_A"]["squared_error"] > results["strict_reversal_B"]["squared_error"]
    assert all(recovery(x, k) > recovery(y, k) for x, y in zip(a, b) for k in range(1, 10))
    results["strict_reversal_A"]["one_token_recovery"] = conditional_recovery(a)
    results["strict_reversal_B"]["one_token_recovery"] = conditional_recovery(b)

    for count in range(2, 10):
        original = [[(F(2 * r + 1, 2 * (count + 1)), 1), (F(r, count + 1), 0)] for r in range(1, count + 1)]
        moved = [[(F(2 * r + 1, 2 * (count + 1)), 1), (F(r + 1 if r < count else 1, count + 1), 0)] for r in range(1, count + 1)]
        assert pooled_law(original) == pooled_law(moved)
        assert conditional_recovery(original) == 1
        assert conditional_recovery(moved) == F(1, count)
        for budget in range(1, 2 * count):
            assert recovery(sum(original, []), budget) == recovery(sum(moved, []), budget)
        results[f"cyclic_{count}"] = {"original_recovery": F(1), "moved_recovery": F(1, count)}

    joint_a = [[(F(5, 14), 1), (F(0), 0), (F(6, 7), 0)], [(F(1, 2), 1), (F(1, 2), 1), (F(3, 7), 0), (F(3, 7), 0)]]
    joint_b = [[(F(5, 14), 1), (F(3, 7), 0), (F(3, 7), 0)], [(F(1, 2), 1), (F(1, 2), 1), (F(0), 0), (F(6, 7), 0)]]
    assert pooled_law(joint_a) == pooled_law(joint_b)
    ma, mb = moments(joint_a), moments(joint_b)
    assert ma == mb
    assert ma["prevalences"] == [F(1, 3), F(1, 2)]
    assert ma["means"] == [F(17, 42), F(13, 28)]
    assert conditional_recovery(joint_a) == F(1, 2)
    assert conditional_recovery(joint_b) == 0
    results["jointly_fixed_means_and_pool"] = {
        **ma,
        "A_one_token_recovery": conditional_recovery(joint_a),
        "B_one_token_recovery": conditional_recovery(joint_b),
        "A_token_weighted_response_recovery": F(4, 7),
        "B_token_weighted_response_recovery": F(0),
        "pooled_law_equal": True,
    }

    uneven = [[(F(1, 4), 0), (F(3, 4), 1)], [(F(1, 5), 1), (F(2, 5), 1), (F(4, 5), 0)], [(F(1, 6), 0)] * 4]
    results["unequal_lengths_token_weights"] = moments(uneven)
    results["unequal_lengths_response_weights"] = moments(uneven, [F(1, 3)] * 3)
    results["unequal_lengths_explicit_weights"] = moments(uneven, [F(1, 5), F(3, 10), F(1, 2)])

    offsets = [F(2), F(-2)]
    shifted = [[(s + offset, z) for s, z in group] for group, offset in zip(a, offsets)]
    for original, moved in zip(a, shifted):
        assert all(recovery(original, k) == recovery(moved, k) for k in range(1, len(original)))
    assert moments(a)["within_covariance"] == moments(shifted)["within_covariance"]
    results["response_offsets"] = {"original_covariance": moments(a)["pooled_covariance"], "shifted_covariance": moments(shifted)["pooled_covariance"], "within_covariance": moments(a)["within_covariance"]}
    results["conditional_posterior"] = posterior_fixture()

    joint_error_laws = {
        "perfect_dependence": [(F(1, 2), (0, 0)), (F(1, 2), (1, 1))],
        "independence": [(F(1, 4), labels) for labels in [(0, 0), (0, 1), (1, 0), (1, 1)]],
        "exactly_one_error": [(F(1, 2), (0, 1)), (F(1, 2), (1, 0))],
    }
    for name, law in joint_error_laws.items():
        prevalence = sum(weight * F(sum(labels), 2) for weight, labels in law)
        assert prevalence == F(1, 2)
        results[f"response_outcome_{name}"] = {"prevalence": prevalence, "any_error_probability": sum(weight for weight, labels in law if any(labels))}

    spans = [(0, 1), (2, 3)]
    span_a = [(F(4, 5), 1), (F(9, 10), 1), (F(1, 5), 1), (F(1, 10), 1), (F(0), 0)]
    span_b = [(F(4, 5), 1), (F(1, 5), 1), (F(9, 10), 1), (F(1, 10), 1), (F(0), 0)]
    assert pooled_law([span_a]) == pooled_law([span_b])
    assert recovery(span_a, 2) == recovery(span_b, 2) == 2
    assert span_completion(span_a, spans, 2) == 1
    assert span_completion(span_b, spans, 2) == 0
    results["structured_span_action"] = {"selected_positive_tokens": 2, "A_complete_spans": 1, "B_complete_spans": 0}
    return results


def serialize(value):
    if isinstance(value, F):
        return {"exact": str(value), "decimal": float(value)}
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize(item) for item in value]
    return value


def rows(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from rows(item, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from rows(item, f"{prefix}.{index}")
    else:
        yield [prefix, str(value), float(value) if isinstance(value, (F, int)) else ""]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    values = fixtures()
    record = {"kind": "analytic_fixtures", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "fixture_count": len(values), "all_checks_passed": True, "results": serialize(values)}
    (args.output / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    with (args.output / "values.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "exact", "decimal"])
        writer.writerows(rows(values))
    print(json.dumps({key: record[key] for key in ["kind", "script_sha256", "fixture_count", "all_checks_passed"]}))


if __name__ == "__main__":
    main()
