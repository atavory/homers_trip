"""Check graded response loss, shared cost, and metric-ordering examples."""

import argparse
from collections import defaultdict
import csv
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path


def add_offsets(scores, offsets):
    return tuple(tuple(score + shift for score in row) for row, shift in zip(scores, offsets))


def project(scores):
    return tuple((sum(row) / len(row),) * len(row) for row in scores)


def readout(labels, scores, threshold):
    flags = tuple(tuple(score >= threshold for score in row) for row in scores)
    return from_flags(labels, flags)


def from_flags(labels, flags):
    response = []
    for truth, selected in zip(labels, flags):
        tp = sum(label and flag for label, flag in zip(truth, selected))
        fp = sum(not label and flag for label, flag in zip(truth, selected))
        response.append({"tp": tp, "fp": fp, "residual": sum(truth) - tp, "checks": sum(selected)})
    return {"flags": flags, "responses": response,
            "tp": sum(row["tp"] for row in response), "fp": sum(row["fp"] for row in response),
            "fn": sum(row["residual"] for row in response),
            "tn": sum(sum(not label for label in truth) for truth in labels) - sum(row["fp"] for row in response),
            "checks": sum(row["checks"] for row in response)}


def comparison(first, second):
    return F(1) if first > second else F(1, 2) if first == second else F(0)


def rank_metrics(labels, scores):
    pairs = [(label, score) for truth, row in zip(labels, scores) for label, score in zip(truth, row)]
    positive = [score for label, score in pairs if label]
    negative = [score for label, score in pairs if not label]
    auc = sum(comparison(a, b) for a in positive for b in negative) / (len(positive) * len(negative))
    groups = defaultdict(list)
    for label, score in pairs:
        groups[score].append(label)
    n_seen, p_seen, ap = 0, 0, F(0)
    for score in sorted(groups, reverse=True):
        at_score = groups[score]
        n_seen += len(at_score)
        p_seen += sum(at_score)
        ap += F(sum(at_score), len(positive)) * F(p_seen, n_seen)
    local_auc = []
    for truth, row in zip(labels, scores):
        p = [score for label, score in zip(truth, row) if label]
        n = [score for label, score in zip(truth, row) if not label]
        local_auc.append(sum(comparison(a, b) for a in p for b in n) / (len(p) * len(n)))
    return {"auc": auc, "ap": ap,
            "brier": sum((score - label) ** 2 for label, score in pairs) / len(pairs),
            "conditional_auc": tuple(local_auc)}


def f1(result):
    return F(2 * result["tp"], 2 * result["tp"] + result["fp"] + result["fn"])


def risk(result, loss, cost):
    return sum(loss(row["residual"]) + cost(row["checks"]) for row in result["responses"]) / len(result["responses"])


def plateaus(labels, scores, loss, cost):
    cutoffs = sorted({F(1)} | {score for row in scores for score in row}, reverse=True)
    rows = []
    for cutoff in cutoffs:
        result = readout(labels, scores, cutoff)
        rows.append({"threshold": cutoff, "readout": result, "f1": f1(result), "risk": risk(result, loss, cost)})
    minimum, maximum_f1 = min(row["risk"] for row in rows), max(row["f1"] for row in rows)
    for first, second in itertools.combinations_with_replacement(rows, 2):
        for numerator in range(5):
            probability = F(numerator, 4)
            assert probability * first["risk"] + (1 - probability) * second["risk"] >= minimum
    return {"plateaus": rows, "minimum_risk": minimum, "maximum_f1": maximum_f1,
            "f1_optimal_risks": tuple(sorted({row["risk"] for row in rows if row["f1"] == maximum_f1})),
            "f1_optimal_residuals": tuple(tuple(v["residual"] for v in row["readout"]["responses"])
                                          for row in rows if row["f1"] == maximum_f1)}


def prefix_ties(labels, scores, loss, cost):
    units = [(r, t) for r, row in enumerate(scores) for t in range(len(row))]
    totals = []
    for count in range(len(units) + 1):
        if count == 0:
            candidates = [set()]
        else:
            boundary = sorted((scores[r][t] for r, t in units), reverse=True)[count - 1]
            above = {unit for unit in units if scores[unit[0]][unit[1]] > boundary}
            tied = [unit for unit in units if scores[unit[0]][unit[1]] == boundary]
            candidates = [above | set(selected) for selected in itertools.combinations(tied, count - len(above))]
        values = []
        for candidate in candidates:
            flags = tuple(tuple((r, t) in candidate for t in range(len(truth))) for r, truth in enumerate(labels))
            values.append(risk(from_flags(labels, flags), loss, cost))
        totals.append({"prefix_size": count, "tie_configurations": len(values),
                       "uniform_tie_risk": sum(values) / len(values), "best_allowed_tie_risk": min(values)})
    return {"prefixes": totals, "minimum_uniform_tie_risk": min(row["uniform_tie_risk"] for row in totals),
            "minimum_arbitrary_tie_risk": min(row["best_allowed_tie_risk"] for row in totals)}


def optimized_case():
    labels = ((1, 0, 0), (1, 1, 1, 1, 0))
    original = ((F(9, 10), F(4, 5), F(4, 5)), (F(2, 5), F(2, 5), F(2, 5), F(2, 5), F(0)))
    shifts = (-F(3, 10), F(11, 20))
    offset = add_offsets(original, shifts)
    loss = lambda e: 1 - F(1, 2) ** e
    cost = lambda k: F(1, 2) * (1 - F(1, 2) ** k) + F(k, 8)
    conditions = {"original": original, "projection": project(original),
                  "offset": offset, "offset_projection": add_offsets(project(original), shifts)}
    assert conditions["offset_projection"] == project(offset)
    for before, after in zip(original, offset):
        assert tuple(value - sum(before) / len(before) for value in before) == tuple(value - sum(after) / len(after) for value in after)
    output = {"labels": labels, "offsets": shifts, "common_threshold": F(23, 40), "conditions": {}}
    for name, scores in conditions.items():
        assert all(0 <= value <= 1 for row in scores for value in row)
        common = readout(labels, scores, output["common_threshold"])
        output["conditions"][name] = {"scores": scores, "means": tuple(sum(row) / len(row) for row in scores),
                                      **rank_metrics(labels, scores), "common_cutoff": common,
                                      "common_cutoff_f1": f1(common), "common_cutoff_risk": risk(common, loss, cost),
                                      **plateaus(labels, scores, loss, cost), "prefix_ties": prefix_ties(labels, scores, loss, cost)}
    o, shifted = (output["conditions"][name] for name in ("original", "offset"))
    assert o["auc"] == F(7, 15) and shifted["auc"] == 1
    assert o["conditional_auc"] == shifted["conditional_auc"] == (F(1), F(1))
    assert o["common_cutoff"]["tp"] == 1 and o["common_cutoff"]["fp"] == 2
    assert shifted["common_cutoff"]["tp"] == 5 and shifted["common_cutoff"]["fp"] == 0
    assert o["common_cutoff_risk"] == F(7, 8) > shifted["common_cutoff_risk"] == F(43, 64)
    assert o["minimum_risk"] == F(21, 32)
    assert shifted["minimum_risk"] == F(43, 64)
    assert shifted["minimum_risk"] - o["minimum_risk"] == F(1, 64)
    assert o["maximum_f1"] == F(5, 6) and shifted["maximum_f1"] == 1
    assert o["f1_optimal_risks"] == (F(57, 64),)
    assert shifted["f1_optimal_risks"] == (F(43, 64),)
    for name in ("original", "offset"):
        assert output["conditions"][name]["prefix_ties"]["minimum_uniform_tie_risk"] == output["conditions"][name]["minimum_risk"]
        assert output["conditions"][name]["prefix_ties"]["minimum_arbitrary_tie_risk"] == output["conditions"][name]["minimum_risk"]
    for name, auc_target in (("projection", F(4, 15)), ("offset_projection", F(11, 15))):
        condition = output["conditions"][name]
        assert condition["auc"] == auc_target and condition["minimum_risk"] == F(23, 32)
        assert condition["prefix_ties"]["minimum_uniform_tie_risk"] == F(23, 32)
    assert output["conditions"]["projection"]["prefix_ties"]["minimum_arbitrary_tie_risk"] == F(21, 32)
    assert output["conditions"]["offset_projection"]["prefix_ties"]["minimum_arbitrary_tie_risk"] == F(23, 32)
    output["loss_values"] = tuple(loss(e) for e in range(5))
    output["cost_values"] = tuple(cost(k) for k in range(6))
    output["optimized_risk_gap"] = F(1, 64)
    output["scope"] = "Global threshold or global rank-prefix controllers; richer response-specific routing can recover the cheap response."
    return output


def shape(values):
    increments = tuple(b - a for a, b in zip(values, values[1:]))
    return {"increments": increments, "nondecreasing": all(value >= 0 for value in increments),
            "discrete_concave": all(a >= b for a, b in zip(increments, increments[1:])),
            "discrete_convex": all(a <= b for a, b in zip(increments, increments[1:]))}


def fixed_case_checks():
    labels = ((1, 1, 1, 0), (1, 1, 1, 0))
    original = (tuple(F(value, 10) for value in (9, 8, 7, 6)), tuple(F(value, 10) for value in (4, 3, 2, 1)))
    offset = add_offsets(original, (-F(1, 5), F(3, 10)))
    first, second = readout(labels, original, F(11, 20)), readout(labels, offset, F(11, 20))
    assert first["tp"] == 3 and first["fp"] == 1
    assert second["tp"] == 4 and second["fp"] == 0
    assert tuple(row["residual"] for row in first["responses"]) == (0, 3)
    assert tuple(row["residual"] for row in second["responses"]) == (1, 1)
    assert first["checks"] == second["checks"] == 4
    cases = [
        ("graded_no_setup", lambda e: 1 - F(1, 2) ** e, lambda k: F(0), F(1, 16)),
        ("graded_setup", lambda e: 1 - F(1, 2) ** e, lambda k: F(bool(k), 5) + F(k, 10), F(13, 80)),
        ("graded_saturating_cost", lambda e: 1 - F(1, 2) ** e, lambda k: F(2 * k, k + 1), F(143, 240)),
        ("both_strictly_concave_no_reversal", lambda e: 1 - F(9, 10) ** e, lambda k: F(1, 10) * (1 - F(1, 2) ** k), -F(59, 8000)),
        ("linear_loss_linear_cost", lambda e: F(e), lambda k: F(k, 10), -F(1, 2)),
        ("linear_loss_large_setup", lambda e: F(e), lambda k: F(2 * bool(k)) + F(k, 10), F(1, 2)),
        ("diminishing_repair_benefit", lambda e: F(1, 2) ** (3 - e) - F(1, 8), lambda k: F(0), -F(5, 16)),
        ("diminishing_repair_benefit_shared_cost", lambda e: F(1, 2) ** (3 - e) - F(1, 8), lambda k: F(bool(k)), F(3, 16)),
    ]
    outputs = []
    for name, loss, cost, expected in cases:
        assert loss(0) == cost(0) == 0
        original_risk, offset_risk = risk(first, loss, cost), risk(second, loss, cost)
        loss_term, cost_term = 2 * loss(1) - loss(3), 2 * cost(2) - cost(4)
        assert offset_risk - original_risk == (loss_term + cost_term) / 2 == expected
        d_loss = tuple(loss(e) - loss(e - 1) for e in range(1, 4))
        d_cost = tuple(cost(k) - cost(k - 1) for k in range(1, 5))
        assert loss_term == d_loss[0] - d_loss[1] - d_loss[2]
        assert cost_term == d_cost[0] + d_cost[1] - d_cost[2] - d_cost[3]
        cost_shape = shape(tuple(cost(k) for k in range(5)))
        assert cost_shape["nondecreasing"] and cost_shape["discrete_concave"]
        assert cost_term >= 0
        old_sweep, new_sweep = plateaus(labels, original, loss, cost), plateaus(labels, offset, loss, cost)
        pointwise_optimum = min(loss(3 - corrected) + cost(corrected) for corrected in range(4))
        assert new_sweep["minimum_risk"] == pointwise_optimum <= old_sweep["minimum_risk"]
        assert old_sweep["maximum_f1"] == F(12, 13) and new_sweep["maximum_f1"] == 1
        assert all(residual == (0, 0) for residual in old_sweep["f1_optimal_residuals"] + new_sweep["f1_optimal_residuals"])
        outputs.append({"name": name, "loss_values": tuple(loss(e) for e in range(4)),
                        "cost_values": tuple(cost(k) for k in range(5)), "loss_shape": shape(tuple(loss(e) for e in range(4))),
                        "cost_shape": cost_shape, "repair_gains_in_order": tuple(loss(e) - loss(e - 1) for e in (3, 2, 1)),
                        "original_risk": original_risk, "offset_risk": offset_risk, "risk_gap": expected,
                        "loss_term": loss_term, "cost_term": cost_term, "strict_reversal": expected > 0,
                        "original_threshold_sweep": old_sweep, "offset_threshold_sweep": new_sweep})
    return {"labels": labels, "original_scores": original, "offset_scores": offset, "threshold": F(11, 20),
            "original_readout": first, "offset_readout": second, "cases": outputs}


def bundle_gain(labels, selected, loss, cost):
    errors = sum(labels)
    repaired = sum(labels[t] for t in selected)
    return loss(errors) - loss(errors - repaired) - cost(len(selected))


def marginal_checks():
    loss = lambda e: 1 - F(1, 2) ** e
    cost = lambda k: F(bool(k), 5) + F(k, 20)
    checks = 0
    for labels in itertools.product((0, 1), repeat=4):
        for t in range(4):
            other = [index for index in range(4) if index != t]
            for membership in itertools.product((0, 1, 2), repeat=3):
                first = {index for index, group in zip(other, membership) if group == 2}
                second = {index for index, group in zip(other, membership) if group >= 1}
                marginals = []
                for selected in (first, second):
                    residual = sum(labels) - sum(labels[index] for index in selected)
                    benefit = loss(residual) - loss(residual - 1) if labels[t] else F(0)
                    formula = benefit - (cost(len(selected) + 1) - cost(len(selected)))
                    actual = bundle_gain(labels, selected | {t}, loss, cost) - bundle_gain(labels, selected, loss, cost)
                    assert actual == formula
                    marginals.append(actual)
                assert marginals[0] <= marginals[1]
                checks += 1
    known = tuple(bundle_gain((1, 1, 1), set(range(k)), loss, cost) for k in range(4))
    correlated = tuple((bundle_gain((1, 1, 1), set(range(k)), loss, lambda j: F(0)) +
                        bundle_gain((0, 0, 0), set(range(k)), loss, lambda j: F(0))) / 2 - cost(k)
                       for k in range(4))
    assert known == (F(0), -F(1, 8), F(3, 40), F(21, 40))
    assert correlated == (F(0), -F(3, 16), -F(9, 80), F(7, 80))
    return {"pointwise_nested_set_checks": checks,
            "known_errors_bundle_gains_by_count": known,
            "correlated_all_errors_or_no_errors_half_each_bundle_gains_by_count": correlated,
            "loss_values": tuple(loss(e) for e in range(5)), "cost_values": tuple(cost(k) for k in range(5)),
            "scope": "Fixed ideal-review labels; zero benefit product for a correct added unit; no independence assumption"}


def encode(value):
    if isinstance(value, F):
        return {"exact": str(value), "decimal": float(value)}
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def tex_number(value):
    value = F(value)
    return str(value.numerator) if value.denominator == 1 else r"\frac{" + str(value.numerator) + "}{" + str(value.denominator) + "}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("analysis/bandha_marginal_response_results"))
    args = parser.parse_args()
    optimized, fixed, marginals = optimized_case(), fixed_case_checks(), marginal_checks()
    result = {"all_checks_passed": True, "optimized_threshold": optimized, "fixed_threshold": fixed,
              "marginal_value": marginals, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Exact finite ideal-review examples, with graded loss and endogenous threshold sets; no empirical degradation claim"}
    values = {"optimized.risk_gap": optimized["optimized_risk_gap"], "optimized.gap": optimized["optimized_risk_gap"],
              "optimized.m": F(1), "optimized.h": F(4)}
    for name, condition in optimized["conditions"].items():
        prefix = "optimized." + name + "."
        for key in ("auc", "ap", "brier", "common_cutoff_f1", "common_cutoff_risk", "minimum_risk", "maximum_f1"):
            values[prefix + key] = condition[key]
        values[prefix + "risk_star"] = condition["minimum_risk"]
        values[prefix + "conditional_auc"] = sum(condition["conditional_auc"]) / len(condition["conditional_auc"])
        for r, row in enumerate(condition["scores"], start=1):
            values[prefix + "mean.r" + str(r)] = condition["means"][r - 1]
            values.update({prefix + "score.r" + str(r) + ".t" + str(t): score for t, score in enumerate(row, start=1)})
        for key in ("tp", "fp", "fn", "tn", "checks"):
            values[prefix + "common." + key] = F(condition["common_cutoff"][key])
        values[prefix + "f1_optimal_risk"] = condition["f1_optimal_risks"][0]
        values[prefix + "uniform_prefix_minimum"] = condition["prefix_ties"]["minimum_uniform_tie_risk"]
        values[prefix + "arbitrary_prefix_minimum"] = condition["prefix_ties"]["minimum_arbitrary_tie_risk"]
    for name, sequence in (("loss", optimized["loss_values"]), ("cost", optimized["cost_values"])):
        values.update({"optimized." + name + "." + str(k): value for k, value in enumerate(sequence)})
    values.update({"optimized.cost_increment." + str(k): optimized["cost_values"][k] - optimized["cost_values"][k - 1]
                   for k in range(1, len(optimized["cost_values"]))})
    for case in fixed["cases"]:
        for key in ("original_risk", "offset_risk", "risk_gap", "loss_term", "cost_term"):
            values["fixed." + case["name"] + "." + key] = case[key]
    values.update({"bundle.known." + str(k): value for k, value in enumerate(marginals["known_errors_bundle_gains_by_count"])})
    values.update({"bundle.correlated." + str(k): value for k, value in enumerate(marginals["correlated_all_errors_or_no_errors_half_each_bundle_gains_by_count"])})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "verification.json").write_text(json.dumps(encode(result), indent=2) + "\n")
    with (args.out / "values.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["quantity", "exact", "decimal"])
        writer.writerows((key, str(value), format(float(value), ".12f")) for key, value in values.items())
    lines = [r"\providecommand{\HomerMarginalResponseValue}[1]{\csname hmr@#1\endcsname}"]
    lines.extend(r"\expandafter\def\csname hmr@" + key + r"\endcsname{" + tex_number(value) + "}" for key, value in values.items())
    (args.out / "values.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps({"all_checks_passed": True, "source_sha256": result["source_sha256"],
                      "optimized_risk_original": str(optimized["conditions"]["original"]["minimum_risk"]),
                      "optimized_risk_offset": str(optimized["conditions"]["offset"]["minimum_risk"]),
                      "marginal_identity_checks": marginals["pointwise_nested_set_checks"]}))


if __name__ == "__main__":
    main()
