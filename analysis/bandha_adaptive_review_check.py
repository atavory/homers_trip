"""Exact adaptive query-and-repair comparison after score mean projection."""

import argparse
from collections import Counter
import csv
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parents[1]
LABELS = ((1, 0, 0), (1, 1, 1, 1, 0))
SCORES = ((F(9, 10), F(4, 5), F(4, 5)), (F(2, 5), F(2, 5), F(2, 5), F(2, 5), F(0)))
OFFSETS = (-F(3, 10), F(11, 20))


def loss(errors):
    return 1 - F(1, 2) ** errors


def cost(queries):
    return F(1, 2) * (1 - F(1, 2) ** queries) + F(queries, 8)


def increment(next_query):
    return cost(next_query) - cost(next_query - 1)


@lru_cache(None)
def query_return(k, e, n):
    assert 0 < e <= n and k >= 0
    probability = F(e, n)
    result = increment(k + 1) + probability * coarse_value(k + 1, e - 1, n - 1)
    if e < n:
        result += (1 - probability) * coarse_value(k + 1, e, n - 1)
    return result


@lru_cache(None)
def coarse_value(k, e, n):
    assert 0 <= e <= n and k >= 0
    return F(0) if e == 0 else min(loss(e), query_return(k, e, n))


@lru_cache(None)
def deterministic_tree_returns(k, e, n):
    if e == 0:
        return frozenset((F(0),))
    probability = F(e, n)
    positive = deterministic_tree_returns(k + 1, e - 1, n - 1)
    negative = deterministic_tree_returns(k + 1, e, n - 1) if e < n else (F(0),)
    returns = {loss(e)}
    returns.update(increment(k + 1) + probability * first + (1 - probability) * second
                   for first, second in itertools.product(positive, negative))
    return frozenset(returns)


@lru_cache(None)
def full_search_value(k, e, n):
    if e == 0:
        return F(0)
    probability = F(e, n)
    value = increment(k + 1) + probability * full_search_value(k + 1, e - 1, n - 1)
    if e < n:
        value += (1 - probability) * full_search_value(k + 1, e, n - 1)
    return value


@lru_cache(None)
def known_label_value(k, e, n):
    if e == 0:
        return F(0)
    actions = [loss(e), increment(k + 1) + known_label_value(k + 1, e - 1, n - 1)]
    if e < n:
        actions.append(increment(k + 1) + known_label_value(k + 1, e, n - 1))
    return min(actions)


def states(initial_errors, initial_units):
    return [(k, e, initial_units - k) for k in range(initial_units + 1)
            for e in range(max(0, initial_errors - k), min(initial_errors, initial_units - k) + 1)]


def response_check(name, initial_errors, initial_units):
    output = []
    for k, e, n in states(initial_errors, initial_units):
        stop = loss(e)
        query = query_return(k, e, n) if e else None
        value = coarse_value(k, e, n)
        all_returns = deterministic_tree_returns(k, e, n)
        assert value == min(all_returns)
        assert value == (min(stop, query) if e else 0)
        found = initial_errors - e
        negatives = k - found
        probability = F(math.comb(initial_errors, found) * math.comb(initial_units - initial_errors, negatives),
                        math.comb(initial_units, k))
        output.append({"k": k, "e": e, "n": n, "stop_loss": stop, "query_return": query,
                       "optimal_remaining_value": value, "optimal_action": "stop" if e == 0 or stop <= query else "query",
                       "distinct_deterministic_tree_returns": len(all_returns),
                       "forced_prefix_state_probability": probability})
    completions = Counter()
    for positives in itertools.combinations(range(initial_units), initial_errors):
        completions[max(positives) + 1] += 1
    total = math.comb(initial_units, initial_errors)
    full_cost = sum(F(frequency, total) * cost(queries) for queries, frequency in completions.items())
    expected_queries = sum(F(frequency, total) * queries for queries, frequency in completions.items())
    assert full_cost == full_search_value(0, initial_errors, initial_units)
    known = known_label_value(0, initial_errors, initial_units)
    assert known == min(loss(initial_errors - repaired) + cost(repaired) for repaired in range(initial_errors + 1))
    return {"name": name, "initial_errors": initial_errors, "initial_units": initial_units,
            "coarse_initial_value": coarse_value(0, initial_errors, initial_units), "known_label_initial_value": known,
            "initial_query_return": query_return(0, initial_errors, initial_units),
            "full_search_cost": full_cost, "full_search_expected_queries": expected_queries,
            "full_search_query_count_law": {queries: F(frequency, total) for queries, frequency in completions.items()},
            "initial_distinct_policy_return_values": sorted(deterministic_tree_returns(0, initial_errors, initial_units)),
            "posterior_states": output}


@lru_cache(None)
def interleaved_value(first, second):
    options = [loss(first[1]) + loss(second[1])]
    for r, state in enumerate((first, second)):
        k, e, n = state
        if e == 0:
            continue
        probability = F(e, n)
        positive = (k + 1, e - 1, n - 1)
        next_positive = (positive, second) if r == 0 else (first, positive)
        value = increment(k + 1) + probability * interleaved_value(*next_positive)
        if e < n:
            negative = (k + 1, e, n - 1)
            next_negative = (negative, second) if r == 0 else (first, negative)
            value += (1 - probability) * interleaved_value(*next_negative)
        options.append(value)
    return min(options)


def interleaving_check():
    checks = []
    for first, second in itertools.product(states(1, 3), states(4, 5)):
        joint = interleaved_value(first, second)
        independent = coarse_value(*first) + coarse_value(*second)
        assert joint == independent
        checks.append({"small_state": first, "big_state": second, "joint_remaining_value": joint})
    assert len(checks) == 60
    return {"posterior_pair_checks": len(checks), "initial_average_risk": interleaved_value((0, 1, 3), (0, 4, 5)) / 2,
            "checks": checks,
            "scope": "Independent within-response permutations, separable loss/cost, no global query budget"}


def auc(labels, scores):
    positive = [score for truth, row in zip(labels, scores) for label, score in zip(truth, row) if label]
    negative = [score for truth, row in zip(labels, scores) for label, score in zip(truth, row) if not label]
    return sum(F(1) if a > b else F(1, 2) if a == b else F(0) for a in positive for b in negative) / (len(positive) * len(negative))


def permuted_scoring_check():
    mean_scores = tuple((sum(row) / len(row),) * len(row) for row in SCORES)
    shifted_means = tuple(tuple(score + shift for score in row) for row, shift in zip(mean_scores, OFFSETS))
    pair_permutations = [sorted(set(itertools.permutations(tuple(zip(truth, row))))) for truth, row in zip(LABELS, SCORES)]
    counts = 0
    for first, second in itertools.product(*pair_permutations):
        labels = tuple(tuple(pair[0] for pair in row) for row in (first, second))
        scores = tuple(tuple(pair[1] for pair in row) for row in (first, second))
        shifted_scores = tuple(tuple(score + shift for score in row) for row, shift in zip(scores, OFFSETS))
        assert auc(labels, scores) == F(7, 15)
        assert auc(labels, shifted_scores) == F(1)
        assert auc(labels, mean_scores) == F(4, 15)
        assert auc(labels, shifted_means) == F(11, 15)
        assert [auc((truth,), (row,)) for truth, row in zip(labels, scores)] == [F(1), F(1)]
        assert [auc((truth,), (row,)) for truth, row in zip(labels, shifted_scores)] == [F(1), F(1)]
        assert [auc((truth,), (row,)) for truth, row in zip(labels, shifted_means)] == [F(1, 2), F(1, 2)]
        for truth, before, after in zip(labels, scores, shifted_scores):
            actual_errors = {index for index, label in enumerate(truth) if label}
            for row in (before, after):
                located = set(sorted(range(len(row)), key=lambda index: row[index], reverse=True)[:sum(truth)])
                assert located == actual_errors
        counts += 1
    assert counts == 15
    return {"labels": LABELS, "original_scores": SCORES, "projected_scores": mean_scores,
            "offset_scores": tuple(tuple(score + shift for score in row) for row, shift in zip(SCORES, OFFSETS)),
            "offset_projected_scores": shifted_means, "offsets": OFFSETS,
            "distinct_independent_pair_permutations": counts,
            "original_auc": F(7, 15), "offset_auc": F(1), "projection_auc": F(4, 15), "offset_projection_auc": F(11, 15)}


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
    parser.add_argument("--out", type=Path, default=Path("analysis/bandha_adaptive_review_results"))
    args = parser.parse_args()
    frozen_files = [DATA_ROOT / "analysis/bandha_marginal_response_check.py"] + [
        DATA_ROOT / "analysis/bandha_marginal_response_results" / name for name in ("verification.json", "values.csv", "values.tex")]
    frozen_hashes = {str(path.relative_to(DATA_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in frozen_files}
    small, big = response_check("small", 1, 3), response_check("big", 4, 5)
    assert small["coarse_initial_value"] == F(1, 2)
    assert big["coarse_initial_value"] == F(15, 16)
    assert small["known_label_initial_value"] == F(3, 8)
    assert big["known_label_initial_value"] == F(15, 16)
    assert small["full_search_cost"] == F(29, 48)
    assert big["full_search_cost"] == F(173, 160)
    assert coarse_value(1, 1, 2) == F(11, 32)
    full, coarse = (small["known_label_initial_value"] + big["known_label_initial_value"]) / 2, (small["coarse_initial_value"] + big["coarse_initial_value"]) / 2
    assert full == F(21, 32) and coarse == F(23, 32) and coarse - full == F(1, 16)
    interleaving, scoring = interleaving_check(), permuted_scoring_check()
    assert interleaving["initial_average_risk"] == coarse
    conditions = {
        name: {"risk_star": full if local else coarse, "auc": scoring[name + "_auc"],
               "conditional_auc": F(1) if local else F(1, 2),
               "information_class": "local labels identified by score ordering and exact counts" if local else
                                    "exact counts and exchangeable hidden positions"}
        for name, local in (("original", True), ("projection", False), ("offset", True), ("offset_projection", False))
    }
    assert tuple(conditions[name]["risk_star"] for name in ("original", "projection", "offset", "offset_projection")) == (
        F(21, 32), F(23, 32), F(21, 32), F(23, 32))
    result = {"all_checks_passed": True, "responses": {"small": small, "big": big},
              "full_information_average_risk": full, "coarse_information_average_risk": coarse, "risk_gap": coarse - full,
              "conditions": conditions,
              "interleaving": interleaving, "scoring": scoring,
              "loss_values": tuple(loss(e) for e in range(5)), "cost_values": tuple(cost(k) for k in range(6)),
              "assumptions": ["Within-response label-score pairs are independently uniformly permuted.",
                              "Coarse observations are constant scores, exact initial counts, and paid query feedback.",
                              "No other uncharged locator or label-relevant text feature is available.",
                              "A query perfectly reveals and repairs its error and incurs its marginal cost.",
                              "Known errors still require a paid repair query; no free bulk repair is allowed.",
                              "Loss and work are separable across responses, with no externally imposed query quota.",
                              "Fine score classes identify the unverified error positions in this analytic model."],
              "scope": "All adaptive score/count/query-feedback policies in a finite analytic review model; no human or neural trial",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "unchanged_hmr_hashes": frozen_hashes}
    values = {"adaptive.full.risk": full, "adaptive.coarse.risk": coarse, "adaptive.gap": coarse - full,
              "adaptive.gap_projection_offset_vs_original": coarse - full,
              "adaptive.original.auc": scoring["original_auc"], "adaptive.projection.auc": scoring["projection_auc"],
              "adaptive.offset.auc": scoring["offset_auc"],
              "adaptive.offset_projection.auc": scoring["offset_projection_auc"]}
    for name, condition in conditions.items():
        values["adaptive." + name + ".risk_star"] = condition["risk_star"]
        values["adaptive." + name + ".conditional_auc"] = condition["conditional_auc"]
    for name, response in (("small", small), ("big", big)):
        prefix = "adaptive." + name + "."
        for key in ("initial_errors", "initial_units", "coarse_initial_value", "known_label_initial_value", "initial_query_return", "full_search_cost", "full_search_expected_queries"):
            values[prefix + key] = F(response[key])
        values[prefix + "optimal_coarse_queries"] = F(0)
        values[prefix + "optimal_fine_queries"] = F(1 if name == "small" else 0)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "verification.json").write_text(json.dumps(encode(result), indent=2) + "\n")
    with (args.out / "values.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["quantity", "exact", "decimal"])
        writer.writerows((key, str(value), format(float(value), ".12f")) for key, value in values.items())
    lines = [r"\providecommand{\HomerAdaptiveReviewValue}[1]{\csname har@#1\endcsname}"]
    lines.extend(r"\expandafter\def\csname har@" + key + r"\endcsname{" + tex_number(value) + "}" for key, value in values.items())
    (args.out / "values.tex").write_text("\n".join(lines) + "\n")
    assert frozen_hashes == {str(path.relative_to(DATA_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in frozen_files}
    print(json.dumps({"all_checks_passed": True, "source_sha256": result["source_sha256"],
                      "fine_risk": str(full), "coarse_risk": str(coarse), "gap": str(coarse - full),
                      "coarse_initial_query_costs": [str(small["initial_query_return"]), str(big["initial_query_return"])]}))


if __name__ == "__main__":
    main()
