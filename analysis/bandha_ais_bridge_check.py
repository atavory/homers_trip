"""Verify pooled detection ties with unequal whole-response source support."""

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = F(1, 2)
UNSUPPORTED = ((1, 0, 1), (1, 0, 1))
SCORES = {
    "concentrated": ((F(9, 10), F(1, 10), F(9, 10)), (F(1, 10), F(1, 10), F(1, 10))),
    "distributed": ((F(9, 10), F(1, 10), F(1, 10)), (F(9, 10), F(1, 10), F(1, 10))),
}


def compare(first, second):
    return F(1) if first > second else F(1, 2) if first == second else F(0)


def auc(labels, scores):
    positive = [score for label, score in zip(labels, scores) if label]
    negative = [score for label, score in zip(labels, scores) if not label]
    assert positive and negative
    return sum(compare(a, b) for a in positive for b in negative) / (len(positive) * len(negative))


def average_precision(labels, scores):
    groups = defaultdict(list)
    for label, score in zip(labels, scores):
        groups[score].append(label)
    total, positive, result = 0, 0, F(0)
    for score in sorted(groups, reverse=True):
        at_score = groups[score]
        total += len(at_score)
        positive += sum(at_score)
        result += F(sum(at_score), sum(labels)) * F(positive, total)
    return result


def perfect_editor(labels, flags):
    revised = tuple(int(label and not flag) for label, flag in zip(labels, flags))
    assert all(revised[i] == labels[i] for i, flag in enumerate(flags) if not flag)
    assert all(revised[i] == 0 for i, flag in enumerate(flags) if flag)
    assert all(revised[i] == 0 for i, label in enumerate(labels) if not label)
    return revised


def readout(labels_by_response, scores_by_response):
    assert len(labels_by_response) == len(scores_by_response)
    flags = tuple(tuple(score > THRESHOLD for score in scores) for scores in scores_by_response)
    revised = tuple(perfect_editor(labels, selected) for labels, selected in zip(labels_by_response, flags))
    flat_labels = tuple(label for labels in labels_by_response for label in labels)
    flat_scores = tuple(score for scores in scores_by_response for score in scores)
    flat_flags = tuple(flag for selected in flags for flag in selected)
    tp = sum(label and flag for label, flag in zip(flat_labels, flat_flags))
    fp = sum(not label and flag for label, flag in zip(flat_labels, flat_flags))
    fn = sum(label and not flag for label, flag in zip(flat_labels, flat_flags))
    tn = sum(not label and not flag for label, flag in zip(flat_labels, flat_flags))
    per_response = []
    for labels, scores, selected, remaining in zip(labels_by_response, scores_by_response, flags, revised):
        intersection = sum(label and flag for label, flag in zip(labels, selected))
        union = sum(label or flag for label, flag in zip(labels, selected))
        assert union > 0
        mean_score = sum(scores) / len(scores)
        per_response.append({
            "original_unsupported": sum(labels), "actions": sum(selected),
            "remaining_unsupported": sum(remaining), "fully_supported": int(not any(remaining)),
            "support_fraction": F(len(labels) - sum(remaining), len(labels)),
            "unsupported_recovery_fraction": F(intersection, sum(labels)),
            "iou": F(intersection, union), "conditional_auc": auc(labels, scores),
            "mean_score": mean_score,
            "centered_scores": tuple(score - mean_score for score in scores),
        })
    response_count = len(per_response)
    remaining_count = sum(sum(labels) for labels in revised)
    complete_count = sum(row["fully_supported"] for row in per_response)
    macro_iou = sum(row["iou"] for row in per_response) / response_count
    actions = sum(flat_flags)
    assert fp == 0
    assert macro_iou == F(tp, 2 * response_count)  # Equal denominator two on this base witness.
    pooled = Counter((score, label) for label, score in zip(flat_labels, flat_scores))
    summary = {
        "true_positives": tp, "false_positives": fp, "false_negatives": fn, "true_negatives": tn,
        "actions": actions, "correction_cost_at_unit_price_one": actions,
        "precision": F(tp, tp + fp), "recall": F(tp, tp + fn),
        "f1": F(2 * tp, 2 * tp + fp + fn), "accuracy": F(tp + tn, len(flat_labels)),
        "pooled_iou": F(tp, tp + fp + fn), "pooled_auc": auc(flat_labels, flat_scores),
        "pooled_ap": average_precision(flat_labels, flat_scores),
        "pooled_brier": sum((score - label) ** 2 for label, score in zip(flat_labels, flat_scores)) / len(flat_labels),
        "macro_iou": macro_iou,
        "macro_conditional_auc": sum(row["conditional_auc"] for row in per_response) / response_count,
        "macro_unsupported_recovery_fraction": sum(row["unsupported_recovery_fraction"] for row in per_response) / response_count,
        "remaining_unsupported_count": remaining_count,
        "final_supported_count": len(flat_labels) - remaining_count,
        "micro_final_support_fraction": F(len(flat_labels) - remaining_count, len(flat_labels)),
        "macro_final_support_fraction": sum(row["support_fraction"] for row in per_response) / response_count,
        "fully_supported_count": complete_count, "ais_success_rate": F(complete_count, response_count),
        "ais_loss": 1 - F(complete_count, response_count),
        "mean_correction_cost": F(actions, response_count),
        "ais_plus_mean_work_loss_at_price_one": 1 - F(complete_count, response_count) + F(actions, response_count),
        "ideal_preservation_assumption": F(1),
    }
    return {"scores": scores_by_response, "selected": flags, "revised_unsupported_labels": revised,
            "per_response": per_response, "summary": summary,
            "pooled_score_label_multiset": [(score, label, count) for (score, label), count in sorted(pooled.items())]}


def operator_check(first, second):
    differences = tuple(tuple(b - a for a, b in zip(a_scores, b_scores))
                        for a_scores, b_scores in zip(first, second))
    assert differences == ((F(0), F(0), -F(4, 5)), (F(4, 5), F(0), F(0)))
    assert all(len(set(delta)) > 1 for delta in differences)
    assert all(len(set(scores)) > 1 for scores in second)
    return {"score_differences_distributed_minus_concentrated": differences,
            "is_pure_response_offset": False,
            "is_offset_after_projection": False,
            "concentrated_means": tuple(sum(scores) / len(scores) for scores in first),
            "distributed_means": tuple(sum(scores) / len(scores) for scores in second),
            "original_prevalences": (F(2, 3), F(2, 3))}


def generalized_check():
    cases = []
    for count in range(2, 8):
        labels = tuple(value for index in range(count) for value in ((1,) if index == count - 1 else (1, 0)))
        assert sum(labels) == count
        for split in range(1, count):
            masks = {
                "concentrated": (tuple(bool(label) for label in labels), (False,) * len(labels)),
                "distributed": (
                    tuple(bool(label) and index // 2 < split for index, label in enumerate(labels)),
                    tuple(bool(label) and index // 2 < count - split for index, label in enumerate(labels)),
                ),
            }
            summaries = {}
            pooled = {}
            for name, flags in masks.items():
                values = tuple(tuple(F(9, 10) if flag else F(1, 10) for flag in selected) for selected in flags)
                pooled[name] = Counter((score, label) for scores in values for label, score in zip(labels, scores))
                intersections = [sum(label and flag for label, flag in zip(labels, selected)) for selected in flags]
                remaining = [count - value for value in intersections]
                summaries[name] = {"tp": sum(intersections), "fn": sum(remaining), "actions": sum(sum(selected) for selected in flags),
                                   "macro_iou": sum(F(value, count) for value in intersections) / 2,
                                   "complete_count": sum(value == 0 for value in remaining)}
            assert pooled["concentrated"] == pooled["distributed"]
            assert summaries["concentrated"]["macro_iou"] == summaries["distributed"]["macro_iou"] == F(1, 2)
            assert summaries["concentrated"]["complete_count"] == 1
            assert summaries["distributed"]["complete_count"] == 0
            cases.append({"unsupported_per_response": count, "distributed_split": split, "summaries": summaries})
    return cases


def completion_bound(count, beta):
    assert count >= 1 and 0 <= beta <= 1
    mean_count = count * beta
    lower, upper = max(F(0), mean_count - count + 1), beta
    upper_law = {0: 1 - beta, count: beta}
    if mean_count >= count - 1:
        lower_law = {count - 1: 1 - lower, count: lower}
    else:
        partial_mass = mean_count / (count - 1)
        lower_law = {0: 1 - partial_mass, count - 1: partial_mass}
    attained = {}
    for name, law, target in (("upper", upper_law, upper), ("lower", lower_law, lower)):
        assert all(0 <= repaired <= count and probability >= 0 for repaired, probability in law.items())
        assert sum(law.values()) == 1
        assert sum(F(repaired, count) * probability for repaired, probability in law.items()) == beta
        assert law.get(count, F(0)) == target
        cohort_size = math.lcm(*(probability.denominator for probability in law.values()))
        frequencies = {repaired: cohort_size * probability for repaired, probability in law.items()}
        assert all(frequency.denominator == 1 for frequency in frequencies.values())
        assert sum(frequencies.values()) == cohort_size
        attained[name] = {"repair_count_law": law, "ais_rate": target,
                          "uniform_cohort_size": cohort_size, "cohort_count_frequencies": frequencies}
    return {"initial_unsupported_per_answer": count, "macro_iou": beta,
            "lower": lower, "upper": upper, "attainments": attained}


def completion_bounds_check(results):
    cases = [completion_bound(count, F(numerator, 12)) for count in range(1, 9) for numerator in range(13)]
    bridge = completion_bound(2, F(1, 2))
    assert bridge["lower"] == 0 and bridge["upper"] == F(1, 2)
    for name, endpoint in (("concentrated", "upper"), ("distributed", "lower")):
        rows = results[name]["per_response"]
        frequencies = Counter(row["original_unsupported"] - row["remaining_unsupported"] for row in rows)
        empirical = {count: F(frequency, len(rows)) for count, frequency in frequencies.items()}
        endpoint_law = {count: mass for count, mass in bridge["attainments"][endpoint]["repair_count_law"].items() if mass}
        assert empirical == endpoint_law
    fixed_cohort = [(first, second, F(int(first == 2) + int(second == 2), 2))
                    for first in range(3) for second in range(3) if first + second == 2]
    assert {rate for _, _, rate in fixed_cohort} == {F(0), F(1, 2)}
    randomized_rates = {F(numerator, 4): F(numerator, 8) for numerator in range(5)}
    assert all(0 <= rate <= F(1, 2) for rate in randomized_rates.values())
    return {"given_information": "Common initial unsupported count m and beta=E[C/m] only",
            "lower_formula": "max(0, m*beta-m+1)", "upper_formula": "beta",
            "scope": "No false flags or new errors; perfect flagged repair; AIS-complete iff C=m",
            "sharpness": "Laws are realizable by stochastic scoring or compatible finite uniform rational cohorts",
            "cases": cases, "bridge_case": bridge,
            "fixed_two_answer_cohort_at_m2_beta_half": fixed_cohort,
            "randomized_bridge_rate_by_concentration_probability": randomized_rates}


def encode(value):
    if isinstance(value, F):
        return {"exact": str(value), "decimal": float(value)}
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def tex_fraction(value):
    value = F(value)
    return str(value.numerator) if value.denominator == 1 else "\\frac{" + str(value.numerator) + "}{" + str(value.denominator) + "}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "analysis/bandha_ais_bridge_results")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    input_paths = [args.reference] if args.reference else []
    original_hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}
    results = {name: readout(UNSUPPORTED, scores) for name, scores in SCORES.items()}
    first, second = (results[name] for name in ("concentrated", "distributed"))
    assert first["pooled_score_label_multiset"] == second["pooled_score_label_multiset"]
    tied = ("true_positives", "false_positives", "false_negatives", "true_negatives", "actions",
            "correction_cost_at_unit_price_one", "precision", "recall", "f1", "accuracy", "pooled_iou",
            "pooled_auc", "pooled_ap", "pooled_brier", "macro_iou", "macro_conditional_auc",
            "macro_unsupported_recovery_fraction", "remaining_unsupported_count", "final_supported_count",
            "micro_final_support_fraction", "macro_final_support_fraction", "mean_correction_cost")
    assert all(first["summary"][name] == second["summary"][name] for name in tied)
    assert first["summary"]["fully_supported_count"] == 1
    assert second["summary"]["fully_supported_count"] == 0
    assert first["summary"]["macro_iou"] == F(1, 2)
    assert first["summary"]["pooled_auc"] == F(3, 4)
    assert first["summary"]["pooled_ap"] == F(5, 6)
    assert first["summary"]["f1"] == F(2, 3)
    assert first["summary"]["micro_final_support_fraction"] == F(2, 3)
    if args.reference:
        private_result = json.loads(args.reference.read_text())
        for name, values in results.items():
            for key in ("true_positives", "false_positives", "false_negatives", "actions", "fully_supported_responses"):
                mapped = "fully_supported_count" if key == "fully_supported_responses" else key
                assert private_result[name][key] == values["summary"][mapped]
            assert private_result[name]["macro_iou"] == str(values["summary"]["macro_iou"])
            assert private_result[name]["response_support_success_rate"] == str(values["summary"]["ais_success_rate"])
            assert private_result[name]["remaining_errors_by_response"] == [row["remaining_unsupported"] for row in values["per_response"]]
            stored_multiset = {(F(pair[0][0]), pair[0][1]): pair[1] for pair in private_result[name]["pooled_score_label_multiset"]}
            assert stored_multiset == {(score, label): count for score, label, count in values["pooled_score_label_multiset"]}
    output = {"all_checks_passed": True, "unsupported_labels": UNSUPPORTED, "threshold": THRESHOLD,
              "results": results, "verified_tied_summaries": tied,
              "operator_relation": operator_check(SCORES["concentrated"], SCORES["distributed"]),
              "generalized_fixtures": generalized_check(),
              "completion_moment_bounds": completion_bounds_check(results),
              "assumptions": ["All responses are interpretable and contain only the represented factual units.",
                              "Unsupported means unsupported by the stipulated source, independently of world truth.",
                              "Flagged unsupported units become source-supported; no other units change.",
                              "Intent and useful correct content are preserved by assumption.",
                              "Every flagged unit costs one; response weights and unit weights are uniform.",
                              "Whole-response AIS succeeds exactly when no represented unsupported units remain."],
              "preservation_scope": "Ideal preservation is stipulated; literal RARR character-edit preservation is not computed.",
              "conditional_scorer_scope": "Conditional AUROC uses half credit for ties; no score_cor value or convention is asserted.",
              "scope": "Specified-consumer nonidentification; no strict pooled misranking or empirical degradation claim.",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "immutable_original_sha256": original_hashes}
    values = {name + "." + key: F(value) for name, result in results.items() for key, value in result["summary"].items()}
    values["contrast.ais_success_rate"] = first["summary"]["ais_success_rate"] - second["summary"]["ais_success_rate"]
    values["contrast.work"] = first["summary"]["mean_correction_cost"] - second["summary"]["mean_correction_cost"]
    bridge_bound = output["completion_moment_bounds"]["bridge_case"]
    values["bound.bridge.m"] = F(bridge_bound["initial_unsupported_per_answer"])
    values["bound.bridge.beta"] = bridge_bound["macro_iou"]
    values["bound.bridge.lower"] = bridge_bound["lower"]
    values["bound.bridge.upper"] = bridge_bound["upper"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "verification.json").write_text(json.dumps(encode(output), indent=2) + "\n")
    with (args.out / "values.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["quantity", "exact", "decimal"])
        writer.writerows((key, str(value), format(float(value), ".12f")) for key, value in values.items())
    lines = [r"\providecommand{\HomerAISBridgeValue}[1]{\csname hais@#1\endcsname}"]
    lines.extend(r"\expandafter\def\csname hais@" + key + r"\endcsname{" + tex_fraction(value) + "}" for key, value in values.items())
    (args.out / "values.tex").write_text("\n".join(lines) + "\n")
    assert original_hashes == {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}
    print(json.dumps({"all_checks_passed": True, "source_sha256": output["source_sha256"],
                      "ais_success_counts": [first["summary"]["fully_supported_count"], second["summary"]["fully_supported_count"]],
                      "macro_iou": str(first["summary"]["macro_iou"]), "output": str(args.out)}))


if __name__ == "__main__":
    main()
