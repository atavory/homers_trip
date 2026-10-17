"""Check history-gated kernel changes with explicit final loss and costs."""

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

import bandha_intervention_check as core
import bandha_sequential_check as existing


PROMPTS = existing.PROMPTS
THRESHOLD = F(1, 2)


def proposal(prompt, prefix, horizon, marked_count, feedback=True):
    if prompt == "error_bearing":
        return int(len(prefix) >= horizon - marked_count)
    if prompt == "critical_clean" and feedback and prefix:
        return int(prefix[0] == 1)
    return 0


def scores(prompt, horizon, marked_count):
    if prompt == "error_bearing":
        return tuple(F(1, 20) if t < horizon - marked_count else
                     F(1, 5) if t == horizon - marked_count else F(11, 20)
                     for t in range(horizon))
    if prompt == "critical_clean":
        return (F(1, 5),) + (F(1, 20),) * (horizon - 1)
    return (F(2, 5),) * (horizon - 1) + (F(4, 5),)


def transformed_scores(prompt, horizon, marked_count, policy):
    values = scores(prompt, horizon, marked_count)
    if policy in ("projection", "offset_projection"):
        values = (sum(values) / horizon,) * horizon
    if policy in ("offset", "offset_projection"):
        shift = F(2, 5) if prompt in PROMPTS[:2] else -F(2, 5)
        values = tuple(value + shift for value in values)
    return values


def rollout(prompt, horizon, marked_count, policy, feedback=True, threshold=THRESHOLD):
    values = transformed_scores(prompt, horizon, marked_count, policy)
    prefix, events = [], []
    for t in range(horizon):
        candidate = proposal(prompt, prefix, horizon, marked_count, feedback)
        selected = policy != "none" and values[t] >= threshold
        accepted = 1 - candidate if selected else candidate
        events.append({"position": t, "prefix": tuple(prefix), "proposal": candidate,
                       "score": values[t], "override": selected, "accepted": accepted})
        prefix.append(accepted)
    return {"prompt": prompt, "events": events, "accepted": tuple(prefix)}


def summarize(runs, query=True):
    events = [event for run in runs for event in run["events"]]
    return {"errors": sum(event["accepted"] for event in events),
            "edits": sum(event["override"] for event in events),
            "queries": len(events) if query else 0,
            "base_kernel_errors": sum(event["proposal"] for event in events)}


def classification(rows):
    pairs = [(label, score >= THRESHOLD) for row in rows for label, score in zip(row.labels, row.scores)]
    tp = sum(label and selected for label, selected in pairs)
    fp = sum(not label and selected for label, selected in pairs)
    fn = sum(label and not selected for label, selected in pairs)
    tn = sum(not label and not selected for label, selected in pairs)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": F(tp, tp + fp) if tp + fp else None,
            "recall": F(tp, tp + fn), "f1": F(2 * tp, 2 * tp + fp + fn),
            "accuracy": F(tp + tn, len(pairs)),
            "auc": core.auc(core.pooled_law(rows, [F(1, 4)] * 4)),
            "ap": existing.average_precision(rows),
            "conditional_auc_mixed": [core.local_auc(row) for row in rows if 0 < sum(row.labels) < len(row.labels)]}


def fixture(horizon, marked_count):
    assert 4 <= marked_count < horizon
    assert 2 * horizon * (marked_count - 1) >= 2 * marked_count + 1
    outputs = {"horizon": horizon, "marked_count": marked_count, "policies": {}}
    original_rows, offset_rows = [], []
    for policy in ("detector", "projection", "offset", "offset_projection"):
        rows = []
        for prompt in PROMPTS:
            baseline = rollout(prompt, horizon, marked_count, "none")
            row_scores = transformed_scores(prompt, horizon, marked_count, policy)
            rows.append(core.row(baseline["accepted"], row_scores))
        metrics = classification(rows)
        thresholds = sorted({value for row in rows for value in row.scores} | {F(1)})
        maximum_f1 = max(existing.f1_at(rows, threshold) for threshold in thresholds)
        if policy == "detector":
            original_rows = rows
        if policy == "offset":
            offset_rows = rows
        policies = {"frozen": metrics, "maximum_frozen_f1": maximum_f1}
        for feedback in (False, True):
            name = "feedback" if feedback else "feedback_disabled"
            runs = [rollout(prompt, horizon, marked_count, policy, feedback) for prompt in PROMPTS]
            policies[name] = {"summary": summarize(runs), "trajectories": runs}
        gate_outcomes = {str(threshold): summarize([rollout(prompt, horizon, marked_count, policy, threshold=threshold)
                                                   for prompt in PROMPTS]) for threshold in thresholds}
        policies["controller_threshold_outcomes"] = gate_outcomes
        policies["minimum_final_errors_over_thresholds"] = min(stats["errors"] for stats in gate_outcomes.values())
        outputs["policies"][policy] = policies
    detector, offset = (outputs["policies"][name] for name in ("detector", "offset"))
    assert detector["frozen"]["f1"] == detector["maximum_frozen_f1"]
    assert offset["frozen"]["f1"] == offset["maximum_frozen_f1"]
    for metric in ("f1", "auc", "ap", "accuracy", "precision", "recall"):
        assert offset["frozen"][metric] > detector["frozen"][metric]
    assert detector["frozen"]["conditional_auc_mixed"] == offset["frozen"]["conditional_auc_mixed"] == [F(1)]
    for original, shifted in zip(original_rows, offset_rows):
        assert all(0 <= value <= 1 for value in shifted.scores)
        assert tuple(value - core.mean(original.scores) for value in original.scores) == tuple(value - core.mean(shifted.scores) for value in shifted.scores)
    assert detector["feedback"]["summary"]["errors"] == 3
    assert offset["feedback"]["summary"]["errors"] == horizon
    assert detector["minimum_final_errors_over_thresholds"] == 3
    assert offset["minimum_final_errors_over_thresholds"] == 1
    assert detector["feedback_disabled"]["summary"]["errors"] == 3
    assert offset["feedback_disabled"]["summary"]["errors"] == 1
    assert detector["feedback"]["summary"]["edits"] == offset["feedback"]["summary"]["edits"] == marked_count + 1
    for policy in (detector, offset):
        selected = lambda branch: [[(event["position"], event["proposal"], event["accepted"])
                                   for event in run["events"] if event["override"]]
                                  for run in policy[branch]["trajectories"]]
        assert selected("feedback") == selected("feedback_disabled")
    baseline = summarize([rollout(prompt, horizon, marked_count, "none") for prompt in PROMPTS], query=False)
    assert baseline["errors"] == marked_count
    outputs["baseline"] = baseline
    outputs["cost_comparisons"] = []
    for edit_cost in (F(0), F(1, 100), F(1, 10), F(1)):
        for query_cost in (F(0), F(1, 100)):
            losses = {}
            for name in ("detector", "offset"):
                stats = outputs["policies"][name]["feedback"]["summary"]
                losses[name] = F(stats["errors"]) + edit_cost * stats["edits"] + query_cost * stats["queries"]
            assert losses["offset"] - losses["detector"] == horizon - 3
            outputs["cost_comparisons"].append({"edit_cost": edit_cost, "query_cost": query_cost,
                                                 "losses": losses, "loss_difference": horizon - 3})
    return outputs


def calibrated_checkpoint_fixture():
    states = [("late", F(7, 10), F(9, 10), F(9, 10)),
              ("critical", F(3, 20), F(3, 5), F(2, 5)),
              ("easy", F(3, 20), F(1, 5), F(2, 5))]
    metrics = {}
    for name, refined in (("coarse", False), ("refined", True)):
        distribution = {}
        for _, weight, risk, coarse in states:
            value = risk if refined else coarse
            for label, mass in ((1, weight * risk), (0, weight * (1 - risk))):
                key = (value, label)
                distribution[key] = distribution.get(key, F(0)) + mass
        for score in {value for value, _ in distribution}:
            positive = distribution.get((score, 1), F(0))
            total = positive + distribution.get((score, 0), F(0))
            assert positive / total == score
        def scores_at(threshold):
            tp = sum(mass for (score, label), mass in distribution.items() if label and score >= threshold)
            fp = sum(mass for (score, label), mass in distribution.items() if not label and score >= threshold)
            fn = sum(mass for (score, label), mass in distribution.items() if label and score < threshold)
            tn = sum(mass for (score, label), mass in distribution.items() if not label and score < threshold)
            return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "f1": 2 * tp / (2 * tp + fp + fn), "accuracy": tp + tn}
        threshold_metrics = scores_at(THRESHOLD)
        assert threshold_metrics["f1"] == max(scores_at(value)["f1"] for value in {score for score, _ in distribution} | {F(1)})
        metrics[name] = {"label_law": distribution, "calibrated": True, "f1_optimal_threshold": THRESHOLD,
                         **threshold_metrics, "auc": core.auc(distribution),
                         "brier": sum(mass * (score - label) ** 2 for (score, label), mass in distribution.items())}
    assert metrics["coarse"]["f1"] == F(126, 145)
    assert metrics["refined"]["f1"] == F(9, 10)
    assert metrics["coarse"]["auc"] == F(39, 50)
    assert metrics["refined"]["auc"] == F(201, 250)
    assert metrics["refined"]["brier"] < metrics["coarse"]["brier"]
    gains = []
    for horizon in (2, 5, 20):
        for cost in (F(0), F(1, 10)):
            values = {}
            for name, refined in (("coarse", False), ("refined", True)):
                value = F(0)
                for state, weight, risk, coarse in states:
                    score = risk if refined else coarse
                    damage = horizon if state == "critical" else 1
                    if score >= THRESHOLD:
                        value += weight * (risk - (1 - risk) * damage - cost)
                values[name] = value
            assert values["refined"] - values["coarse"] == F(3, 20) * (F(3, 5) - F(2, 5) * horizon - cost)
            assert values["refined"] < values["coarse"]
            gains.append({"horizon": horizon, "override_cost": cost, "net_improvement_over_keep": values})
    return {"scope": "One checkpoint then common continuation; calibrated context-information example, not completed-response offset",
            "states": states, "metrics": metrics, "continuation_gains": gains}


def constant_gate_regret_fixture():
    values = []
    positive_mass, adverse_mass = F(4, 5), F(1, 5)
    for horizon in (1, 2, 4, 20):
        for cost in (F(0), F(1, 10)):
            positive_advantage, adverse_advantage = 1 - cost, -horizon - cost
            a = positive_mass * max(positive_advantage, 0)
            b = adverse_mass * max(-adverse_advantage, 0)
            fine = a
            constant = max(a - b, 0)
            assert fine - constant == min(a, b)
            always_active = a - b
            assert fine - always_active == b
            values.append({"horizon": horizon, "override_cost": cost,
                           "positive_advantage": positive_advantage, "adverse_advantage": adverse_advantage,
                           "positive_return_mass": a, "adverse_return_mass": b,
                           "fine_gate_return": fine, "optimal_constant_gate_return": constant,
                           "optimal_constant_gate_regret": min(a, b), "always_active_gate_regret": b})
    return {"scope": "One fixed sampled checkpoint law and common future policy",
            "positive_state_mass": positive_mass, "adverse_state_mass": adverse_mass,
            "values": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("analysis/bandha_downstream_loss_results"))
    args = parser.parse_args()
    outputs = [fixture(horizon, marked_count) for horizon, marked_count in ((5, 4), (6, 4), (10, 4), (20, 4), (20, 5))]
    small = outputs[0]
    detector, offset = (small["policies"][name]["frozen"] for name in ("detector", "offset"))
    assert detector["f1"] == F(2, 3)
    assert offset["f1"] == F(8, 9)
    assert detector["auc"] == F(95, 128)
    assert offset["auc"] == F(127, 128)
    assert detector["ap"] == F(31, 60)
    assert offset["ap"] == F(19, 20)
    assert outputs[-1]["policies"]["detector"]["frozen"]["auc"] == F(653, 750)
    assert outputs[-1]["policies"]["offset"]["feedback"]["summary"]["errors"] == 20
    result = {"all_checks_passed": True, "scope": "Exact finite generators and action counts; no LLM run", "fixtures": outputs,
              "calibrated_checkpoint": calibrated_checkpoint_fixture(),
              "constant_gate_regret": constant_gate_regret_fixture(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "helper_sha256": {Path(helper.__file__).name: hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest()
                                for helper in (core, existing)}}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "verification.json").write_text(json.dumps(core.serializable(result), indent=2) + "\n")
    print(json.dumps({"all_checks_passed": True, "fixtures": len(outputs), "source_sha256": result["source_sha256"]}))


if __name__ == "__main__":
    main()
