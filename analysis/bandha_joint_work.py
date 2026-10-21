"""Evaluate capped errors, capped work and normalized joint regret."""
import argparse
from functools import lru_cache
import json
import multiprocessing
from pathlib import Path
import shutil
import time

import numpy as np
from bandha_cap_evidence import Allocation, digest, regrets


@lru_cache(maxsize=50000)
def selected_tie_law(size, population, chosen):
    lower, upper = max(0, size - population + chosen), min(size, chosen)
    x = np.arange(lower, upper, dtype=float)
    ratios = (np.log(size - x) - np.log(x + 1) + np.log(chosen - x)
              - np.log(population - size - chosen + x + 1))
    logs = np.r_[0., np.cumsum(ratios)]
    mass = np.exp(logs - logs.max())
    mass /= mass.sum()
    return lower, mass


class JointAllocation(Allocation):
    def __init__(self, *args):
        super().__init__(*args)
        self.word_responses = np.repeat(np.arange(self.responses), self.units)
        self.maximum_work = int(self.units.max())

    def joint(self, source_weights):
        errors, metadata, _ = self.evaluate(source_weights, .2, return_histogram=True)
        threshold = metadata["threshold"]
        population, chosen = metadata["tie_population"], metadata["selected_ties"]
        weights = source_weights[self.response_sources]
        above = np.bincount(self.word_responses[self.scores > threshold], minlength=self.responses)
        tied = np.bincount(self.word_responses[self.scores == threshold], minlength=self.responses)
        if chosen == population:
            checks = above + tied
            histogram = np.bincount(checks, weights=weights, minlength=self.maximum_work + 1)
            read_words = float(np.dot(weights * (checks > 0), self.units))
        else:
            histogram = np.bincount(above, weights=weights, minlength=self.maximum_work + 1)
            read_words = float(np.dot(weights * (above > 0), self.units))
            active = np.flatnonzero((tied > 0) & (weights > 0))
            np.add.at(histogram, above[active], -weights[active])
            states = {}
            for r in active:
                key = int(above[r]), int(tied[r])
                mass, reading = states.get(key, (0, 0))
                states[key] = mass + int(weights[r]), reading + int(weights[r] * self.units[r])
            for (a, t), (mass, reading) in states.items():
                lower, probabilities = selected_tie_law(t, population, chosen)
                histogram[a + lower:a + lower + len(probabilities)] += mass * probabilities
                if a == 0:
                    read_words += reading * (1 - probabilities[0] if lower == 0 else 1.)
        denominator = int(weights.sum())
        if not np.isclose(histogram.sum(), denominator, atol=1e-7, rtol=0):
            raise ValueError("Work histogram lost response mass.")
        distribution = histogram / histogram.sum()
        counts = np.arange(len(distribution))
        cdf = np.cumsum(distribution)
        work = np.cumsum(distribution[1:] * counts[1:]) + counts[1:] * np.maximum(0., 1 - cdf[1:])
        expected_checks = metadata["selected_units"] / denominator
        if not np.isclose(work[-1], expected_checks, atol=1e-8, rtol=0):
            raise ValueError("Expected work does not match selected-word budget.")
        work[-1] = expected_checks
        metadata.update({"expected_opened_responses": float(work[0] * denominator),
                         "expected_read_response_words": read_words,
                         "mean_checks": expected_checks})
        return errors, work, metadata


def block_values(block):
    start, weights = block
    errors, work = [], []
    for w in weights:
        outcomes = [a.joint(w) for a in ALLOCATIONS.values()]
        errors.append([x[0] for x in outcomes])
        work.append([x[1] for x in outcomes])
    return start, np.array(errors), np.array(work)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--error-results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--draw-limit", type=int)
    args = ap.parse_args()
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    index = json.loads(args.index.read_text())
    source_ids = sorted({r["source_id"] for r in index})
    source_lookup = {s: i for i, s in enumerate(source_ids)}
    response_sources = np.array([source_lookup[r["source_id"]] for r in index])
    global ALLOCATIONS
    with np.load(args.scores, allow_pickle=False) as z:
        names = [n for n in z.files if n not in ["labels", "response_offsets"]]
        ALLOCATIONS = {n: JointAllocation(z[n], z["labels"], z["response_offsets"], response_sources)
                       for n in names}
    with np.load(args.error_results / "source_weights.npz", allow_pickle=False) as z:
        assert list(z["source_ids"]) == source_ids
        weights = z["weights"]
    if args.draw_limit:
        weights = weights[:args.draw_limit]
    point_outcomes = [a.joint(np.ones(len(source_ids), dtype=int)) for a in ALLOCATIONS.values()]
    error_point = np.array([x[0] for x in point_outcomes])
    work_point = np.array([x[1] for x in point_outcomes])
    old = json.loads((args.error_results / "results.json").read_text())
    old_point = np.load(args.error_results / "curves.npz")["point"]
    if old["methods"] != names or np.max(np.abs(error_point - old_point)) > 1e-10:
        raise ValueError("The prior error comparison changed.")
    draws = len(weights)
    boot_error = np.empty((draws, *error_point.shape))
    boot_work = np.empty((draws, *work_point.shape))
    blocks = [(s, weights[s:s + 100]) for s in range(0, draws, 100)]
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        completed = 0
        for start, e, k in pool.imap_unordered(block_values, blocks):
            boot_error[start:start + len(e)] = e
            boot_work[start:start + len(k)] = k
            completed += len(e)
            if completed % 500 == 0 or completed == draws:
                print(json.dumps({"completed": completed, "draws": draws,
                                  "seconds": round(time.monotonic() - started, 1)}), flush=True)
    prior_regrets = np.load(args.error_results / "bootstrap_summaries.npz")["regrets"][:draws]
    replay_error = float(np.max(np.abs(regrets(boot_error) - prior_regrets)))
    if replay_error > 1e-9:
        raise ValueError("Prior source-bootstrap error regrets changed.")
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
    points = [np.array([a[i] - a[j] for i, j in pairs]) for a in [error_point, work_point]]
    differences = [np.stack([a[:, i] - a[:, j] for i, j in pairs], axis=1)
                   for a in [boot_error, boot_work]]
    standard_deviations = [a.std(axis=0, ddof=1) for a in differences]
    maxima = np.zeros(draws)
    for b, p, sd in zip(differences, points, standard_deviations):
        valid = sd > 1e-10
        standardized = np.zeros_like(b)
        np.divide(b - p, sd, out=standardized, where=valid)
        maxima = np.maximum(maxima, np.abs(standardized).max(axis=(1, 2)))
    critical = float(np.quantile(maxima, .95))
    lower = [p - critical * sd for p, sd in zip(points, standard_deviations)]
    upper = [p + critical * sd for p, sd in zip(points, standard_deviations)]
    regret_low = np.zeros((2, len(names)))
    regret_high = np.zeros((2, len(names)))
    for component in range(2):
        for p, (i, j) in enumerate(pairs):
            regret_low[component, i] = max(regret_low[component, i], lower[component][p].max())
            regret_high[component, i] = max(regret_high[component, i], upper[component][p].max())
            regret_low[component, j] = max(regret_low[component, j], (-upper[component][p]).max())
            regret_high[component, j] = max(regret_high[component, j], (-lower[component][p]).max())
    error_regret, work_regret = regrets(error_point), regrets(work_point)
    joint_regret = np.maximum(error_regret, work_regret)
    base = next(iter(ALLOCATIONS.values()))
    before_hist = np.bincount(base.before, minlength=base.maximum + 1) / base.responses
    before_curve = np.cumsum(1 - np.cumsum(before_hist[:-1]))
    no_work_error_regret = float((before_curve - error_point.min(axis=0)).max())
    no_work_check = {
        "scope": "Fixed-benchmark point comparison after adding the unchanged-response, zero-check procedure; no population assertion.",
        "initial_mean_errors": float(np.dot(np.arange(len(before_hist)), before_hist)),
        "no_work_error_and_joint_regret": no_work_error_regret,
        "positive_work_joint_regrets": {n: float(max(error_regret[j], work_point[j, -1])) for j, n in enumerate(names)},
        "no_work_minimizes_joint_regret": bool(no_work_error_regret < min(max(error_regret[j], work_point[j, -1]) for j in range(len(names)))),
        "reason": "With a zero-work comparator, worst normalized work regret is mean checks. The full theta range includes cost-only loss and linear work cost.",
    }
    summaries = {n: {**point_outcomes[j][2],
                     "error_regret": float(error_regret[j]), "work_regret": float(work_regret[j]),
                     "joint_regret": float(joint_regret[j]),
                     "error_bounds": [float(regret_low[0, j]), float(regret_high[0, j])],
                     "work_bounds": [float(regret_low[1, j]), float(regret_high[1, j])],
                     "joint_bounds": [float(regret_low[:, j].max()), float(regret_high[:, j].max())]}
                 for j, n in enumerate(names)}
    np.savez_compressed(args.out / "curves.npz", error_point=error_point, work_point=work_point,
                        pairs=pairs, error_differences=points[0], work_differences=points[1],
                        error_lower=lower[0], error_upper=upper[0], work_lower=lower[1], work_upper=upper[1])
    np.savez_compressed(args.out / "bootstrap_summaries.npz", error_regret=regrets(boot_error),
                        work_regret=regrets(boot_work), max_statistics=maxima)
    shutil.copy2(args.error_results / "source_weights.npz", args.out / "source_weights.npz")
    result = {"methods": names, "responses": len(index), "sources": len(source_ids), "draws": draws,
              "primary_budget": .2, "maximum_error_cap": error_point.shape[1],
              "maximum_work_cap": work_point.shape[1], "critical_max_t": critical,
              "summaries": summaries, "minimum_error_regret_methods": [names[i] for i in np.flatnonzero(error_regret == error_regret.min())],
              "minimum_joint_regret_methods": [names[i] for i in np.flatnonzero(joint_regret == joint_regret.min())],
              "maximum_prior_error_regret_replay_error": replay_error,
              "zero_work_sanity_check": no_work_check,
              "normalization": "Independent concave error/work shapes each have unit first increment; combined loss is (1-theta)*error + theta*work, theta in [0,1]. Joint regret is max(error regret,work regret).",
              "scope": "Fixed five-procedure set; ideal annotation coverage, one check per selected word; common count-based concave losses/costs. Human review time is not measured.",
              "sampling": "Paired source draws from the prior experiment; recompute global allocation and integrate cutoff ties before error and work losses. Scores remain fixed.",
              "uncertainty": "Approximate simultaneous 95% source-bootstrap max-t bands over both outcome components, all listed pairs and all integer caps.",
              "scores_sha256": digest(args.scores), "index_sha256": digest(args.index),
              "source_weights_sha256": digest(args.error_results / "source_weights.npz"),
              "script_sha256": digest(__file__), "elapsed_seconds": time.monotonic() - started}
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"summaries": summaries, "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
