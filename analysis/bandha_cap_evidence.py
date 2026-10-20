"""Compare capped annotation residuals with source-clustered uncertainty."""
import argparse
from functools import lru_cache
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from scipy.stats import hypergeom


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Allocation:
    def __init__(self, scores, labels, offsets, response_sources):
        self.scores = scores
        self.labels = labels
        self.offsets = offsets
        self.response_sources = response_sources
        self.responses = len(offsets) - 1
        self.units = np.diff(offsets)
        word_responses = np.repeat(np.arange(self.responses), self.units)
        self.order = np.argsort(-scores, kind="stable")
        self.sorted_scores = scores[self.order]
        self.sorted_sources = response_sources[word_responses[self.order]]
        positives = np.flatnonzero(labels)
        positives = positives[np.argsort(-scores[positives], kind="stable")]
        self.positive_scores = scores[positives]
        self.positive_responses = word_responses[positives]
        self.before = np.bincount(self.positive_responses, minlength=self.responses)
        self.maximum = int(self.before.max())

    def evaluate(self, source_weights, budget, return_histogram=False):
        response_weights = source_weights[self.response_sources]
        denominator = response_weights.sum()
        sorted_weights = source_weights[self.sorted_sources]
        cumulative = np.cumsum(sorted_weights)
        selected = int(round(budget * int(cumulative[-1])))
        if not 0 < selected < cumulative[-1]:
            raise ValueError("Use an interior allocation budget.")
        position = np.searchsorted(cumulative, selected, side="left")
        threshold = self.sorted_scores[position]
        left = np.searchsorted(-self.sorted_scores, -threshold, side="left")
        right = np.searchsorted(-self.sorted_scores, -threshold, side="right")
        above = int(cumulative[left - 1]) if left else 0
        population = int(cumulative[right - 1]) - above
        chosen = selected - above
        pleft = np.searchsorted(-self.positive_scores, -threshold, side="left")
        pright = np.searchsorted(-self.positive_scores, -threshold, side="right")
        below = np.bincount(self.positive_responses[pright:], minlength=self.responses)
        tied = np.bincount(self.positive_responses[pleft:pright], minlength=self.responses)
        histogram = np.bincount(below, weights=response_weights, minlength=self.maximum + 1)
        if chosen == population:
            expected_tp = float(np.dot(self.before - below, response_weights))
        else:
            active = np.flatnonzero((tied > 0) & (response_weights > 0))
            np.add.at(histogram, below[active], -response_weights[active])
            states = {}
            for r in active:
                key = (int(below[r]), int(tied[r]))
                states[key] = states.get(key, 0) + int(response_weights[r])
            for (misses, errors), mass in states.items():
                probabilities = tie_probabilities(errors, population, chosen)
                histogram[misses:misses + errors + 1] += mass * probabilities
            expected_tp = float(np.dot(self.before - below - tied, response_weights)
                                + chosen / population * np.dot(tied, response_weights))
        if not np.isclose(histogram.sum(), denominator, atol=1e-7, rtol=0):
            raise ValueError("Residual histogram lost response mass.")
        tail = 1 - np.cumsum(histogram[:-1]) / denominator
        curve = np.cumsum(tail)
        if np.min(tail) < -1e-10:
            raise ValueError("Negative residual tail probability.")
        if not np.isclose(curve[-1], np.dot(np.arange(len(histogram)), histogram) / denominator,
                          atol=1e-9, rtol=0):
            raise ValueError("Capped endpoint differs from the mean.")
        originals = np.dot(self.before, response_weights)
        f1 = 2 * expected_tp / (originals + selected)
        meta = {"threshold": float(threshold), "selected_units": selected,
                "responses": int(denominator), "tie_population": population,
                "selected_ties": chosen, "word_f1": float(f1),
                "error_containing_fraction": float(curve[0]), "mean_remaining": float(curve[-1])}
        return (curve, meta, histogram / denominator) if return_histogram else curve


@lru_cache(maxsize=50000)
def tie_probabilities(errors, population, chosen):
    selected_errors = np.arange(errors + 1)
    pmf = hypergeom.pmf(selected_errors, population, errors, chosen)
    if not np.isclose(pmf.sum(), 1, atol=1e-8, rtol=0):
        raise ValueError("Hypergeometric probability calculation failed.")
    return pmf[::-1] / pmf.sum()


def regrets(curves):
    return (curves - curves.min(axis=-2, keepdims=True)).max(axis=-1)


def bootstrap_block(block):
    start, weights = block
    values = np.array([[a.evaluate(w, .2) for a in WORKER_ALLOCATIONS.values()] for w in weights])
    return start, values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    index = json.loads(args.index.read_text())
    source_ids = sorted({r["source_id"] for r in index})
    source_lookup = {s: i for i, s in enumerate(source_ids)}
    response_sources = np.array([source_lookup[r["source_id"]] for r in index])
    with np.load(args.scores, allow_pickle=False) as raw:
        names = [n for n in raw.files if n not in ["labels", "response_offsets"]]
        allocations = {n: Allocation(raw[n], raw["labels"], raw["response_offsets"], response_sources)
                       for n in names}
    maximum = max(a.maximum for a in allocations.values())
    weights = np.ones(len(source_ids), dtype=np.int16)
    point, metadata, histograms, budget_sweep = {}, {}, {}, {}
    for budget in [.1, .2, .3]:
        c, m = [], {}
        for name, allocation in allocations.items():
            curve, meta, hist = allocation.evaluate(weights, budget, return_histogram=True)
            c.append(np.pad(curve, (0, maximum - len(curve)), constant_values=curve[-1]))
            m[name] = meta
            if budget == .2:
                histograms[name] = hist.tolist()
        budget_sweep[str(budget)] = {"metadata": m, "regrets": dict(zip(names, regrets(np.array(c)).tolist()))}
        if budget == .2:
            point = np.array(c)
            metadata = m
    rng = np.random.default_rng(args.seed)
    draws = rng.multinomial(len(source_ids), np.full(len(source_ids), 1 / len(source_ids)),
                            size=args.draws).astype(np.int16)
    np.savez_compressed(args.out / "source_weights.npz", weights=draws, source_ids=source_ids)
    boot = np.empty((args.draws, len(names), maximum))
    global WORKER_ALLOCATIONS
    WORKER_ALLOCATIONS = allocations
    blocks = [(start, draws[start:start + 125]) for start in range(0, args.draws, 125)]
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        completed = 0
        for start, values in pool.imap_unordered(bootstrap_block, blocks):
            boot[start:start + len(values)] = values
            completed += len(values)
            if completed % 500 == 0 or completed == args.draws:
                print(json.dumps({"completed_draws": completed, "draws": args.draws,
                                  "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    np.save(args.out / "bootstrap_curves.npy", boot)
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
    point_differences = np.array([point[i] - point[j] for i, j in pairs])
    boot_differences = np.stack([boot[:, i] - boot[:, j] for i, j in pairs], axis=1)
    sd = boot_differences.std(axis=0, ddof=1)
    valid = sd > 1e-12
    standardized = np.zeros_like(boot_differences)
    np.divide(boot_differences - point_differences, sd, out=standardized, where=valid)
    maxima = np.abs(standardized).max(axis=(1, 2))
    critical = float(np.quantile(maxima, .95))
    lower, upper = point_differences - critical * sd, point_differences + critical * sd
    comparison = {}
    regret_lower = np.zeros(len(names))
    regret_upper = np.zeros(len(names))
    for p, (i, j) in enumerate(pairs):
        d = point_differences[p]
        nz = np.flatnonzero(np.abs(d) > 1e-10)
        crossings = [[int(a + 1), int(b + 1)] for a, b in zip(nz[:-1], nz[1:]) if d[a] * d[b] < 0]
        comparison[names[i] + "__minus__" + names[j]] = {
            "minimum": float(d.min()), "maximum": float(d.max()), "sign_changes": crossings,
            "dominance_i_fixed_benchmark": bool(d.max() <= 1e-10),
            "dominance_j_fixed_benchmark": bool(d.min() >= -1e-10),
            "dominance_i_simultaneous_upper_nonpositive": bool(upper[p].max() <= 0),
            "dominance_j_simultaneous_lower_nonnegative": bool(lower[p].min() >= 0),
            "maximum_disadvantage_interval_i": [float(lower[p].max()), float(upper[p].max())]}
        regret_lower[i] = max(regret_lower[i], lower[p].max())
        regret_upper[i] = max(regret_upper[i], upper[p].max())
        regret_lower[j] = max(regret_lower[j], (-upper[p]).max())
        regret_upper[j] = max(regret_upper[j], (-lower[p]).max())
    boot_regret = regrets(boot)
    np.savez_compressed(args.out / "bootstrap_summaries.npz", regrets=boot_regret, max_statistics=maxima)
    np.savez_compressed(args.out / "curves.npz", point=point, pairs=pairs,
                        differences=point_differences, lower=lower, upper=upper)
    overall = regrets(point)
    summaries = {n: {"regret": float(overall[j]),
                     "regret_simultaneous_interval": [float(regret_lower[j]), float(regret_upper[j])],
                     "bootstrap_minimum_regret_frequency": float((boot_regret.argmin(axis=1) == j).mean()),
                     **metadata[n]} for j, n in enumerate(names)}
    previous = [names.index(n) for n in ["generator", "lexical", "lexical_metadata"] if n in names]
    original = dict(zip([names[i] for i in previous], regrets(point[previous]).tolist()))
    results = {"methods": names, "responses": len(index), "sources": len(source_ids),
               "maximum_error_count": maximum, "primary_budget": .2, "draws": args.draws,
               "seed": args.seed, "critical_max_t": critical, "summaries": summaries,
               "original_three_method_regrets": original, "comparisons": comparison,
               "budget_sweep": budget_sweep, "histograms": histograms,
               "inference": "Approximate simultaneous 95% paired source-bootstrap max-t bands across all listed pairs and integer caps.",
               "policy_resampling": "Recompute the exact global 20% allocation, cutoff and uniform cutoff-tie law within every source resample; keep trained scores fixed.",
               "tie_scope": "Marginal hypergeometric probabilities integrated before nonlinear loss; shared global ties need no independence assumption.",
               "fit_resampling": False, "correction_scope": "Ideal targeted correction of annotated whitespace words; no actual edited outputs.",
               "scores_sha256": digest(args.scores), "index_sha256": digest(args.index),
               "source_weights_sha256": digest(args.out / "source_weights.npz"),
               "script_sha256": digest(__file__), "elapsed_seconds": time.monotonic() - started}
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"summaries": summaries, "elapsed_seconds": results["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
