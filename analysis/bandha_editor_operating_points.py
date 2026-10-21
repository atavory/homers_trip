"""Source uncertainty for marginal-bound editor frontiers at word budgets."""
import argparse
from functools import lru_cache
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np


def capped_means(means, weights, maximum):
    means = np.maximum(means, 0.)
    floor = np.floor(means).astype(int)
    fraction = means - floor
    histogram = (np.bincount(floor, weights=weights * (1 - fraction), minlength=maximum + 2)
                 + np.bincount(floor + 1, weights=weights * fraction, minlength=maximum + 2))
    return np.cumsum(1 - np.cumsum(histogram[:-1]) / weights.sum())[:maximum]


class Allocation:
    def __init__(self, scores, labels, offsets, response_sources, budgets, repair):
        self.sources = response_sources
        self.responses = len(offsets) - 1
        word_response = np.repeat(np.arange(self.responses), np.diff(offsets))
        order = np.argsort(-scores, kind="stable")
        self.scores = scores[order]
        self.word_sources = response_sources[word_response[order]]
        self.word_responses = word_response[order]
        positive_order = order[labels[order].astype(bool)]
        self.positive_scores = scores[positive_order]
        self.positive_responses = word_response[positive_order]
        self.original = np.bincount(word_response[labels.astype(bool)], minlength=self.responses)
        self.maximum = int(np.diff(offsets).max())
        self.budgets, self.repair = budgets, repair

    @lru_cache(maxsize=1000)
    def counts(self, threshold):
        left = np.searchsorted(-self.scores, -threshold, side="left")
        right = np.searchsorted(-self.scores, -threshold, side="right")
        pl = np.searchsorted(-self.positive_scores, -threshold, side="left")
        pr = np.searchsorted(-self.positive_scores, -threshold, side="right")
        above = np.bincount(self.word_responses[:left], minlength=self.responses)
        tied = np.bincount(self.word_responses[left:right], minlength=self.responses)
        tp = np.bincount(self.positive_responses[:pl], minlength=self.responses)
        tied_tp = np.bincount(self.positive_responses[pl:pr], minlength=self.responses)
        return left, right, tp, above - tp, tied_tp, tied - tied_tp

    def frontier(self, base, fp, weights, reference):
        def disadvantage(q):
            return float((capped_means(base + q * fp, weights, self.maximum) - reference).max())
        if disadvantage(1.) <= 1e-11:
            return 1.
        lo, hi = 0., 1.
        for _ in range(26):
            mid = (lo + hi) / 2
            if disadvantage(mid) <= 1e-11:
                lo = mid
            else:
                hi = mid
        return lo

    def evaluate(self, source_weights):
        weights = source_weights[self.sources]
        cumulative = np.cumsum(source_weights[self.word_sources])
        word_total = int(cumulative[-1])
        reference = capped_means(self.original, weights, self.maximum)
        rows = []
        for budget in self.budgets:
            selected = round(budget * word_total)
            pos = np.searchsorted(cumulative, selected, side="left")
            threshold = float(self.scores[pos])
            left, right, tp, fp, tied_tp, tied_fp = self.counts(threshold)
            above = int(cumulative[left - 1]) if left else 0
            pool = int(cumulative[right - 1]) - above
            share = (selected - above) / pool
            TP, FP = tp + share * tied_tp, fp + share * tied_fp
            assert np.isclose(np.dot(weights, TP + FP), selected, atol=1e-7, rtol=0)
            q = self.frontier(self.original - self.repair * TP, FP, weights, reference)
            rows.append([q, float(np.dot(weights, TP) / selected), selected])
        return np.array(rows)


def block(args):
    start, weights = args
    return start, np.array([ALLOC.evaluate(w)[:, 0] for w in weights])


def coupling_comparison(path):
    from bandha_editor_frontier import OutcomeModel
    triplets = np.load(path)
    result = {}
    for name in triplets.files:
        model = OutcomeModel(triplets[name])
        def curve(p, q):
            f, t, a = model.fn, model.tp, model.fp
            return sum(prob * capped_means(counts, model.weights, model.maximum)
                       for counts, prob in [(f, p * (1-q)), (f+a, p*q),
                                            (f+t, (1-p)*(1-q)), (f+t+a, (1-p)*q)])
        checks = 0
        for p in [.5, .9, 1.]:
            for q in [0., .02, .1]:
                model.evaluate(p, q)
                coupled = curve(p, q)
                assert np.all(coupled <= model.last_curve + 1e-9)
                assert abs(coupled[-1] - model.last_curve[-1]) < 1e-9
                checks += 2
        independent = model.evaluate(.9, .02)
        difference = curve(.9, .02) - model.reference
        lo, hi = 0., 1.
        for _ in range(32):
            mid = (lo + hi) / 2
            if (curve(.9, mid) - model.reference).max() <= 1e-11:
                lo = mid
            else:
                hi = mid
        result[name] = {"independent_p09_q002": independent,
                        "grouped_p09_q002_B": float(difference.max()),
                        "grouped_frontier_p09": lo,
                        "grouped_frontier_limiting_cap": int((curve(.9, hi) - model.reference).argmax() + 1),
                        "checks": checks,
                        "model": "All TP repair together; all FP corrupt together; the two group events are independent. Matched marginal p,q; benchmark points, not measured editor behavior."}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--source-draws", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repair-lower", type=float, default=.9)
    args = parser.parse_args()
    if not 0 <= args.repair_lower <= 1:
        parser.error("Repair lower bound must be between zero and one.")
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    index = json.loads((args.inputs / "response_index.json").read_text())
    sources = sorted({r["source_id"] for r in index})
    lookup = {s: i for i, s in enumerate(sources)}
    response_sources = np.array([lookup[r["source_id"]] for r in index])
    budgets = [.01, .02, .05, .1, .2, .3, .5]
    scores = np.load(args.inputs / "trained_scores.npz")
    global ALLOC
    ALLOC = Allocation(scores["lettuce"], scores["labels"], scores["response_offsets"],
                       response_sources, budgets, args.repair_lower)
    source_draws = np.load(args.source_draws)
    assert list(source_draws["source_ids"]) == sources
    weights = source_draws["weights"]
    point = ALLOC.evaluate(np.ones(len(sources), dtype=int))
    rng = np.random.default_rng(20261021)
    for _ in range(30):
        means = rng.uniform(0, ALLOC.maximum, 12)
        w = rng.integers(1, 5, len(means))
        direct = np.average(np.minimum(means[:, None], np.arange(1, ALLOC.maximum + 1)), weights=w, axis=0)
        assert np.allclose(capped_means(means, w, ALLOC.maximum), direct, atol=1e-10, rtol=0)
    bootstrap = np.empty((len(weights), len(budgets)))
    blocks = [(i, weights[i:i + 100]) for i in range(0, len(weights), 100)]
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        completed = 0
        for start, values in pool.imap_unordered(block, blocks):
            bootstrap[start:start + len(values)] = values
            completed += len(values)
            if completed % 1000 == 0 or completed == len(weights):
                print(json.dumps({"completed": completed, "draws": len(weights), "seconds": round(time.monotonic() - started, 1)}), flush=True)
    intervals = np.quantile(bootstrap, [.025, .975], axis=0).T
    rows = {str(b): {"q_sufficient": float(point[i, 0]), "source_interval": intervals[i].tolist(),
                     "precision": float(point[i, 1]), "selected_words": int(point[i, 2])}
            for i, b in enumerate(budgets)}
    delta = bootstrap[:, 0] - bootstrap[:, 1]
    result = {"repair_lower_bound": args.repair_lower, "reference": "Unchanged response",
              "budgets": rows, "sources": len(sources), "responses": len(index), "draws": len(weights),
              "maximum_cap": ALLOC.maximum,
              "one_percent_minus_two_percent": {"point": float(point[0, 0] - point[1, 0]), "source_interval": np.quantile(delta, [.025, .975]).tolist()},
              "scope": "Numerical sufficient frontier for benchmark-average concave error loss under local editing and marginal repair/harm bounds; no independence model and no measured editor. Approximate pointwise source-bootstrap intervals conditional on frozen detector and specified repair lower bound.",
              "tie_handling": "Expected confusion counts under uniform cutoff-tie allocation, including source clones; no hypergeometric distribution is needed for this Jensen bound.",
              "direct_capping_checks": 30,
              "scores_sha256": hashlib.sha256((args.inputs / "trained_scores.npz").read_bytes()).hexdigest(),
              "source_draws_sha256": hashlib.sha256(args.source_draws.read_bytes()).hexdigest(),
              "seconds": time.monotonic() - started}
    triplets = args.inputs.parent / "bandha_editor_frontier_results/triplets.npz"
    if triplets.exists():
        result["coupling_comparison"] = coupling_comparison(triplets)
    delta = bootstrap[:, budgets.index(.05)] - bootstrap[:, budgets.index(.2)]
    result["five_percent_minus_twenty_percent"] = {
        "point": float(point[budgets.index(.05), 0] - point[budgets.index(.2), 0]),
        "source_interval": np.quantile(delta, [.025, .975]).tolist(),
    }
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(args.out / "bootstrap.npz", q_sufficient=bootstrap, budgets=budgets)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    figure, ax = plt.subplots(figsize=(5.4, 3.2))
    x = 100 * np.array(budgets)
    ax.errorbar(x, 100 * point[:, 0], yerr=100 * np.array([point[:, 0] - intervals[:, 0], intervals[:, 1] - point[:, 0]]), fmt="o-", capsize=3)
    ax.set_xscale("log");ax.set_yscale("log")
    ax.set_xticks(x, [str(int(v)) for v in x])
    ax.set_xlabel("Word inspection budget (%)")
    ax.set_ylabel("Sufficient tolerated corruption q (%)")
    ax.set_title("Repair probability at least 0.9")
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout();figure.savefig(args.out / "operating_points.pdf", bbox_inches="tight")
    figure.savefig(args.out / "operating_points.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
