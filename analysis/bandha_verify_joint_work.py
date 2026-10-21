"""Verify work expectations and joint-regret identities independently."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom
from bandha_cap_evidence import regrets
from bandha_joint_work import JointAllocation, selected_tie_law


def exhaustive(scores, labels, offsets, sources, weights):
    copies = [r for r in range(len(sources)) for _ in range(weights[sources[r]])]
    ss = np.concatenate([scores[offsets[r]:offsets[r + 1]] for r in copies])
    yy = np.concatenate([labels[offsets[r]:offsets[r + 1]] for r in copies])
    oo = np.r_[0, np.cumsum([offsets[r + 1] - offsets[r] for r in copies])]
    k = round(.2 * len(ss))
    threshold = np.sort(ss)[-k]
    base = ss > threshold
    ties = np.flatnonzero(ss == threshold)
    chosen = k - int(base.sum())
    e, w = [], []
    for subset in itertools.combinations(ties, chosen):
        selected = base.copy()
        selected[list(subset)] = True
        errors = [yy[a:b][~selected[a:b]].sum() for a, b in zip(oo[:-1], oo[1:])]
        checks = [selected[a:b].sum() for a, b in zip(oo[:-1], oo[1:])]
        e.append([np.minimum(errors, t).mean() for t in range(1, 4)])
        w.append([np.minimum(checks, t).mean() for t in range(1, 4)])
    return np.mean(e, axis=0), np.mean(w, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    args = ap.parse_args()
    checks = 0
    for size, population, chosen in [(1, 1, 1), (12, 60, 2), (157, 157, 143),
                                     (350, 59082, 17422), (900, 1500, 1300)]:
        lo, p = selected_tie_law(size, population, chosen)
        direct = hypergeom.pmf(np.arange(lo, lo + len(p)), population, size, chosen)
        assert np.max(np.abs(p - direct)) < 1e-10
        assert np.isclose(np.dot(np.arange(lo, lo + len(p)), p), chosen * size / population, atol=1e-8)
        checks += 2
    labels = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0])
    offsets = np.arange(0, 13, 3)
    sources = np.array([0, 0, 1, 1])
    rng = np.random.default_rng(612004)
    for scores in [np.ones(12), np.repeat([4, 2, 3, 1], 3), rng.integers(0, 3, 12)]:
        for weights in [np.array([1, 1]), np.array([0, 2]), np.array([2, 0])]:
            a = JointAllocation(scores, labels, offsets, sources)
            e, w, _ = a.joint(weights)
            direct_e, direct_w = exhaustive(scores, labels, offsets, sources, weights)
            assert np.allclose(e, direct_e, atol=1e-12)
            assert np.allclose(w, direct_w, atol=1e-12)
            checks += 2
    report = json.loads((args.results / "results.json").read_text())
    curves = np.load(args.results / "curves.npz")
    ep, wp = curves["error_point"], curves["work_point"]
    names = report["methods"]
    with np.load(args.inputs / "trained_scores.npz", allow_pickle=False) as z:
        oo = z["response_offsets"]
        word_responses = np.repeat(np.arange(len(oo) - 1), np.diff(oo))
        k = round(.2 * len(z["labels"]))
        for name in ["lettuce", "lettuce_mean"]:
            selected = np.zeros(len(z[name]), dtype=bool)
            selected[np.argsort(-z[name], kind="stable")[:k]] = True
            work = np.bincount(word_responses[selected], minlength=len(oo) - 1)
            direct = np.array([np.minimum(work, t).mean() for t in range(1, wp.shape[1] + 1)])
            assert np.allclose(direct, wp[names.index(name)], atol=1e-10)
            checks += len(direct)
        before = np.add.reduceat(z["labels"], oo[:-1])
        before_curve = np.array([np.minimum(before, t).mean() for t in range(1, ep.shape[1] + 1)])
    er, wr = regrets(ep), regrets(wp)
    joint = np.maximum(er, wr)
    for j, name in enumerate(names):
        assert np.isclose(joint[j], report["summaries"][name]["joint_regret"])
        vertex = ep if er[j] >= wr[j] else wp
        t = int(np.argmax(vertex[j] - vertex.min(axis=0)))
        assert np.isclose(joint[j], vertex[j, t] - vertex[:, t].min())
        checks += 2
    for _ in range(100):
        ae, ak = rng.dirichlet(np.ones(ep.shape[1])), rng.dirichlet(np.ones(wp.shape[1]))
        theta = rng.random()
        loss = (1 - theta) * (ep @ ae) + theta * (wp @ ak)
        assert np.all(loss - loss.min() <= joint + 1e-10)
        checks += len(names)
    de = (ep[:, None, :] - ep[None, :, :]).max(axis=2)
    dk = (wp[:, None, :] - wp[None, :, :]).max(axis=2)
    for price in [0., .1, 1., 10.]:
        for j in range(len(names)):
            values = de[j] + price * dk[j]
            i = int(np.argmax(values))
            t = int(np.argmax(ep[j] - ep[i]))
            u = int(np.argmax(wp[j] - wp[i]))
            losses = ep[:, t] + price * wp[:, u]
            assert np.isclose(losses[j] - losses.min(), values[i], atol=1e-10)
            checks += 1
    augmented_error = np.vstack([ep, before_curve])
    augmented_work = np.vstack([wp, np.zeros(wp.shape[1])])
    augmented_joint = np.maximum(regrets(augmented_error), regrets(augmented_work))
    sanity = report["zero_work_sanity_check"]
    for j, name in enumerate(names):
        assert np.isclose(augmented_joint[j], sanity["positive_work_joint_regrets"][name])
        checks += 1
    assert np.isclose(augmented_joint[-1], sanity["no_work_error_and_joint_regret"])
    result = {"all_checks_passed": True, "checks": checks + 1, "exhaustive_allocation_fixtures": 9,
              "source_bootstrap_error_replay_maximum": report["maximum_prior_error_regret_replay_error"],
              "zero_work_sanity_verified": True}
    (args.results / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
