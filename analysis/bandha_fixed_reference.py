"""Compare frozen procedures with one fixed reference without mixing units."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bandha_cap_evidence import Allocation, digest
from bandha_render_cap_evidence import LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.results / "results.json").read_text())
    names = report["methods"]
    reference = names.index("lettuce_mean")
    curves = np.load(args.results / "curves.npz")
    summaries = {}
    for j, name in enumerate(names):
        row = {key: value for key, value in report["summaries"][name].items()
               if key not in ["error_regret", "work_regret", "joint_regret", "joint_bounds"]}
        for key in ["error", "work"]:
            difference = curves[key + "_point"][j] - curves[key + "_point"][reference]
            if j == reference:
                lower = upper = np.zeros_like(difference)
            else:
                i, k = sorted([j, reference])
                pair = curves["pairs"].tolist().index([i, k])
                lo, hi = curves[key + "_lower"][pair], curves[key + "_upper"][pair]
                lower, upper = (lo, hi) if j < reference else (-hi, -lo)
            row[key + "_disadvantage"] = float(difference.max())
            row[key + "_bounds"] = [float(lower.max()), float(upper.max())]
        summaries[name] = row

    index = json.loads((args.inputs / "response_index.json").read_text())
    sources = sorted({row["source_id"] for row in index})
    lookup = {name: j for j, name in enumerate(sources)}
    response_sources = np.array([lookup[row["source_id"]] for row in index])
    saved = np.load(args.inputs / "trained_scores.npz")
    allocations = [Allocation(saved[n], saved["labels"], saved["response_offsets"], response_sources)
                   for n in names]
    weights = np.ones(len(sources), dtype=int)
    sweep = {}
    checks = 0
    for budget in [.1, .2, .3]:
        outcomes = [allocation.evaluate(weights, budget, return_histogram=True) for allocation in allocations]
        errors = np.array([outcome[0] for outcome in outcomes])
        differences = errors - errors[reference]
        values = differences.max(axis=1)
        rng = np.random.default_rng(20261021)
        for j in range(len(names)):
            for _ in range(30):
                coefficients = rng.dirichlet(np.ones(errors.shape[1]))
                assert differences[j] @ coefficients <= values[j] + 1e-10
                checks += 1
            assert np.isclose(differences[j, differences[j].argmax()], values[j])
            checks += 1
        if budget == .2:
            assert np.allclose(errors, curves["error_point"], atol=1e-10, rtol=0)
            checks += 1
        sweep[str(budget)] = {
            "selected_words": outcomes[0][1]["selected_units"],
            "mean_checks": outcomes[0][1]["selected_units"] / len(index),
            "error_disadvantages": {n: float(values[j]) for j, n in enumerate(names)},
            "full_error_dominance": bool(np.all(errors[names.index("lettuce")] <= errors + 1e-10)),
        }
    full = names.index("lettuce")
    for comparator in range(len(names)):
        if comparator == full:
            continue
        i, k = sorted([full, comparator])
        pair = curves["pairs"].tolist().index([i, k])
        upper = curves["error_upper"][pair] if full < comparator else -curves["error_lower"][pair]
        assert upper.max() < 0
        checks += 1
    scaled_joint = {
        str(scale): {n: max(report["summaries"][n]["error_regret"],
                           scale * report["summaries"][n]["work_regret"]) for n in names}
        for scale in [1., .01]
    }
    output = {
        "reference": "lettuce_mean", "reference_scope": "One frozen checkpoint mean control shared by all procedures; at each declared check budget, the same selection rule is applied to this reference.",
        "methods": names, "responses": len(index), "sources": len(sources),
        "primary_budget": .2, "summaries": summaries, "budget_sweep": sweep,
        "scaled_joint_sensitivity": scaled_joint, "zero_work_check": report["zero_work_sanity_check"],
        "uncertainty": "20% point: maxima of existing simultaneous source-bootstrap pairwise bounds. 10% and 30%: benchmark points only, no population interval.",
        "work_axis": "Mean selected-word checks; not measured review time and not equal concave preparation cost.",
        "source_results_sha256": digest(args.results / "results.json"),
        "curves_sha256": digest(args.results / "curves.npz"),
        "verification": {"checks_passed": checks, "saved_primary_curves_reproduced": True},
    }
    (args.out / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    gen = args.paper / "generated/fixed_reference"
    figdir = args.paper / "figures/fixed_reference"
    gen.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    values = [r"\newcommand{\Joint" + key + "}{" + str(value) + "}"
              for key, value in [("Responses", len(index)), ("Sources", len(sources)),
                                 ("Draws", report["draws"]), ("WorkCaps", report["maximum_work_cap"])]]
    (gen / "values.tex").write_text("\n".join(values) + "\n")
    table = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Procedure & Mean $E$ & Opened & $B^E$ [bounds] & $B^K$\\", r"\midrule"]
    for name in names:
        row = summaries[name]
        lo, hi = row["error_bounds"]
        table.append(f"{LABELS[name]} & {row['mean_remaining']:.3f} & {row['expected_opened_responses']:.0f} & {row['error_disadvantage']:.3f} [{lo:.3f}, {hi:.3f}] & {row['work_disadvantage']:.3f}" + r"\\")
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (gen / "readouts.tex").write_text("\n".join(table) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 3, figsize=(8, 3.2))
    pair = curves["pairs"].tolist().index([names.index("lettuce"), reference])
    for ax, key, title in zip(axes[:2], ["error", "work"], ["Remaining errors", "Checking work"]):
        x = np.arange(1, curves[key + "_point"].shape[1] + 1)
        ax.fill_between(x, curves[key + "_lower"][pair], curves[key + "_upper"][pair], color="#2a7f95", alpha=.18)
        ax.plot(x, curves[key + "_differences"][pair], color="#126779", lw=1.5)
        ax.axhline(0, color="#555555", lw=.7)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
        ax.set_xlabel("Error cap $t$" if key == "error" else "Work cap $u$")
        ax.set_title(title)
        ax.set_ylabel("Full minus mean control")
    for name in names:
        x = [row["mean_checks"] for row in sweep.values()]
        y = [row["error_disadvantages"][name] for row in sweep.values()]
        axes[2].plot(x, y, marker="o", ms=3, label=LABELS[name])
    axes[2].axhline(0, color="#555555", lw=.7)
    axes[2].set_xlabel("Mean word checks per response", fontsize=8)
    axes[2].set_ylabel("Worst error disadvantage $B^E$")
    axes[2].set_title("Fixed-reference check curve", fontsize=9)
    figure.legend(*axes[2].get_legend_handles_labels(), loc="lower center",
                  ncol=3, frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(rect=(0, .19, 1, 1))
    figure.savefig(figdir / "comparisons.pdf", bbox_inches="tight")
    figure.savefig(figdir / "comparisons.png", dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({"checks_passed": checks, "full_error_disadvantage": summaries["lettuce"]["error_disadvantage"], "full_error_bounds": summaries["lettuce"]["error_bounds"], "budget_sweep": sweep}))


if __name__ == "__main__":
    main()
