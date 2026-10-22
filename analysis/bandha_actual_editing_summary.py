"""Summarize saved, actually edited responses using their assessed outcomes."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ("original", "top20", "cutoff05")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    cases = json.loads((args.results / "cases.json").read_text())
    assessments = []
    duplicate_disagreements = []
    operations = collections.defaultdict(list)
    for case in cases:
        assessment = json.loads((args.results / "assessments" / (case["id"] + ".json")).read_text())
        texts = {"original": case["original"]}
        for policy in METHODS[1:]:
            output = json.loads((args.results / "outputs" / (case["id"] + "_" + policy + ".json")).read_text())
            texts[policy] = output["response"]
            operations[policy].append({"model_called": output["model_called"],
                "text_changed": output["response"] != case["original"],
                "returned_replacements": len(output["replacements"]),
                "flagged_words": case["flagged_words"][policy]})
        mapping = assessment["candidate_mapping"]
        canonical = {}
        for candidate in sorted(mapping):
            policy = mapping[candidate]
            canonical.setdefault(texts[policy], policy)
        canonical[case["original"]] = "original"
        normalized = {}
        for method in METHODS:
            chosen = canonical[texts[method]]
            outcome = assessment["outcomes"][chosen]
            raw = assessment["outcomes"][method]
            if any(raw[k] != outcome[k] for k in ("errors", "uncertain_claims", "preservation_errors")):
                duplicate_disagreements.append({"id": case["id"], "method": method,
                    "canonical_method": chosen, "raw_errors": raw["errors"], "canonical_errors": outcome["errors"]})
            normalized[method] = outcome
        assessments.append({"id": case["id"], "outcomes": normalized})
    errors = np.array([[a["outcomes"][m]["errors"] for a in assessments] for m in METHODS])
    uncertain = np.array([[a["outcomes"][m]["uncertain_claims"] for a in assessments] for m in METHODS])
    preservation = np.array([[a["outcomes"][m]["preservation_errors"] for a in assessments] for m in METHODS])
    unsupported = np.array([[a["outcomes"][m]["unsupported_claims"] for a in assessments] for m in METHODS])
    cap = np.arange(1, max(1, int((errors + uncertain).max())) + 1)
    capped = np.minimum(errors[:, :, None], cap)
    curves = capped.mean(axis=1)
    rng = np.random.default_rng(20261022)
    strata = np.array([c["stratum"] for c in cases])
    selected = np.concatenate([rng.choice(np.flatnonzero(strata == s),
        size=(args.draws, int((strata == s).sum())), replace=True) for s in sorted(set(strata))], axis=1)
    bootstrap = capped[:, selected, :].mean(axis=2)
    factual_capped = np.minimum(unsupported[:, :, None], cap)
    factual_curves = factual_capped.mean(axis=1)
    factual_bootstrap = factual_capped[:, selected, :].mean(axis=2)
    original_gold = np.array([c["gold_original_has_error"] for c in cases])
    rows = {}
    for i, method in enumerate(METHODS):
        delta = errors[i] - errors[0]
        bad_delta = (errors[i] > 0).astype(int) - (errors[0] > 0)
        excess = curves[i] - curves[0]
        bdraw = (bootstrap[i] - bootstrap[0]).max(axis=1)
        factual_delta = unsupported[i] - unsupported[0]
        factual_bdraw = (factual_bootstrap[i] - factual_bootstrap[0]).max(axis=1)
        rows[method] = {"mean_errors": float(errors[i].mean()),
            "mean_unsupported_claims": float(unsupported[i].mean()),
            "fraction_with_unsupported_claims": float((unsupported[i] > 0).mean()),
            "fraction_with_errors": float((errors[i] > 0).mean()),
            "mean_uncertain_claims": float(uncertain[i].mean()),
            "mean_preservation_errors": float(preservation[i].mean()),
            "mean_error_change": float(delta.mean()),
            "mean_error_change_interval": np.quantile(delta[selected].mean(axis=1), [.025, .975]).tolist(),
            "bad_response_fraction_change": float(bad_delta.mean()),
            "bad_response_change_interval": np.quantile(bad_delta[selected].mean(axis=1), [.025, .975]).tolist(),
            "B_error": float(excess.max()),
            "B_error_interval": np.quantile(bdraw, [.025, .975]).tolist(),
            "factual_mean_change": float(factual_delta.mean()),
            "factual_mean_change_interval": np.quantile(factual_delta[selected].mean(axis=1), [.025, .975]).tolist(),
            "B_factual": float((factual_curves[i] - factual_curves[0]).max()),
            "B_factual_interval": np.quantile(factual_bdraw, [.025, .975]).tolist(),
            "caps": cap.tolist(), "L": curves[i].tolist(),
            "L_unsupported_only": np.minimum(unsupported[i, :, None], cap).mean(axis=0).tolist(),
            "L_uncertainty_included": np.minimum((errors + uncertain)[i, :, None], cap).mean(axis=0).tolist()}
        if i:
            rows[method]["operations"] = {k: sum(x[k] for x in operations[method]) for k in operations[method][0]}
            load = np.array([x["flagged_words"] for x in operations[method]])
            work_caps = np.arange(1, max(1, int(load.max())) + 1)
            rows[method]["work_caps"] = work_caps.tolist()
            rows[method]["W_selected_word_load"] = np.minimum(load[:, None], work_caps).mean(axis=0).tolist()
            rows[method]["W_model_requests_at_cap1"] = rows[method]["operations"]["model_called"] / len(cases)
    summary = {"responses": len(cases), "distinct_sources": len({c["source_id"] for c in cases}),
        "methods": rows, "bootstrap_draws": args.draws,
        "bootstrap_scope": "Approximate stratified source percentile intervals on a balanced pilot; excludes assessor uncertainty and repeated-generation variability",
        "error_unit": "Assessor-enumerated unsupported atomic propositions plus omitted supported original targets; ambiguous claims reported separately",
        "baseline_gold_agreement": float(((errors[0] > 0) == original_gold).mean()),
        "baseline_gold_agreement_by_stratum": {s: float(((errors[0] > 0) == original_gold)[strata == s].mean()) for s in sorted(set(strata))},
        "duplicate_assessment_disagreements": duplicate_disagreements,
        "identity_rule": "Unchanged texts use the original assessment; other identical texts use their first blinded candidate assessment. Raw assessment disagreements retained.",
        "model_scope": "Actual GPT-4.1 span replacements; independent Claude Sonnet 4.6 automatic assessment, not human post-edit gold labels"}
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.results / "normalized_assessments.json").write_text(json.dumps(assessments, indent=2, ensure_ascii=False) + "\n")
    np.savez_compressed(args.results / "bootstrap.npz", selected=selected, cap=cap, error_counts=errors,
                        uncertain_counts=uncertain, preservation_counts=preservation, curves=curves, bootstrap_curves=bootstrap)
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    labels = ("Unedited", "Top 20% flags", "Word score at least .5")
    for i, label in enumerate(labels):
        axes[0].plot(cap, curves[i], marker="o", markersize=3, label=label, color=f"C{i}")
        if i:
            interval = np.quantile(bootstrap[i] - bootstrap[0], [.025, .975], axis=0)
            line = axes[1].plot(cap, curves[i] - curves[0], marker="o", markersize=3, label=label, color=f"C{i}")[0]
            axes[1].fill_between(cap, *interval, color=line.get_color(), alpha=.18)
    axes[0].set_ylabel("Mean capped factuality/preservation events")
    axes[1].set_ylabel("Difference from unedited")
    axes[1].axhline(0, color="black", linewidth=.7)
    for axis in axes:
        axis.set_xlabel("Error cap")
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    figure.suptitle("Executed correction pilot: automated outcome assessment", fontsize=11)
    figure.tight_layout()
    figure.savefig(args.results / "actual_outcomes.pdf")
    figure.savefig(args.results / "actual_outcomes.png", dpi=160)
    for axis in axes:
        axis.clear()
    for i, label in enumerate(labels):
        axes[0].plot(cap, factual_curves[i], marker="o", markersize=3, label=label, color=f"C{i}")
        if i:
            interval = np.quantile(factual_bootstrap[i] - factual_bootstrap[0], [.025, .975], axis=0)
            line = axes[1].plot(cap, factual_curves[i] - factual_curves[0], marker="o", markersize=3, label=label, color=f"C{i}")[0]
            axes[1].fill_between(cap, *interval, color=line.get_color(), alpha=.18)
    axes[0].set_ylabel("Mean capped unsupported claims")
    axes[1].set_ylabel("Difference from unedited")
    axes[1].axhline(0, color="black", linewidth=.7)
    for axis in axes:
        axis.set_xlabel("Error cap")
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    figure.suptitle("Executed correction pilot: factuality proxy", fontsize=11)
    figure.tight_layout()
    figure.savefig(args.results / "factual_outcomes.pdf")
    figure.savefig(args.results / "factual_outcomes.png", dpi=160)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
