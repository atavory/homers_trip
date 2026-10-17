"""Run response-only controls through pinned span evaluation functions."""

import argparse
import ast
import collections
import csv
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import spearmanr


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def released_function(path, name):
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    env = {"np": np, "spearmanr": spearmanr, "tqdm": lambda iterable, **kwargs: iterable}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    return env[name], hashlib.sha256(ast.dump(node).encode()).hexdigest()


def load_ragtruth(root):
    source_file = root / "dataset/source_info.jsonl"
    response_file = root / "dataset/response.jsonl"
    with source_file.open() as handle:
        sources = {row["source_id"]: row for row in map(json.loads, handle)}
    rows = []
    source_splits = collections.defaultdict(set)
    with response_file.open() as handle:
        for raw in map(json.loads, handle):
            text = raw["response"]
            spans = sorted((label["start"], label["end"]) for label in raw["labels"])
            assert all(0 <= a < b <= len(text) for a, b in spans)
            merged = []
            for start, end in spans:
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((start, end))
            source = sources[raw["source_id"]]
            is_validation = int(hashlib.sha256(("homers-dream-dev-v1:" + raw["source_id"]).encode()).hexdigest()[:8], 16) % 5 == 0
            split = "test" if raw["split"] == "test" else "validation" if is_validation else "train"
            source_splits[raw["source_id"]].add(raw["split"])
            rows.append({
                "id": raw["id"], "source_id": raw["source_id"], "split": split,
                "quality": raw["quality"], "text": text, "labels": raw["labels"],
                "key": (source["task_type"], raw["model"]),
                "n": len(text), "g": sum(b - a for a, b in spans),
                "gold_union_characters": sum(b - a for a, b in merged),
            })
    assert all(len(values) == 1 for values in source_splits.values())
    return rows, {"response_sha256": sha(response_file), "source_sha256": sha(source_file)}


def train_rates(rows):
    fit = [row for row in rows if row["split"] == "train" and row["quality"] == "good" and row["n"]]
    totals = collections.defaultdict(lambda: [0, 0, 0, 0])
    for row in fit:
        atom = totals[row["key"]]
        atom[0] += row["g"]
        atom[1] += row["n"]
        atom[2] += int(row["g"] > 0)
        atom[3] += 1
    prevalence = sum(row["g"] for row in fit) / sum(row["n"] for row in fit)
    any_rate = sum(row["g"] > 0 for row in fit) / len(fit)
    rates = {key: {"metadata_char": (g + 1000 * prevalence) / (n + 1000), "metadata_any": (marked + 10 * any_rate) / (count + 10)} for key, (g, n, marked, count) in totals.items()}
    return rates, {"train_responses": len(fit), "train_sources": len({row["source_id"] for row in fit}), "char_smoothing": 1000, "response_smoothing": 10, "char_prevalence": prevalence, "response_any_rate": any_rate}


def best_subset(rows, score):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[score(row)].append(row)
    gold_total = sum(row["g"] for row in rows)
    best = {"f1": Fraction(0), "threshold": math.inf, "ids": set()}
    selected = set()
    gold = length = 0
    for value in sorted(groups, reverse=True):
        for row in groups[value]:
            selected.add(row["id"])
            gold += row["g"]
            length += row["n"]
        f1 = Fraction(2 * gold, gold_total + length) if gold_total + length else Fraction(0)
        if f1 > best["f1"]:
            best = {"f1": f1, "threshold": value, "ids": set(selected)}
    return best


class WholeResponseDetector:
    def __init__(self, selected):
        self.selected = selected

    def predict_prompt(self, prompt, answer, output_format):
        assert output_format == "spans"
        return [{"start": 0, "end": len(answer)}] if prompt in self.selected else []


def score_selected(rows, selected, evaluator):
    samples = [SimpleNamespace(prompt=row["id"], answer=row["text"], labels=row["labels"]) for row in rows]
    result = evaluator(WholeResponseDetector(selected), samples)
    gold_total = sum(row["g"] for row in rows)
    marked_gold = sum(row["g"] for row in rows if row["id"] in selected)
    marked_length = sum(row["n"] for row in rows if row["id"] in selected)
    independent = {"precision": marked_gold / marked_length if marked_length else 0, "recall": marked_gold / gold_total if gold_total else 0, "f1": 2 * marked_gold / (gold_total + marked_length) if gold_total + marked_length else 0}
    assert all(abs(result[key] - independent[key]) < 1e-12 for key in independent)
    return {**result, "flagged_responses": len(selected), "flagged_characters": marked_length, "covered_gold_characters": marked_gold}


def run_mu(root, iou, correlation):
    results = []
    inputs = []
    for path in sorted((root / "splits/test_labeled/v1").glob("*.jsonl")):
        items = []
        with path.open() as handle:
            for raw in map(json.loads, handle):
                ref = {key: raw[key] for key in ("id", "hard_labels", "soft_labels")}
                ref["text_len"] = len(raw["model_output_text"])
                if not ref["text_len"]:
                    raise ValueError("Empty output needs explicit separate handling")
                for strategy in ("mark_none", "mark_all"):
                    predicted = {"id": ref["id"], "hard_labels": [], "soft_labels": []}
                    if strategy == "mark_all":
                        predicted.update(hard_labels=[[0, ref["text_len"]]], soft_labels=[{"start": 0, "end": ref["text_len"], "prob": 1.0}])
                    items.append({"strategy": strategy, "iou": iou(ref, predicted), "cor": correlation(ref, predicted), "hard_empty": not ref["hard_labels"]})
        for strategy in ("mark_none", "mark_all"):
            chosen = [item for item in items if item["strategy"] == strategy]
            results.append({"file": path.name, "strategy": strategy, "responses": len(chosen), "hard_empty_responses": sum(item["hard_empty"] for item in chosen), "iou_macro": float(np.mean([item["iou"] for item in chosen])), "cor_macro": float(np.mean([item["cor"] for item in chosen]))})
        inputs.append({"file": str(path.relative_to(root)), "sha256": sha(path)})
    assert inputs
    mixed = {"id": "fixture", "text_len": 2, "hard_labels": [[0, 1]], "soft_labels": [{"start": 0, "end": 1, "prob": 1.0}]}
    perfect = {"id": "fixture", "hard_labels": [[0, 1]], "soft_labels": [{"start": 0, "end": 1, "prob": 1.0}]}
    tiny = {"id": "fixture", "hard_labels": [], "soft_labels": [{"start": 0, "end": 1, "prob": 1e-10}]}
    clean = {"id": "fixture", "text_len": 2, "hard_labels": [], "soft_labels": []}
    all_positive = {"id": "fixture", "text_len": 2, "hard_labels": [[0, 2]], "soft_labels": [{"start": 0, "end": 2, "prob": 1.0}]}
    full = {"id": "fixture", "hard_labels": [[0, 2]], "soft_labels": [{"start": 0, "end": 2, "prob": 1.0}]}
    empty = {"id": "fixture", "hard_labels": [], "soft_labels": []}
    fixtures = {"clean_empty_iou": iou(clean, empty), "clean_full_iou": iou(clean, full), "clean_full_correlation": correlation(clean, full), "all_positive_empty_correlation": correlation(all_positive, empty), "perfect_correlation": correlation(mixed, perfect), "strict_positive_rescaling_correlation": correlation(mixed, tiny)}
    return results, inputs, fixtures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ragtruth", type=Path, required=True)
    parser.add_argument("--lettuce-evaluator", type=Path, required=True)
    parser.add_argument("--mu-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    evaluator, lettuce_ast = released_function(args.lettuce_evaluator, "evaluate_detector_char_level")
    mu_source = args.mu_root / "participant_kit/scorer.py"
    iou, iou_ast = released_function(mu_source, "score_iou")
    correlation, correlation_ast = released_function(mu_source, "score_cor")
    rows, inputs = load_ragtruth(args.ragtruth)
    rates, fit = train_rates(rows)
    validation = [row for row in rows if row["split"] == "validation" and row["quality"] == "good" and row["n"]]
    fallback = {"metadata_char": fit["char_prevalence"], "metadata_any": fit["response_any_rate"]}
    scores = {name: lambda row, name=name: rates.get(row["key"], fallback)[name] for name in fallback}
    thresholds = {name: best_subset(validation, score)["threshold"] for name, score in scores.items()}
    result_rows = []
    populations = []
    for population in ("test_good", "test_all_qualities"):
        test = [row for row in rows if row["split"] == "test" and row["n"] and (population != "test_good" or row["quality"] == "good")]
        populations.append({"population": population, "responses": len(test), "sources": len({row["source_id"] for row in test}), "characters": sum(row["n"] for row in test), "gold_span_length_sum": sum(row["g"] for row in test), "gold_union_characters": sum(row["gold_union_characters"] for row in test), "responses_with_overlapping_gold": sum(row["g"] != row["gold_union_characters"] for row in test)})
        controls = {"mark_none": set(), "mark_all": {row["id"] for row in test}}
        controls.update({name: {row["id"] for row in test if score(row) >= thresholds[name]} for name, score in scores.items()})
        ceiling = best_subset(test, lambda row: Fraction(row["g"], row["n"]))
        controls["label_informed_whole_response_ceiling"] = ceiling["ids"]
        for name, selected in controls.items():
            metrics = score_selected(test, selected, evaluator)
            if name == "label_informed_whole_response_ceiling":
                assert abs(metrics["f1"] - float(ceiling["f1"])) < 1e-12
            result_rows.append({"population": population, "control": name, **metrics})
    mu_results, mu_inputs, fixtures = run_mu(args.mu_root, iou, correlation)
    report = {"script_sha256": sha(Path(__file__)), "input_sha256": inputs, "fit": fit, "validation_responses": len(validation), "validation_selected_thresholds": thresholds, "populations": populations, "ragtruth_results": result_rows, "mu_results": mu_results, "mu_inputs": mu_inputs, "mu_fixtures": fixtures, "released_function_sources": {"lettuce_sha256": sha(args.lettuce_evaluator), "lettuce_function_ast_sha256": lettuce_ast, "mu_sha256": sha(mu_source), "mu_iou_ast_sha256": iou_ast, "mu_cor_ast_sha256": correlation_ast}, "scope": "Exact pinned evaluator function bodies, with model calls replaced by all-or-empty response spans. Metadata sees task/generator only at prediction. The ceiling uses held-out labels and is not a deployable baseline. No LettuceDetect neural model or repair policy was run. No claim of reproducing its published cohort/model scores."}
    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, table in (("ragtruth_controls.csv", result_rows), ("mu_constant_controls.csv", mu_results)):
        with (args.out / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(table)
    print(json.dumps({"ragtruth": result_rows, "mu_fixtures": fixtures}))


if __name__ == "__main__":
    main()
