"""Evaluate matched response-operation logs at declared utility and work prices."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

def evaluate(path, reference, draws=2000, seed=20261014):
    methods = {}
    with Path(path).open() as stream:
        reader = csv.DictReader(stream)
        required = {"method", "response_id", "source_id", "utility", "cost"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing operation-log columns.")
        for row in reader:
            if not all((row[field] or "").strip() for field in ("method", "response_id", "source_id")):
                raise ValueError("Method, response and source IDs must be nonempty.")
            key = row["response_id"]
            records = methods.setdefault(row["method"], {})
            if key in records:
                raise ValueError("Duplicate response within a method.")
            values = {field: float(row[field]) for field in ("utility", "cost")}
            values["weight"] = float(row.get("weight") or 1)
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError("Nonfinite outcome or price.")
            if values["cost"] < 0 or values["weight"] <= 0:
                raise ValueError("Costs must be nonnegative; eligible response weights positive.")
            records[key] = {**values, "source_id": row["source_id"]}
    if reference not in methods or not methods[reference]:
        raise ValueError("Reference method absent or empty.")
    ids = sorted(methods[reference])
    for method, records in methods.items():
        if set(records) != set(ids):
            raise ValueError("Paired assessment requires identical response sets.")
        if any(records[key]["source_id"] != methods[reference][key]["source_id"] or
               records[key]["weight"] != methods[reference][key]["weight"] for key in ids):
            raise ValueError("Methods must share source IDs and target weights.")
    sources = sorted({methods[reference][key]["source_id"] for key in ids})
    source_index = {source: i for i, source in enumerate(sources)}
    masses = np.zeros(len(sources))
    for key in ids:
        row = methods[reference][key]
        masses[source_index[row["source_id"]]] += row["weight"]
    weights = np.random.default_rng(seed).multinomial(len(sources),
                    np.full(len(sources), 1/len(sources)), size=draws)
    denominator = weights @ masses
    estimates, results = {}, {}
    for method, records in methods.items():
        atoms = np.zeros((len(sources), 2))
        for key in ids:
            row = records[key]
            atoms[source_index[row["source_id"]]] += row["weight"] * np.array([row["utility"], row["cost"]])
        mean = atoms.sum(axis=0) / masses.sum()
        boot = (weights @ atoms) / denominator[:,None]
        value = boot[:,0] - boot[:,1]
        estimates[method] = value
        results[method] = {"utility":float(mean[0]),"priced_work":float(mean[1]),
            "net_value":float(mean[0]-mean[1]),
            "net_value_ci95":np.quantile(value,[.025,.975]).tolist()}
    contrasts = {method:{"point":results[method]["net_value"]-results[reference]["net_value"],
        "ci95":np.quantile(estimate-estimates[reference],[.025,.975]).tolist()}
        for method,estimate in estimates.items() if method != reference}
    return {"scope":"Fixed-policy logged outcomes; not optimal risk or causal identification",
        "methods":results,"reference":reference,"paired_contrasts":contrasts,
        "responses":len(ids),"sources":len(sources),"bootstrap_draws":draws,"seed":seed,
        "uncertainty":"Paired complete-source percentile intervals; pointwise, policy fixed",
        "input_sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest()}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--outcomes",type=Path,required=True)
    parser.add_argument("--reference",required=True)
    parser.add_argument("--policy-description",required=True)
    parser.add_argument("--cost-description",required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if not args.policy_description.strip() or not args.cost_description.strip():
        raise ValueError("Policy and cost descriptions are required.")
    result=evaluate(args.outcomes,args.reference)
    result["policy_description"]=args.policy_description
    result["cost_description"]=args.cost_description
    result["script_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"operation_evaluation.json").write_text(json.dumps(result,indent=2)+"\n")

if __name__=="__main__":
    main()
