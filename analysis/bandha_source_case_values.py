"""Format source-verified RARR comparisons without rerunning the authors' study."""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.records.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        assert Decimal(row["after"]) - Decimal(row["before"]) == Decimal(row["difference_percentage_points"])
    record = next(row for row in rows if row["metric"] == "downstream_answer_accuracy" and row["cohort_label"] == "StrategyQA")
    values = {"sqa.before": record["before"], "sqa.after": record["after"], "sqa.change": record["difference_percentage_points"]}
    args.out.mkdir(parents=True, exist_ok=True)
    tex = [r"\newcommand{\SourceCaseValue}[1]{\csname SourceCaseValue@#1\endcsname}"]
    tex += [r"\expandafter\def\csname SourceCaseValue@" + k + r"\endcsname{" + v + "}" for k,v in sorted(values.items())]
    (args.out/"values.tex").write_text("\n".join(tex)+"\n")
    (args.out/"verification.json").write_text(json.dumps({"source": record,"values": values,
        "record_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Published values transcribed from verified primary record; no new model evaluation"},indent=2)+"\n")

if __name__ == "__main__":
    main()
