"""Transcribe published FRED tables and render declared evaluation requirements."""

import argparse
import csv
import hashlib
import json
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.tables = []
        self.captions = {}
        self.table = None
        self.row = None
        self.cell = None
        self.caption = None
        self.text = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag not in {"br", "img", "meta", "link", "hr", "input", "wbr"}:
            self.stack.append((tag, attrs.get("id", "")))
        parent = next((ident for _, ident in reversed(self.stack)
                       if ".T" in ident), "")
        if tag == "table":
            self.table = {"id": parent, "rows": []}
        if tag == "figcaption":
            self.caption = (parent, [])
        if self.table is not None:
            if tag == "tr":
                self.row = []
            if tag in {"th", "td"}:
                self.cell = []

    def handle_data(self, text):
        self.text.append(text)
        if self.cell is not None:
            self.cell.append(text)
        if self.caption is not None:
            self.caption[1].append(text)

    def handle_endtag(self, tag):
        if tag == "figcaption" and self.caption is not None:
            ident, parts = self.caption
            self.captions[ident] = " ".join("".join(parts).split())
            self.caption = None
        if self.table is not None:
            if tag in {"td", "th"} and self.cell is not None:
                self.row.append(" ".join("".join(self.cell).split()))
                self.cell = None
            if tag == "tr" and self.row is not None:
                self.table["rows"].append(self.row)
                self.row = None
            if tag == "table":
                self.tables.append(self.table)
                self.table = None
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                self.stack = self.stack[:i]
                break


def unique(items):
    if len(items) != 1:
        raise ValueError(f"Expected one matching table or row, got {len(items)}")
    return items[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fred-html", type=Path, required=True)
    parser.add_argument("--adequacy-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.fred_html.read_bytes()
    html = Tables()
    html.feed(raw.decode())
    detection = unique([t for t in html.tables if t["id"].startswith("S3.T2")])
    editing = unique([t for t in html.tables if t["id"].startswith("S3.T4")])
    for prefix in ["S3.T2", "S3.T4"]:
        captions = [text for ident, text in html.captions.items()
                    if ident.startswith(prefix)]
        assert any("FAVA" in text for text in captions), captions
    header = unique([row for row in detection["rows"] if "Ov." in row])
    overall = header.index("Ov.")
    edit_header = unique([row for row in editing["rows"]
                          if "gpt-4-turbo" in row])
    models = {"phi4": "Phi4*", "o3": "o3"}
    values = {}
    records = []
    normalize = lambda value: value.replace("∗", "*").replace(" ", "")
    for model, label in models.items():
        drow = unique([row for row in detection["rows"]
                       if row and normalize(row[0]) == label])
        erow = unique([row for row in editing["rows"]
                       if row and normalize(row[0]) == label])
        value = str(Decimal(drow[overall]))
        values[f"fred.detection.{model}"] = value
        records.append(["detection_overall_f1", label, value, detection["id"]])
        for judge in ["gpt-4-turbo", "o3-mini", "Llama-scout"]:
            value = str(Decimal(erow[edit_header.index(judge)]))
            values[f"fred.editing.{judge}.{model}"] = value
            records.append([f"edited_factuality_{judge}", label, value, editing["id"]])
    assert Decimal(values["fred.detection.phi4"]) > Decimal(values["fred.detection.o3"])
    for judge in ["gpt-4-turbo", "o3-mini", "Llama-scout"]:
        assert Decimal(values[f"fred.editing.{judge}.phi4"]) < Decimal(values[f"fred.editing.{judge}.o3"])
    visible = " ".join(" ".join(html.text).split())
    assert "outperforms o3 in detection tasks" in visible
    assert "o3 shows superior performance in editing" in visible
    with (args.out / "published_values.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["assessment", "model", "reported_value", "source_table"])
        writer.writerows(records)
    macros = [r"\newcommand{\PublishedCaseValue}[1]{\csname PublishedCaseValue@#1\endcsname}"]
    macros += [r"\expandafter\def\csname PublishedCaseValue@" + key + r"\endcsname{" + value + "}"
               for key, value in sorted(values.items())]
    (args.out / "values.tex").write_text("\n".join(macros) + "\n")
    table = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
             r"Reported assessment & Phi4$^*$ & o3\\", r"\midrule",
             r"Detection F1 & \textbf{\PublishedCaseValue{fred.detection.phi4}} & \PublishedCaseValue{fred.detection.o3}\\",
             r"Edited factuality (GPT-4-turbo) & \PublishedCaseValue{fred.editing.gpt-4-turbo.phi4} & \textbf{\PublishedCaseValue{fred.editing.gpt-4-turbo.o3}}\\",
             r"\bottomrule", r"\end{tabular}"]
    (args.out / "fred_ordering.tex").write_text("\n".join(table) + "\n")
    with args.adequacy_csv.open() as f:
        adequacy = {row["id"]: row for row in csv.DictReader(f)}
    render = [
        ("point_highlighting", "Independent highlights",
         "Confusion counts, cutoff and common unit costs",
         "AUROC or F1 for unspecified costs"),
        ("actual_editing", "Targeted revision",
         "Changed output, support, preservation and work",
         "Original detection counts or localization"),
        ("generation_retrieval", "Retrieval/generation",
         "Final task outcome and trajectory work",
         "Original gate counts or token labels"),
        ("structured_matching", "Exact span/claim matching",
         "Boundaries, types, matching and duplicates",
         "Pooled character counts"),
    ]
    selected = [adequacy[key] for key, *_ in render]
    (args.out / "adequacy_selected.json").write_text(json.dumps(selected, indent=2) + "\n")
    rows = [r"\begin{tabular}{@{}p{.20\linewidth}p{.35\linewidth}p{.39\linewidth}@{}}",
            r"\toprule", r"{\raggedright Declared use\par} & {\raggedright Assessment needs\par} & {\raggedright Insufficient alone\par}\\",
            r"\midrule"]
    for key, label, needed, omitted in render:
        assert adequacy[key]["status"]
        rows.append(" & ".join(r"{\raggedright " + text + r"\par}"
                               for text in [label, needed, omitted]) + r"\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (args.out / "adequacy_table.tex").write_text("\n".join(rows) + "\n")
    verification = {
        "source_url": "https://arxiv.org/html/2507.20930v2",
        "fred_html_sha256": hashlib.sha256(raw).hexdigest(),
        "adequacy_csv_sha256": hashlib.sha256(args.adequacy_csv.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_tables": [detection, editing],
        "values": values,
        "all_three_editing_judges_prefer_o3": True,
        "authors_state_detection_editing_disagreement": True,
        "adequacy_kind": "Derived requirements under declared action/loss assumptions, not empirical findings",
        "scope": "Published task-ordering disagreement; not a matched-item or common-editor pooling ablation",
    }
    (args.out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")


if __name__ == "__main__":
    main()
