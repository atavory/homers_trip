# Homer's Trip

Reproduction code for **Homer's Trip: Simpson's Paradox in Token-Level Hallucination Metrics**.

The experiments compare pooled classification, within-response ranking, and annotation recovery at matched inspection costs. They include response-constant priors, a lexical score, response offsets, response means, and retrospective label-preserving permutations.

## Reproduction

Create the recorded scientific Python environment:

```bash
conda env create -f environment.yml
conda activate homers-trip
python scripts/reproduce.py
```

The pipeline downloads the two RAGTruth input files at the revision in `datasets.json` and verifies their SHA256 hashes. It then runs the finite theory checks, fitting and validation, all 2,000 source-group bootstrap draws, the alternative-evaluation experiments, independent selection checks, plotting, and paper aggregation.

The frozen source-disjoint split uses salt `homers-dream-dev-v1`; its name is retained to reproduce the original partitions. The bootstrap seed is 20261014. The two retrospective permutation seeds are 20261015 and 20261016. Fitting and validation-selected thresholds are held fixed during test uncertainty estimation.

## Outputs

Under `results/`:

- `expanded/`: eight scorers, unrounded results, bootstrap estimates, and the pooled/conditional figure.
- `alternatives/`: conditional AP, word and annotation-span recovery, one-class alarm costs, permutation controls, bootstrap estimates, and the inspection figure.
- `paper_aggregates/`: CSVs and generated LaTeX table fragments.
- `toy/`: computed construction values, exact class means, and generated vector diagram.
- `paper_values/`: inline LaTeX values and their CSV source mapping.
- `reproduction_manifest.json`: commands and code hashes.

`analysis/aggregate_paper.py` formats recorded experiment outputs. Its manifest records input hashes and every generated output hash. `all_statistics.csv` retains exact result paths and values; `table_sources.csv` maps table cells to the source results.

For a saved run:

```bash
python analysis/aggregate_paper.py \
  --expanded results/expanded/expanded_results.json \
  --alternatives results/alternatives/results.json \
  --out results/paper_aggregates
```

## Saved-prediction replay

`analysis/replay_predictions.py` evaluates original token predictions and distinguishes response means taken over eligible rows from means taken over all scored rows.

Supply a CSV with columns:

```text
scorer,response_id,source_id,token_index,label,score,eligible
```

Labels and eligibility use 0/1. Each row retains its original token index, response and source identity. The script computes pooled, cross-response, and conditional ranking and writes unrounded JSON and CSV outputs with input and code hashes.

```bash
python analysis/replay_predictions.py \
  --predictions predictions.csv --out results/archived_replay
```

Internal data, CSVs, private runs, and their provenance are maintained in the companion data project. This repository contains the public code and pinned public-dataset specification.

## Data source

RAGTruth: https://github.com/ParticleMedia/RAGTruth

The study's units are eligible whitespace-delimited words, marked positive by overlap with supplied annotation spans. These experiments measure recovery of annotations. Retrospective label-permutation controls use test labels to diagnose an evaluation property.

## Historical summary formatting

`analysis/format_reported_archive.py --summary reported_summary.csv --out results/archive_summary` formats retained archival summaries and records their input hash. Its outputs are marked `reported_summary_only`. Original prediction replay remains the separate `replay_predictions.py` workflow.

## Restored archival replay

The archived detector inputs have been recovered and validated against their original manifests. They use RAGTruth revision `c103204b9ce28d6bbad859304bf30de72b8ed8fe`, distinct from the current word-level study. Internal data preparation retains all scored rows, the original eligibility masks, and the five-seed raw-logit ensemble definition.

Given the reviewer input CSV and preserved bootstrap/threshold files:

```bash
python analysis/replay_predictions.py --predictions test_predictions.csv --out results/archived_replay
python analysis/replay_bootstrap.py --predictions test_predictions.csv --bootstrap bootstrap_inputs.json --thresholds thresholds.json --out results/archived_replay
python analysis/aggregate_archive.py --metrics results/archived_replay/results.json --bootstrap results/archived_replay/bootstrap_results.json --provenance input_provenance.json --out results/paper_aggregates
```

Run `replay_predictions.py` first to produce the point-metric `results.json`. Source seed 360035 and its 500 source draws are preserved. The source bootstrap holds the original fits and validation thresholds fixed. Reported-summary formatting is a historical record; manuscript spreads are calculated from replayed unrounded values. Historical audit notes predate this recovery.

## Decisions and score relationships

The decision framework distinguishes global token decisions, response screening, and token inspection inside a selected response. A policy specifies its feasible actions, costs and outcome. The public pipeline now also runs:

```bash
python analysis/decision_framework_check.py --output results/decision_framework
python analysis/decision_relationships.py --ragtruth data/RAGTruth --baseline results/expanded/expanded_results.json --out results/decision_relationships
```

The first command generates exact rational checks and counterexamples. The second measures pooled score-label covariance, its within-response and between-response components, association of response means with hallucination prevalence, and probability prediction-error components. It uses the eight frozen scorers and the original paired source-bootstrap draws, under both token and uniform-response weighting. Its CSV and TeX outputs are generated by the script.

These quantities describe specified statistical relationships. The same-response pair fraction is not an information fraction, and covariance does not determine every action utility. Actual editing or decoding utility requires evaluating the resulting outputs.

## Shadow manuscript aggregation

The parallel shadow manuscript uses the same frozen RAGTruth study and a metric-general analysis of response risk and token discrimination. Generate its construction and table sources with:

```bash
python analysis/bandha_intervention_check.py --out results/intervention
python analysis/aggregate_shadow.py --expanded results/expanded/expanded_results.json --alternatives results/alternatives/results.json --relationships results/decision_relationships/results.json --fixtures results/intervention/verification.json --out results/shadow
```

The main reproduction pipeline produces the expanded, alternative and relationship results. The shadow aggregation writes four LaTeX tables, inline values and a CSV mapping every new display to its unrounded source. Its manifest records dataset, fitting and source-bootstrap consistency checks and input/output hashes.
