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
