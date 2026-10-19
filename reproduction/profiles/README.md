# Response profile experiments

The scripts calculate remaining-annotation and flagged-work profiles, 10,000 paired-source intervals, simultaneous function-family bounds, correction sensitivity, and validation-selected value rules. They operate on fixed counts and declared correction models, not measured human or neural editing.

The anonymized reviewer supplement supplies frozen inputs and the RAGTruth snapshot. Use the supplied environment file. From this repository, with those inputs available:

```bash
python3 -I -B analysis/bandha_profile_experiments.py --inputs path/to/profile_inputs --out outputs/profiles --ragtruth path/to/RAGTruth
python3 -I -B analysis/bandha_verify_profile_experiments.py --inputs path/to/profile_inputs --results outputs/profiles --ragtruth path/to/RAGTruth
python3 -I -B analysis/bandha_profile_aggregate.py --results outputs/profiles --out outputs/profiles/manuscript
python3 -I -B analysis/bandha_plot_profile_experiments.py --results outputs/profiles --out outputs/profiles/figures
```

The preparation script builds portable inputs from the existing frozen-response pipeline. Records and draw files are copied exactly, and metadata retains numeric references and source hashes.

## Interpretation

The original operating points use validation-selected F1 cutoffs. Changing an evaluation parameter does not change their flags. Value-rule tuning uses only training and validation information; test labels assess the chosen rule. Both comparison types are retained.

Correction probability parameters are assumed, not estimated. Unit-independent and response-coupled models share marginal correction rates but can differ in nonlinear usefulness. Source inference conditions on those assumptions and on fitted scores and selected cutoffs.

The count logs explicitly distinguish original annotation misses, flags, and modeled expected remaining errors. Taking a power of an expected count is not a substitute for the expected generating function.

Intervals are jointly adjusted across the declared profile, correction, tuning and transport comparisons. Root summaries retain absent and multiple crossings rather than selecting only successful unique-root draws. A printed grid region is not a unique population-root confidence interval.

All sixty value-tuning and forty-eight ideal-policy transport comparisons are reported, including negative results and no-check policies. Counts, outcomes, source weights and all manuscript aggregates accompany the private reviewer artifacts.

The verifier accepts both the pipeline's `source_weights.npy` and the compressed
`source_weights.npz` supplied with saved results. It verifies the hash of the
uncompressed draw matrix in either case.

The count-parameterized pooling example can be checked independently with
`python -I analysis/bandha_parametric_pooling.py --out parametric_results`.
It uses exact fractions and only the Python standard library.
