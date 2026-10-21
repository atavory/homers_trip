# Fixed-reference error evaluation with separate work

Date: 2026-10-21. This is the current revision of the Meaningful Evaluation paper. It supersedes the normalized joint maximum as the recommended scalar; the earlier joint results remain reproducible diagnostics.

## 1. What changed

1. Lead with capped-error dominance, including the trained detector's supported improvement over every comparator at the primary operating point.
2. Use signed worst-case excess error loss against a fixed reference, B_E = max_t[L_j(t)-L_reference(t)]. Negative values certify improvement throughout the normalized concave error-loss class. This is signed disadvantage, not classical nonnegative regret against the best listed method.
3. All procedures share the same frozen LettuceDetect response-mean reference. Its checkpoint, information and selection rule are fixed; its operating point follows the declared check budget. This shared diagnostic reference is explicit and is not claimed to be a universal benchmark baseline.
4. Report B_K and the work curve separately. No maximum combining error and work units is used to rank procedures. A word-check axis describes a count constraint, not equal concave costs or measured review time.
5. Adding a competitor changes neither fixed-reference point score. Changing the reference, assessed units, weights or normalization can change it. Simultaneous uncertainty still refers to a declared family of comparisons.

## 2. Primary fixed-reference results

2,675 RAGTruth test/good responses from 450 sources, 68,195 selected words per procedure, ideal targeted correction, uniform cutoff-tie expectations. Source bands reuse the frozen 5,000 paired source draws.

| Procedure | Mean remaining errors | Opened responses | Error disadvantage [95% bounds] | Work disadvantage |
|---|---:|---:|---:|---:|
| Generator metadata | 3.265 | 885 | 1.271 [0.569, 2.236] | 5.182 |
| Lexical | 3.093 | 2674 | 1.099 [0.438, 2.097] | 16.439 |
| Lexical + metadata | 2.676 | 1848 | 0.681 [0.290, 1.408] | 9.070 |
| LettuceDetect | 0.878 | 2338 | -0.107 [-0.147, -0.068] | 12.788 |
| LettuceDetect mean (reference) | 1.995 | 484 | 0.000 [0.000, 0.000] | 0.000 |

The full detector has lower error loss than all four comparators throughout the declared class; simultaneous source-bootstrap bounds support this at 20% inspection. Full versus mean is an error improvement and a work trade-off, not evidence that either universally wins on unknown costs.

## 3. Declared check levels

The report adds 10%, 20% and 30% word-check operating points without new model inference. The trained detector dominates all comparators on the benchmark at all three. Only the 20% point has population uncertainty. The right-hand plot reports fixed-reference error disadvantage against mean checks per response; the error and work cap plots remain separate. The reference is the same fixed policy family at each level, so this curve reports relative performance and need not be monotone in the amount of work.

## 4. Why the joint maximum was dropped

The identity max(R_E,R_K) under the stated combined normalization is mathematically correct and finite. The editorial failure was to recommend that normalization as the benchmark decision rule despite its dependence on the relative units. Its work ranges dominate four of five primary rows; multiplying the work coordinate by 0.01 changes the selected procedure. Adding a zero-work procedure also makes it prefer doing nothing in this particular candidate set. These calculations are retained as reasons not to prescribe that scalar. They do not invalidate capped-mixture mathematics and do not prove that no-work is best at known prices or in every operating-point set.

For a declared physical work price lambda, worst excess cost against the fixed reference is B_E + lambda B_K. Unrestricted lambda still gives infinity when B_K is positive. Work on an axis removes an assumed price only when the axis declares the actual constrained resource; raw checks do not recover unknown preparation costs.

## 5. Approval and evidence scope

The recorded enthusiasm quote concerns the hierarchical narrative presented to the user. It is not numerical validation or approval of all tables, and it does not authorize treating a superseded scalar recommendation as settled. The former outline is retained as historical narrative approval; this revision follows the later critique.

Current inference remains one trained checkpoint on one benchmark, perfect correction of marked words, and modeled one-check-per-selected-word work. It does not measure neural corrected-output quality or human review time. The capped basis and robust comparison principles are established mathematics.

## Reproduction

```sh
python analysis/bandha_fixed_reference.py --inputs analysis/bandha_cap_inputs --results analysis/bandha_joint_work_results --out analysis/bandha_fixed_reference_results --paper /path/to/paper
```

The report reproduces saved primary curves and verifies 470 cap-attainment/mixture comparisons across the three check levels. It retains original approximate source-bootstrap bands; no fresh inference or bootstrap rerun is hidden in this revision.
