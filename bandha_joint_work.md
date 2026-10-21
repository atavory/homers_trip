# Unified error and optional-work evaluation

Detection-assisted and direct correction use one outcome framework: assessed remaining errors E and work K when work is part of the objective. Ignoring work does not require equal work. The procedure type changes how outcomes are obtained, not the worst-case comparison principle.

## Criterion

Report L_j(t)=E[min(E_j,t)] and W_j(u)=E[min(K_j,u)]. Common increasing concave component shapes have zero loss at zero and unit first increments. The combined benchmark convention is (1-theta)*error loss + theta*work cost, theta in [0,1]. Its first increments sum to one. Worst normalized joint regret is max(error regret,work regret), with both regrets measured against the same fixed comparison set. This is finite but changes the yardstick across physical prices; it does not recover unknown dollar costs.

For a specified work price lambda in first-error units, regret is max_i[D_E(j,i)+lambda D_K(j,i)]. Unrestricted physical prices can give infinite regret. The normalized convention, a bounded price range, or a declared resource constraint are distinct choices.

## Executed comparison

Same frozen predictions, 2,675 RAGTruth test/good responses, 450 sources, 68,195 selected words per procedure. No new model inference or fitting. One flag incurs one modeled check. Selected errors are perfectly corrected without damage: ideal annotation coverage, not edited-output quality or measured human time.

| Fixed 20% procedure | Error regret | Work regret | Normalized joint regret [joint 95% bounds] |
|---|---:|---:|---:|
| Generator metadata | 2.387 | 5.182 | 5.182 [4.206, 6.313] |
| Lexical | 2.215 | 16.439 | 16.439 [15.205, 17.693] |
| Lexical + metadata | 1.798 | 9.070 | 9.070 [7.649, 10.554] |
| LettuceDetect | 0.000 | 12.788 | 12.788 [11.607, 13.985] |
| LettuceDetect mean | 1.116 | 0.000 | 1.116 [0.716, 1.517] |

Fine scores minimize error-only regret. The mean control minimizes normalized joint regret among these five operating points. Fine scores open 2,338 responses, versus 484 for the mean, despite identical check totals. Mean residual errors favor fine scores (0.878 versus 1.995). Work's endpoint means agree; its capped curve distinguishes shared preparation.

The original 5,000 paired source draws are reused. Every replicate recomputes the global selection cutoff and integrates error and work tie laws. All earlier error-only bootstrap regrets reproduce exactly. Joint max-t bounds cover both outcomes, ten pairs, 256 error caps and 359 work caps. They are approximate source-bootstrap bounds conditional on fixed trained scores.

## Zero-work check

Adding the unchanged-response/no-check procedure changes the candidate set. It has zero work, so each positive-work procedure's worst normalized work regret is mean checks, 25.493 here. Its own error and joint regret is 4.013. The fully weight-agnostic convention therefore selects no work in this enlarged finite set. These are exact fixed-benchmark point comparisons, not population recommendations.

This exposes the convention's conservativeness. The class includes cost-only loss and linear checking cost. It does not imply no work is best at known prices or across additional operating points. Activity decisions require the relevant alternatives and objective constraints. A raw check-count budget does not establish equal preparation cost or review time.

## Reproduction

Requires NumPy, SciPy and Matplotlib; no neural dependencies for this extension.

```sh
python analysis/bandha_joint_work.py --scores analysis/bandha_cap_inputs/trained_scores.npz --index analysis/bandha_cap_inputs/response_index.json --error-results analysis/bandha_cap_trained_results --out analysis/bandha_joint_work_results --workers 4
python analysis/bandha_verify_joint_work.py --inputs analysis/bandha_cap_inputs --results analysis/bandha_joint_work_results
```

The independent verifier checks 1,282 values, including nine exhaustively enumerated source/selection fixtures, hypergeometric probabilities, direct work curves, loss-mixture bounds, fixed-price attainment and the zero-work comparison. Source draws and curves are retained. Model generation costs and human review time are not measured. Common count-based costs are assumed; context-dependent costs need context outcomes.
