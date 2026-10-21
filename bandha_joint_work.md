# Unified error and optional-work evaluation

## User-approved paper hierarchy — 2026-10-21

The user explicitly approved the hierarchical outline below: **“i am absolutely loving the fuck out of this! Mark this in the md file.”** This records approval of the current narrative and structure. Use this outline as the current writing reference when consulting older notes; the earlier aggregation-only, generating-function and fixed-local-budget narratives do not define the present paper.

**Central question:** How should we compare hallucination procedures when we know the shape of error damage and checking cost, but not their exact values?

### 1. Introduction

- **1.1. Response setting.** There are R responses, each containing n_r assessed items: words, spans, sentences or claims. A procedure leaves E_r errors and, when checking work matters, incurs K_r units of work.
- **1.2. Practical rationale.** One remaining error can substantially compromise an answer; additional errors may cause less incremental damage. This motivates increasing concave error loss, equivalently decreasing convex usefulness. Checking can share preparation, motivating increasing concave checking cost. These are assumptions about particular uses, not universal laws.
- **1.3. Existing evaluations and their limits.** Distinguish pooled item metrics, response-conditioned metrics, mean remaining errors, error-free response rate, and particular correction/review outcomes. Each can answer a legitimate question; none alone determines performance across the proposed damage and cost classes.
- **1.4. Contribution.** Report capped-error and capped-work curves, identify robust dominance, and calculate worst-case regret. Detection and correction use one framework; work is included when the objective requires it.

### 2. Capped Outcomes and Robust Comparison

#### 2.1. What we propose

- **2.1.1. Evaluated outcomes.** Correctors supply assessed returned-output errors. Detectors require a declared selection/correction policy. Perfect targeted correction without collateral damage or changes to unselected-item correctness gives residual errors equal to false negatives. Actual editors require actual output assessment.
- **2.1.2. Two curves.** Report L_j(t)=E[min(E_j,t)] and W_j(u)=E[min(K_j,u)]. Apply caps within responses, then average. Cap one gives the error-containing or checked-response fraction; sufficiently large caps give mean errors or mean work. Caps describe saturation of damage or cost, not inspection budgets.
- **2.1.3. Normalization.** Normalize the first error's damage and first work unit's cost to one. When both matter, use (1-theta)*ell(E)+theta*c(K), theta in [0,1]. This is an explicit benchmark convention for unknown relative importance.

#### 2.2. Why these reports are useful

- **2.2.1. Coverage of the declared shape class.** Every normalized increasing concave count function is a convex combination of capped counts. The curves determine all expected losses and costs in that class. Credit the established representation; the evaluation application is our contribution.
- **2.2.2. Dominance.** An error curve everywhere below another establishes preference under every admissible error-loss function. Crossings expose dependence on error valuation.
- **2.2.3. Scalar comparison.** Against a named finite set S, R_j^E=max_t[L_j(t)-min_{i in S}L_i(t)]. This measures the worst disadvantage under uncertain error damage. Lower regret gives a minimax-regret ranking.
- **2.2.4. Optional work.** Define R_j^K analogously. Under complete uncertainty in the normalized combined objective, R_j^joint=max(R_j^E,R_j^K). The scalar depends on alternatives, operating points and normalization; it does not preserve every application's preferred ranking.

### 3. Experiments

- **3.1. Five fixed procedures on RAGTruth.** Evaluate 2,675 responses from 450 sources using a trained LettuceDetect checkpoint, its response-mean control, and three simpler scorers. Each selects exactly 20% of words. Correction is idealized annotation coverage; checking work is modeled.
- **3.2. Equal checks do not mean equal concave costs.** The trained detector leaves 0.878 errors per response, versus 1.995 for its mean control. Both select 68,195 words, but open 2,338 versus 484 responses.
- **3.3. Changed preference.** The trained detector wins the error-only comparison. Its mean control minimizes normalized joint regret among the five fixed procedures. Source-clustered uncertainty supports the distinctions.
- **3.4. Conservatism and alternatives.** Adding an unchanged-output, zero-work alternative changes the joint recommendation to doing nothing on this benchmark. Unrestricted uncertainty includes work-dominated objectives. State relevant alternatives and constraints; do not suppress this sensitivity result.

### 4. Discussion

- **4.1. Takeaway.** Report curves showing dependence on diminishing damage and shared checking costs. When a scalar is needed, declare the uncertainty class and use explicit worst-case regret.
- **4.2. Benchmark declarations.** Specify assessed targets, error/work units, correction assumptions, response weights, operating points, alternatives and normalization.
- **4.3. Limits.** One benchmark and trained checkpoint; ideal correction rather than measured edited-output quality; modeled checking rather than measured review time; common concave count functions rather than every application's objective.

### Appendices

- **A.** Reproduction, tie averaging, source-clustered uncertainty, physical prices and resource constraints.
- **B.** The parametric Simpson reversal: pooled AUROC can reverse within-response localization rankings. This supports the distinction between evaluation targets; it is not the central contribution.

---

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
