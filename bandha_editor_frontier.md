# Imperfect correction: editor frontier audit

Supplementary analysis, 2026-10-21. This checks the proposed extension; it is not a replacement paper spine or a claim about a measured editor. The current paper's -0.107 comparison uses the mean control. This note uses the **original unedited responses** as the fixed baseline for correction benefit.

## 1. Same metric, different outcomes

For fixed flags under item-local editing, with unselected-item correctness preserved,

    E_r = FN_r + failed repairs + corrupted false positives.

If each flagged error is independently repaired with probability p and each flagged clean item independently corrupted with probability q, the exact count law is

    E_r ~ FN_r + Binomial(TP_r, 1-p) + Binomial(FP_r, q).

The metric remains L(t;p,q) = average_r E[min(E_r,t)]. Cap possible outcomes and then average them. Capping the mean is generally an upper bound, not the exact expected capped outcome.

Keep the per-response FN/TP/FP triplets for this pre-editor calculation. Overall TP/FP totals suffice for mean changes but do not identify the capped curves. Ideal L/W curves alone do not retain the needed joint counts. The existing paired reference labels and frozen predictions let us recover them.

## 2. Corrections to the proposed argument

- Mean improvement requires p*TP > q*FP. For positive denominators, FP/TP < p/q and precision > q/(p+q).
- The mean-error corruption boundary is q_linear(p)=min(1,p*TP/FP), not p*FP/TP.
- Increasing p makes absolute loss nonincreasing; increasing q makes it nondecreasing. The inequalities need not be strict.
- This gives monotonicity of disadvantage when the baseline is fixed and parameter-independent, as the unedited baseline is. Comparing two flagging procedures processed by the same parameterized editor need not give monotone disadvantage: one can benefit more quickly from increasing p than the other.
- A positive exact B=max_t[L(t)-L_ref(t)] exhibits an admitted capped loss for which the pipeline is worse on the benchmark average. It does not establish worse performance for every loss, universal unsafety, or a population/deployment guarantee.
- A nonpositive B concerns average loss over evaluated responses. It does not promise that no originally clean response is corrupted.
- All-zero remaining errors have probability 1{FN_r=0} * p^TP_r * (1-q)^FP_r under the independent model. Partial detection cannot clear a response with FN_r>0.

Against the unedited baseline, the numerical dominance frontier is q_star(p)=sup{q:B(p,q)<=0}. It is nondecreasing in p, can be flat or zero, and is at most the mean-error boundary. A different fixed baseline can make the feasible set empty; assigning zero to such an empty set would falsely imply that q=0 is feasible.

## 3. Marginal rates do not imply independent editing

Two flagged errors repaired with marginal probability 0.5 illustrate the issue. Independent repairs leave at least one error with probability 0.75. Joint success/failure leaves one with probability 0.5; a balanced scheme repairing exactly one always leaves one with probability 1. All have mean remaining errors 1. The same p and q therefore do not determine exact capped losses without a dependence model.

## 4. A simpler conservative guarantee

Independence is unnecessary for the mean formula. If the stated repair and harm bounds hold per flagged item, then

    E[E_r] <= m_r(p,q) = FN_r + (1-p)TP_r + q FP_r.

Jensen and monotonicity give

    L(t;p,q) <= U(t;p,q) = average_r min(m_r(p,q),t).

Consequently U_B(p,q)=max_t[U(t;p,q)-L_ref(t)]<=0 is a sufficient guarantee of no greater average loss for every admitted increasing concave error-loss function. It allows arbitrary dependence among edits. A positive U_B is an inconclusive guarantee test, not proof that the actual editor harms outputs.

This supplies a conservative frontier without binomial convolutions. Monotonicity also covers the rectangle p>=p_min, q<=q_max by checking its worst corner. The premise is a uniform/per-item marginal bound within this local-edit model; global average repair/harm rates, even on the same benchmark, do not identify their distribution across responses or automatically establish uniform bounds. Unrestricted p,q allow an editor that never repairs and always corrupts, so no useful strict improvement guarantee can cover the whole unit square.

## 5. Checked RAGTruth example

Same frozen trained detector, 2,675 responses. Two deterministic word policies: the paper's top-20% rule and a word-score cutoff 0.5. The latter is a word mapping of frozen subtoken scores, not a replication of the release's span-F1 evaluation. No new model inference and no executed editor.

At p=0.9, q=0.01 under independent local edits:

| Word policy | Flags | Mean-error change per response | Change in error-containing response fraction | B against unedited output | Conservative U_B |
|---|---:|---:|---:|---:|---:|
| 20% checks | 68,195 | -3.397 | +1.16 percentage points | 0.012 | 0.071 |
| Cutoff 0.5 | 12,307 | -2.240 | -4.43 percentage points | -0.044 | -0.031 |

The 20% rule reduces expected error words by about 9,086 overall but increases the expected number of responses containing an error by about 31. The cutoff-0.5 rule improves both summaries; its negative conservative bound also supports the entire concave class under the stated marginal bounds, irrespective of edit dependence. These are conditional fixed-benchmark calculations, not measured editor quality or population intervals.

At p=0.9, numerical tolerated-q frontiers are approximately:

| Word policy | Mean-error boundary | Independent-edit frontier | Conservative sufficient frontier |
|---|---:|---:|---:|
| 20% checks | 0.16812 | 0.00886 | 0.00489 |
| Cutoff 0.5 | 1.00000 | 0.08508 | 0.04106 |

The curve can be published without choosing the eventual editor's exact p,q. A consumer can use relevant reliability bounds to locate a conditional guarantee region. It cannot turn global reliability estimates into an assumption-free certificate for arbitrary contextual editing.

Harm expands the possible remaining-error support: the 20% policy can leave up to 264 error words, exceeding the ideal experiment's 256. The computation includes those additional caps.

## 6. Hypergeometric handling

Untied scores require ordinary counts. In the primary trained-detector policy the cutoff pool has one word. The mean control creates a 157-word tied pool and selects 143. Exact expectation under uniform fixed-size tie selection uses a hypergeometric law for that control. Generator and lexical controls also tie. This is an implementation detail for the declared tie policy, not a requirement that all detector scores be discrete. With deterministic threshold policies, no such tie randomization is needed.

## Reproduction

```sh
python analysis/bandha_editor_frontier.py --scores analysis/bandha_cap_inputs/trained_scores.npz --out analysis/bandha_editor_frontier_results
```

The script checks 212 identities/inequalities, including independent direct convolution on small fixtures, a two-response counterexample, Jensen bounds, and frontier ordering. The plotted frontier points are numerical estimates with crossing brackets; those brackets are not confidence intervals, and line interpolation is not a continuum certificate. Source/population uncertainty and editor calibration uncertainty have not been estimated.

[Frontier figure](analysis/bandha_editor_frontier_results/frontiers.pdf)
