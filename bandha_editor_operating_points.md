# Operating-point and editor-dependence audit

Supplementary checked evidence, 2026-10-21. This strengthens the conditional editor-frontier analysis. It does not adopt a claim that actual LLM editors universally harm outputs, and it does not replace the current paper's outcome-report framework.

## 1. What survives the critique

The capped-report mathematics is sound. Under a specified editing law and a fixed unedited baseline, editor-reliability frontiers are meaningful. A mean-error improvement can coexist with a worse admitted capped loss. The conditional operating-point gap is worth reporting.

## 2. What the proposed headline overstates

1. The three low-precision generator/lexical baselines do not establish the trained detector's frontier. At p=0.9 the checked trained model's independent-edit q frontier is about 0.00886 at 20% inspection and 0.08508 at word-score cutoff 0.5. These are different from the approximate 0.002 figures quoted for simpler scorers.
2. The paper's -0.107 ideal result uses the mean control as reference. A frontier relative to doing nothing is a different comparison and cannot be substituted for it silently.
3. "Any fallible corrector makes things worse" is false: p<1 with q=0 can leave errors uncorrected but cannot add errors under local repair without damage. Its errors never exceed the original count, realization by realization.
4. No per-flagged-clean-word corruption rate has been measured for an actual editor in these experiments. Claims that no LLM editor is below 0.2%, or that 2% is optimistic, are unsupported and must not be inferred from response-level rewriting or hallucination rates.
5. The first-cap asymmetry concerns particular changes. A last-error repair 1->0 has the same unit benefit as the unit damage from 0->1. A false alarm in an already erroneous response can add nothing at cap one. There is no universal strict ordering of all false alarms versus all repairs.
6. Positive exact disadvantage means some admitted loss is worse, not every loss and not universal deployment failure. Exceeding a tolerance by a factor of ten is not ten times the harm.

## 3. The coupling direction

More concentrated error counts can LOWER expected increasing concave damage at a fixed mean. Independence is not universally optimistic. We checked a specific grouped law: all flagged-error repairs share a Bernoulli event with success p; all false-positive corruptions share an event with probability q; the two type-group events are independent.

This gives four count outcomes FN, FN+FP, FN+TP and FN+TP+FP with probabilities p(1-q), pq, (1-p)(1-q), and (1-p)q. Compared with independent item flips, the mean count is unchanged and every checked capped expectation is no greater. These are models, not empirical editor observations.

| Trained-detector word policy | Independent B at p=.9,q=.02 | Grouped B at the same p,q | Independent q frontier at p=.9 | Grouped q frontier |
|---|---:|---:|---:|---:|
| 20% inspection | 0.09660 | -0.20915 | 0.00886 | 0.16142 |
| Score cutoff 0.5 | -0.03540 | -0.09122 | 0.08508 | 0.63063 |

Thus the 20% policy has opposite loss-class verdicts under two laws with the same p,q and mean errors. Arbitrary dependence can also be less favorable; the conservative Jensen frontier protects against that possibility. One must state the dependence model or use a justified bound.

The grouped 20% frontier's binding cap is 88. A TMAX=40 calculation misses that constraint. To claim the entire concave class, cover the full possible error support, including the linear endpoint; truncating caps can only weaken the test and raise its apparent tolerance. The present operating-point calculation uses all 359 caps.

## 4. Source uncertainty and operating points

We computed the sufficient marginal-bound frontier with 5,000 paired resamples of the 450 RAGTruth sources. The trained word scores stay fixed. Every sample recomputes the global word allocation and uses expected TP/FP counts under uniform cutoff-tie allocation, including source clones. For this Jensen bound, no binomial or hypergeometric distribution is needed.

The repair lower bound is 0.9. The following q values are numerical sufficient guarantees of no greater benchmark-average error loss throughout the normalized concave class under local editing and the stated per-item marginal bounds. They are NOT estimates of the editor's actual q. Intervals are approximate pointwise source-bootstrap intervals, not simultaneous bands, finite-sample guarantees, or editor-calibration intervals.

| Word budget | Word precision | Sufficient q | 95% source interval |
|---|---:|---:|---:|
| 1% | 0.758 | 0.06407 | [0.03250, 0.21160] |
| 2% | 0.674 | 0.07658 | [0.04238, 0.14059] |
| 5% | 0.455 | 0.02729 | [0.02140, 0.03470] |
| 10% | 0.277 | 0.01110 | [0.00937, 0.01306] |
| 20% | 0.157 | 0.00489 | [0.00425, 0.00558] |
| 30% | 0.111 | 0.00316 | [0.00275, 0.00358] |
| 50% | 0.071 | 0.00181 | [0.00159, 0.00204] |

The paired 5%-minus-20% tolerance difference is 0.02240, with source interval [0.01706, 0.02919].

At the benchmark point, 1% has higher precision than 2% but a lower sufficient tolerance. Their paired tolerance interval [-0.03934, 0.08959] includes zero, so the population ordering is unresolved. One cannot infer a monotone frontier or an optimal budget from precision alone.

At small budgets, the uncertainty is appreciable despite large ratios to some mean-only boundaries. Noise cannot be dismissed because a point ratio looks large. A maximum over noisy cap differences and an inverse frontier are different estimators; no blanket upward-bias assertion about the frontier follows. The deterministic optimism from omitting constraints is a separate issue.

A raw word budget specifies a count constraint, not equal review time or preparation cost. Marginal-rate guarantees apply to average loss over evaluated responses, not to every individual answer. Global average editor rates alone do not establish uniform per-item or per-response repair/harm bounds. Actual deployment conclusions require assessed edited outputs or relevant calibration evidence in the same units.

## 5. What the paper should claim

Mean error reduction does not establish improvement in response usefulness under diminishing error damage. The outcome report exposes that distinction; editor-reliability requirements can be evaluated as a compact application of the same capped curves. Operating points, dependence assumptions, baseline, error units and uncertainty remain explicit. This is stronger than an unsupported statement that every fallible editor or fine-grained detector is harmful.

## Reproduction

```sh
python analysis/bandha_editor_operating_points.py --inputs analysis/bandha_cap_inputs --source-draws analysis/bandha_cap_trained_results/source_weights.npz --out analysis/bandha_editor_operating_point_results --workers 4
```

The implementation checks capped means against 30 independent dense computations, checks allocation totals in every resample, and checks matched-mean grouped versus independent editing over 18 parameter settings. All evaluated caps are retained. No new model inference or actual editor execution is included.

[Operating-point figure](analysis/bandha_editor_operating_point_results/operating_points.pdf)
