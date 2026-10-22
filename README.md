Executed correction pilot: see [the report](bandha_actual_editing.md) for actual outputs, capped factuality/workload reports and assessment limitations.

**Current paper revision:** [Fixed-reference error evaluation with separate work](bandha_fixed_reference.md). The joint maximum is retained as a diagnostic, not the proposed benchmark ranking.

## Current unified error/work experiment

See [bandha_joint_work.md](bandha_joint_work.md) for the capped-work extension, normalized joint regret, paired source bounds and zero-work sensitivity check. Both detection and correction use the same framework, with work included according to the objective.

## Latest capped-error experiment

[Trained-detector evidence and reproduction](bandha_cap_evidence.md): released LettuceDetect predictions on 2,675 RAGTruth responses, capped remaining-error curves, worst-case regret and 5,000 paired source-bootstrap draws. The error outcomes assume ideal correction of selected words.

# Sub-Response Hallucination Detection Evaluation

Code for **Evaluating Sub-Response Hallucination Detection Metrics:
From Pooled Scores to Grouped Recovery**.

## Current-paper reproduction

Use environment-review.yml (Python 3.12, NumPy and SciPy).
Acquire pinned sources with scripts/fetch_short_inputs.py, or pass local
copies whose hashes match datasets.json and reproduction/short/package_sources.json.

Run:
```
python -I -B scripts/reproduce_short.py --ragtruth inputs/RAGTruth --fred-html inputs/fred_v2.html --out results/short
```

The focused pipeline refits the transparent predictors, regenerates all
2,000 complete-source bootstrap draws, reproduces the model/work tables,
transcribes published FRED values, verifies graded and affine review cases,
and runs the unchanged pinned character evaluator.
It also tests an analytic operation-log fixture. No neural inference is run.
Output quantities are checked against retained full-precision references.

## Operation logs

analysis/evaluate_operations.py accepts:
method,response_id,source_id,utility,cost,weight.
Weight is optional and defaults to one; eligible weights must be positive.
Utility must assess the returned response. Cost must already be priced in
the same utility units. Provide policy and cost descriptions.

The evaluator reports net value and its components, with paired complete-source
uncertainty. It rejects different case sets, source IDs or target weights.
It evaluates fixed policies, not the optimal value of unobserved actions.

## Source records and prior work

reproduction/short holds the pinned inputs, source records and numeric references.
The FRED HTML excerpt contains only attributed tables and the authors' stated
comparison; its original-primary hash is retained.
The character verifier retains its upstream MIT license and an exact certificate
for the whole-response marking ceiling. That ceiling is not a floor or additive
localization contribution.
The explicit review experiment and proper-score qualifications accompany the paper.

## Historical analyses

Earlier complete campaigns, archived prediction replay and source controls
remain available. Their commands and provenance are in docs/legacy_reproduction.md.
The current pipeline selects the results used by the short paper.

## Pooled Detection Counts

The introduction's exact illustration verifies identical pooled precision, recall
and F1 with different response-completion and checking patterns. The policy
checks only flagged units and repairs them perfectly; it is distinct from the
optimal-review theorem.

```bash
python analysis/sleek_detection_credit_example.py --out results/sleek_detection_credit_example
```

The output records input arrays, exact counts, checks, script and output hashes.
