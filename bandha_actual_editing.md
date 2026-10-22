# Actual correction pilot

Status: executed and assessed on all 48 frozen cases. Results below are automated assessment proxies, not post-edit human gold labels.

We execute a constrained GPT-4.1 corrector on frozen LettuceDetect flags, comparing corpus-wide top-20% selection and word score at least .5. Selected contiguous word intervals are the only replaceable text. Unselected text is preserved by code. The editor receives source evidence and predicted flags, never gold error annotations.

The pilot has 48 responses from 48 distinct sources: eight per task/original-error stratum across Summary, QA and Data2txt. Cases are hash-selected before running models, restricted to at most 220 response words and 14,000 evidence characters. Evidence is not truncated. This is a balanced pilot, not a prevalence-weighted estimate for the full benchmark.

A different model, Claude Sonnet 4.6, assesses original and returned text in randomized candidate order with policy names hidden. It enumerates unsupported atomic propositions and omissions of supported original targets; ambiguous assessments are recorded separately. These are automated assessment proxies, not new human gold annotations. Original RAGTruth annotations are used for sampling and later assessor-agreement checks, not supplied to either model.

The original response is the common fixed baseline. We retain prompts, actual replacements, returned text, blinded assessments, IDs and source groups. Capped curves will be calculated from assessed returned-output error counts; p/q coin flips are not used. Realized operation counts and flagged-word checks are reported as proxies, not human review time.

This is a controlled detector-guided LLM correction pipeline, not a claimed reproduction of FAVA, RARR or GenAudit. Model aliases are recorded; the serving API does not provide pinned weight revisions.

## Run

```sh
python analysis/bandha_actual_editing.py --ragtruth data/RAGTruth --inputs analysis/bandha_cap_inputs --out analysis/bandha_actual_editing_results
```

The live runner uses one-shot model calls through Meta CLI and Mr. Gitties. The saved prompt/output records permit independent assessment without access to that serving interface.

## Executed results

| Procedure | Mean unsupported claims | Fraction with unsupported claims | Mean lost supported targets | Editor cases called |
|---|---:|---:|---:|---:|
| Unedited | 1.333 | 0.625 | 0.000 | 0 |
| Top-20% flags | 0.604 | 0.375 | 0.333 | 43 |
| Word score at least .5 | 0.792 | 0.479 | 0.000 | 21 |

Top-20% flagging yields fewer assessor-rated unsupported claims but more lost supported information. These are distinct outcomes; neither licenses a universal preference or a real-editor safety claim.

The primary factuality curves use unsupported atomic propositions as E. Preservation is reported separately. The retained alternative composite curves count unsupported propositions plus omitted supported targets, an explicit extra assessment convention rather than an assumed universal value scale.

The fixed reference is the unedited response. Factual B and approximate 95% stratified source-percentile intervals:

- top20: B = -0.250, interval [-0.396, -0.104].
- cutoff05: B = -0.146, interval [-0.229, -0.062].

Intervals use 5,000 paired source draws preserving the six sampling strata. They condition on frozen generations and automatic assessment; they omit assessor error and model-generation variability. Pointwise curve intervals are not simultaneous confidence bands.

The assessor's original-response binary judgments agree with RAGTruth labels on 38/48 responses (79.2%). Original labels concern annotated word spans; the new assessments enumerate propositions. This agreement is an assessment diagnostic, not validation of the returned outputs as gold. Strong claims require checked post-edit labels.

## Validation and audit

All 96 policy outputs are retained; 64 cases involved an editor call, while unflagged responses were preserved without a call. Every applied replacement references an allowed selected span, and unselected text is preserved by code. The assessor receives no gold labels or policy names. Candidate order is hash-randomized. Exact candidate quotations, target-ID references and the rule that the original cannot omit its own original targets are checked. Invalid responses were retried; old responses remain in the request/archive records.

Case 4065 repeatedly extracted source facts absent from the original as preservation targets. Assistant inspection of the original and source supplied three supported original substrings; the target override and evidence are disclosed in `analysis/bandha_actual_editing_results/source_audit/4065_target_override.json`. Factuality verdicts were still produced by the independent assessor. This is not a human annotation study.

Identical returned texts use a common assessment: the original assessment for unchanged text, otherwise the first candidate in the blinded order. Raw assessments remain available. No correction or threshold was selected using these outcomes. No repair/harm probabilities were fitted or simulated. The source-disjoint balanced pilot excludes long inputs and is not an estimate of full-benchmark prevalence or deployed performance.

## Saved deliverables

- `analysis/bandha_actual_editing_results/summary.json`: factuality, preservation, scalar and interval readouts.
- `analysis/bandha_actual_editing_results/factual_outcomes.pdf`: capped unsupported-claim curves and differences from unedited outputs.
- `analysis/bandha_actual_editing_results/actual_outcomes.pdf`: explicitly combined factuality/preservation event curves.
- `analysis/bandha_actual_editing_results/cases.json`, `outputs/`, `assessments/`, and `requests/`: source evidence, actual returned text, judgments and raw requests.
- `analysis/bandha_actual_editing_results/bootstrap.npz`: paired draws and curve arrays.

Recompute the report without model inference:

```sh
python analysis/bandha_actual_editing_summary.py --results analysis/bandha_actual_editing_results
```

The executed pilot is summarized in Section 3 of the live manuscript. Its larger output records and plots remain in the supplement; the manuscript still has seven main pages.
