# Actual correction pilot

Status: prepared; no empirical outcome claim yet.

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
