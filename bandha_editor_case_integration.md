# Conditional editor example integrated into the live paper

The paper now includes the checked imperfect-correction example and the operating-point reliability requirements, alongside the fixed-reference error and work curves.

- Section 2 gives a sufficient bound over every admitted increasing concave error loss using lower repair and upper corruption bounds. It does not require independent edits.
- Section 3 compares two policies on the same frozen LettuceDetect scores. At hypothetical repair probability .9 and corruption probability .02, top-20% inspection removes more errors overall but increases error-containing responses by 9.66 percentage points. Word-score cutoff .5 decreases that fraction by 3.54 points.
- Word F1 already favors cutoff .5 (.529 versus .264). This example establishes a mean-error versus response-damage disagreement, not universal F1 failure.
- The cutoff policy has conservative worst excess loss -.0193 versus unedited responses under the stated local repair/harm bounds.
- The operating-point plot uses 5,000 paired source resamples, all 359 caps, and expected cutoff-tie confusion counts. Its intervals are pointwise source intervals; they do not estimate editor rates.

These are conditional calculations on frozen predictions. No editor was executed, no repair/corruption rates were measured, and word-check budgets do not establish equal review costs. Direct correction studies supply assessed errors in the returned outputs instead.

The manuscript builds at seven main pages, two reference pages and two appendix pages. The former combined error/work scalar is not recommended.

## Reproduce the manuscript table and figure

From the repository root, run the existing scientific Python environment with:

```sh
python analysis/bandha_editor_manuscript.py --joint-results analysis/bandha_joint_work_results --editor-results analysis/bandha_editor_frontier_results --operating-results analysis/bandha_editor_operating_point_results --paper /path/to/paper --out analysis/bandha_editor_manuscript_results
```

The figure distinguishes simultaneous intervals for the ideal error/work comparison from pointwise intervals for the reliability frontier. See `bandha_editor_frontier.md` and `bandha_editor_operating_points.md` for assumptions and the independent checks.
