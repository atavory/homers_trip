# Current character-control replay

Run from this directory with the main review package's scientific Python
environment (Python 3.12, NumPy 2.5.4, SciPy 1.18.1):

```bash
python -I verify_character_controls.py \
  --ragtruth /path/to/RAGTruth \
  --out results
```

The explicit cache path needs only `dataset/response.jsonl` and
`dataset/source_info.jsonl`. The input hashes are checked against `datasets.json`.
Use a new or empty output directory, and do not run with `-O`.

The adapter imports the unchanged public `bandha_span_evaluator_control.py`,
calls its original loader, metadata fitting, threshold selection and control
functions, and loads only `evaluate_detector_char_level` from the vendored
source extract. The function body is unchanged; its exact AST hash is checked.
Postponed annotations and a progress-iterator stub avoid importing any neural
dependencies. No Torch, sklearn, model weights, inference, Mu-SHROOM dataset or
Git metadata are required.

The evaluator was obtained from LettuceDetect revision
`7947f415b9e1fbecb0b97fee9e2e9e80305cb196`. The raw source URL, original full-file
SHA256, function AST SHA256 and extract SHA256 are recorded in
`vendor/source_manifest.json`. The MIT license, copyright KR Labs 2025, is
preserved in `vendor/LICENSE_LettuceDetect`.

Outputs:

- `results.json`: full precision controls, fitting/threshold details, selected
  response IDs, independent arithmetic and global optimality certificates.
- `ragtruth_controls.csv`: mark-none, mark-all, metadata character/any-error,
  and held-out-label whole-response ceiling on the main good-quality panel
  and the all-quality sensitivity panel.
- `character_values.tex`: current character control bindings, including
  `character.metadata_char.f1` = `0.108` and
  `character.label_informed_whole_response_ceiling.f1` = `0.363`.

The direct check independently re-reads the raw JSONLs and computes exact
fractions from raw gold-span lengths, covered gold mass and selected response
length. It also proves the label-informed optimum over every whole-response
subset using a signed-advantage bound at the attained F1. This is independent
of the original sorted-prefix ceiling implementation.

Gold lengths and predicted/gold pair intersections follow the released raw-span
evaluator; overlapping annotations can contribute repeated mass. The main panel
has gold-span length sum 78,515 and merged gold union 78,180. These are separate
from the paper's binary word mask. The .363 result is a label-informed upper
bound for this whole-response strategy. It is not a deployable score, a floor,
or an additive attribution for a published neural detector. No published neural
score is rerun or compared under an unmatched protocol here.
