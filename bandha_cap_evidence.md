# Capped remaining-error evaluation

The active paper compares remaining annotation errors across increasing concave damage functions normalized to unit first-error loss. The capped-error curve reports expected min(E,t); worst-case regret is the greatest capped disadvantage relative to the best whole procedure in a named comparison set. Checking costs are a separate component of detection evaluation.

## Executed evidence

- 2,675 RAGTruth test/good responses, 450 source documents, 340,973 whitespace words and 13,083 positive word labels.
- Newly executed offline inference with KRLabsOrg/lettucedect-base-modernbert-en-v1, checkpoint revision bbd77832f52f9bd87546a3924c032467921f5c34. PyTorch 2.2.2 CPU float32, Transformers 4.49.0. All answer words receive scores; the longest input is 2,402 tokens, below the 4,096-token limit.
- Five procedures: generator metadata, lexical, lexical + metadata, LettuceDetect, and LettuceDetect response mean. All select exactly 68,195 words at the primary 20% operating point.
- Every selected error is assumed fixed without damage. These are ideal annotation-coverage outcomes, not assessed edited outputs.
- 5,000 paired source resamples recompute the global allocation, cutoff and uniform tie law, conditional on fixed fitted scores. Joint approximate max-t bounds cover all ten pairs and all 256 integer caps.

LettuceDetect improves on its mean control at every observed cap, and the simultaneous bounds support this dominance. Mean remaining words decrease from 1.995 to 0.878; error-containing responses decrease from 19.6% to 8.9%. The full checkpoint also dominates the three transparent baselines under those bounds.

In the five-procedure comparison, regrets [simultaneous bounds] are: generator 2.387 [1.665, 3.229]; lexical 2.215 [1.586, 3.042]; combined 1.798 [1.305, 2.385]; LettuceDetect 0 [0, 0]; response mean 1.116 [0.786, 1.447]. Zero regret refers to the listed alternatives and does not mean zero remaining errors.

In the original three-procedure set, regrets are 0.589, 0.417, and 0.118 respectively. Generator and combined cross between caps 11 and 12; simultaneous bounds support opposite preferences at small and large caps. Regret depends on the comparison set. Combined dominates lexical on the observed benchmark; uncertainty does not establish that dominance at every cap for future responses.

## Reproduction

Requires NumPy, SciPy and Matplotlib. Optional checkpoint inference requires PyTorch 2.2.2 and Transformers 4.49.0. Frozen word scores, paired response/source IDs, source draws, curves and scalar outputs are included. Large regenerated bootstrap-curve caches are omitted.

```sh
python analysis/bandha_verify_cap_evidence.py --inputs analysis/bandha_cap_inputs --saved-exact analysis/bandha_cap_inputs/exact_reference.json --out analysis/bandha_cap_baseline_results/verification.json
python analysis/bandha_cap_evidence.py --scores analysis/bandha_cap_inputs/trained_scores.npz --index analysis/bandha_cap_inputs/response_index.json --out analysis/bandha_cap_trained_results --draws 5000 --workers 4
```

The baseline verifier checks 18 exhaustive allocation fixtures and previously rationally checked curves, mixture identities and regret: 1,014 checks. The trained verifier independently reconstructs both trained curves with exact rational cutoff laws, checks 529 values and compares an independent 12-response inference run; maximum rational curve error is 3.2e-15 and batch score difference is zero.

`analysis/bandha_cap_trained_results/results.json` contains all five-procedure results. `analysis/bandha_cap_baseline_results/results.json` contains the original three-procedure comparison. The same source resampling seed is used. `analysis/bandha_prepare_cap_inputs.py` reconstructs baseline fits and merges inference shards.

The method and experiments in the active paper use capped errors and regret. Earlier profile, generating-function and editor-sensitivity experiments remain archived material. The current source bundle includes only files used by the active build.
