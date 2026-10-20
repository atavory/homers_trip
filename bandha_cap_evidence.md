# Capped remaining-error evidence

This experiment compares ideal annotation coverage on 2,675 RAGTruth responses from 450 sources. It uses a fixed 20% global inspection rule, with exact cutoff-tie expectations. It does not measure actual edited outputs.

For each procedure the curve is the expected number of remaining annotated words capped at each integer t. Its first point is the error-containing fraction; its final point is mean remaining words. Worst-case regret is the greatest capped disadvantage relative to the best whole procedure in the named comparison set. This equals regret over increasing concave losses normalized to unit first-error damage.

The original three-procedure set has regrets 0.118 (lexical + metadata), 0.417 (lexical), and 0.589 (generator metadata). Joint source-bootstrap regret bounds are respectively [0.066, 0.197], [0.165, 0.713], and [0.230, 0.950]. Generator and combined support opposite preferences at small and large caps. Combined dominates lexical in the fixed benchmark, but simultaneous bounds do not establish population dominance at every cap.

## Reproduction

Requires NumPy, SciPy, and Matplotlib. The optional checkpoint inference requires PyTorch 2.2.2 and Transformers 4.49.0; model revision bbd77832f52f9bd87546a3924c032467921f5c34 is pinned in the script. Frozen baseline predictions are included.

```sh
python analysis/bandha_verify_cap_evidence.py --inputs analysis/bandha_cap_inputs --saved-exact analysis/bandha_cap_inputs/exact_reference.json --out analysis/bandha_cap_baseline_results/verification.json
python analysis/bandha_cap_evidence.py --scores analysis/bandha_cap_inputs/baseline_scores.npz --index analysis/bandha_cap_inputs/response_index.json --out analysis/bandha_cap_baseline_results --draws 5000 --workers 4
```

The verifier checks 18 exhaustively enumerated allocation fixtures, all previously rationally checked capped curves, and loss-mixture/regret identities: 1,014 checks in total. Every bootstrap replicate resamples complete sources and recomputes the allocation cutoff and uniform tie law. Joint max-t bounds cover all listed pairs and caps. They are approximate bootstrap intervals, conditional on fixed fitted scores. Scores, source IDs, source draws, and scalar/curve outputs are retained. Large bootstrap-curve caches are regenerated rather than included.

The trained-detector extension is being executed separately and will have its own fixed comparison set and results. Do not compare regret numbers across different candidate sets.
