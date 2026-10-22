# Prior-art and novelty clarification: Dolan and Moré

The original paper and publisher metadata were checked on 2026-10-21. Verified reference: Elizabeth D. Dolan and Jorge J. Moré, “Benchmarking Optimization Software with Performance Profiles,” Mathematical Programming 91(2), 201–213 (2002), DOI 10.1007/s101070100263. Original text: arXiv cs/0102001v2, Section 2.

## What is established prior art

- Reporting curves rather than choosing one arbitrary summary is established benchmarking practice.
- The Dolan–Moré profile is a CDF of performance ratios relative to the best listed solver on each problem. It can use metrics other than runtime.
- Our capped expectation curve is not that CDF: its axis is the saturation count in a loss, and its ordinate is expected capped residual errors. Increasing-concave ordering, capped mixtures and their expectation identities are also classical; the manuscript already credits them.
- Absolute curves and a fixed-reference scalar avoid competitor-set dependence by design. This is a useful evaluation convention, not a new mathematical principle.
- Reporting two performance coordinates is not itself a novelty claim.

## Supported contribution

The paper specifies how to assess residual errors and optional work for detection-guided or direct correction, computes comparisons throughout a declared loss class, and reports fixed-reference worst excess loss. Its frozen trained-detector experiment demonstrates improvement throughout that class at the primary ideal-correction operating point. Its conditional editor example shows that greater mean error removal can coincide with more error-containing responses, and supplies sufficient reliability requirements with paired source uncertainty without requiring an edit-independence model.

No editor was executed and no actual repair/corruption rates were measured. The q≈.002 values concern selected simple-scorer policies, not every trained-detector policy. The trained cutoff-.5 policy is a counterexample to the claim that every fallible editor makes things worse. Ordinary word F1 already prefers cutoff .5 in the demonstrated policy comparison; this is not a universal detection-metric failure result.

## AIS agreement interpretation

AIS motivates a complete-support endpoint under compatible assessment and target coverage. Its reported annotation disagreements do not establish that this endpoint is harder to measure than every other cap.

Under independent per-claim disagreements with probability epsilon, 1-(1-epsilon)^k is the probability of at least one claim disagreement. It is generally only an upper bound on disagreement between conjunctive response labels: an unsupported claim agreed upon by both annotators can make both labels negative regardless of disagreements elsewhere. A length trend in response agreement or crowd–expert gaps therefore needs additional assumptions and empirical verification. No such new measurement claim was added.

## Live manuscript changes

The introduction now cites AIS for complete support and GenAudit for response/evidence review. The discussion cites Dolan–Moré and states the exact distinction between its performance-ratio CDF and capped absolute outcomes. No empirical thesis was changed.
