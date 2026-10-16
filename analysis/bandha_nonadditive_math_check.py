"""Check finite witnesses for recovery utility and mixture-order claims."""

import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path


def law(pairs):
    output = {}
    for value, mass in pairs:
        value, mass = F(value), F(mass)
        output[value] = output.get(value, F(0)) + mass
    assert all(0 <= value <= 1 and mass >= 0 for value, mass in output.items())
    assert sum(output.values()) == 1
    return output


def integer_moment(distribution, exponent):
    assert exponent > 0
    return sum(mass * value ** exponent for value, mass in distribution.items())


def cdf(distribution, value):
    return sum(mass for location, mass in distribution.items() if location <= value)


def cdf_contrasts(first, second):
    return [(value, cdf(first, value) - cdf(second, value))
            for value in sorted(first.keys() | second.keys())]


def moment_endpoints(distribution):
    return {
        "touch": 1 - distribution.get(F(0), F(0)),
        "complete": distribution.get(F(1), F(0)),
    }


def encode(value):
    if isinstance(value, F):
        return {"exact": str(value), "decimal": float(value)}
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def make_results():
    # Positive at all powers, although neither recovery law dominates the other.
    power_a = law([(F(1, 4), F(1, 2)), (1, F(1, 2))])
    power_b = law([(0, F(1, 5)), (F(1, 2), F(4, 5))])
    gaps = cdf_contrasts(power_a, power_b)
    assert any(gap > 0 for _, gap in gaps)
    assert any(gap < 0 for _, gap in gaps)
    # With t = 2^(-alpha), gap = (t - 4/5)^2 / 2 + 9/50.
    for numerator in range(101):
        t = F(numerator, 100)
        polynomial = t * t / 2 - F(4, 5) * t + F(1, 2)
        assert polynomial == (t - F(4, 5)) ** 2 / 2 + F(9, 50)
        assert polynomial >= F(9, 50)

    # Both alpha=1 and alpha=infinity gaps are positive; alpha=2 is negative.
    roots_a = law([(F(1, 4), F(100, 103)), (1, F(3, 103))])
    roots_b = law([(0, F(63, 103)), (F(1, 2), F(40, 103))])
    root_gaps = {alpha: integer_moment(roots_a, alpha) - integer_moment(roots_b, alpha)
                 for alpha in (1, 2, 3, 4)}
    assert root_gaps[1] == F(8, 103)
    assert root_gaps[2] == -F(3, 412)
    assert root_gaps[4] > 0
    assert moment_endpoints(roots_a)["complete"] - moment_endpoints(roots_b)["complete"] == F(3, 103)
    for numerator in range(101):
        t = F(numerator, 100)
        polynomial = (100 * t * t - 40 * t + 3) / 103
        assert polynomial == F(100, 103) * (t - F(1, 10)) * (t - F(3, 10))

    # Strictly better recovery probability for each coordinate can lose completion.
    joint_a = {(1, 0): F(1, 2), (0, 1): F(1, 2)}
    joint_b = {(1, 1): F(2, 5), (0, 0): F(3, 5)}
    marginals_a = [sum(mass * outcome[i] for outcome, mass in joint_a.items()) for i in range(2)]
    marginals_b = [sum(mass * outcome[i] for outcome, mass in joint_b.items()) for i in range(2)]
    assert all(a > b for a, b in zip(marginals_a, marginals_b))
    complete_a = joint_a.get((1, 1), F(0))
    complete_b = joint_b.get((1, 1), F(0))
    assert complete_a < complete_b

    # Equal marginals also leave both complete and touched recovery undetermined.
    equal_b = {(1, 1): F(1, 2), (0, 0): F(1, 2)}
    equal_marginals = [sum(mass * outcome[i] for outcome, mass in equal_b.items()) for i in range(2)]
    assert equal_marginals == marginals_a
    assert equal_b.get((1, 1), F(0)) == F(1, 2)

    # Method-specific mixture weights can reverse groupwise utility dominance.
    utilities_a, utilities_b = (F(9, 10), F(3, 10)), (F(4, 5), F(1, 5))
    weights_a, weights_b = (F(1, 10), F(9, 10)), (F(9, 10), F(1, 10))
    assert all(a > b for a, b in zip(utilities_a, utilities_b))
    pooled_a = sum(weight * utility for weight, utility in zip(weights_a, utilities_a))
    pooled_b = sum(weight * utility for weight, utility in zip(weights_b, utilities_b))
    assert pooled_a == F(9, 25) < pooled_b == F(37, 50)

    # The grouped unit and its weights determine whether U_1 is word recall.
    sizes, recovered = (1, 3), (1, 0)
    pooled_word = F(sum(recovered), sum(sizes))
    macro_span_fraction = sum(F(found, size) for found, size in zip(recovered, sizes)) / 2
    uniform_complete = F(sum(found == size for found, size in zip(recovered, sizes)), 2)
    length_complete = sum(F(size, sum(sizes)) * (found == size)
                          for found, size in zip(recovered, sizes))
    assert pooled_word == length_complete == F(1, 4)
    assert macro_span_fraction == uniform_complete == F(1, 2)

    # Exact without-replacement null, and the failure of effective-length inversion.
    n, budget, span_size = 20, 2, 2
    fraction = F(budget, n)
    all_selected = F(math.comb(n - span_size, budget - span_size), math.comb(n, budget))
    assert all_selected == F(1, 190)
    assert all_selected / fraction == F(1, 19)
    beta = F(1, 10)
    heterogeneous_retention = (beta ** 0 + beta ** 2) / 2
    assert heterogeneous_retention == F(101, 200)
    effective_size = 1 + math.log(float(heterogeneous_retention)) / math.log(float(beta))
    assert effective_size < 2  # True mean span size of the length-1/length-3 mixture.

    return {
        "all_checks_passed": True,
        "power_order_without_fosd": {
            "a": power_a, "b": power_b, "cdf_a_minus_b": gaps,
            "gap_polynomial": "(t - 4/5)^2 / 2 + 9/50, t=2^(-alpha)",
            "strict_lower_bound": F(9, 50),
        },
        "same_sign_endpoints_two_roots": {
            "a": roots_a, "b": roots_b, "integer_moment_gaps": root_gaps,
            "endpoint_gaps": {key: moment_endpoints(roots_a)[key] - moment_endpoints(roots_b)[key]
                              for key in ("touch", "complete")},
            "gap_polynomial": "100/103 * (t - 1/10) * (t - 3/10), t=2^(-alpha)",
            "positive_alpha_roots": [math.log2(10 / 3), math.log2(10)],
        },
        "coordinate_improvement_completion_reversal": {
            "joint_a": joint_a, "joint_b": joint_b,
            "marginals_a": marginals_a, "marginals_b": marginals_b,
            "complete_a": complete_a, "complete_b": complete_b,
            "critical_alpha": math.log(0.4) / math.log(0.5),
        },
        "equal_marginals_different_joints": {
            "joint_a": joint_a, "joint_b": equal_b,
            "shared_marginals": marginals_a, "complete_a": complete_a,
            "complete_b": equal_b.get((1, 1), F(0)),
        },
        "simpson_method_specific_weights": {
            "utilities_a": utilities_a, "utilities_b": utilities_b,
            "weights_a": weights_a, "weights_b": weights_b,
            "pooled_a": pooled_a, "pooled_b": pooled_b,
        },
        "unit_and_weighting_mismatch": {
            "span_sizes": sizes, "recovered_words": recovered,
            "pooled_word_recall": pooled_word,
            "uniform_mean_span_fraction": macro_span_fraction,
            "uniform_complete_span_recall": uniform_complete,
            "length_weighted_complete_span_recall": length_complete,
        },
        "uniform_selection_null": {
            "n": n, "budget": budget, "span_size": span_size,
            "expected_recovery_fraction": fraction,
            "complete_probability": all_selected,
            "complete_to_fraction_ratio": all_selected / fraction,
            "independent_sampling_approximation": fraction ** (span_size - 1),
            "heterogeneous_length_1_3_retention": heterogeneous_retention,
            "heterogeneous_mean_length": 2,
            "effective_length_from_retention": effective_size,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("analysis/bandha_nonadditive_math_results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    result = make_results()
    result["checker_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (args.out / "verification.json").write_text(json.dumps(encode(result), indent=2) + "\n")
    print(json.dumps({"all_checks_passed": True, "checker_sha256": result["checker_sha256"],
                      "output": str(args.out / "verification.json")}))


if __name__ == "__main__":
    main()
