"""
statistical_inference.py

Unit 3: Math for Machine Learning

Objective:
- Practice statistical inference concepts
- Understand sampling and the Central Limit Theorem
- Calculate confidence intervals
- Perform a basic hypothesis test
"""

import numpy as np


# -------------------------------
# 1. CREATE POPULATION
# -------------------------------

np.random.seed(42)

population = np.random.normal(
    loc=70,
    scale=10,
    size=10000
)

print("Population Statistics:")
print(
    f"Mean: {population.mean():.2f}"
)

print(
    f"Standard Deviation: "
    f"{population.std():.2f}"
)


# -------------------------------
# 2. DRAW A SAMPLE
# -------------------------------

sample = np.random.choice(
    population,
    size=100,
    replace=False
)

sample_mean = sample.mean()
sample_std = sample.std(ddof=1)

print("\nSample Statistics:")

print(
    f"Sample Mean: "
    f"{sample_mean:.2f}"
)

print(
    f"Sample Standard Deviation: "
    f"{sample_std:.2f}"
)


# -------------------------------
# 3. STANDARD ERROR
# -------------------------------

standard_error = (
    sample_std
    / np.sqrt(len(sample))
)

print("\nStandard Error:")
print(
    f"{standard_error:.4f}"
)


# -------------------------------
# 4. CONFIDENCE INTERVAL
# -------------------------------

confidence_level = 0.95

# Approximate z-value for 95% CI
z_value = 1.96

margin_of_error = (
    z_value * standard_error
)

lower_bound = (
    sample_mean - margin_of_error
)

upper_bound = (
    sample_mean + margin_of_error
)

print("\n95% Confidence Interval:")

print(
    f"Lower Bound: "
    f"{lower_bound:.2f}"
)

print(
    f"Upper Bound: "
    f"{upper_bound:.2f}"
)


# -------------------------------
# 5. CENTRAL LIMIT THEOREM
# -------------------------------

sample_means = []

for _ in range(1000):

    random_sample = np.random.choice(
        population,
        size=30,
        replace=False
    )

    sample_means.append(
        random_sample.mean()
    )

sample_means = np.array(
    sample_means
)

print("\nCentral Limit Theorem:")

print(
    f"Mean of Sample Means: "
    f"{sample_means.mean():.2f}"
)

print(
    f"Std of Sample Means: "
    f"{sample_means.std():.2f}"
)


# -------------------------------
# 6. HYPOTHESIS TESTING
# -------------------------------

# H0: Population mean = 70
# H1: Population mean != 70

hypothesized_mean = 70

z_score = (
    sample_mean - hypothesized_mean
) / standard_error

print("\nHypothesis Test:")

print(
    f"Sample Mean: "
    f"{sample_mean:.2f}"
)

print(
    f"Hypothesized Mean: "
    f"{hypothesized_mean:.2f}"
)

print(
    f"Z-Score: "
    f"{z_score:.4f}"
)


# -------------------------------
# 7. APPROXIMATE TWO-TAILED P-VALUE
# -------------------------------

# Approximate standard normal CDF
normal_cdf = (
    0.5
    * (
        1
        + np.math.erf(
            abs(z_score)
            / np.sqrt(2)
        )
    )
)

p_value = 2 * (
    1 - normal_cdf
)

print(
    f"P-Value: "
    f"{p_value:.6f}"
)


# -------------------------------
# 8. DECISION
# -------------------------------

alpha = 0.05

if p_value < alpha:

    print(
        "\nDecision: Reject the null hypothesis."
    )

else:

    print(
        "\nDecision: Fail to reject the null hypothesis."
    )


# -------------------------------
# 9. SIMULATION OF SAMPLE MEANS
# -------------------------------

print("\nSample Mean Distribution:")

print(
    f"Minimum: "
    f"{sample_means.min():.2f}"
)

print(
    f"Maximum: "
    f"{sample_means.max():.2f}"
)

print(
    f"Mean: "
    f"{sample_means.mean():.2f}"
)

print(
    f"Standard Deviation: "
    f"{sample_means.std():.2f}"
)


# -------------------------------
# 10. ML APPLICATIONS
# -------------------------------

print("\nML Applications:")

print(
    "- Estimating population statistics"
)

print(
    "- A/B testing"
)

print(
    "- Model performance comparison"
)

print(
    "- Feature analysis"
)

print(
    "- Experiment evaluation"
)

print(
    "- Understanding uncertainty"
)


# -------------------------------
# 11. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print(
        "\nstatistical_inference.py "
        "executed successfully"
)
