"""
probability_distributions.py

Unit 3: Math for Machine Learning

Objective:
- Practice common probability distributions
- Calculate mean and variance
- Generate random samples
- Understand their applications in ML
"""

import numpy as np


# -------------------------------
# 1. RANDOM SEED
# -------------------------------

np.random.seed(42)


# -------------------------------
# 2. BERNOULLI DISTRIBUTION
# -------------------------------

# A Bernoulli variable has only
# two possible outcomes: 0 or 1.

probability_success = 0.7

bernoulli_samples = np.random.binomial(
    n=1,
    p=probability_success,
    size=10
)

print("Bernoulli Samples:")
print(bernoulli_samples)

print(
    "Sample Mean:",
    round(bernoulli_samples.mean(), 4)
)


# -------------------------------
# 3. BINOMIAL DISTRIBUTION
# -------------------------------

# Number of successes in multiple
# independent Bernoulli trials.

binomial_samples = np.random.binomial(
    n=10,
    p=0.7,
    size=10
)

print("\nBinomial Samples:")
print(binomial_samples)

print(
    "Sample Mean:",
    round(binomial_samples.mean(), 4)
)


# -------------------------------
# 4. UNIFORM DISTRIBUTION
# -------------------------------

uniform_samples = np.random.uniform(
    low=0,
    high=10,
    size=10
)

print("\nUniform Samples:")
print(np.round(uniform_samples, 2))

print(
    "Sample Mean:",
    round(uniform_samples.mean(), 4)
)

print(
    "Sample Variance:",
    round(uniform_samples.var(), 4)
)


# -------------------------------
# 5. NORMAL DISTRIBUTION
# -------------------------------

mean = 50
std = 10

normal_samples = np.random.normal(
    loc=mean,
    scale=std,
    size=1000
)

print("\nNormal Distribution:")

print(
    "Sample Mean:",
    round(normal_samples.mean(), 2)
)

print(
    "Sample Standard Deviation:",
    round(normal_samples.std(), 2)
)


# -------------------------------
# 6. Z-SCORE
# -------------------------------

value = 70

z_score = (
    value - mean
) / std

print("\nZ-Score:")

print(
    f"Value: {value}"
)

print(
    f"Z-Score: {z_score:.2f}"
)


# -------------------------------
# 7. STANDARD NORMAL DISTRIBUTION
# -------------------------------

standard_normal = np.random.normal(
    loc=0,
    scale=1,
    size=1000
)

print("\nStandard Normal Distribution:")

print(
    "Mean:",
    round(standard_normal.mean(), 3)
)

print(
    "Standard Deviation:",
    round(standard_normal.std(), 3)
)


# -------------------------------
# 8. EMPIRICAL PROBABILITY
# -------------------------------

# Estimate probability that a
# standard normal value lies
# between -1 and +1.

inside_range = np.sum(
    (standard_normal >= -1)
    &
    (standard_normal <= 1)
)

estimated_probability = (
    inside_range
    / len(standard_normal)
)

print(
    "\nEstimated P(-1 <= X <= 1):",
    round(estimated_probability, 4)
)


# -------------------------------
# 9. PERCENTILES
# -------------------------------

percentiles = np.percentile(
    normal_samples,
    [25, 50, 75]
)

print("\nNormal Distribution Percentiles:")

print(
    "25th Percentile:",
    round(percentiles[0], 2)
)

print(
    "50th Percentile:",
    round(percentiles[1], 2)
)

print(
    "75th Percentile:",
    round(percentiles[2], 2)
)


# -------------------------------
# 10. ML APPLICATIONS
# -------------------------------

print("\nML Applications:")

print(
    "- Bernoulli → Binary classification outcomes"
)

print(
    "- Binomial → Number of successes across trials"
)

print(
    "- Normal → Noise, features, statistical modeling"
)

print(
    "- Uniform → Random initialization and simulation"
)

print(
    "- Z-Score → Feature standardization"
)


# -------------------------------
# 11. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print(
        "\nprobability_distributions.py "
        "executed successfully"
)
