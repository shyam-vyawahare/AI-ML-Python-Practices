"""
bayes_theorem.py

Unit 3: Math for Machine Learning

Objective:
- Understand conditional probability
- Implement Bayes' theorem
- Calculate prior, likelihood, evidence, and posterior
- Apply Bayes' theorem to a simple classification problem
"""

# -------------------------------
# 1. BASIC PROBABILITY
# -------------------------------

total_students = 100

python_students = 60
ml_students = 40

prob_python = python_students / total_students
prob_ml = ml_students / total_students

print("Basic Probability")
print(f"P(Python) = {prob_python:.2f}")
print(f"P(ML)     = {prob_ml:.2f}")


# -------------------------------
# 2. CONDITIONAL PROBABILITY
# -------------------------------

# Students who know Python
# and are interested in ML

python_and_ml = 30

prob_ml_given_python = (
    python_and_ml / python_students
)

print("\nConditional Probability")

print(
    f"P(ML | Python) = "
    f"{prob_ml_given_python:.2f}"
)


# -------------------------------
# 3. BAYES' THEOREM
# -------------------------------

def bayes_theorem(
    prior,
    likelihood,
    evidence
):
    """
    Bayes' theorem:

    P(A | B) =
    P(B | A) * P(A) / P(B)
    """

    return (
        likelihood * prior
    ) / evidence


# -------------------------------
# 4. SIMPLE BAYES EXAMPLE
# -------------------------------

# Probability of having a disease
prior_disease = 0.01

# Probability of positive test
# when disease is present
likelihood_positive_given_disease = 0.95

# Probability of positive test
evidence_positive = 0.05

posterior = bayes_theorem(
    prior_disease,
    likelihood_positive_given_disease,
    evidence_positive
)

print("\nBayes' Theorem Example")

print(
    f"P(Disease | Positive Test) = "
    f"{posterior:.4f}"
)


# -------------------------------
# 5. ML-STYLE CLASSIFICATION
# -------------------------------

# Suppose we want to determine
# whether an email is spam.

# Prior probabilities
p_spam = 0.30
p_not_spam = 0.70

# Probability of seeing the word
# "offer" in each class
p_offer_given_spam = 0.80
p_offer_given_not_spam = 0.10

# Evidence:
#
# P(Offer) =
# P(Offer|Spam)P(Spam)
# +
# P(Offer|NotSpam)P(NotSpam)

p_offer = (
    p_offer_given_spam * p_spam
    +
    p_offer_given_not_spam * p_not_spam
)


# Posterior probability
p_spam_given_offer = (
    p_offer_given_spam * p_spam
) / p_offer


print("\nSpam Classification Example")

print(
    f"P(Spam)              = {p_spam:.2f}"
)

print(
    f"P(Not Spam)          = {p_not_spam:.2f}"
)

print(
    f"P(Offer | Spam)      = "
    f"{p_offer_given_spam:.2f}"
)

print(
    f"P(Offer | Not Spam)  = "
    f"{p_offer_given_not_spam:.2f}"
)

print(
    f"P(Offer)             = "
    f"{p_offer:.4f}"
)

print(
    f"P(Spam | Offer)      = "
    f"{p_spam_given_offer:.4f}"
)


# -------------------------------
# 6. MAKE CLASSIFICATION
# -------------------------------

prediction = (
    "Spam"
    if p_spam_given_offer > 0.5
    else "Not Spam"
)

print(
    f"\nPrediction: {prediction}"
)


# -------------------------------
# 7. MULTIPLE FEATURES
# -------------------------------

# A simplified Naive Bayes-style example.

p_spam = 0.30
p_not_spam = 0.70

# Features:
# "offer" and "free"

p_offer_spam = 0.80
p_offer_not_spam = 0.10

p_free_spam = 0.70
p_free_not_spam = 0.05

# Naive Bayes assumes conditional
# independence between features.

spam_score = (
    p_spam
    * p_offer_spam
    * p_free_spam
)

not_spam_score = (
    p_not_spam
    * p_offer_not_spam
    * p_free_not_spam
)


# Normalize scores

total_score = (
    spam_score
    + not_spam_score
)

spam_probability = (
    spam_score / total_score
)

not_spam_probability = (
    not_spam_score / total_score
)


print("\nNaive Bayes-Style Example")

print(
    f"Spam Probability     : "
    f"{spam_probability:.4f}"
)

print(
    f"Not Spam Probability : "
    f"{not_spam_probability:.4f}"
)


# -------------------------------
# 8. CONCEPT SUMMARY
# -------------------------------

print("\nConcept Summary:")

print(
    "- Prior       → Probability before observing evidence"
)

print(
    "- Likelihood  → Probability of evidence given a class"
)

print(
    "- Evidence    → Overall probability of the evidence"
)

print(
    "- Posterior   → Updated probability after evidence"
)

print(
    "- Bayes       → Prior × Likelihood / Evidence"
)

print(
    "- Naive Bayes → Uses conditional independence assumption"
)


# -------------------------------
# 9. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print(
        "\nbayes_theorem.py "
        "executed successfully"
)
