"""
probability_simulation.py

Unit 3: Math for Machine Learning

Objective:
- Understand probability via simulation
- Build intuition using randomness and experiments
- Learn concepts used in stochastic ML systems
"""

import numpy as np


# -------------------------------
# 1. BASIC PROBABILITY (COIN TOSS)
# -------------------------------

def coin_toss(trials: int = 1000):
    tosses = np.random.choice(["H", "T"], size=trials)
    heads_prob = np.mean(tosses == "H")
    tails_prob = np.mean(tosses == "T")
    return heads_prob, tails_prob


# -------------------------------
# 2. DICE ROLL SIMULATION
# -------------------------------

def dice_roll(trials: int = 1000):
    rolls = np.random.randint(1, 7, size=trials)
    probabilities = {}
    for face in range(1, 7):
        probabilities[face] = np.mean(rolls == face)
    return probabilities


# -------------------------------
# 3. CONDITIONAL PROBABILITY
# -------------------------------

def conditional_probability(trials: int = 10000):
    dice = np.random.randint(1, 7, size=trials)

    event_A = dice > 3          # dice > 3
    event_B = dice % 2 == 0     # dice is even

    p_A = np.mean(event_A)
    p_B = np.mean(event_B)
    p_A_given_B = np.mean(event_A[event_B])

    return p_A, p_B, p_A_given_B


# -------------------------------
# 4. BAYES INTUITION (SIMULATION)
# -------------------------------

def bayes_simulation(trials: int = 10000):
    disease = np.random.choice([0, 1], size=trials, p=[0.99, 0.01])
    test_positive = []

    for d in disease:
        if d == 1:
            test_positive.append(np.random.rand() < 0.9)   # true positive
        else:
            test_positive.append(np.random.rand() < 0.05)  # false positive

    test_positive = np.array(test_positive)

    prob_disease_given_positive = np.mean(disease[test_positive] == 1)
    return prob_disease_given_positive


# -------------------------------
# 5. EXPECTATION (AVERAGE OUTCOME)
# -------------------------------

def expected_value(trials: int = 10000):
    dice = np.random.randint(1, 7, size=trials)
    return np.mean(dice)


# -------------------------------
# 6. LAW OF LARGE NUMBERS
# -------------------------------

def law_of_large_numbers():
    samples = np.random.randint(1, 7, size=10000)
    running_means = [np.mean(samples[:i]) for i in range(1, 10000)]
    return running_means


# -------------------------------
# 7. MONTE CARLO ESTIMATION (π)
# -------------------------------

def estimate_pi(points: int = 100000):
    x = np.random.rand(points)
    y = np.random.rand(points)

    inside_circle = (x ** 2 + y ** 2) <= 1
    pi_estimate = 4 * np.mean(inside_circle)
    return pi_estimate


# -------------------------------
# 8. RANDOMNESS IN ML
# -------------------------------

def random_initial_weights(size: int = 5):
    return np.random.randn(size)


# -------------------------------
# 9. STOCHASTIC BEHAVIOR
# -------------------------------

def stochastic_gradient_simulation():
    gradients = np.random.normal(0, 1, size=1000)
    return np.mean(gradients), np.std(gradients)


# -------------------------------
# 10. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("Coin toss:", coin_toss())
    print("Dice roll:", dice_roll())
    print("Conditional probability:", conditional_probability())
    print("Bayes simulation:", bayes_simulation())
    print("Expected dice value:", expected_value())
    print("Estimated Pi:", estimate_pi())
    print("probability_simulation.py executed successfully")
