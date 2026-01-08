"""
statistics.py

Unit 3: Math for Machine Learning

Objective:
- Implement core statistical concepts using code
- Build intuition for data distribution and spread
- Understand statistics used in ML preprocessing and evaluation
"""

import numpy as np


# -------------------------------
# 1. SAMPLE DATASET
# -------------------------------

data = np.array([70, 75, 80, 85, 90, 95])


# -------------------------------
# 2. MEAN (AVERAGE)
# -------------------------------

mean = np.mean(data)


# -------------------------------
# 3. MEDIAN
# -------------------------------

median = np.median(data)


# -------------------------------
# 4. VARIANCE
# -------------------------------

variance = np.var(data)


# -------------------------------
# 5. STANDARD DEVIATION
# -------------------------------

std_dev = np.std(data)


# -------------------------------
# 6. MIN, MAX, RANGE
# -------------------------------

min_val = np.min(data)
max_val = np.max(data)
data_range = max_val - min_val


# -------------------------------
# 7. PERCENTILES
# -------------------------------

q25 = np.percentile(data, 25)
q50 = np.percentile(data, 50)
q75 = np.percentile(data, 75)


# -------------------------------
# 8. Z-SCORE NORMALIZATION
# -------------------------------

z_scores = (data - mean) / std_dev


# -------------------------------
# 9. MIN-MAX NORMALIZATION
# -------------------------------

min_max_scaled = (data - min_val) / (max_val - min_val)


# -------------------------------
# 10. POPULATION vs SAMPLE STATS
# -------------------------------

population_variance = np.var(data)
sample_variance = np.var(data, ddof=1)


# -------------------------------
# 11. RANDOM VARIABLES (SIMULATION)
# -------------------------------

normal_distribution = np.random.normal(loc=0, scale=1, size=1000)
uniform_distribution = np.random.uniform(low=0, high=1, size=1000)


# -------------------------------
# 12. LAW OF LARGE NUMBERS (INTUITION)
# -------------------------------

means = []
for i in range(1, 1000):
    means.append(np.mean(normal_distribution[:i]))


# -------------------------------
# 13. REAL ML PATTERN
# -------------------------------

# Standardizing features before training
features = np.array([50, 60, 70, 80, 90])

features_mean = np.mean(features)
features_std = np.std(features)

standardized_features = (features - features_mean) / features_std


# -------------------------------
# 14. NUMERICAL STABILITY CHECK
# -------------------------------

epsilon = 1e-8
safe_std = features_std + epsilon


# -------------------------------
# 15. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("statistics.py executed successfully")
