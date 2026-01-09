"""
knn_classifier.py

Unit 4: Machine Learning Algorithms

Objective:
- Implement K-Nearest Neighbors from scratch
- Understand distance-based classification
- Analyze effect of k on predictions
"""

import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# -------------------------------
# 1. DISTANCE FUNCTION
# -------------------------------

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# -------------------------------
# 2. KNN CLASSIFIER (SCRATCH)
# -------------------------------

class KNNClassifierScratch:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []

        for x in X:
            distances = [
                euclidean_distance(x, x_train)
                for x_train in self.X_train
            ]

            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]

            most_common = Counter(k_labels).most_common(1)
            predictions.append(most_common[0][0])

        return np.array(predictions)


# -------------------------------
# 3. CREATE DATASET
# -------------------------------

X = np.array([
    [21, 8.1],
    [22, 8.7],
    [23, 7.9],
    [24, 9.1],
    [22, 8.4],
    [25, 9.0]
])

y = np.array([0, 1, 0, 1, 1, 1])


# -------------------------------
# 4. TRAIN SCRATCH MODEL
# -------------------------------

knn_scratch = KNNClassifierScratch(k=3)
knn_scratch.fit(X, y)

y_pred_scratch = knn_scratch.predict(X)


# -------------------------------
# 5. TRAIN SCIKIT-LEARN MODEL
# -------------------------------

knn_sklearn = KNeighborsClassifier(n_neighbors=3)
knn_sklearn.fit(X, y)

y_pred_sklearn = knn_sklearn.predict(X)


# -------------------------------
# 6. EVALUATION
# -------------------------------

accuracy_scratch = accuracy_score(y, y_pred_scratch)
accuracy_sklearn = accuracy_score(y, y_pred_sklearn)


# -------------------------------
# 7. EFFECT OF k (BIAS–VARIANCE)
# -------------------------------

def evaluate_k_values(X, y, k_values):
    results = {}
    for k in k_values:
        model = KNNClassifierScratch(k=k)
        model.fit(X, y)
        preds = model.predict(X)
        results[k] = accuracy_score(y, preds)
    return results


k_analysis = evaluate_k_values(X, y, k_values=[1, 3, 5])


# -------------------------------
# 8. RESULTS SUMMARY
# -------------------------------

results = {
    "Scratch Accuracy": accuracy_scratch,
    "Sklearn Accuracy": accuracy_sklearn,
    "K Analysis": k_analysis
}


# -------------------------------
# 9. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    for key, value in results.items():
        print(f"{key}: {value}")

    print("knn_classifier.py executed successfully")
