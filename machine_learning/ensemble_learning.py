"""
ensemble_learning.py

Unit 4: Machine Learning

Objective:
- Learn ensemble learning techniques
- Compare multiple ensemble classifiers
- Understand voting, bagging, random forests, and boosting
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    VotingClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier
)


# -------------------------------
# 1. LOAD DATASET
# -------------------------------

iris = load_iris()

X = iris.data
y = iris.target

print("Dataset Shape:")
print(X.shape)


# -------------------------------
# 2. TRAIN / TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------------
# 3. BASE MODELS
# -------------------------------

logistic_model = LogisticRegression(
    max_iter=1000
)

tree_model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)


# -------------------------------
# 4. VOTING CLASSIFIER
# -------------------------------

voting_model = VotingClassifier(
    estimators=[
        ("logistic", logistic_model),
        ("tree", tree_model)
    ],
    voting="soft"
)

voting_model.fit(
    X_train,
    y_train
)

voting_predictions = voting_model.predict(
    X_test
)

voting_accuracy = accuracy_score(
    y_test,
    voting_predictions
)


# -------------------------------
# 5. BAGGING
# -------------------------------

bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=4
    ),
    n_estimators=50,
    random_state=42
)

bagging_model.fit(
    X_train,
    y_train
)

bagging_predictions = bagging_model.predict(
    X_test
)

bagging_accuracy = accuracy_score(
    y_test,
    bagging_predictions
)


# -------------------------------
# 6. RANDOM FOREST
# -------------------------------

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

random_forest.fit(
    X_train,
    y_train
)

forest_predictions = random_forest.predict(
    X_test
)

forest_accuracy = accuracy_score(
    y_test,
    forest_predictions
)


# -------------------------------
# 7. GRADIENT BOOSTING
# -------------------------------

gradient_boosting = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gradient_boosting.fit(
    X_train,
    y_train
)

boosting_predictions = gradient_boosting.predict(
    X_test
)

boosting_accuracy = accuracy_score(
    y_test,
    boosting_predictions
)


# -------------------------------
# 8. COMPARE MODELS
# -------------------------------

results = {
    "Voting Classifier": voting_accuracy,
    "Bagging": bagging_accuracy,
    "Random Forest": forest_accuracy,
    "Gradient Boosting": boosting_accuracy
}

print("\nModel Comparison:")

for model_name, accuracy in results.items():
    print(
        f"{model_name:20s}: "
        f"{accuracy:.4f}"
    )


# -------------------------------
# 9. RANDOM FOREST FEATURE IMPORTANCE
# -------------------------------

feature_importance = random_forest.feature_importances_

print("\nRandom Forest Feature Importance:")

for feature, importance in zip(
    iris.feature_names,
    feature_importance
):
    print(
        f"{feature:25s}: "
        f"{importance:.4f}"
    )


# -------------------------------
# 10. CONCEPT SUMMARY
# -------------------------------

print("\nEnsemble Learning Concepts:")

print("- Voting     -> Combines predictions from different models")
print("- Bagging    -> Trains models on different data samples")
print("- Random Forest -> Combines many decision trees")
print("- Boosting   -> Sequentially improves weak learners")


# -------------------------------
# 11. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print(
        "\nensemble_learning.py "
        "executed successfully"
)
