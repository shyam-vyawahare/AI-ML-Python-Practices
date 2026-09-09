"""
Feature Scaling & Preprocessing Pipeline

Practice:
- Separating numerical and categorical features
- StandardScaler
- OneHotEncoder
- ColumnTransformer
- Pipeline
- Train/test preprocessing
- Preventing data leakage
- Transforming unseen data
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------
# 1. Sample Dataset
# ---------------------------------------------------------

data = pd.DataFrame(
    {
        "age": [22, 25, 28, 35, 42, 45, 52, 60, 23, 31],
        "income": [
            25000,
            32000,
            40000,
            55000,
            70000,
            85000,
            95000,
            120000,
            28000,
            50000,
        ],
        "city": [
            "Mumbai",
            "Pune",
            "Mumbai",
            "Delhi",
            "Pune",
            "Delhi",
            "Mumbai",
            "Delhi",
            "Pune",
            "Mumbai",
        ],
        "experience_level": [
            "Junior",
            "Junior",
            "Mid",
            "Mid",
            "Senior",
            "Senior",
            "Senior",
            "Senior",
            "Junior",
            "Mid",
        ],
        "purchased": [
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
        ],
    }
)

print("Original Dataset:")
print(data)


# ---------------------------------------------------------
# 2. Separate Features and Target
# ---------------------------------------------------------

X = data.drop("purchased", axis=1)
y = data["purchased"]

print("\nFeatures:")
print(X)

print("\nTarget:")
print(y)


# ---------------------------------------------------------
# 3. Identify Feature Types
# ---------------------------------------------------------

numeric_features = [
    "age",
    "income",
]

categorical_features = [
    "city",
    "experience_level",
]


# ---------------------------------------------------------
# 4. Create Numerical Preprocessor
# ---------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
    ]
)


# ---------------------------------------------------------
# 5. Create Categorical Preprocessor
# ---------------------------------------------------------

categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)


# ---------------------------------------------------------
# 6. Combine Preprocessors
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features,
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features,
        ),
    ]
)


# ---------------------------------------------------------
# 7. Transform Training Data
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

print("\nOriginal Training Shape:")
print(X_train.shape)

print("\nTransformed Training Shape:")
print(X_train_transformed.shape)


# ---------------------------------------------------------
# 8. Inspect Generated Feature Names
# ---------------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

print("\nGenerated Features:")

for feature in feature_names:
    print(feature)


# ---------------------------------------------------------
# 9. Convert Transformed Data Back to DataFrame
# ---------------------------------------------------------

X_train_df = pd.DataFrame(
    X_train_transformed,
    columns=feature_names,
    index=X_train.index,
)

print("\nTransformed Training Data:")
print(X_train_df)


# ---------------------------------------------------------
# 10. Build Complete ML Pipeline
# ---------------------------------------------------------
# The preprocessing and model now become one object.

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
            ),
        ),
    ]
)


# ---------------------------------------------------------
# 11. Train the Complete Pipeline
# ---------------------------------------------------------

model_pipeline.fit(
    X_train,
    y_train,
)


# ---------------------------------------------------------
# 12. Make Predictions
# ---------------------------------------------------------

predictions = model_pipeline.predict(X_test)

print("\nPredictions:")
print(predictions)

print("\nActual Values:")
print(y_test.to_numpy())


# ---------------------------------------------------------
# 13. Evaluate Model
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions,
)

print(f"\nAccuracy: {accuracy:.2f}")


# ---------------------------------------------------------
# 14. Transform Completely New Data
# ---------------------------------------------------------
# Notice that Bangalore was never present in the
# training data.
#
# handle_unknown="ignore" prevents the encoder from
# crashing.

new_customers = pd.DataFrame(
    {
        "age": [29, 48],
        "income": [45000, 90000],
        "city": ["Bangalore", "Mumbai"],
        "experience_level": ["Junior", "Senior"],
    }
)

new_predictions = model_pipeline.predict(
    new_customers
)

print("\nNew Customers:")
print(new_customers)

print("\nPredictions for New Customers:")
print(new_predictions)


# ---------------------------------------------------------
# 15. Demonstrate the Correct Preprocessing Flow
# ---------------------------------------------------------

print("\nCorrect ML Preprocessing Flow:")

print(
    """
Raw Data
   ↓
Separate Features / Target
   ↓
Train / Test Split
   ↓
Identify Feature Types
   ↓
Numerical Features → StandardScaler
   ↓
Categorical Features → OneHotEncoder
   ↓
ColumnTransformer
   ↓
ML Model
   ↓
Predictions
"""
)


# ---------------------------------------------------------
# 16. Important Leakage Rule
# ---------------------------------------------------------

print("\nData Leakage Rule:")

print(
    "Fit preprocessing only on training data."
)

print(
    "Use transform() on validation/test/new data."
)

print(
    "A Pipeline automatically keeps preprocessing "
    "and model training together."
)
