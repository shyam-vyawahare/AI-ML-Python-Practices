"""
feature_engineering.py

Unit 4: Machine Learning Algorithms

Objective:
- Prepare features correctly for ML models
- Apply scaling and encoding techniques
- Avoid common data leakage mistakes
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


# -------------------------------
# 1. SAMPLE DATASET
# -------------------------------

data = {
    "age": [21, 22, 23, 24, 22, 25],
    "cgpa": [8.1, 8.7, 7.9, 9.1, 8.4, 9.0],
    "branch": ["ECE", "CSE", "ECE", "CSE", "ME", "CSE"],
    "placed": [0, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)


# -------------------------------
# 2. FEATURE / LABEL SPLIT
# -------------------------------

X = df.drop("placed", axis=1)
y = df["placed"]


# -------------------------------
# 3. TRAIN / TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------------
# 4. NUMERICAL FEATURE SCALING
# -------------------------------

num_features = ["age", "cgpa"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[num_features])
X_test_scaled = scaler.transform(X_test[num_features])


# -------------------------------
# 5. MIN-MAX SCALING (ALTERNATIVE)
# -------------------------------

minmax = MinMaxScaler()
X_train_minmax = minmax.fit_transform(X_train[num_features])


# -------------------------------
# 6. CATEGORICAL ENCODING
# -------------------------------

encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

X_train_cat = encoder.fit_transform(X_train[["branch"]])
X_test_cat = encoder.transform(X_test[["branch"]])


# -------------------------------
# 7. COMBINE FEATURES
# -------------------------------

X_train_final = np.hstack([X_train_scaled, X_train_cat])
X_test_final = np.hstack([X_test_scaled, X_test_cat])


# -------------------------------
# 8. FEATURE SHAPE CHECK
# -------------------------------

assert X_train_final.shape[1] == X_test_final.shape[1], "Feature mismatch"


# -------------------------------
# 9. FEATURE NAME TRACKING
# -------------------------------

feature_names = (
    num_features +
    list(encoder.get_feature_names_out(["branch"]))
)


# -------------------------------
# 10. REAL-WORLD ML READY OUTPUT
# -------------------------------

ml_ready_data = {
    "X_train": X_train_final,
    "X_test": X_test_final,
    "y_train": y_train.values,
    "y_test": y_test.values,
    "features": feature_names
}


# -------------------------------
# 11. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("Final Feature Matrix Shape:", X_train_final.shape)
    print("Feature Names:", feature_names)
    print("feature_engineering.py executed successfully")
