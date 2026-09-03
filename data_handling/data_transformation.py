"""
Data Transformation for Machine Learning

Practice:
- Min-Max Scaling
- Standardization (Z-score)
- Robust Scaling
- Log Transformation
- Binning / Discretization
- Custom feature transformations
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
)


# ---------------------------------------------------------
# 1. Sample Dataset
# ---------------------------------------------------------

data = pd.DataFrame(
    {
        "age": [18, 22, 25, 30, 35, 40, 50, 65],
        "income": [18000, 22000, 28000, 35000, 45000, 60000, 90000, 250000],
        "experience": [0, 1, 2, 5, 8, 12, 20, 35],
    }
)

print("Original Data:")
print(data)


# ---------------------------------------------------------
# 2. Min-Max Scaling
# ---------------------------------------------------------

minmax_scaler = MinMaxScaler()

data["income_minmax"] = minmax_scaler.fit_transform(
    data[["income"]]
)

print("\nMin-Max Scaled Income:")
print(data[["income", "income_minmax"]])


# ---------------------------------------------------------
# 3. Standardization (Z-score)
# ---------------------------------------------------------

standard_scaler = StandardScaler()

data["income_standardized"] = standard_scaler.fit_transform(
    data[["income"]]
)

print("\nStandardized Income:")
print(data[["income", "income_standardized"]])


# ---------------------------------------------------------
# 4. Robust Scaling
# ---------------------------------------------------------

robust_scaler = RobustScaler()

data["income_robust"] = robust_scaler.fit_transform(
    data[["income"]]
)

print("\nRobust Scaled Income:")
print(data[["income", "income_robust"]])


# ---------------------------------------------------------
# 5. Log Transformation
# ---------------------------------------------------------

data["income_log"] = np.log1p(data["income"])

print("\nLog Transformed Income:")
print(data[["income", "income_log"]])


# ---------------------------------------------------------
# 6. Binning / Discretization
# ---------------------------------------------------------

bins = [0, 25, 40, 60, np.inf]
labels = ["Young", "Adult", "Middle-aged", "Senior"]

data["age_group"] = pd.cut(
    data["age"],
    bins=bins,
    labels=labels,
    right=False,
)

print("\nAge Groups:")
print(data[["age", "age_group"]])


# ---------------------------------------------------------
# 7. Quantile-based Binning
# ---------------------------------------------------------

data["income_level"] = pd.qcut(
    data["income"],
    q=4,
    labels=["Low", "Medium", "High", "Very High"],
)

print("\nIncome Quantile Groups:")
print(data[["income", "income_level"]])


# ---------------------------------------------------------
# 8. Custom Feature Transformation
# ---------------------------------------------------------

data["experience_squared"] = data["experience"] ** 2

print("\nPolynomial Feature:")
print(data[["experience", "experience_squared"]])


# ---------------------------------------------------------
# 9. Applying Transformation to Multiple Columns
# ---------------------------------------------------------

numeric_columns = ["age", "income", "experience"]

scaler = StandardScaler()

scaled_features = scaler.fit_transform(data[numeric_columns])

scaled_data = pd.DataFrame(
    scaled_features,
    columns=numeric_columns,
)

print("\nMultiple Columns Standardized:")
print(scaled_data)


# ---------------------------------------------------------
# 10. Transformation Summary
# ---------------------------------------------------------

print("\nFinal Dataset:")
print(data)

print("\nTransformation Guide:")
print("Min-Max Scaling     -> Scales values to a fixed range (usually 0 to 1)")
print("Standardization     -> Centers data around mean=0 and std=1")
print("Robust Scaling      -> Less affected by outliers")
print("Log Transformation  -> Reduces strong right-skew")
print("Binning             -> Converts continuous values into categories")
print("Polynomial Feature  -> Captures non-linear relationships")
