"""
missing_data_strategies.py

Unit 2: Data Handling

Objective:
- Detect missing values
- Practice different missing-data strategies
- Prepare datasets for machine learning
"""

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer


# -------------------------------
# 1. CREATE DATASET WITH MISSING VALUES
# -------------------------------

data = {
    "Name": ["Amit", "Priya", "Rahul", "Sneha", "Vikas", "Neha"],
    "Age": [21, np.nan, 23, 22, np.nan, 24],
    "CGPA": [8.1, 8.7, np.nan, 8.4, 9.0, np.nan],
    "Internships": [1, 2, 0, np.nan, 3, 1]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)


# -------------------------------
# 2. DETECT MISSING VALUES
# -------------------------------

print("\nMissing Values:")
print(df.isna())

print("\nMissing Value Count:")
print(df.isna().sum())


# -------------------------------
# 3. MISSING VALUE PERCENTAGE
# -------------------------------

missing_percentage = (
    df.isna().mean() * 100
).round(2)

print("\nMissing Value Percentage:")
print(missing_percentage)


# -------------------------------
# 4. DROP ROWS WITH MISSING VALUES
# -------------------------------

dropped_rows = df.dropna()

print("\nAfter Dropping Rows:")
print(dropped_rows)


# -------------------------------
# 5. FILL WITH CONSTANT VALUE
# -------------------------------

constant_filled = df.copy()

constant_filled["Age"] = (
    constant_filled["Age"].fillna(0)
)

print("\nAge Filled With 0:")
print(constant_filled)


# -------------------------------
# 6. MEAN IMPUTATION
# -------------------------------

mean_filled = df.copy()

mean_filled["Age"] = (
    mean_filled["Age"].fillna(
        mean_filled["Age"].mean()
    )
)

print("\nAge Filled With Mean:")
print(mean_filled)


# -------------------------------
# 7. MEDIAN IMPUTATION
# -------------------------------

median_filled = df.copy()

median_filled["CGPA"] = (
    median_filled["CGPA"].fillna(
        median_filled["CGPA"].median()
    )
)

print("\nCGPA Filled With Median:")
print(median_filled)


# -------------------------------
# 8. MODE IMPUTATION
# -------------------------------

mode_filled = df.copy()

mode_value = mode_filled["Internships"].mode()[0]

mode_filled["Internships"] = (
    mode_filled["Internships"].fillna(mode_value)
)

print("\nInternships Filled With Mode:")
print(mode_filled)


# -------------------------------
# 9. FORWARD FILL
# -------------------------------

forward_filled = df.copy()

forward_filled = forward_filled.ffill()

print("\nForward Filled Dataset:")
print(forward_filled)


# -------------------------------
# 10. BACKWARD FILL
# -------------------------------

backward_filled = df.copy()

backward_filled = backward_filled.bfill()

print("\nBackward Filled Dataset:")
print(backward_filled)


# -------------------------------
# 11. INTERPOLATION
# -------------------------------

interpolated = df.copy()

interpolated["CGPA"] = (
    interpolated["CGPA"].interpolate()
)

print("\nInterpolated CGPA:")
print(interpolated["CGPA"])


# -------------------------------
# 12. SCIKIT-LEARN IMPUTATION
# -------------------------------

numeric_columns = [
    "Age",
    "CGPA",
    "Internships"
]

imputer = SimpleImputer(strategy="median")

imputed_values = imputer.fit_transform(
    df[numeric_columns]
)

imputed_df = pd.DataFrame(
    imputed_values,
    columns=numeric_columns
)

print("\nSimpleImputer Result:")
print(imputed_df)


# -------------------------------
# 13. VERIFY NO MISSING VALUES
# -------------------------------

print("\nMissing Values After Imputation:")
print(imputed_df.isna().sum())


# -------------------------------
# 14. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print(
        "\nmissing_data_strategies.py "
        "executed successfully"
)
