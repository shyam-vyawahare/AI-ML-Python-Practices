"""
data_cleaning.py

Unit 2: Data Handling

Objective:
- Practice real-world data cleaning
- Remove duplicates
- Standardize inconsistent values
- Handle invalid data
- Detect simple outliers
"""

import numpy as np
import pandas as pd


# -------------------------------
# 1. CREATE MESSY DATASET
# -------------------------------

data = {
    "Name": [
        " Amit ",
        "Priya",
        "rahul",
        "Sneha",
        "Priya",
        "Vikas"
    ],
    "Department": [
        "Computer",
        "computer ",
        "Electronics",
        "ELECTRONICS",
        "computer ",
        "Mechanical"
    ],
    "Age": [
        21,
        22,
        23,
        150,
        22,
        -5
    ],
    "CGPA": [
        8.1,
        8.7,
        7.9,
        9.2,
        8.7,
        12.0
    ]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)


# -------------------------------
# 2. REMOVE EXTRA SPACES
# -------------------------------

df["Name"] = df["Name"].str.strip()

df["Department"] = (
    df["Department"]
    .str.strip()
    .str.title()
)

print("\nAfter String Cleaning:")
print(df)


# -------------------------------
# 3. STANDARDIZE TEXT
# -------------------------------

df["Name"] = df["Name"].str.title()

print("\nStandardized Names:")
print(df["Name"])


# -------------------------------
# 4. FIND DUPLICATES
# -------------------------------

duplicates = df.duplicated()

print("\nDuplicate Rows:")
print(df[duplicates])


# -------------------------------
# 5. REMOVE DUPLICATES
# -------------------------------

df = df.drop_duplicates()

print("\nAfter Removing Duplicates:")
print(df)


# -------------------------------
# 6. VALIDATE AGE
# -------------------------------

valid_age = df["Age"].between(18, 30)

print("\nInvalid Age Records:")
print(df[~valid_age])


# -------------------------------
# 7. REPLACE INVALID AGE
# -------------------------------

df.loc[
    ~df["Age"].between(18, 30),
    "Age"
] = np.nan


# -------------------------------
# 8. VALIDATE CGPA
# -------------------------------

valid_cgpa = df["CGPA"].between(0, 10)

print("\nInvalid CGPA Records:")
print(df[~valid_cgpa])


# -------------------------------
# 9. REPLACE INVALID CGPA
# -------------------------------

df.loc[
    ~df["CGPA"].between(0, 10),
    "CGPA"
] = np.nan


# -------------------------------
# 10. HANDLE INVALID VALUES
# -------------------------------

df["Age"] = df["Age"].fillna(
    df["Age"].median()
)

df["CGPA"] = df["CGPA"].fillna(
    df["CGPA"].median()
)


# -------------------------------
# 11. OUTLIER DETECTION
# -------------------------------

Q1 = df["CGPA"].quantile(0.25)
Q3 = df["CGPA"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df["CGPA"] < lower_bound) |
    (df["CGPA"] > upper_bound)
]

print("\nCGPA Outliers:")
print(outliers)


# -------------------------------
# 12. FINAL DATASET
# -------------------------------

print("\nFinal Clean Dataset:")
print(df)


# -------------------------------
# 13. DATA VALIDATION
# -------------------------------

print("\nValidation:")
print("Missing values:")
print(df.isna().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nData Types:")
print(df.dtypes)


# -------------------------------
# 14. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\ndata_cleaning.py executed successfully")
