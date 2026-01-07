"""
data_cleaning.py

Unit 2: Data Handling for AI & ML

Objective:
- Clean, validate, and prepare real-world data
- Handle missing values, duplicates, outliers
- Build ML-safe datasets
"""

import pandas as pd
import numpy as np


# -------------------------------
# 1. SAMPLE DIRTY DATASET
# -------------------------------

data = {
    "name": ["Amit", "Neha", "Rahul", "Neha", None],
    "age": [22, 21, None, 21, 23],
    "cgpa": [8.1, 8.7, 15.0, 8.7, -1.0],
    "placed": [True, False, True, False, True]
}

df = pd.DataFrame(data)


# -------------------------------
# 2. BASIC DATA HEALTH CHECK
# -------------------------------

df.info()
missing_values = df.isnull().sum()


# -------------------------------
# 3. DROPPING DUPLICATES
# -------------------------------

df_no_duplicates = df.drop_duplicates()


# -------------------------------
# 4. HANDLING MISSING VALUES
# -------------------------------

# Fill numeric columns with mean
df_filled = df_no_duplicates.copy()
df_filled["age"] = df_filled["age"].fillna(df_filled["age"].mean())

# Drop rows where name is missing (critical column)
df_filled = df_filled.dropna(subset=["name"])


# -------------------------------
# 5. FIXING INVALID VALUES
# -------------------------------

# CGPA should be between 0 and 10
df_filled["cgpa"] = df_filled["cgpa"].clip(0, 10)


# -------------------------------
# 6. DATA TYPE CORRECTION
# -------------------------------

df_filled["placed"] = df_filled["placed"].astype(int)


# -------------------------------
# 7. OUTLIER DETECTION (IQR METHOD)
# -------------------------------

Q1 = df_filled["cgpa"].quantile(0.25)
Q3 = df_filled["cgpa"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_no_outliers = df_filled[
    (df_filled["cgpa"] >= lower_bound) &
    (df_filled["cgpa"] <= upper_bound)
]


# -------------------------------
# 8. STRING NORMALIZATION
# -------------------------------

df_no_outliers["name"] = df_no_outliers["name"].str.strip().str.title()


# -------------------------------
# 9. COLUMN RENAMING (CONSISTENCY)
# -------------------------------

df_cleaned = df_no_outliers.rename(columns={
    "cgpa": "cgpa_score"
})


# -------------------------------
# 10. FINAL SANITY CHECK
# -------------------------------

df_cleaned.info()


# -------------------------------
# 11. ML-READY SPLIT
# -------------------------------

features = df_cleaned[["age", "cgpa_score"]]
labels = df_cleaned["placed"]


# -------------------------------
# 12. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("data_cleaning.py executed successfully")
