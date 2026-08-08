"""
categorical_data.py

Unit 2: Data Handling

Objective:
- Understand categorical data in Pandas
- Practice category types and ordering
- Encode categorical features for ML
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# -------------------------------
# 1. CREATE DATASET
# -------------------------------

data = {
    "Name": ["Amit", "Priya", "Rahul", "Sneha", "Vikas", "Neha"],
    "Department": [
        "Computer",
        "Electronics",
        "Computer",
        "Mechanical",
        "Electronics",
        "Computer"
    ],
    "Performance": [
        "Excellent",
        "Good",
        "Average",
        "Good",
        "Excellent",
        "Average"
    ]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)


# -------------------------------
# 2. CHECK UNIQUE CATEGORIES
# -------------------------------

print("\nUnique Departments:")
print(df["Department"].unique())

print("\nUnique Performance Levels:")
print(df["Performance"].unique())


# -------------------------------
# 3. CATEGORY DATA TYPE
# -------------------------------

df["Department"] = df["Department"].astype("category")

print("\nData Types:")
print(df.dtypes)


# -------------------------------
# 4. CATEGORY FREQUENCY
# -------------------------------

print("\nDepartment Frequency:")
print(df["Department"].value_counts())


# -------------------------------
# 5. ORDERED CATEGORIES
# -------------------------------

performance_order = [
    "Average",
    "Good",
    "Excellent"
]

df["Performance"] = pd.Categorical(
    df["Performance"],
    categories=performance_order,
    ordered=True
)

print("\nOrdered Performance:")
print(df["Performance"])


# -------------------------------
# 6. SORT BY CATEGORY
# -------------------------------

sorted_df = df.sort_values("Performance")

print("\nStudents Sorted by Performance:")
print(sorted_df)


# -------------------------------
# 7. LABEL ENCODING
# -------------------------------

encoder = LabelEncoder()

performance_encoded = encoder.fit_transform(
    df["Performance"].astype(str)
)

print("\nLabel Encoding:")
print(performance_encoded)

print("\nLabel Mapping:")
for number, category in enumerate(encoder.classes_):
    print(f"{category} -> {number}")


# -------------------------------
# 8. ONE-HOT ENCODING
# -------------------------------

department_encoded = pd.get_dummies(
    df["Department"],
    dtype=int
)

print("\nOne-Hot Encoded Department:")
print(department_encoded)


# -------------------------------
# 9. COMBINE ENCODED FEATURES
# -------------------------------

encoded_df = pd.concat(
    [
        df[["Name"]],
        department_encoded
    ],
    axis=1
)

print("\nFinal Encoded Data:")
print(encoded_df)


# -------------------------------
# 10. ML-READY DATA
# -------------------------------

X = pd.concat(
    [
        department_encoded,
        pd.Series(
            performance_encoded,
            name="Performance"
        )
    ],
    axis=1
)

print("\nML-Ready Features:")
print(X)


# -------------------------------
# 11. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\ncategorical_data.py executed successfully")
