"""
Feature Encoding for Machine Learning

Practice:
- Label Encoding
- Ordinal Encoding
- One-Hot Encoding
- Handling unknown categories
- Avoiding unnecessary encoding
- Comparing different encoding strategies
"""

import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
)


# ---------------------------------------------------------
# 1. Sample Dataset
# ---------------------------------------------------------

data = pd.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "city": ["Mumbai", "Pune", "Delhi", "Pune", "Mumbai"],
        "education": ["Bachelor", "Master", "PhD", "Bachelor", "Master"],
        "satisfaction": ["Low", "Medium", "High", "Medium", "High"],
    }
)

print("Original Dataset:")
print(data)


# ---------------------------------------------------------
# 2. Label Encoding
# ---------------------------------------------------------
# LabelEncoder converts categories into integer labels.
#
# Example:
# Bachelor -> 0
# Master   -> 1
# PhD      -> 2
#
# Best suited for target labels rather than nominal
# input features.

label_encoder = LabelEncoder()

data["education_label"] = label_encoder.fit_transform(
    data["education"]
)

print("\nLabel Encoded Education:")
print(data[["education", "education_label"]])

print("\nLabel Mapping:")
for label, value in zip(
    label_encoder.classes_,
    range(len(label_encoder.classes_)),
):
    print(f"{label} -> {value}")


# ---------------------------------------------------------
# 3. Ordinal Encoding
# ---------------------------------------------------------
# Ordinal encoding is appropriate when categories have
# a meaningful order.
#
# Low < Medium < High

ordinal_encoder = OrdinalEncoder(
    categories=[["Low", "Medium", "High"]]
)

data["satisfaction_ordinal"] = ordinal_encoder.fit_transform(
    data[["satisfaction"]]
).astype(int)

print("\nOrdinal Encoded Satisfaction:")
print(data[["satisfaction", "satisfaction_ordinal"]])


# ---------------------------------------------------------
# 4. One-Hot Encoding
# ---------------------------------------------------------
# One-hot encoding creates a separate binary column
# for every category.
#
# Mumbai -> [1, 0, 0]
# Pune   -> [0, 1, 0]
# Delhi  -> [0, 0, 1]

one_hot_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore",
)

city_encoded = one_hot_encoder.fit_transform(
    data[["city"]]
)

city_columns = one_hot_encoder.get_feature_names_out(
    ["city"]
)

city_encoded_df = pd.DataFrame(
    city_encoded,
    columns=city_columns,
    index=data.index,
)

print("\nOne-Hot Encoded City:")
print(city_encoded_df)


# ---------------------------------------------------------
# 5. Combine Encoded Data with Original Dataset
# ---------------------------------------------------------

data_encoded = pd.concat(
    [data, city_encoded_df],
    axis=1,
)

print("\nDataset After Encoding:")
print(data_encoded)


# ---------------------------------------------------------
# 6. Transform New / Unknown Categories
# ---------------------------------------------------------
# handle_unknown="ignore" prevents an error when new
# categories appear during inference.

new_data = pd.DataFrame(
    {
        "city": ["Mumbai", "Bangalore", "Pune"]
    }
)

new_city_encoded = one_hot_encoder.transform(
    new_data[["city"]]
)

new_city_encoded_df = pd.DataFrame(
    new_city_encoded,
    columns=city_columns,
)

print("\nEncoding New Data:")
print(new_city_encoded_df)


# ---------------------------------------------------------
# 7. Why Label Encoding Can Be Dangerous
# ---------------------------------------------------------

example = pd.DataFrame(
    {
        "city": ["Mumbai", "Pune", "Delhi"]
    }
)

bad_encoder = LabelEncoder()

example["city_encoded"] = bad_encoder.fit_transform(
    example["city"]
)

print("\nPotentially Misleading Label Encoding:")
print(example)

print(
    "\nThe numerical values do NOT mean that one city "
    "is greater than another."
)


# ---------------------------------------------------------
# 8. Ordinal vs Nominal Data
# ---------------------------------------------------------

ordinal_example = pd.DataFrame(
    {
        "level": [
            "Beginner",
            "Intermediate",
            "Advanced",
            "Expert",
        ]
    }
)

level_encoder = OrdinalEncoder(
    categories=[
        [
            "Beginner",
            "Intermediate",
            "Advanced",
            "Expert",
        ]
    ]
)

ordinal_example["level_encoded"] = level_encoder.fit_transform(
    ordinal_example[["level"]]
).astype(int)

print("\nProper Ordinal Encoding:")
print(ordinal_example)


# ---------------------------------------------------------
# 9. Encoding Using Pandas
# ---------------------------------------------------------

pandas_one_hot = pd.get_dummies(
    data["city"],
    prefix="city",
    dtype=int,
)

print("\nPandas One-Hot Encoding:")
print(pandas_one_hot)


# ---------------------------------------------------------
# 10. Practical Encoding Rules
# ---------------------------------------------------------

print("\nEncoding Rules:")

print("1. Nominal categories -> One-Hot Encoding")
print("2. Ordered categories -> Ordinal Encoding")
print("3. Target labels      -> Label Encoding")
print("4. Unknown categories -> handle_unknown='ignore'")
print("5. Do not assign fake numerical order to nominal data")


# ---------------------------------------------------------
# 11. Final Dataset
# ---------------------------------------------------------

print("\nFinal Dataset:")
print(data_encoded)
