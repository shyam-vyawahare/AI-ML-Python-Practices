"""
pandas_basics.py

Unit 2: Data Handling for AI & ML

Objective:
- Learn Pandas for structured data handling
- Perform real-world dataset operations
- Prepare data for ML pipelines
"""

import pandas as pd


# -------------------------------
# 1. CREATING A DATAFRAME
# -------------------------------

data = {
    "name": ["Amit", "Neha", "Rahul", "Sneha"],
    "age": [22, 21, 23, 22],
    "cgpa": [8.1, 8.7, 7.9, 8.4],
    "placed": [True, False, True, False]
}

df = pd.DataFrame(data)


# -------------------------------
# 2. BASIC DATA INSPECTION
# -------------------------------

df.head()
df.tail()
df.info()
df.describe()


# -------------------------------
# 3. COLUMN ACCESS
# -------------------------------

names = df["name"]
ages = df.age


# -------------------------------
# 4. ROW ACCESS (iloc / loc)
# -------------------------------

first_row = df.iloc[0]
selected_rows = df.loc[df["age"] > 21]


# -------------------------------
# 5. FILTERING DATA (ML CORE)
# -------------------------------

high_cgpa = df[df["cgpa"] >= 8.5]
placed_students = df[df["placed"] == True]


# -------------------------------
# 6. ADDING & MODIFYING COLUMNS
# -------------------------------

df["age_next_year"] = df["age"] + 1
df["cgpa_scaled"] = df["cgpa"] / 10


# -------------------------------
# 7. DROPPING COLUMNS
# -------------------------------

df_dropped = df.drop(columns=["age_next_year"])


# -------------------------------
# 8. HANDLING MISSING VALUES
# -------------------------------

df_with_nan = df.copy()
df_with_nan.loc[2, "cgpa"] = None

filled = df_with_nan["cgpa"].fillna(df_with_nan["cgpa"].mean())
df_with_nan["cgpa"] = filled


# -------------------------------
# 9. SORTING DATA
# -------------------------------

sorted_by_cgpa = df.sort_values(by="cgpa", ascending=False)


# -------------------------------
# 10. VALUE COUNTS (CLASS BALANCE)
# -------------------------------

placement_counts = df["placed"].value_counts()


# -------------------------------
# 11. GROUPBY (VERY IMPORTANT)
# -------------------------------

avg_cgpa_by_placement = df.groupby("placed")["cgpa"].mean()


# -------------------------------
# 12. APPLY FUNCTION (CUSTOM LOGIC)
# -------------------------------

def cgpa_label(cgpa):
    return "Excellent" if cgpa >= 8.5 else "Good"

df["performance"] = df["cgpa"].apply(cgpa_label)


# -------------------------------
# 13. REAL-WORLD ML PREP PATTERN
# -------------------------------

features = df[["age", "cgpa"]]
labels = df["placed"]


# -------------------------------
# 14. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("pandas_basics.py executed successfully")
