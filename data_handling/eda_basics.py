"""
eda_basics.py

Unit 2: Data Handling for AI & ML

Objective:
- Perform Exploratory Data Analysis (EDA)
- Understand data distribution and patterns
- Make informed decisions before ML modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# 1. SAMPLE DATASET
# -------------------------------

data = {
    "age": [21, 22, 23, 22, 24, 21, 23, 22],
    "cgpa": [8.1, 8.7, 7.9, 8.4, 9.1, 8.0, 7.8, 8.6],
    "placed": [1, 0, 1, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)


# -------------------------------
# 2. BASIC SHAPE & SUMMARY
# -------------------------------

rows, cols = df.shape
summary = df.describe()


# -------------------------------
# 3. DATA TYPES & NULL CHECK
# -------------------------------

df.info()
null_check = df.isnull().sum()


# -------------------------------
# 4. TARGET VARIABLE ANALYSIS
# -------------------------------

class_distribution = df["placed"].value_counts()
class_ratio = df["placed"].value_counts(normalize=True)


# -------------------------------
# 5. FEATURE DISTRIBUTIONS
# -------------------------------

plt.figure()
plt.hist(df["age"], bins=5)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

plt.figure()
plt.hist(df["cgpa"], bins=5)
plt.title("CGPA Distribution")
plt.xlabel("CGPA")
plt.ylabel("Count")
plt.show()


# -------------------------------
# 6. FEATURE VS TARGET
# -------------------------------

plt.figure()
plt.scatter(df["cgpa"], df["placed"])
plt.title("CGPA vs Placement")
plt.xlabel("CGPA")
plt.ylabel("Placed")
plt.show()


# -------------------------------
# 7. CORRELATION ANALYSIS
# -------------------------------

correlation_matrix = df.corr()


# -------------------------------
# 8. OUTLIER CHECK (VISUAL)
# -------------------------------

plt.figure()
plt.boxplot(df["cgpa"])
plt.title("CGPA Outlier Check")
plt.ylabel("CGPA")
plt.show()


# -------------------------------
# 9. SIMPLE INSIGHTS (LOGIC)
# -------------------------------

avg_cgpa_placed = df[df["placed"] == 1]["cgpa"].mean()
avg_cgpa_not_placed = df[df["placed"] == 0]["cgpa"].mean()


# -------------------------------
# 10. ML DECISION SUPPORT
# -------------------------------

features = df[["age", "cgpa"]]
labels = df["placed"]


# -------------------------------
# 11. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("eda_basics.py executed successfully")
