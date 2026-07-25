"""
pandas_merge_groupby.py

Unit 2: Data Handling

Objective:
- Learn DataFrame merging
- Practice GroupBy operations
- Perform aggregations
"""

import pandas as pd

# -------------------------------
# 1. CREATE DATAFRAMES
# -------------------------------

employees = pd.DataFrame({
    "EmployeeID": [101, 102, 103, 104],
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "DepartmentID": [1, 2, 1, 3],
    "Salary": [50000, 65000, 55000, 70000]
})

departments = pd.DataFrame({
    "DepartmentID": [1, 2, 3],
    "Department": ["HR", "Engineering", "Finance"]
})

print("Employees:")
print(employees)

print("\nDepartments:")
print(departments)

# -------------------------------
# 2. MERGE DATAFRAMES
# -------------------------------

merged = pd.merge(
    employees,
    departments,
    on="DepartmentID",
    how="inner"
)

print("\nMerged DataFrame:")
print(merged)

# -------------------------------
# 3. GROUP BY DEPARTMENT
# -------------------------------

grouped = merged.groupby("Department")

print("\nAverage Salary by Department:")
print(grouped["Salary"].mean())

# -------------------------------
# 4. MULTIPLE AGGREGATIONS
# -------------------------------

summary = grouped["Salary"].agg(
    ["count", "min", "max", "mean", "sum"]
)

print("\nDepartment Salary Summary:")
print(summary)

# -------------------------------
# 5. SORT RESULTS
# -------------------------------

sorted_summary = summary.sort_values(
    by="mean",
    ascending=False
)

print("\nSorted by Average Salary:")
print(sorted_summary)

# -------------------------------
# 6. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\npandas_merge_groupby.py executed successfully")
