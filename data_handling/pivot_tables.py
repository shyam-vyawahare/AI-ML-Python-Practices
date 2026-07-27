"""
pivot_tables.py

Unit 2: Data Handling

Objective:
- Learn how to create pivot tables
- Summarize data using different aggregation functions
- Analyze grouped datasets efficiently
"""

import pandas as pd

# -------------------------------
# 1. CREATE DATAFRAME
# -------------------------------

sales = pd.DataFrame({
    "Region": ["North", "North", "South", "South",
               "East", "East", "West", "West"],
    "Category": ["Electronics", "Furniture",
                 "Electronics", "Furniture",
                 "Electronics", "Furniture",
                 "Electronics", "Furniture"],
    "Sales": [1200, 800, 1500, 950, 1100, 700, 1400, 900],
    "Quantity": [10, 6, 12, 8, 9, 5, 11, 7]
})

print("Original Data:")
print(sales)

# -------------------------------
# 2. TOTAL SALES BY REGION
# -------------------------------

pivot_sales = pd.pivot_table(
    sales,
    values="Sales",
    index="Region",
    aggfunc="sum"
)

print("\nTotal Sales by Region:")
print(pivot_sales)

# -------------------------------
# 3. AVERAGE SALES BY REGION
# -------------------------------

pivot_average = pd.pivot_table(
    sales,
    values="Sales",
    index="Region",
    aggfunc="mean"
)

print("\nAverage Sales by Region:")
print(pivot_average)

# -------------------------------
# 4. MULTI-LEVEL PIVOT TABLE
# -------------------------------

pivot_multi = pd.pivot_table(
    sales,
    values="Sales",
    index="Region",
    columns="Category",
    aggfunc="sum",
    fill_value=0
)

print("\nSales by Region and Category:")
print(pivot_multi)

# -------------------------------
# 5. MULTIPLE AGGREGATIONS
# -------------------------------

pivot_stats = pd.pivot_table(
    sales,
    values="Sales",
    index="Region",
    aggfunc=["sum", "mean", "max"]
)

print("\nSales Statistics:")
print(pivot_stats)

# -------------------------------
# 6. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\npivot_tables.py executed successfully")
