"""
handling_datetime.py

Unit 2: Data Handling

Objective:
- Learn datetime operations in Pandas
- Extract date components
- Calculate date differences
- Filter records by date
"""

import pandas as pd

# -------------------------------
# 1. CREATE DATAFRAME
# -------------------------------

data = {
    "OrderID": [101, 102, 103, 104],
    "OrderDate": [
        "2026-07-20",
        "2026-07-22",
        "2026-07-25",
        "2026-07-28"
    ],
    "Sales": [2500, 1800, 3200, 2750]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# -------------------------------
# 2. CONVERT TO DATETIME
# -------------------------------

df["OrderDate"] = pd.to_datetime(df["OrderDate"])

print("\nData Types:")
print(df.dtypes)

# -------------------------------
# 3. EXTRACT DATE COMPONENTS
# -------------------------------

df["Year"] = df["OrderDate"].dt.year
df["Month"] = df["OrderDate"].dt.month
df["Day"] = df["OrderDate"].dt.day
df["Weekday"] = df["OrderDate"].dt.day_name()

print("\nExtracted Date Components:")
print(df)

# -------------------------------
# 4. DATE DIFFERENCE
# -------------------------------

today = pd.Timestamp("2026-08-01")

df["DaysSinceOrder"] = (today - df["OrderDate"]).dt.days

print("\nDays Since Order:")
print(df[["OrderID", "DaysSinceOrder"]])

# -------------------------------
# 5. FILTER DATA
# -------------------------------

recent_orders = df[df["OrderDate"] >= "2026-07-23"]

print("\nOrders on or After 23 July 2026:")
print(recent_orders)

# -------------------------------
# 6. SORT BY DATE
# -------------------------------

sorted_df = df.sort_values("OrderDate", ascending=False)

print("\nSorted by Date:")
print(sorted_df)

# -------------------------------
# 7. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\nhandling_datetime.py executed successfully")
