"""
modules_and_packages.py

Unit 1: Python Basics

Objective:
- Learn how modules work in Python
- Import built-in and custom modules
- Understand different import styles
"""

import math
import random

# -------------------------------
# 1. USING BUILT-IN MODULES
# -------------------------------

number = 25

print(f"Square Root of {number}: {math.sqrt(number)}")
print(f"Value of π: {math.pi:.4f}")

# -------------------------------
# 2. RANDOM MODULE
# -------------------------------

print("\nRandom Integer:", random.randint(1, 100))

colors = ["Red", "Blue", "Green", "Black"]

print("Random Choice:", random.choice(colors))

# -------------------------------
# 3. IMPORT SPECIFIC FUNCTIONS
# -------------------------------

from math import factorial, pow

print("\nFactorial of 5:", factorial(5))
print("2 raised to 5:", pow(2, 5))

# -------------------------------
# 4. IMPORT WITH ALIAS
# -------------------------------

import statistics as stats

marks = [82, 90, 78, 88, 95]

print("\nMean Marks:", stats.mean(marks))
print("Median Marks:", stats.median(marks))

# -------------------------------
# 5. CUSTOM MODULE EXAMPLE
# -------------------------------

print("\nExample Custom Module Structure:")

print("""
project/
│── main.py
│── calculator.py

calculator.py
--------------
def add(a, b):
    return a + b

main.py
-------
from calculator import add

print(add(5, 3))
""")

# -------------------------------
# 6. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\nmodules_and_packages.py executed successfully")
