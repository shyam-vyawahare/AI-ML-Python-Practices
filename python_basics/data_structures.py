"""
data_structures.py

Objective:
- Practical revision of Python data structures
- AI/ML-oriented usage patterns
- Emphasis on readability, performance, and correctness
"""

# -------------------------------
# 1. LISTS (ORDERED, MUTABLE)
# -------------------------------

numbers = [1, 2, 3, 4, 5]

# Append & extend
numbers.append(6)
numbers.extend([7, 8])

# List comprehension (HEAVILY USED IN ML)
squared = [n ** 2 for n in numbers]

# Conditional comprehension
even_numbers = [n for n in numbers if n % 2 == 0]


# -------------------------------
# 2. TUPLES (ORDERED, IMMUTABLE)
# -------------------------------

coordinates = (10.5, 20.3)

# Tuple unpacking (very common)
x, y = coordinates


# -------------------------------
# 3. SETS (UNORDERED, UNIQUE)
# -------------------------------

labels = {"cat", "dog", "bird", "cat"}

# Fast membership check
if "dog" in labels:
    pass

# Set operations (useful in preprocessing)
a = {1, 2, 3}
b = {3, 4, 5}

union = a | b
intersection = a & b
difference = a - b


# -------------------------------
# 4. DICTIONARIES (KEY-VALUE CORE)
# -------------------------------

student = {
    "name": "Ultrex",
    "branch": "ECE",
    "cgpa": 8.5
}

# Access with safety
cgpa = student.get("cgpa", 0.0)

# Update values
student["cgpa"] = 8.7

# Looping through dicts (very common)
for key, value in student.items():
    pass


# -------------------------------
# 5. NESTED DATA STRUCTURES
# -------------------------------

dataset = [
    {"id": 1, "label": "spam"},
    {"id": 2, "label": "ham"},
    {"id": 3, "label": "spam"}
]

# Extract labels (ML-style)
labels = [row["label"] for row in dataset]


# -------------------------------
# 6. IMMUTABILITY VS MUTABILITY
# -------------------------------

original = [1, 2, 3]
alias = original

alias.append(4)

# original is modified due to shared reference


# -------------------------------
# 7. COPYING STRUCTURES (IMPORTANT)
# -------------------------------

import copy

shallow_copy = original.copy()
deep_copy = copy.deepcopy(original)


# -------------------------------
# 8. REAL-WORLD MINI PATTERN
# -------------------------------

# Frequency count (used in NLP, stats, logs)
text = "ai ml ai data ml ai".split()

frequency = {}

for word in text:
    frequency[word] = frequency.get(word, 0) + 1


# -------------------------------
# 9. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("data_structures.py executed successfully")
