"""
numpy_basics.py

Unit 2: Data Handling for AI & ML

Objective:
- Introduce NumPy as the core numerical engine
- Replace Python loops with vectorized operations
- Build array intuition required for ML & DL frameworks
"""

import numpy as np


# -------------------------------
# 1. NUMPY ARRAY VS PYTHON LIST
# -------------------------------

python_list = [1, 2, 3, 4, 5]
numpy_array = np.array(python_list)

print(type(python_list))
print(type(numpy_array))


# -------------------------------
# 2. ARRAY CREATION METHODS
# -------------------------------

zeros = np.zeros((2, 3))
ones = np.ones((3, 2))
identity = np.eye(3)
range_array = np.arange(0, 10, 2)
linspace_array = np.linspace(0, 1, 5)


# -------------------------------
# 3. DATA TYPES (IMPORTANT FOR ML)
# -------------------------------

float_array = np.array([1, 2, 3], dtype=np.float32)
int_array = np.array([1, 2, 3], dtype=np.int32)

print(float_array.dtype)
print(int_array.dtype)


# -------------------------------
# 4. BASIC ARRAY PROPERTIES
# -------------------------------

matrix = np.array([[1, 2, 3], [4, 5, 6]])

print(matrix.shape)   # rows, columns
print(matrix.ndim)    # number of dimensions
print(matrix.size)    # total elements


# -------------------------------
# 5. INDEXING & SLICING
# -------------------------------

arr = np.array([10, 20, 30, 40, 50])

first = arr[0]
slice_part = arr[1:4]


# -------------------------------
# 6. VECTORISED OPERATIONS
# -------------------------------

data = np.array([1, 2, 3, 4])

squared = data ** 2
added = data + 10
multiplied = data * 3


# -------------------------------
# 7. BROADCASTING
# -------------------------------

matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

broadcast_add = matrix + 10


# -------------------------------
# 8. AGGREGATION FUNCTIONS
# -------------------------------

values = np.array([10, 20, 30, 40])

total = np.sum(values)
mean = np.mean(values)
maximum = np.max(values)
minimum = np.min(values)


# -------------------------------
# 9. RESHAPING ARRAYS
# -------------------------------

flat = np.arange(1, 7)
reshaped = flat.reshape((2, 3))


# -------------------------------
# 10. RANDOM DATA (ML SIMULATION)
# -------------------------------

random_uniform = np.random.rand(3, 3)
random_normal = np.random.randn(3, 3)
random_integers = np.random.randint(0, 10, size=(2, 4))


# -------------------------------
# 11. BOOLEAN MASKING
# -------------------------------

numbers = np.array([5, 10, 15, 20, 25])
mask = numbers > 15
filtered = numbers[mask]


# -------------------------------
# 12. AVOIDING PYTHON LOOPS (KEY)
# -------------------------------

# Bad (loop-based)
squared_loop = []
for x in numbers:
    squared_loop.append(x ** 2)

# Good (vectorized)
squared_vectorized = numbers ** 2


# -------------------------------
# 13. REAL-WORLD ML PATTERN
# -------------------------------

# Feature scaling (min-max normalization)
data = np.array([2, 4, 6, 8, 10])

min_val = np.min(data)
max_val = np.max(data)

normalized = (data - min_val) / (max_val - min_val)


# -------------------------------
# 14. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("numpy_basics.py executed successfully")
