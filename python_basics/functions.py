"""
functions.py

Objective:
- Revise Python functions with production and ML relevance
- Emphasize clarity, reusability, and clean interfaces
"""

# -------------------------------
# 1. BASIC FUNCTION
# -------------------------------

def greet(name: str) -> str:
    return f"Hello, {name}"


# -------------------------------
# 2. FUNCTION WITH DEFAULT VALUES
# -------------------------------

def calculate_accuracy(correct: int, total: int = 1) -> float:
    if total == 0:
        return 0.0
    return correct / total


# -------------------------------
# 3. MULTIPLE RETURN VALUES
# -------------------------------

def min_max(values: list):
    return min(values), max(values)


# -------------------------------
# 4. *args (VARIABLE POSITIONAL)
# -------------------------------

def average(*numbers: float) -> float:
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


# -------------------------------
# 5. **kwargs (KEYWORD ARGUMENTS)
# -------------------------------

def build_config(**kwargs) -> dict:
    return kwargs


# -------------------------------
# 6. TYPE HINTS (ML-FRIENDLY)
# -------------------------------

def scale_value(value: float, factor: float) -> float:
    return value * factor


# -------------------------------
# 7. LAMBDA FUNCTIONS
# -------------------------------

square = lambda x: x ** 2


# -------------------------------
# 8. HIGHER-ORDER FUNCTIONS
# -------------------------------

def apply_function(func, data: list):
    return [func(x) for x in data]


# -------------------------------
# 9. DOCSTRING STANDARD
# -------------------------------

def normalize(value: float, max_value: float) -> float:
    """
    Normalize a value between 0 and 1.
    """
    if max_value == 0:
        return 0.0
    return value / max_value


# -------------------------------
# 10. REAL-WORLD ML PATTERN
# -------------------------------

def threshold_predictions(predictions: list, threshold: float = 0.5) -> list:
    return [1 if p >= threshold else 0 for p in predictions]


# -------------------------------
# 11. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print(greet("Ultrex"))
    print("functions.py executed successfully")
