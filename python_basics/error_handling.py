"""
error_handling.py

Objective:
- Understand Python exception handling
- Build safe, predictable execution flows
- Prevent silent failures in ML & data pipelines
"""

# -------------------------------
# 1. BASIC TRY / EXCEPT
# -------------------------------

try:
    result = 10 / 2
except ZeroDivisionError:
    result = None


# -------------------------------
# 2. MULTIPLE EXCEPT BLOCKS
# -------------------------------

def safe_convert(value):
    try:
        return int(value)
    except ValueError:
        return None
    except TypeError:
        return None


# -------------------------------
# 3. EXCEPT WITH ERROR OBJECT
# -------------------------------

try:
    x = int("abc")
except ValueError as e:
    error_message = str(e)


# -------------------------------
# 4. ELSE BLOCK (CLEAN SUCCESS PATH)
# -------------------------------

try:
    y = int("123")
except ValueError:
    y = None
else:
    y = y * 2


# -------------------------------
# 5. FINALLY BLOCK (CLEANUP)
# -------------------------------

try:
    file = open("sample.txt", "w")
    file.write("AI & ML Python Practices")
finally:
    file.close()


# -------------------------------
# 6. RAISING CUSTOM ERRORS
# -------------------------------

def validate_probability(p: float):
    if not 0.0 <= p <= 1.0:
        raise ValueError("Probability must be between 0 and 1")


# -------------------------------
# 7. CUSTOM EXCEPTION CLASS
# -------------------------------

class ModelNotTrainedError(Exception):
    pass


def predict(model):
    if model is None:
        raise ModelNotTrainedError("Model must be trained before prediction")
    return "prediction"


# -------------------------------
# 8. SAFE DATA PIPELINE PATTERN
# -------------------------------

def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 0.0


# -------------------------------
# 9. AVOIDING SILENT FAILURES
# -------------------------------

def load_data(path: str):
    if not path:
        raise FileNotFoundError("Data path not provided")
    return "data_loaded"


# -------------------------------
# 10. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("error_handling.py executed successfully")
