"""
syntax_revision.py

Objective:
- Fast Python syntax revision
- AI/ML-oriented coding style
- Focus on clarity, correctness, and patterns

Rule:
If a concept doesn’t help in real-world Python or ML workflows, it doesn’t belong here.
"""

# -------------------------------
# 1. VARIABLES & BASIC DATA TYPES
# -------------------------------

name = "Ultrex"
age = 22
cgpa = 8.5
is_engineer = True

# Python is dynamically typed
dynamic_var = 10
dynamic_var = "Now I'm a string"


# -------------------------------
# 2. MULTI-VARIABLE ASSIGNMENT
# -------------------------------

x, y, z = 10, 20, 30
a = b = c = 0


# -------------------------------
# 3. TYPE CHECKING (IMPORTANT FOR ML)
# -------------------------------

value = 3.14
print(type(value))  # <class 'float'>


# -------------------------------
# 4. BASIC INPUT (COMMENTED FOR SCRIPT SAFETY)
# -------------------------------

# user_name = input("Enter your name: ")
# print(f"Welcome, {user_name}")


# -------------------------------
# 5. TYPE CASTING (VERY IMPORTANT)
# -------------------------------

num_str = "100"
num_int = int(num_str)
num_float = float(num_int)


# -------------------------------
# 6. BOOLEAN LOGIC
# -------------------------------

is_valid = True
has_permission = False

if is_valid and not has_permission:
    pass  # placeholder for logic


# -------------------------------
# 7. COMPARISON & IDENTITY
# -------------------------------

a = 256
b = 256

print(a == b)   # Value comparison
print(a is b)   # Memory reference comparison


# -------------------------------
# 8. NONE TYPE (COMMON IN ML PIPELINES)
# -------------------------------

model = None

if model is None:
    print("Model not initialized")


# -------------------------------
# 9. BASIC OUTPUT FORMATTING
# -------------------------------

score = 92.45678
print(f"Score: {score:.2f}")


# -------------------------------
# 10. PYTHON EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("syntax_revision.py executed successfully")
