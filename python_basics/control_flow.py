"""
control_flow.py

Objective:
- Revise Python control flow with real-world relevance
- Focus on clean logic, early exits, and readability
- ML / data-pipeline friendly patterns
"""

# -------------------------------
# 1. BASIC CONDITIONALS
# -------------------------------

score = 78

if score >= 90:
    grade = "A"
elif score >= 75:
    grade = "B"
elif score >= 60:
    grade = "C"
else:
    grade = "Fail"


# -------------------------------
# 2. EARLY RETURN PATTERN
# -------------------------------

def validate_age(age: int) -> bool:
    if age < 0:
        return False
    if age < 18:
        return False
    return True


# -------------------------------
# 3. FOR LOOPS (DATA-DRIVEN)
# -------------------------------

data = [10, 20, 30, 40]

total = 0
for value in data:
    total += value


# -------------------------------
# 4. ENUMERATE (INDEX + VALUE)
# -------------------------------

for index, value in enumerate(data):
    pass


# -------------------------------
# 5. RANGE WITH STEP (COMMON)
# -------------------------------

for i in range(0, 10, 2):
    pass


# -------------------------------
# 6. WHILE LOOP (CONTROLLED USE)
# -------------------------------

counter = 0
while counter < 3:
    counter += 1


# -------------------------------
# 7. BREAK & CONTINUE (FILTERING)
# -------------------------------

numbers = [1, 2, 3, 4, 5, 6]

for n in numbers:
    if n == 4:
        break
    if n % 2 == 0:
        continue
    pass


# -------------------------------
# 8. MATCH-CASE (PYTHON 3.10+)
# -------------------------------

command = "train"

match command:
    case "train":
        action = "Training model"
    case "test":
        action = "Testing model"
    case "deploy":
        action = "Deploying model"
    case _:
        action = "Unknown command"


# -------------------------------
# 9. TRY / EXCEPT FLOW
# -------------------------------

def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None


# -------------------------------
# 10. LOOP + CONDITIONAL (ML STYLE)
# -------------------------------

predictions = [0.2, 0.6, 0.8, 0.4]

labels = []
for p in predictions:
    labels.append(1 if p >= 0.5 else 0)


# -------------------------------
# 11. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("control_flow.py executed successfully")
