"""
early_stopping.py

Unit 5: Deep Learning Fundamentals

Objective:
- Understand Early Stopping
- Prevent overfitting during training
- Simulate validation loss monitoring
"""

import numpy as np


# -------------------------------
# 1. SIMULATED VALIDATION LOSS
# -------------------------------

validation_loss = [
    0.82,
    0.70,
    0.61,
    0.55,
    0.52,
    0.51,
    0.50,
    0.51,
    0.53,
    0.56,
    0.60
]


# -------------------------------
# 2. EARLY STOPPING PARAMETERS
# -------------------------------

patience = 3
best_loss = float("inf")
counter = 0


# -------------------------------
# 3. TRAINING SIMULATION
# -------------------------------

for epoch, loss in enumerate(validation_loss, start=1):

    print(f"Epoch {epoch:02d} | Validation Loss: {loss:.2f}")

    if loss < best_loss:
        best_loss = loss
        counter = 0
        print("✓ Best model updated")

    else:
        counter += 1
        print(f"No improvement ({counter}/{patience})")

    if counter >= patience:
        print("\nEarly stopping triggered.")
        break


# -------------------------------
# 4. FINAL RESULT
# -------------------------------

print(f"\nBest Validation Loss: {best_loss:.2f}")


# -------------------------------
# 5. WHY EARLY STOPPING?
# -------------------------------

print("\nBenefits:")
print("- Prevents overfitting")
print("- Saves computation time")
print("- Keeps the best-performing model")


# -------------------------------
# 6. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\nearly_stopping.py executed successfully")
