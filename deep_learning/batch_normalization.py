"""
batch_normalization.py

Unit 5: Deep Learning Fundamentals

Objective:
- Learn Batch Normalization
- Compare outputs in training and evaluation modes
- Understand where BatchNorm fits in a neural network
"""

import torch
import torch.nn as nn


# -------------------------------
# 1. SAMPLE INPUT
# -------------------------------

X = torch.randn(8, 10)


# -------------------------------
# 2. MODEL WITH BATCH NORMALIZATION
# -------------------------------

class BatchNormNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(10, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),

            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.network(x)


# -------------------------------
# 3. CREATE MODEL
# -------------------------------

model = BatchNormNetwork()


# -------------------------------
# 4. TRAINING MODE
# -------------------------------

model.train()
train_output = model(X)


# -------------------------------
# 5. EVALUATION MODE
# -------------------------------

model.eval()

with torch.no_grad():
    eval_output = model(X)


# -------------------------------
# 6. COMPARE OUTPUTS
# -------------------------------

print("Training Output:")
print(train_output)

print("\nEvaluation Output:")
print(eval_output)


# -------------------------------
# 7. MODEL INFORMATION
# -------------------------------

print("\nModel Architecture:")
print(model)


# -------------------------------
# 8. WHY BATCH NORMALIZATION?
# -------------------------------

print("\nBenefits:")
print("- Faster convergence")
print("- More stable training")
print("- Helps reduce internal covariate shift")
print("- Often improves generalization")


# -------------------------------
# 9. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\nbatch_normalization.py executed successfully")
