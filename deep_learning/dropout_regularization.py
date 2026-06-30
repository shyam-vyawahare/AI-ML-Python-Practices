"""
dropout_regularization.py

Unit 5: Deep Learning Fundamentals

Objective:
- Understand Dropout Regularization
- Prevent overfitting in neural networks
- Build a simple PyTorch model with Dropout
"""

import torch
import torch.nn as nn


# -------------------------------
# 1. SAMPLE INPUT
# -------------------------------

X = torch.rand((5, 10))


# -------------------------------
# 2. MODEL WITH DROPOUT
# -------------------------------

class DropoutNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),

            # Randomly disables neurons during training
            nn.Dropout(p=0.5),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Dropout(p=0.3),

            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.network(x)


# -------------------------------
# 3. CREATE MODEL
# -------------------------------

model = DropoutNetwork()


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
# 6. OBSERVE THE DIFFERENCE
# -------------------------------

print("Training Output:")
print(train_output)

print("\nEvaluation Output:")
print(eval_output)


# -------------------------------
# 7. MODEL SUMMARY
# -------------------------------

print("\nModel Architecture:")
print(model)


# -------------------------------
# 8. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\ndropout_regularization.py executed successfully")
