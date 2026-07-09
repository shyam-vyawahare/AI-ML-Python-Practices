"""
learning_rate_scheduler.py

Unit 5: Deep Learning Fundamentals

Objective:
- Learn Learning Rate Scheduling
- Observe how learning rates change during training
- Improve optimization using PyTorch schedulers
"""

import torch
import torch.nn as nn
import torch.optim as optim


# -------------------------------
# 1. SAMPLE DATA
# -------------------------------

X = torch.randn(100, 5)
y = torch.randn(100, 1)


# -------------------------------
# 2. MODEL
# -------------------------------

model = nn.Sequential(
    nn.Linear(5, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)


# -------------------------------
# 3. LOSS & OPTIMIZER
# -------------------------------

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)


# -------------------------------
# 4. LEARNING RATE SCHEDULER
# -------------------------------

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)


# -------------------------------
# 5. TRAINING LOOP
# -------------------------------

epochs = 15

for epoch in range(epochs):

    optimizer.zero_grad()

    predictions = model(X)

    loss = criterion(predictions, y)

    loss.backward()

    optimizer.step()

    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Loss: {loss.item():.4f} | "
        f"Learning Rate: {current_lr:.5f}"
    )


# -------------------------------
# 6. WHY USE A SCHEDULER?
# -------------------------------

print("\nBenefits:")
print("- Faster convergence")
print("- More stable optimization")
print("- Better final model performance")
print("- Common practice in deep learning")


# -------------------------------
# 7. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\nlearning_rate_scheduler.py executed successfully")
