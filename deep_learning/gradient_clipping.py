"""
gradient_clipping.py

Unit 5: Deep Learning Fundamentals

Objective:
- Understand Gradient Clipping
- Prevent exploding gradients
- Learn how PyTorch clips gradients during training
"""

import torch
import torch.nn as nn
import torch.optim as optim


# -------------------------------
# 1. CREATE SAMPLE DATA
# -------------------------------

X = torch.randn(100, 5)
y = torch.randn(100, 1)


# -------------------------------
# 2. DEFINE MODEL
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
optimizer = optim.Adam(model.parameters(), lr=0.01)


# -------------------------------
# 4. TRAINING LOOP
# -------------------------------

epochs = 10

for epoch in range(epochs):

    optimizer.zero_grad()

    predictions = model(X)

    loss = criterion(predictions, y)

    loss.backward()

    # ---------------------------
    # Gradient Clipping
    # ---------------------------
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    print(f"Epoch {epoch + 1:02d} | Loss: {loss.item():.4f}")


# -------------------------------
# 5. WHY GRADIENT CLIPPING?
# -------------------------------

print("\nBenefits:")
print("- Prevents exploding gradients")
print("- Stabilizes training")
print("- Commonly used in deep networks")
print("- Essential for sequence models and LLMs")


# -------------------------------
# 6. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\ngradient_clipping.py executed successfully")
