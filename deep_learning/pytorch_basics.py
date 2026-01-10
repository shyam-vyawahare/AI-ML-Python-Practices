"""
pytorch_basics.py

Unit 5: Deep Learning Fundamentals

Objective:
- Learn PyTorch fundamentals
- Understand tensors, autograd, and modules
- Train a simple neural network using PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim


# -------------------------------
# 1. TENSORS (NUMPY-LIKE)
# -------------------------------

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.tensor([[1.0], [0.0]])

print("Tensor x:")
print(x)
print("Tensor y:")
print(y)


# -------------------------------
# 2. AUTOGRAD (GRADIENT TRACKING)
# -------------------------------

w = torch.randn((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

y_pred = x @ w + b
loss = torch.mean((y - y_pred) ** 2)

loss.backward()

print("\nGradients:")
print("dw:", w.grad)
print("db:", b.grad)


# -------------------------------
# 3. SIMPLE NEURAL NETWORK
# -------------------------------

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.activation(self.output(x))
        return x


# -------------------------------
# 4. DATASET (XOR-LIKE)
# -------------------------------

X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])


# -------------------------------
# 5. MODEL, LOSS, OPTIMIZER
# -------------------------------

model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)


# -------------------------------
# 6. TRAINING LOOP (PYTORCH STYLE)
# -------------------------------

epochs = 1000

for epoch in range(epochs):
    optimizer.zero_grad()          # reset gradients
    outputs = model(X)             # forward pass
    loss = criterion(outputs, y)   # compute loss
    loss.backward()                # backprop
    optimizer.step()               # update weights

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")


# -------------------------------
# 7. EVALUATION
# -------------------------------

with torch.no_grad():
    predictions = model(X)
    print("\nFinal Predictions:")
    print(predictions.round())


# -------------------------------
# 8. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\npytorch_basics.py executed successfully")
