"""
weight_initialization.py

Unit 5: Deep Learning Fundamentals

Objective:
- Learn common weight initialization techniques
- Compare default, Xavier, and He initialization
- Understand their impact on deep learning models
"""

import torch
import torch.nn as nn


# -------------------------------
# 1. DEFINE MODEL
# -------------------------------

class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


model = SimpleNetwork()


# -------------------------------
# 2. XAVIER INITIALIZATION
# -------------------------------

def initialize_xavier(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


# -------------------------------
# 3. HE INITIALIZATION
# -------------------------------

def initialize_he(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(
                layer.weight,
                nonlinearity="relu"
            )
            nn.init.zeros_(layer.bias)


# -------------------------------
# 4. APPLY INITIALIZATION
# -------------------------------

print("Applying Xavier Initialization...")
initialize_xavier(model)

print("\nFirst Layer Weights (Xavier):")
print(model.fc1.weight[:2])


print("\nApplying He Initialization...")
initialize_he(model)

print("\nFirst Layer Weights (He):")
print(model.fc1.weight[:2])


# -------------------------------
# 5. SAMPLE FORWARD PASS
# -------------------------------

X = torch.randn(4, 10)

output = model(X)

print("\nModel Output:")
print(output)


# -------------------------------
# 6. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\nweight_initialization.py executed successfully")
