"""
cnn_basics.py

Unit 5: Deep Learning Fundamentals

Objective:
- Learn the basic architecture of a CNN
- Understand convolution, ReLU, pooling, and flattening
- Build a simple image classifier using PyTorch
"""

import torch
import torch.nn as nn


# -------------------------------
# 1. DEFINE CNN MODEL
# -------------------------------

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -------------------------------
# 2. CREATE MODEL
# -------------------------------

model = SimpleCNN()

print("CNN Architecture:")
print(model)


# -------------------------------
# 3. SAMPLE INPUT
# -------------------------------

# Batch of 8 grayscale images (28x28)
images = torch.randn(8, 1, 28, 28)


# -------------------------------
# 4. FORWARD PASS
# -------------------------------

outputs = model(images)

print("\nInput Shape :", images.shape)
print("Output Shape:", outputs.shape)


# -------------------------------
# 5. PARAMETER COUNT
# -------------------------------

total_params = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_params = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(f"\nTotal Parameters    : {total_params}")
print(f"Trainable Parameters: {trainable_params}")


# -------------------------------
# 6. PREDICTED CLASSES
# -------------------------------

predictions = torch.argmax(outputs, dim=1)

print("\nPredicted Classes:")
print(predictions)


# -------------------------------
# 7. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\ncnn_basics.py executed successfully")
