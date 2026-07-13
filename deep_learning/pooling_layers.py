"""
pooling_layers.py

Unit 5: Deep Learning Fundamentals

Objective:
- Learn different pooling layers
- Compare MaxPool, AvgPool, and AdaptiveAvgPool
- Observe changes in feature map dimensions
"""

import torch
import torch.nn as nn


# -------------------------------
# 1. SAMPLE FEATURE MAP
# -------------------------------

feature_map = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)

print("Original Feature Map:")
print(feature_map)


# -------------------------------
# 2. MAX POOLING
# -------------------------------

max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

max_output = max_pool(feature_map)

print("\nMax Pooling Output:")
print(max_output)


# -------------------------------
# 3. AVERAGE POOLING
# -------------------------------

avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

avg_output = avg_pool(feature_map)

print("\nAverage Pooling Output:")
print(avg_output)


# -------------------------------
# 4. ADAPTIVE AVERAGE POOLING
# -------------------------------

adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

adaptive_output = adaptive_pool(feature_map)

print("\nAdaptive Average Pooling Output:")
print(adaptive_output)


# -------------------------------
# 5. SHAPE COMPARISON
# -------------------------------

print("\nShape Comparison")
print(f"Original : {feature_map.shape}")
print(f"MaxPool  : {max_output.shape}")
print(f"AvgPool  : {avg_output.shape}")
print(f"Adaptive : {adaptive_output.shape}")


# -------------------------------
# 6. WHY POOLING?
# -------------------------------

print("\nPooling Benefits:")
print("- Reduces feature map size")
print("- Lowers computational cost")
print("- Helps reduce overfitting")
print("- Makes features more robust to small translations")


# -------------------------------
# 7. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\npooling_layers.py executed successfully")
