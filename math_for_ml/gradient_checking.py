"""
gradient_checking.py

Unit 3: Math for Machine Learning

Objective:
- Understand numerical gradient checking
- Compare analytical and numerical gradients
- Verify gradient implementations
"""

import numpy as np


# -------------------------------
# 1. DEFINE FUNCTION
# -------------------------------

def function(x):
    """
    f(x) = x^2 + 3x + 2
    """
    return x**2 + 3 * x + 2


# -------------------------------
# 2. ANALYTICAL GRADIENT
# -------------------------------

def analytical_gradient(x):
    """
    Derivative:

    f(x) = x^2 + 3x + 2

    f'(x) = 2x + 3
    """
    return 2 * x + 3


# -------------------------------
# 3. NUMERICAL GRADIENT
# -------------------------------

def numerical_gradient(
    function,
    x,
    epsilon=1e-5
):
    """
    Approximate derivative using
    the central difference method.
    """

    forward = function(x + epsilon)
    backward = function(x - epsilon)

    return (
        forward - backward
    ) / (2 * epsilon)


# -------------------------------
# 4. TEST VALUES
# -------------------------------

test_values = [
    -3.0,
    -1.0,
    0.0,
    2.0,
    5.0
]


print("Gradient Checking")
print("=" * 50)


# -------------------------------
# 5. COMPARE GRADIENTS
# -------------------------------

for x in test_values:

    analytical = analytical_gradient(x)

    numerical = numerical_gradient(
        function,
        x
    )

    difference = abs(
        analytical - numerical
    )

    print(f"\nx = {x}")

    print(
        f"Analytical Gradient: "
        f"{analytical:.8f}"
    )

    print(
        f"Numerical Gradient : "
        f"{numerical:.8f}"
    )

    print(
        f"Difference          : "
        f"{difference:.10f}"
    )


# -------------------------------
# 6. GRADIENT CHECK FUNCTION
# -------------------------------

def check_gradient(
    function,
    analytical_gradient,
    x,
    epsilon=1e-5,
    tolerance=1e-7
):

    analytical = analytical_gradient(x)

    numerical = numerical_gradient(
        function,
        x,
        epsilon
    )

    difference = abs(
        analytical - numerical
    )

    return difference < tolerance


# -------------------------------
# 7. RUN GRADIENT CHECK
# -------------------------------

print("\nGradient Check Results")
print("=" * 50)

for x in test_values:

    passed = check_gradient(
        function,
        analytical_gradient,
        x
    )

    status = "PASS" if passed else "FAIL"

    print(
        f"x = {x:5.1f} → {status}"
    )


# -------------------------------
# 8. VECTOR GRADIENT EXAMPLE
# -------------------------------

def vector_function(x):
    """
    f(x) = x₁² + 2x₂² + 3x₃²
    """
    return np.sum(
        np.array([1, 2, 3]) * x**2
    )


def vector_analytical_gradient(x):
    """
    Gradient:

    [2x₁, 4x₂, 6x₃]
    """
    coefficients = np.array([1, 2, 3])

    return 2 * coefficients * x


def vector_numerical_gradient(
    function,
    x,
    epsilon=1e-5
):

    gradient = np.zeros_like(
        x,
        dtype=float
    )

    for i in range(len(x)):

        x_forward = x.copy()
        x_backward = x.copy()

        x_forward[i] += epsilon
        x_backward[i] -= epsilon

        gradient[i] = (
            function(x_forward)
            - function(x_backward)
        ) / (2 * epsilon)

    return gradient


# -------------------------------
# 9. TEST VECTOR GRADIENT
# -------------------------------

x = np.array([
    2.0,
    3.0,
    4.0
])

analytical_vector = (
    vector_analytical_gradient(x)
)

numerical_vector = (
    vector_numerical_gradient(
        vector_function,
        x
    )
)

difference_vector = np.linalg.norm(
    analytical_vector
    - numerical_vector
)


print("\nVector Gradient Check")
print("=" * 50)

print(
    "Analytical:",
    np.round(
        analytical_vector,
        8
    )
)

print(
    "Numerical  :",
    np.round(
        numerical_vector,
        8
    )
)

print(
    "Difference :",
    f"{difference_vector:.10f}"
)


# -------------------------------
# 10. CONCEPT SUMMARY
# -------------------------------

print("\nConcept Summary:")

print(
    "- Analytical Gradient → Derived mathematically"
)

print(
    "- Numerical Gradient → Approximated using finite differences"
)

print(
    "- Central Difference → Uses f(x+ε) and f(x-ε)"
)

print(
    "- Gradient Checking → Validates gradient implementations"
)


# -------------------------------
# 11. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print(
        "\ngradient_checking.py "
        "executed successfully"
)
