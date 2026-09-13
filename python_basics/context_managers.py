"""
Python Context Managers

Practice:
- The `with` statement
- File context managers
- __enter__ and __exit__
- Creating custom context managers
- contextlib.contextmanager
- Exception handling inside context managers
- Practical resource-management patterns
"""

from contextlib import contextmanager
import time


# ---------------------------------------------------------
# 1. Basic `with` Statement
# ---------------------------------------------------------
# The `with` statement automatically handles cleanup.

print("1. Basic Context Manager:")

with open("context_demo.txt", "w") as file:
    file.write("Hello from a context manager.")

print("File automatically closed after leaving the block.")


# ---------------------------------------------------------
# 2. Reading the File
# ---------------------------------------------------------

print("\n2. Reading File:")

with open("context_demo.txt", "r") as file:
    content = file.read()

print(content)
print("File closed:", file.closed)


# ---------------------------------------------------------
# 3. Why Context Managers Matter
# ---------------------------------------------------------
# Without a context manager, you have to remember
# to close the resource manually.

print("\n3. Manual Resource Management:")

file = open("context_manual.txt", "w")

try:
    file.write("Manual file handling.")

finally:
    file.close()

print("File closed manually:", file.closed)


# ---------------------------------------------------------
# 4. Custom Context Manager Using a Class
# ---------------------------------------------------------

class DatabaseConnection:
    def __enter__(self):
        print("Opening database connection.")

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        print("Closing database connection.")

        if exc_type is not None:
            print("An exception occurred:")
            print(exc_value)

        return False


print("\n4. Custom Context Manager:")

with DatabaseConnection() as connection:
    print("Using database connection.")


# ---------------------------------------------------------
# 5. Context Manager with an Exception
# ---------------------------------------------------------

print("\n5. Context Manager with Exception:")

try:
    with DatabaseConnection():
        print("Performing database operation.")

        raise ValueError(
            "Something went wrong."
        )

except ValueError as error:
    print("Caught outside context manager:", error)


# ---------------------------------------------------------
# 6. Understanding __enter__ and __exit__
# ---------------------------------------------------------

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()

        print("Timer started.")

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.end = time.perf_counter()

        self.elapsed = self.end - self.start

        print(
            f"Timer stopped. "
            f"Elapsed time: {self.elapsed:.6f} seconds"
        )

        return False


print("\n6. Timer Context Manager:")

with Timer() as timer:
    time.sleep(0.5)


# ---------------------------------------------------------
# 7. Context Manager Using contextlib
# ---------------------------------------------------------
# contextlib allows us to create context managers
# without writing a full class.

@contextmanager
def temporary_message(message):
    print(f"Starting: {message}")

    try:
        yield

    finally:
        print(f"Finished: {message}")


print("\n7. contextlib.contextmanager:")

with temporary_message("Processing data"):
    print("Data processing is happening...")


# ---------------------------------------------------------
# 8. Context Manager Returning a Value
# ---------------------------------------------------------

@contextmanager
def managed_resource(resource_name):
    print(f"Opening resource: {resource_name}")

    resource = {
        "name": resource_name,
        "status": "open",
    }

    try:
        yield resource

    finally:
        resource["status"] = "closed"

        print(
            f"Closing resource: {resource_name}"
        )


print("\n8. Returning a Resource:")

with managed_resource("Model") as resource:
    print("Resource:", resource)
    print("Using resource...")


# ---------------------------------------------------------
# 9. Exception-Safe Cleanup
# ---------------------------------------------------------

@contextmanager
def safe_operation(operation):
    print(f"\nStarting operation: {operation}")

    try:
        yield

    except Exception as error:
        print(
            f"Operation failed: {error}"
        )

        raise

    finally:
        print(
            f"Cleaning up after: {operation}"
        )


print("\n9. Exception-Safe Cleanup:")

try:
    with safe_operation("Data preprocessing"):
        print("Preprocessing data...")

        raise RuntimeError(
            "Invalid input data."
        )

except RuntimeError:
    print("Exception handled by caller.")


# ---------------------------------------------------------
# 10. Practical ML Example
# ---------------------------------------------------------
# Context managers can be useful for temporarily
# changing an environment or resource state.

@contextmanager
def model_inference_mode(model_name):
    print(
        f"\nEntering inference mode for {model_name}"
    )

    try:
        yield

    finally:
        print(
            f"Leaving inference mode for {model_name}"
        )


print("\n10. ML-Style Context Manager:")

with model_inference_mode("TextClassifier"):
    print("Running predictions...")


# ---------------------------------------------------------
# 11. Nested Context Managers
# ---------------------------------------------------------

print("\n11. Nested Context Managers:")

with open("nested_demo.txt", "w") as file:

    with Timer():
        file.write("Nested context manager example.")

print("Both resources were cleaned up.")


# ---------------------------------------------------------
# 12. Multiple Context Managers
# ---------------------------------------------------------

print("\n12. Multiple Context Managers:")

with (
    open("file_a.txt", "w") as file_a,
    open("file_b.txt", "w") as file_b,
):
    file_a.write("File A")
    file_b.write("File B")

print("Both files were automatically closed.")


# ---------------------------------------------------------
# 13. Context Manager Lifecycle
# ---------------------------------------------------------

class LifecycleDemo:
    def __enter__(self):
        print("→ __enter__()")
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        print("→ __exit__()")


print("\n13. Context Manager Lifecycle:")

with LifecycleDemo():
    print("→ Inside with block")


# ---------------------------------------------------------
# 14. Practical Takeaways
# ---------------------------------------------------------

print("\n14. Context Manager Takeaways:")

print("1. `with` guarantees cleanup.")
print("2. __enter__() runs when entering the block.")
print("3. __exit__() runs when leaving the block.")
print("4. __exit__() also runs when an exception occurs.")
print("5. @contextmanager simplifies custom context managers.")
print("6. Context managers are useful for files, databases,")
print("   locks, network connections, temporary state, and resources.")
