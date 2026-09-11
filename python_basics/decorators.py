"""
Python Decorators

Practice:
- Functions as first-class objects
- Basic decorators
- functools.wraps
- *args and **kwargs
- Decorators with arguments
- Multiple decorators
- Practical logging and timing examples
"""

from functools import wraps
import time


# ---------------------------------------------------------
# 1. Functions as First-Class Objects
# ---------------------------------------------------------

def greet(name):
    return f"Hello, {name}!"


# A function can be assigned to another variable.
say_hello = greet

print("Function as Object:")
print(say_hello("Ultrex"))


# ---------------------------------------------------------
# 2. Basic Decorator
# ---------------------------------------------------------

def log_call(func):
    @wraps(func)
    def wrapper():
        print(f"Calling: {func.__name__}")

        result = func()

        print(f"Finished: {func.__name__}")

        return result

    return wrapper


@log_call
def welcome():
    print("Welcome to Python decorators!")


print("\nBasic Decorator:")
welcome()


# ---------------------------------------------------------
# 3. Decorator with Function Arguments
# ---------------------------------------------------------

def log_arguments(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\nCalling: {func.__name__}")
        print(f"Arguments: {args}")
        print(f"Keyword Arguments: {kwargs}")

        result = func(*args, **kwargs)

        print(f"Result: {result}")

        return result

    return wrapper


@log_arguments
def add(a, b):
    return a + b


print("\nDecorator with Arguments:")
add(10, 20)


# ---------------------------------------------------------
# 4. Using **kwargs
# ---------------------------------------------------------

@log_arguments
def create_user(name, age, city="Unknown"):
    return {
        "name": name,
        "age": age,
        "city": city,
    }


print("\nDecorator with Keyword Arguments:")

user = create_user(
    "Alice",
    25,
    city="Mumbai",
)

print(user)


# ---------------------------------------------------------
# 5. Preserving Function Metadata
# ---------------------------------------------------------

def simple_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@simple_decorator
def calculate_square(number):
    """Return the square of a number."""
    return number ** 2


print("\nFunction Metadata:")
print("Name:", calculate_square.__name__)
print("Documentation:", calculate_square.__doc__)


# ---------------------------------------------------------
# 6. Timing Decorator
# ---------------------------------------------------------

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        end_time = time.perf_counter()

        elapsed = end_time - start_time

        print(
            f"{func.__name__} took "
            f"{elapsed:.6f} seconds"
        )

        return result

    return wrapper


@measure_time
def slow_operation():
    time.sleep(0.5)

    return "Operation completed"


print("\nTiming Decorator:")

result = slow_operation()

print(result)


# ---------------------------------------------------------
# 7. Decorator with Its Own Arguments
# ---------------------------------------------------------

def repeat(times):
    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            result = None

            for _ in range(times):
                result = func(*args, **kwargs)

            return result

        return wrapper

    return decorator


@repeat(3)
def say_message():
    print("Hello!")


print("\nParameterized Decorator:")
say_message()


# ---------------------------------------------------------
# 8. Multiple Decorators
# ---------------------------------------------------------

def uppercase(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        return result.upper()

    return wrapper


def add_exclamation(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        return result + "!"

    return wrapper


@uppercase
@add_exclamation
def message():
    return "python is powerful"


print("\nMultiple Decorators:")
print(message())


# ---------------------------------------------------------
# 9. Simple Authentication Decorator
# ---------------------------------------------------------

def require_admin(func):
    @wraps(func)
    def wrapper(user_role, *args, **kwargs):

        if user_role != "admin":
            print("Access denied.")

            return None

        return func(
            user_role,
            *args,
            **kwargs,
        )

    return wrapper


@require_admin
def delete_database(user_role):
    print("Database deleted.")


print("\nAuthentication Decorator:")

delete_database("user")
delete_database("admin")


# ---------------------------------------------------------
# 10. Practical Validation Decorator
# ---------------------------------------------------------

def require_positive(func):
    @wraps(func)
    def wrapper(number, *args, **kwargs):

        if number <= 0:
            raise ValueError(
                "Number must be positive."
            )

        return func(
            number,
            *args,
            **kwargs,
        )

    return wrapper


@require_positive
def calculate_log_input(number):
    return number


print("\nValidation Decorator:")

print(calculate_log_input(10))

try:
    calculate_log_input(-5)

except ValueError as error:
    print("Error:", error)


# ---------------------------------------------------------
# 11. Decorator Execution Flow
# ---------------------------------------------------------

def trace(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        print("1. Before function")

        result = func(*args, **kwargs)

        print("3. After function")

        return result

    return wrapper


@trace
def example():
    print("2. Inside function")


print("\nDecorator Execution Flow:")
example()


# ---------------------------------------------------------
# 12. Decorators Are Functions Returning Functions
# ---------------------------------------------------------

def decorator(func):
    def wrapper(*args, **kwargs):
        print("Before")
        result = func(*args, **kwargs)
        print("After")

        return result

    return wrapper


def normal_function():
    print("Inside")


decorated_function = decorator(normal_function)

print("\nManual Decoration:")

decorated_function()


# ---------------------------------------------------------
# 13. Practical Takeaways
# ---------------------------------------------------------

print("\nDecorator Takeaways:")

print("1. A decorator modifies function behavior.")
print("2. @decorator is syntactic sugar for function = decorator(function).")
print("3. *args and **kwargs make decorators reusable.")
print("4. functools.wraps preserves function metadata.")
print("5. Decorators are useful for logging, timing, validation,")
print("   authentication, caching, retries, and access control.")
