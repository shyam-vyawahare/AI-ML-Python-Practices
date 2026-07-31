"""
iterators_generators.py

Unit 1: Python Basics

Objective:
- Understand iterables and iterators
- Learn generators using yield
- Practice generator expressions
"""


# -------------------------------
# 1. ITERABLE
# -------------------------------

numbers = [10, 20, 30, 40]

print("Iterable:")
print(numbers)


# -------------------------------
# 2. ITERATOR
# -------------------------------

iterator = iter(numbers)

print("\nIterator:")

print(next(iterator))
print(next(iterator))
print(next(iterator))
print(next(iterator))


# -------------------------------
# 3. GENERATOR FUNCTION
# -------------------------------

def square_generator(limit):
    for number in range(1, limit + 1):
        yield number ** 2


print("\nSquares using Generator:")

for value in square_generator(5):
    print(value)


# -------------------------------
# 4. GENERATOR EXPRESSION
# -------------------------------

cube_generator = (x ** 3 for x in range(1, 6))

print("\nCubes using Generator Expression:")

for cube in cube_generator:
    print(cube)


# -------------------------------
# 5. MEMORY COMPARISON
# -------------------------------

list_data = [x for x in range(1000)]
generator_data = (x for x in range(1000))

print("\nMemory Comparison")

print("List Type      :", type(list_data))
print("Generator Type :", type(generator_data))


# -------------------------------
# 6. PRACTICAL EXAMPLE
# -------------------------------

def even_numbers(limit):
    for number in range(limit + 1):
        if number % 2 == 0:
            yield number


print("\nEven Numbers:")

for number in even_numbers(20):
    print(number, end=" ")

print()


# -------------------------------
# 7. STOP ITERATION EXAMPLE
# -------------------------------

iterator = iter([1, 2])

print("\nStopIteration Example")

try:
    while True:
        print(next(iterator))
except StopIteration:
    print("Iterator exhausted.")


# -------------------------------
# 8. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\niterators_generators.py executed successfully")
