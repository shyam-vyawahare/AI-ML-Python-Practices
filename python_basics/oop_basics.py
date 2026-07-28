"""
oop_basics.py

Unit 1: Python Basics

Objective:
- Learn Object-Oriented Programming (OOP)
- Understand classes, objects, inheritance, and method overriding
"""


# -------------------------------
# 1. BASE CLASS
# -------------------------------

class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        print(f"{self.name} makes a sound.")


# -------------------------------
# 2. INHERITED CLASS
# -------------------------------

class Dog(Animal):
    def speak(self):
        print(f"{self.name} says Woof!")


# -------------------------------
# 3. ANOTHER INHERITED CLASS
# -------------------------------

class Cat(Animal):
    def speak(self):
        print(f"{self.name} says Meow!")


# -------------------------------
# 4. CREATE OBJECTS
# -------------------------------

dog = Dog("Buddy")
cat = Cat("Luna")

dog.speak()
cat.speak()


# -------------------------------
# 5. POLYMORPHISM
# -------------------------------

animals = [
    Dog("Rocky"),
    Cat("Kitty"),
    Animal("Unknown")
]

print("\nAnimal Sounds:")

for animal in animals:
    animal.speak()


# -------------------------------
# 6. CLASS WITH ADDITIONAL METHODS
# -------------------------------

class Student:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

    def grade(self):
        if self.marks >= 90:
            return "A"
        elif self.marks >= 75:
            return "B"
        elif self.marks >= 60:
            return "C"
        return "D"

    def display(self):
        print(
            f"{self.name} | Marks: {self.marks} | Grade: {self.grade()}"
        )


student = Student("Alice", 88)
student.display()


# -------------------------------
# 7. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\noop_basics.py executed successfully")
