"""
file_handling.py

Unit 1: Python Basics

Objective:
- Learn file handling in Python
- Read, write, and append files
- Handle file-related exceptions
"""

# -------------------------------
# 1. WRITE TO A FILE
# -------------------------------

file_name = "sample.txt"

with open(file_name, "w") as file:
    file.write("Welcome to AI & ML Python Practices!\n")
    file.write("Learning Python step by step.\n")

print("File created and written successfully.")

# -------------------------------
# 2. READ THE FILE
# -------------------------------

with open(file_name, "r") as file:
    content = file.read()

print("\nFile Content:")
print(content)

# -------------------------------
# 3. APPEND NEW DATA
# -------------------------------

with open(file_name, "a") as file:
    file.write("Appending a new line.\n")

print("New data appended successfully.")

# -------------------------------
# 4. READ LINE BY LINE
# -------------------------------

print("\nReading Line by Line:")

with open(file_name, "r") as file:
    for line in file:
        print(line.strip())

# -------------------------------
# 5. HANDLE FILE ERRORS
# -------------------------------

try:
    with open("missing_file.txt", "r") as file:
        print(file.read())

except FileNotFoundError:
    print("\nError: File does not exist.")

# -------------------------------
# 6. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\nfile_handling.py executed successfully")
