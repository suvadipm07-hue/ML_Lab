import pandas as pd

data = {
    "Student_Name": ["suvadip", "Sneha", "Akash", "Mohit", "Rahul"],
    "Roll_Number": [32, 33, 34, 35, 36],
    "Marks": [86, 65, 74, 82, 79],
    "Attendance": [8, 10, 3, 9, 10]
}

df = pd.DataFrame(data)

def grade(marks):
    if marks >= 80:
        return "A"
    elif marks >= 70:
        return "B"
    elif marks >= 60:
        return "C"
    else:
        return "D"

df["Grade"] = df["Marks"].apply(grade)

print("Student Data with Grade:")
print(df)

