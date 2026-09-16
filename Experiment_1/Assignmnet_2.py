
import pandas as pd

data={
    "Student_Name":["suvadip","Sneha","Akash","Mohit","Rahul"],
    "Roll_Number":[32,33,34,35,36],
    "Marks":[86,65,74,82,79],
    "Attendance":[8,10,3,9,10]
    }
df=pd.DataFrame(data)
print("Data of student who have got marks over 80:")
print(df[df["Marks"]>80])
