import pandas as pd

students = {
    "Name": ["Ram", "Shyam", "Varinda", "Aadidev"],
    "Age": [20, 21, 19, 22],
    "Marks": [85, 92, 100, 33]
}

df = pd.DataFrame(students)

print("Original DataFrame:")
print(df)

print("\nShape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nSummary Statistics:")
print(df.describe())

print("\nFirst Two Rows:")
print(df.head(2))

print("\nLast Two Rows:")
print(df.tail(2))

print("\nNames Column:")
print(df["Name"])

print("\nFirst Student:")
print(df.loc[0])

print("\nStudents with Marks > 85:")
print(df[df["Marks"] > 85])

df["Passed"] = df["Marks"] >= 80
print(df)

print("\nAverage Marks:")
print(df["Marks"].mean())

print("\nMaximum Marks:")
print(df["Marks"].max())

print("\nSort by Marks (Descending):")
print(df.sort_values(by="Marks", ascending=False))
