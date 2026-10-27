import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(".\\dataset.csv")

# Distribution of scores
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(df["MathScore"].dropna(), bins=20, color="red")
axes[0].set_title("Math Score")

axes[1].hist(df["ReadingScore"].dropna(), bins=20, color="blue")
axes[1].set_title("Reading Score")

axes[2].hist(df["WritingScore"].dropna(), bins=20, color="green")
axes[2].set_title("Writing Score")

plt.tight_layout()
plt.show()

# Average scores by gender
grouped = df.groupby("Gender")[["MathScore", "ReadingScore", "WritingScore"]].mean()

x = range(len(grouped))
width = 0.25

plt.figure(figsize=(10, 6))

plt.bar([i - width for i in x], grouped["MathScore"], width, label="Math")
plt.bar(x, grouped["ReadingScore"], width, label="Reading")
plt.bar([i + width for i in x], grouped["WritingScore"], width, label="Writing")

plt.xticks(x, grouped.index)
plt.xlabel("Gender")
plt.ylabel("Average Score")
plt.title("Average Scores by Gender")
plt.legend()
plt.tight_layout()
plt.show()


# Missing values
missing = df.isnull().sum()
missing = missing[missing > 0]

plt.figure(figsize=(8, 5))

bars = plt.bar(missing.index, missing.values)
for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(bar.get_height()),
        ha="center",
        va="bottom"
    )

plt.xlabel("Column")
plt.ylabel("Number of Missing Values")
plt.title("Missing Values by Column")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
