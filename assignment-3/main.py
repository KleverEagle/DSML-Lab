import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv(".\\dataset-insuracne.csv")

# Pre-plotting
df["bmi"].plot(
    kind="density",
    figsize=(8, 5)
)
plt.xlabel("BMI")
plt.ylabel("Density")
plt.title("Distribution of value")
plt.show()


#Min-Max Scaling / Normalisation
clms = ["age","charges"]
scaler = MinMaxScaler()
df[clms] = scaler.fit_transform(df[clms])


#Standardisation / Z - Score
scaler = StandardScaler()
df[["bmi"]] = scaler.fit_transform(df[["bmi"]])

# print(df["bmi"].mean())
# print(df["bmi"].std(ddof=0))

print(df.describe())

df["bmi"].plot(
    kind="density",
    figsize=(8, 5)
)
plt.xlabel("BMI")
plt.ylabel("Density")
plt.title("Distribution of value")
plt.show()



