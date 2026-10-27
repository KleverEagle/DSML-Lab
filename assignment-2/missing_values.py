import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

df = pd.read_csv(".\\dataset.csv")

columns = ["ParentEduc", "Gender", "EthnicGroup","TestPrep"]

# Find rows with missing values in any of these columns
df = df.dropna(subset=columns).reset_index(drop=True)
print(df.head(10))

# Using Most Frequent instead of removing missing values
# df = pd.read_csv(".\\dataset.csv")
# columns = ["ParentEduc", "Gender", "EthnicGroup"]
# imputer = SimpleImputer(strategy="most_frequent")
# df[columns] = imputer.fit_transform(df[columns])

# Filling missing numbers with mean, median and mode
df["MathScore"] = df["MathScore"].fillna(df["MathScore"].mean())
df["ReadingScore"] = df["ReadingScore"].fillna(df["ReadingScore"].median())
df["WritingScore"] = df["WritingScore"].fillna(df["WritingScore"].mode())

# Encoding

# Label Encoding for Test Prep
le = LabelEncoder()
df["TestPrep"] = le.fit_transform(df["TestPrep"])
print(dict(zip(le.classes_, le.transform(le.classes_))))

# Manual Label Encoding
# df["TestPrep"] = df["TestPrep"].map({
#     "none": 0,
#     "completed": 1
# })

# One Hot Encoding for Ethnic Group
encoder = OneHotEncoder(sparse_output=False)
ethnic_encoded = encoder.fit_transform(df[["EthnicGroup"]])
ethnic_df = pd.DataFrame(
    ethnic_encoded,
    columns=encoder.get_feature_names_out(["EthnicGroup"]),
    index=df.index
)
df = pd.concat([df, ethnic_df], axis=1)
df = df.drop("EthnicGroup", axis=1)

print(df.head(10))

# Dummy variables for ParentEduc
df = pd.get_dummies(
    df,
    columns=["ParentEduc"],
    drop_first=True,
    dtype=int
)
print(df.head(10))

print(df.columns.tolist())
df.to_csv("processed_data.csv", index=False)




