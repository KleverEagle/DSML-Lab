
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("breast_cancer_knn_dataset.csv")

X = df.drop(columns=["target"])
y = df["target"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Compare K values using 5-fold cross-validation
k_values = [3, 5, 7, 9]
results = {}

print("5-Fold Cross-Validation Results:")

for k in k_values:
    model = make_pipeline(
        MinMaxScaler(),
        KNeighborsClassifier(n_neighbors=k)
    )

    scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="accuracy"
    )

    results[k] = scores.mean()
    print(f"K={k}: Mean Accuracy = {scores.mean():.4f}")

# Select best K
best_k = max(results, key=results.get)
print(f"\nBest K: {best_k}")

# Train final model using the full training set
final_model = make_pipeline(
    MinMaxScaler(),
    KNeighborsClassifier(n_neighbors=best_k)
)

final_model.fit(X_train, y_train)

# Evaluate on the test set
y_pred = final_model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["malignant", "benign"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
