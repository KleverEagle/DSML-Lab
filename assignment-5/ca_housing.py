import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, r2_score

df = pd.read_csv('california_housing.csv')

X = df.drop(columns=['Target_Reg', 'Target_Class'])
y_reg = df['Target_Reg']
y_class = df['Target_Class']

X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
    X, y_reg, y_class, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lin_model = LinearRegression()
lin_model.fit(X_train_scaled, yr_train)
yr_pred = lin_model.predict(X_test_scaled)
reg_accuracy = r2_score(yr_test, yr_pred)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, yc_train)
yc_pred = log_model.predict(X_test_scaled)

conf_mat = confusion_matrix(yc_test, yc_pred)
class_acc = accuracy_score(yc_test, yc_pred)
precision = precision_score(yc_test, yc_pred)
recall = recall_score(yc_test, yc_pred)

print("--- Regression Performance (Linear Regression) ---")
print(f"(R-squared): {reg_accuracy:.4f}\n")

print("--- Classification Performance (Logistic Regression) ---")
print(f"Accuracy:  {class_acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print("Confusion Matrix:")
print(conf_mat)