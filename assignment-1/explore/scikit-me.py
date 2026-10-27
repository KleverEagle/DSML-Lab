from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Features: [Hours Studied, Attendance]
X = [[2, 60], [3, 65], [4, 70], [5, 75], [6, 80], [7, 85], [8, 90], [9, 95]]

# Pass / Fail
y = [0, 0, 0, 1, 1, 1, 1, 1]

print("Features:")
for sample in X:
    print(sample)

print("\nLabels:")
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

print("\nPredictions:", predictions)
print("Actual Labels:", y_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, predictions))

# Predict a new student's result
new_student = [[6, 78]]
result = model.predict(new_student)

print("\nNew Student:", new_student[0])
print("Prediction:", "Pass" if result[0] == 1 else "Fail")
