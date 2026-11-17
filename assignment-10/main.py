import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# 1. Load & Split the Wine Dataset
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='cultivar')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Construct & Train the Decision Tree
dt_wine = DecisionTreeClassifier(
    criterion='entropy',
    # max_depth=3,
    # min_samples_split=4,
    random_state=42,
)
dt_wine.fit(X_train, y_train)

# 3. Evaluate the Model
y_pred = dt_wine.predict(X_test)
print(f'Test Accuracy: {accuracy_score(y_test, y_pred):.4f}\n')
print(
    'Classification Report:\n',
    classification_report(y_test, y_pred, target_names=data.target_names),
)

# 4. Print Text-Based Decision Rules
print(
    'Learned Decision Rules:\n',
    export_text(dt_wine, feature_names=list(data.feature_names)),
)

plot_tree(
    dt_wine,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
    rounded=True,
    fontsize=9,
)

cm = confusion_matrix(y_test, y_pred)

print(cm)

plt.tight_layout()
plt.savefig('wine_decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()