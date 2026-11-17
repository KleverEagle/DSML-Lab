import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# 1. Load dataset (569 samples: 212 Malignant [0], 357 Benign [1])
data = load_breast_cancer()
X, y = data.data, data.target

# Overall class distribution
overall_ratio = np.bincount(y) / len(y)
print(f"Dataset Class Distribution -> Class 0: {overall_ratio[0]:.2%}, Class 1: {overall_ratio[1]:.2%}\n")

# Initialize model (max_iter increased for convergence)
model = LogisticRegression(max_iter=5000, random_state=42)

# 2. Standard K-Fold Cross-Validation (with shuffle to avoid ordered data bias)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print("--- Standard K-Fold (k=5) ---")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    val_dist = np.bincount(y[val_idx]) / len(val_idx)
    print(f"Fold {fold} Val Class Ratio -> Class 0: {val_dist[0]:.2%}, Class 1: {val_dist[1]:.2%} | Accuracy: {kf_scores[fold-1]:.4f}")
print(f"Mean Accuracy: {kf_scores.mean():.4f} (± {kf_scores.std():.4f})\n")

# 3. Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

print("--- Stratified K-Fold (k=5) ---")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    val_dist = np.bincount(y[val_idx]) / len(val_idx)
    print(f"Fold {fold} Val Class Ratio -> Class 0: {val_dist[0]:.2%}, Class 1: {val_dist[1]:.2%} | Accuracy: {skf_scores[fold-1]:.4f}")
print(f"Mean Accuracy: {skf_scores.mean():.4f} (± {skf_scores.std():.4f})")