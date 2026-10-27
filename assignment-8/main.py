from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pandas as pd

# Load dataset
data=load_breast_cancer()
X=data.data
y=data.target

# Split into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

# Compare different kernels
kernels=["linear","poly","rbf"]
results=[]

for kernel in kernels:
    model=make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel, degree=3, C=1.0, gamma="scale")
    )

    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)

    results.append({
        "Kernel":kernel.capitalize(),
        "Accuracy":accuracy_score(y_test,y_pred),
        "Precision":precision_score(y_test,y_pred),
        "Recall":recall_score(y_test,y_pred),
        "F1-score":f1_score(y_test,y_pred)
    })

# Display comparison
results_df=pd.DataFrame(results)
print(results_df.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
