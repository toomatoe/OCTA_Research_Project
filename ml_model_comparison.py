import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Update this path to point to your actual data file
df = pd.read_csv("data/INL.csv")  # Place your CSV file in a 'data' folder

# Split data
X = df.drop('0', axis=1)
y = df['0']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define classifiers
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier()
}

cv_results = {}

# Train and evaluate each model
for name, model in models.items():
    # 5-fold cross-validation on training data
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_score = cv_scores.mean()
    cv_results[name] = mean_score
    print(f"{name} 5-fold CV Accuracy: {mean_score:.4f} (+/- {cv_scores.std():.4f})")
    # Fit and evaluate on test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(f"Confusion Matrix for {name}:")
    print(cm)
    print(f"Test Accuracy for {name}: {acc:.4f}\n")

# Find the best model by validation accuracy
best_model = max(cv_results, key=cv_results.get)
print(f"Best model by validation accuracy: {best_model} ({cv_results[best_model]:.4f})")