# Day 13 - Machine Learning (Model Comparison and Selection)

# What I Learned:
# Different machine learning models can produce different results on the same dataset.
# Comparing multiple models helps in selecting the best one.

# Key Concept:
# Model comparison involves training multiple models and evaluating their performance
# using metrics like accuracy.

# Models Compared:
# - Logistic Regression
# - Decision Tree
# - Random Forest

# Important Points:
# - No single model is best for all problems
# - Model performance depends on data
# - Evaluation metrics help in selecting the best model
# - Random Forest often performs better due to ensemble learning

# Conclusion:
# Model comparison is essential to identify the most suitable algorithm
# for a given dataset and problem.

# Day 13 - Model Comparison and Selection

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris["data"], columns=iris["feature_names"])
df["target"] = iris["target"]

# Features and labels
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(max_depth=3),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# Train and evaluate
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy}")

# Best model
best_model = max(results, key=results.get)
print("\nBest Model:", best_model)
