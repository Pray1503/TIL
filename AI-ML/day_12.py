# Day 12 - Machine Learning (Hyperparameter Tuning)

# What I Learned:
# Hyperparameter tuning is the process of selecting the best parameters
# for a machine learning model to improve its performance.

# Key Concept:
# Models have parameters that control their behavior. Instead of manually
# choosing them, GridSearchCV tests multiple combinations and selects the best one.

# How It Works:
# GridSearchCV performs cross-validation on different parameter combinations
# and selects the one with the highest performance.

# Important Points:
# - Hyperparameters are set before training
# - GridSearchCV automates parameter selection
# - Cross-validation ensures reliable results
# - Improves model accuracy and generalization

# Conclusion:
# Hyperparameter tuning helps in optimizing model performance and is an
# essential step in building high-quality machine learning models.

# Day 12 - Hyperparameter Tuning using GridSearchCV

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

# Model
model = RandomForestClassifier(random_state=42)

# Hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [2, 3, 5, None],
    "min_samples_split": [2, 5, 10],
}

# Grid Search
grid_search = GridSearchCV(
    estimator=model, param_grid=param_grid, cv=5, scoring="accuracy"
)

# Train
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Best model
best_model = grid_search.best_estimator_

# Predict
y_pred = best_model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy after tuning:", accuracy)
