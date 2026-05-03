# Day 14 - Machine Learning (ML Pipeline)

# What I Learned:
# An ML pipeline automates the entire machine learning workflow including
# data preprocessing and model training.

# Key Concept:
# A pipeline ensures that all steps such as scaling and model training
# are applied consistently and in the correct order.

# Pipeline Steps:
# - Data preprocessing (scaling)
# - Model training
# - Prediction

# Important Points:
# - Pipelines reduce code complexity
# - Prevent data leakage
# - Ensure reproducibility
# - Useful for production systems

# Conclusion:
# Using pipelines helps in building clean, efficient, and reliable
# machine learning workflows.

# Day 14 - End-to-End ML Pipeline

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

# Create Pipeline
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
)

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("Pipeline Model Accuracy:", accuracy)
