# Day 16 - Machine Learning (Mini Project - Iris Prediction System)

# What I Learned:
# A complete machine learning system includes training, saving, loading,
# and making predictions based on user input.

# Key Concept:
# Mini projects combine multiple ML steps into a single working application.

# Project Workflow:
# - Load dataset
# - Train model
# - Save model
# - Load model
# - Take user input
# - Predict output

# Important Points:
# - Projects demonstrate practical ML skills
# - Combines preprocessing, modeling, and prediction
# - Useful for portfolio and resume
# - Shows end-to-end understanding

# Conclusion:
# Mini projects help in applying machine learning concepts in a real-world
# scenario and are essential for becoming job-ready.

# Day 16 - Mini Project (Iris Prediction System)

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# -----------------------------
# Train & Save Model
# -----------------------------
iris = load_iris()
X = iris["data"]
y = iris["target"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "iris_model.pkl")

# -----------------------------
# Load Model
# -----------------------------
loaded_model = joblib.load("iris_model.pkl")

# -----------------------------
# User Input
# -----------------------------
print("\nEnter flower measurements:")

sepal_length = float(input("Sepal length: "))
sepal_width = float(input("Sepal width: "))
petal_length = float(input("Petal length: "))
petal_width = float(input("Petal width: "))

# Create input DataFrame
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]])

# Predict
prediction = loaded_model.predict(input_data)

# Map output
target_names = iris["target_names"]
result = target_names[prediction[0]]

print("\nPredicted Flower Type:", result)
