# Day 15 - Machine Learning (Model Saving and Loading)

# What I Learned:
# After training a machine learning model, it can be saved to a file
# and reused later without retraining.

# Key Concept:
# Model persistence allows storing trained models and loading them
# whenever needed using tools like joblib or pickle.

# Steps:
# - Train the model
# - Save the model to a file
# - Load the model later
# - Use it for prediction

# Important Points:
# - Saves time by avoiding retraining
# - Useful in deployment and production systems
# - joblib is efficient for large models
# - Loaded model behaves exactly like the original model

# Conclusion:
# Model saving and loading is essential for deploying machine learning
# models and using them in real-world applications.

# Day 15 - Model Saving and Loading

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

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

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

# Load model
loaded_model = joblib.load("model.pkl")

# Predict using loaded model
y_pred = loaded_model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy using loaded model:", accuracy)
