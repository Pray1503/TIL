# Day 10 - Machine Learning (Data Preprocessing and Cleaning)

# What I Learned:
# Data preprocessing is a crucial step in machine learning where raw data
# is cleaned and transformed before training a model.

# Key Concept:
# Real-world data often contains missing values, noise, or inconsistent formats.
# Preprocessing helps in improving data quality and model performance.

# Data Cleaning:
# Missing values can be handled by replacing them with mean, median, or mode values.

# Feature Scaling:
# Scaling ensures that all features are on a similar range, which helps models
# perform better and converge faster.

# Important Points:
# - Missing data must be handled before training
# - Feature scaling improves model performance
# - Clean data leads to better predictions
# - Preprocessing is essential for real-world datasets

# Conclusion:
# Proper data preprocessing improves the accuracy and reliability of machine
# learning models and is a fundamental step in any ML pipeline.

# Day 10 - Data Preprocessing and Cleaning

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create sample dataset with missing values
data = {
    "feature1": [1, 2, np.nan, 4, 5],
    "feature2": [5, np.nan, 2, 4, 3],
    "target": [0, 1, 0, 1, 0],
}

df = pd.DataFrame(data)

print("Original Data:\n", df)

# -----------------------------
# Handling Missing Values
# -----------------------------
df.fillna(df.mean(), inplace=True)

print("\nAfter Handling Missing Values:\n", df)

# -----------------------------
# Features and Labels
# -----------------------------
X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
