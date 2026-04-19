# Day 09 - Machine Learning (Real CSV Dataset)

# What I Learned:
# Machine learning models can be trained using real-world datasets stored in CSV format.
# CSV files are commonly used for storing structured data.

# Key Concept:
# Instead of relying on built-in datasets, real-world data is loaded from CSV files
# using pandas and then processed for training and evaluation.

# Data Handling:
# The dataset is read using pandas, and features and labels are separated
# before training the model.

# Important Points:
# - CSV is a common format for real-world datasets
# - pandas is used to load and manipulate data
# - Data must be cleaned and structured before training
# - Train-test split is essential for evaluation

# Evaluation:
# Model performance is measured using accuracy after making predictions on test data.

# Conclusion:
# Working with CSV datasets is an essential step toward real-world machine learning
# applications and project development.

# Day 09 - Machine Learning with Real CSV Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset from CSV
# (For now we use sklearn dataset and save it as CSV first)
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris["data"], columns=iris["feature_names"])
df["target"] = iris["target"]

# Save to CSV (simulating real dataset usage)
df.to_csv("iris_dataset.csv", index=False)

# Read CSV
data = pd.read_csv("iris_dataset.csv")

# Features and labels
X = data.drop("target", axis=1)
y = data["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy on CSV dataset:", accuracy)
