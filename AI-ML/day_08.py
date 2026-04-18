# Day 08 - Machine Learning (Confusion Matrix and Evaluation Metrics)

# What I Learned:
# Evaluation metrics help in understanding how well a machine learning model performs.
# A confusion matrix provides a detailed breakdown of correct and incorrect predictions.

# Key Concept:
# A confusion matrix compares actual values with predicted values and shows how many
# predictions were correct or incorrect for each class.

# Metrics:
# Precision measures how many predicted positives are actually correct.
# Recall measures how many actual positives are correctly identified.
# F1-score is the balance between precision and recall.

# Important Points:
# - Accuracy alone is not enough to evaluate a model
# - Confusion matrix gives detailed insights
# - Precision and recall are important for imbalanced datasets
# - Classification report summarizes all metrics

# Conclusion:
# Using evaluation metrics like confusion matrix, precision, and recall provides
# a deeper understanding of model performance beyond just accuracy.

# Day 08 - Confusion Matrix and Evaluation Metrics

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

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

# Predictions
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", report)
