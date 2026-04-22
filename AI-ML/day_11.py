# Day 11 - Machine Learning (Feature Engineering)

# What I Learned:
# Feature engineering is the process of creating new features from existing data
# to improve model performance.

# Key Concept:
# Better features help the model understand patterns more effectively. Instead of
# only using raw data, new meaningful features are created.

# Feature Engineering Techniques:
# - Creating ratios (e.g., petal length / petal width)
# - Combining features
# - Transforming existing features

# Important Points:
# - Good features improve model accuracy
# - Domain knowledge helps in creating useful features
# - Not all features are useful, selection is important
# - Feature engineering is a key step in real-world ML

# Conclusion:
# Feature engineering enhances the quality of data and helps models make better
# predictions, making it one of the most important steps in machine learning.

# Day 11 - Feature Engineering with Real Dataset

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris["data"], columns=iris["feature_names"])
df["target"] = iris["target"]

# -----------------------------
# Feature Engineering
# -----------------------------
# Create new feature: petal ratio
df["petal_ratio"] = df["petal length (cm)"] / df["petal width (cm)"]

# Create new feature: sepal ratio
df["sepal_ratio"] = df["sepal length (cm)"] / df["sepal width (cm)"]

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

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy after Feature Engineering:", accuracy)
