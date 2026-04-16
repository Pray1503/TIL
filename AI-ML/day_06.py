# Day 06 - Machine Learning (Random Forest)

# What I Learned:
# Random Forest is an ensemble learning algorithm that combines multiple
# Decision Trees to improve performance and reduce overfitting.

# Key Concept:
# Instead of relying on a single Decision Tree, Random Forest builds multiple
# trees using different subsets of data and features, and then combines their
# predictions (majority voting).

# How It Works:
# Each tree is trained on a random subset of the data (bagging), and
# predictions from all trees are aggregated to produce the final output.

# Important Points:
# - Random Forest reduces overfitting compared to a single Decision Tree
# - It improves accuracy and stability
# - n_estimators defines the number of trees
# - Works well on both small and large datasets

# Comparison:
# Decision Trees can overfit easily, while Random Forest generalizes better
# by averaging multiple trees.

# Evaluation:
# Accuracy is used to compare the performance of Decision Tree and Random Forest models.

# Conclusion:
# Random Forest is a powerful and widely used algorithm that provides better
# generalization and performance compared to individual Decision Trees.

# Day 06 - Machine Learning (Random Forest)

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris["data"], columns=iris["feature_names"])
df["target"] = iris["target"]

# Features and labels
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Decision Tree (for comparison)
# -----------------------------
dt_model = DecisionTreeClassifier(max_depth=3)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

# -----------------------------
# Random Forest
# -----------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Decision Tree Accuracy:", dt_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
