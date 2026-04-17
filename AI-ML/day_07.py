# Day 07 - Machine Learning (Feature Importance and Visualization)

# What I Learned:
# Feature importance helps in understanding which features contribute the most
# to the model's predictions. It provides insight into how the model makes decisions.

# Key Concept:
# Random Forest calculates feature importance based on how much each feature
# reduces impurity across all trees. Higher importance means the feature has
# more influence on predictions.

# Visualization:
# Data visualization helps in interpreting model results. Bar plots can be used
# to compare feature importance values effectively.

# Important Points:
# - Feature importance improves model interpretability
# - Random Forest provides built-in importance scores
# - Visualization makes analysis easier
# - Helps in feature selection

# Conclusion:
# Understanding feature importance allows better insights into the model and
# helps in improving performance by focusing on the most relevant features.

# Day 07 - Feature Importance and Visualization

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris["data"], columns=iris["feature_names"])
df["target"] = iris["target"]

# Features and labels
X = df.drop("target", axis=1)
y = df["target"]

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Feature Importance
importances = model.feature_importances_
features = X.columns

# Create DataFrame for visualization
importance_df = pd.DataFrame(
    {"Feature": features, "Importance": importances}
).sort_values(by="Importance", ascending=False)

print(importance_df)

# Plot
plt.figure()
plt.bar(importance_df["Feature"], importance_df["Importance"])
plt.xticks(rotation=45)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance (Random Forest)")
plt.show()
