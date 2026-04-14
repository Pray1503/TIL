"""This script compares supervised and unsupervised learning."""

# Day 04 - Machine Learning (Supervised + Unsupervised)

# What I Learned:
# Machine Learning can be broadly divided into supervised and unsupervised learning.
# Supervised learning uses labeled data to train models, while unsupervised learning
# works on unlabeled data to find patterns or groupings.

# Key Concept:
# In supervised learning, the model learns from input-output pairs and predicts
# the correct output. In unsupervised learning, the model identifies hidden
# structures in data without predefined labels.

# Supervised Learning:
# I used Logistic Regression for classification. The dataset was split into
# training and testing sets, and the model was evaluated using accuracy.

# Unsupervised Learning:
# I used KMeans clustering to group data into clusters based on similarity.
# This does not require labeled data.

# Important Points:
# - Supervised learning requires labeled data
# - Unsupervised learning works without labels
# - Train-test split is important for evaluation
# - Accuracy is used to measure classification performance
# - Clustering helps in discovering patterns in data

# Conclusion:
# Understanding both supervised and unsupervised learning is essential
# for building machine learning models and solving real-world problems.

# Day 04 - Machine Learning (Supervised + Unsupervised)

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris(as_frame=False)
df = pd.DataFrame(data=iris["data"], columns=iris["feature_names"])
df["target"] = iris["target"]

# Features and labels
X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------
# SUPERVISED LEARNING (Classification)
# -----------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)  # This scales your data to have a mean of 0
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Supervised Learning Accuracy:", accuracy)

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris["target_names"]))

# -----------------------------
# UNSUPERVISED LEARNING (Clustering)
# -----------------------------

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

print("Unsupervised Learning (KMeans) Clusters:", clusters[:10])
