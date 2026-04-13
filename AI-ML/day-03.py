# Day 03 - Online Machine Learning (Real Dataset)

# What I Learned:
# SGDClassifier can be used with real-world datasets by training the model 
# incrementally using partial_fit(). Instead of using the entire dataset at once, 
# data is divided into smaller batches to simulate streaming data.

# Key Concept:
# Online learning allows models to update continuously as new data arrives, 
# making it suitable for real-time and large-scale applications.

# Dataset Used:
# I used the Iris dataset, which is a standard dataset for classification problems. 
# It contains multiple features and target classes.

# Important Points:
# - partial_fit() enables incremental learning
# - First call requires passing all possible classes
# - Data should be shuffled to avoid bias
# - Training is done in chunks (batch-wise learning)

# Evaluation:
# Model performance can be measured using accuracy, which tells how well 
# the model predicts the correct class labels.

# Conclusion:
# Online learning can be applied to real datasets effectively by simulating 
# data streams using batches, making it practical for real-world ML applications.


                                     
import numpy as np
from sklearn.linear_model import SGDRegressor

# Initialize the model
model = SGDRegressor()

# Simulated data: 2 features per sample [feature1, feature2]
# Example: [Area, Rooms]
X_batches = [
    np.array([[1200, 3], [1500, 3]]),
    np.array([[1800, 4], [2100, 4]])
]

# Targets (e.g., Prices)
y_batches = [
    np.array([250000, 300000]),
    np.array([350000, 400000])
]

# Incremental learning loop
for X, y in zip(X_batches, y_batches):
    model.partial_fit(X, y)

# Predict for a new house: 2000 sq ft, 4 rooms
prediction = model.predict([[2000, 4]])
print(f"Predicted Price: {prediction[0]}")
