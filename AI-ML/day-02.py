
#Day 02 - Online Machine Learning (Regression)

#What I Learned:
#SGDRegressor is used for predicting continuous values using
#Stochastic Gradient Descent. It supports online learning using partial_fit().

#Key Difference:
#Unlike classification, regression predicts numeric outputs.

#Code Example:

# day02_online_regression.py

from sklearn.linear_model import SGDRegressor
import numpy as np

# Initialize model
model = SGDRegressor(max_iter=100, learning_rate='constant', eta0=0.01, random_state=42)

# Simulated streaming data (y = 2x + noise)
X_batches = [
    np.array([[1], [2]]),
    np.array([[3], [4]]),
    np.array([[5], [6]])
]

y_batches = [
    np.array([2.1, 3.9]),
    np.array([6.2, 7.8]),
    np.array([10.1, 11.9])
]

# Train incrementally
for X, y in zip(X_batches, y_batches):
    model.partial_fit(X, y)

# Predict
prediction = model.predict([[7]])
print("Prediction for x=7:", prediction)
