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
