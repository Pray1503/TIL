# Day 01 - Online vs Batch Machine Learning

## What I Learned
- Batch learning uses entire dataset at once
- Online learning updates model continuously with new data

## Key Difference
- Batch → static, slower updates
- Online → dynamic, real-time updates

## Example
- Batch: Training a model once on dataset
- Online: Updating model with streaming data

## Summary
Learned difference between online and batch learning

## Example Code 
```python
# online_ml.py

from sklearn.linear_model import SGDClassifier
import numpy as np

# Initialize model (logistic regression using SGD)
model = SGDClassifier(loss="log_loss")

# Example: data comes in chunks (streaming style)
X_batches = [
    np.array([[1, 2], [2, 3]]),
    np.array([[3, 4], [4, 5]]),
    np.array([[5, 6], [6, 7]])
]

y_batches = [
    np.array([0, 0]),
    np.array([1, 1]),
    np.array([1, 1])
]

# First call needs classes defined
model.partial_fit(X_batches[0], y_batches[0], classes=np.array([0, 1]))

# Continue learning incrementally
for X, y in zip(X_batches[1:], y_batches[1:]):
    model.partial_fit(X, y)

# Test prediction
prediction = model.predict([[2, 2]])
print("Prediction:", prediction)

```
