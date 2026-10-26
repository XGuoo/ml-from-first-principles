"""Linear regression: forward, MSE, gradients, batch gradient descent, and the normal equation."""
import numpy as np

x = np.random.random(size=(10, 10))
w = np.random.random(size=(10, 1))
b = np.random.random(size=(10, 1))
noise = np.random.random(size=(10, 1))

def predict(x, w, b):
    return x * w + b

y = predict(x, w, b) + noise
y_hat = predict(x, w, b)

def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def gradients(x, y, y_hat):
   
    dl_dy = 2 * (y_hat-y) / len(y)
    dl_dw = x.T @ dl_dy
    db = np.sum(dl_dy)
    return db

print(gradients(x, y, y_hat))
