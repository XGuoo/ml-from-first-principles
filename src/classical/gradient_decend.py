import numpy as np

x = np.random.random(size=(10, 10))
w = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
b = 1
# noise = np.random.normal(size=(10, 1), mean=0, std=0.1)
noise = np.random.normal(loc=0, scale=0.1, size=(10, 1))
y = w @ x + b + noise

def predict(x, w, b):
    return x @ w + b

def mse(y, y_hat):
    return np.mean(( y_hat-y) ** 2)

def gradient(x, y, y_hat):
    dl_dy = 2 * (y_hat - y) / len(y)
    dl_dw = x.T @ dl_dy
    db = np.sum(dl_dy)
    return dl_dw, db

lr = 0.01
steps =100

for i in range(steps):
    y_hat = predict(x, w, b)
    loss = mse(y, y_hat)
    print(f"Step {i+1}, Loss: {loss}")
    dl_dw, db = gradient(x, y, y_hat)
    w = w - lr * dl_dw
    b = b - lr * db

print(f"Final Weight: {w}")
print(f"Final Bias: {b}")