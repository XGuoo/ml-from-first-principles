"""Linear regression: forward, MSE, gradients, batch gradient descent, and the normal equation."""
import numpy as np

x = np.random.random(size=(10, 10))
w = np.random.random(size=(10, 1))
w_true = w.copy()

b = 0
# b = np.random.random(size=(10, 1))
noise = np.random.random(size=(10, 1))
eps = 1e-6

def predict(x, w, b):
    return x @ w + b

y = predict(x, w_true, b) + noise
w = np.zeros_like(w_true)


def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def gradients(x, y, y_hat):
    dl_dy = 2 * (y_hat-y) / len(y)
    dl_dw = x.T @ dl_dy
    db = np.sum(dl_dy)
    return dl_dw,db

y_hat = predict(x,w,b)
w_plus = w.copy()
w_plus[0, 0] += eps
w_minus = w.copy()
w_minus[0, 0] -= eps
numeric = (mse(y, predict(x, w_plus, b)) - mse(y, predict(x, w_minus, b))) / (2 * eps)
dl_dw, db = gradients(x, y, y_hat)
print(numeric, dl_dw[0, 0])

epoch = 100
lr = 0.01
for i in range (epoch):
    y_hat = predict(x,w,b)
    dl_dw,db = gradients(x,y,y_hat)
    w = w - lr * dl_dw
    b = b - lr * db


print("w,w_true :",(w,w_true))

print("y,y_hat :",(y,y_hat))

