import numpy as np

def mse(y,y_hat):
    return np.mean((y_hat-y)**2)

w_true = np.array([[1.0],[2.0],[3.0]])

x=np.random.default_rng(0).normal(size=(20, 3))

b_true = 1.2

def pred(x,w,b):
    return x@w+b

y_true = x@w_true+b_true

w = np.zeros_like(w_true)
b = 5
lr =2e-3
steps =3000

def autograd(y,y_hat):
    dl_dy = 2*(y_hat-y)/len(y) #(20,1)
    dy_dw = x  #(20,3)
    dl_dw = dy_dw.T@dl_dy
    dl_db = np.sum(dl_dy)
    return dl_dw,dl_db

def fit(x,y_true,w,b):
    for step in range(steps):
        y_hat = x@w+b
        dl_dw,dl_db = autograd(y_true,y_hat)
        w = w -  dl_dw*lr
        b = b - dl_db*lr
        print(mse(y_true,y_hat))
        print("pred:",pred(x,w,b).T,"true",y_true.T)
        print("w = ",w)

fit(x,y_true,w,b)
print(x.shape)

x_new_feat = np.ones((20,1))
print(x_new_feat.shape)

x_new=np.hstack((x,x_new_feat))

print(x.shape)



w_aug, residuals, rank, s = np.linalg.lstsq(x_new, y_true, rcond=None)

print(w_aug.ravel())
print(mse(y_true, x_new @ w_aug))

print(w_aug, residuals, rank, s)