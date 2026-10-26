# Linear regression

One sample has \(d\) features and one target. The prediction is a weighted sum of those features, plus a bias:

$$
\hat{y}_i = x_{i1}w_1 + x_{i2}w_2 + \cdots + x_{id}w_d + b
$$

For \(n\) samples this is the matrix product

$$
\hat{y} = Xw + b
$$

| Array | Shape | Meaning |
|---|---|---|
| \(X\) | \((n, d)\) | \(n\) samples, \(d\) features each |
| \(w\) | \((d, 1)\) | one weight per feature |
| \(b\) | scalar | one bias for the single output |
| \(y, \hat{y}\) | \((n, 1)\) | one target and one prediction per sample |

`@` is the matrix product. The inner dimensions must match, so \((n, d)\) times \((d, 1)\) is \((n, 1)\). `*` multiplies entries in corresponding positions and does not sum the features of a sample.

A square \(w\) with shape \((d, d)\) is a legal matrix product with \(X\), but the result has \(d\) outputs per sample. This model has one output, so \(w\) and \(y\) are columns.

## Loss

Mean squared error:

$$
L = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2
$$

The factor \(1/n\) keeps the loss from growing just because there are more samples.

## Gradient

\(w\) affects \(L\) only through \(\hat{y}\). For one sample and one weight, the chain rule is

\[
\frac{\partial L}{\partial w}
= \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w}
\]

Differentiating \((\hat{y}-y)^2\) with respect to \(\hat{y}\) gives \(2(\hat{y}-y)\). Differentiating \(xw+b\) with respect to \(w\) gives \(x\).

With \(n\) samples the loss is an average, so each sample contributes

$$
g_i = \frac{\partial L}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)
$$

\(g\) has shape \((n, 1)\). Sample \(i\) changes \(w\) by \(g_i x_i\), because increasing \(w\) by a tiny amount changes that prediction by \(x_i\) times as much. The full gradient is the sum of those contributions:

$$
\frac{\partial L}{\partial w} = \sum_{i=1}^{n} g_i x_i
$$

With \(d\) features, weight \(w_j\) appears in sample \(i\) only as \(x_{ij}w_j\). Every other feature is constant, so

$$
\frac{\partial \hat{y}_i}{\partial w_j} = x_{ij}
$$

and

$$
\frac{\partial L}{\partial w_j} = \sum_{i=1}^{n} g_i x_{ij}
$$

The sum is feature column \(j\) of \(X\), multiplied entry by entry with \(g\) and then added up. Row \(j\) of \(X^\top\) is that same column, so all \(d\) weights fit in one product:

$$
\frac{\partial L}{\partial w} = X^\top g
$$

\(X^\top\) is \((d, n)\) and \(g\) is \((n, 1)\), so the gradient is \((d, 1)\), the same shape as \(w\).

The bias has \(\partial \hat{y}_i / \partial b = 1\) for every sample, so there is no feature to multiply:

\[
\frac{\partial L}{\partial b} = \sum_{i=1}^{n} g_i
\]

In NumPy, with `g = 2 * (y_hat - y) / n`:

```python
dw = X.T @ g
db = np.sum(g)
```

Gradient descent then updates the parameters by stepping against the gradient:

\[
w \leftarrow w - \eta \frac{\partial L}{\partial w}, \qquad
b \leftarrow b - \eta \frac{\partial L}{\partial b}
\]

## Finite-difference check

For a parameter \(\theta\), compare the analytic gradient with a small step \(\epsilon\):

$$
\frac{L(\theta+\epsilon) - L(\theta-\epsilon)}{2\epsilon}
$$

Check every entry of \(w\), and check \(b\). The relative error should be about \(10^{-5}\) or smaller.
n