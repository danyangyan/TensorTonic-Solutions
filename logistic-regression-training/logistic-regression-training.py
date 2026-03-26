import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.array(X)
    y = np.array(y)
    n_samples,n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    for step in range(steps):
        pred = _sigmoid(X@w+b)
        grad_w = 1/n_features*(X.T@(pred-y))
        grad_b = (pred-y).mean()

        w = w-lr*grad_w
        b = b-lr*grad_b

    return w,b
    