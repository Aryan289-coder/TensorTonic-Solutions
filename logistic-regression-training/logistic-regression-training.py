import numpy as np

def sigmoid(z):
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    X = np.array(X)
    y = np.array(y)
    n, d = X.shape

    w = np.zeros(d)
    b = 0.0

    for _ in range(steps):
        z = X.dot(w) + b
        y_hat = sigmoid(z)
        error = y_hat - y
        gw = X.T.dot(error) / n
        gb = error.mean()
        w -= lr * gw
        b -= lr * gb
    return w, b

def predict(X, w, b, threshold=0.5):
    X = np.array(X)
    probs = sigmoid(X.dot(w) + b)
    return (probs >= threshold).astype(int)

def accuracy(y_true, y_pred):
    return (np.array(y_true) == np.array(y_pred)).mean() * 100