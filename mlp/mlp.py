from __future__ import annotations
from typing import Optional

import numpy as np

from .activations import sigmoid, d_sigmoid
from .losses import mse, d_mse

class MLP():

    W1: np.ndarray; b1: np.ndarray
    W2: np.ndarray; b2: np.ndarray
    
    Z1: np.ndarray | None; A1: np.ndarray | None
    Z2: np.ndarray | None
    
    def __init__(self,
                 n_inputs: int,
                 n_hidden: int,
                 n_outputs: int,
                 activation: str = "sigmoid",
                 loss: str = "mse",
                 seed: Optional[int] = None):

        if seed is not None:
            np.random.seed(seed)

        # Prosta inicjalizacja wag
        self.W1 = np.random.randn(n_inputs, n_hidden) * 0.01
        self.b1 = np.zeros((1,n_hidden))
        self.W2 = np.random.randn(n_hidden, n_outputs) * 0.01
        self.b2 = np.zeros((1,n_outputs))

        # Wyniki pośrednie
        self.Z1 = None
        self.A1 = None
        self.Z2 = None

        # Prosta obsługa wyboru funkcji aktywacji i straty
        if activation == "sigmoid":
            self.activation = sigmoid
            self.d_activation = d_sigmoid
        else:
            raise ValueError(f"Nieznana funkcja aktywacji: {activation}")

        if loss == "mse":
            self.loss_fn = mse
            self.d_loss_fn = d_mse
        else:
            raise ValueError(f"Nieznana funkcja straty: {loss}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.activation(self.Z1)
        
        self.Z2 = self.A1 @ self.W2 + self.b2

        return self.Z2

    def backward(self, X: np.ndarray, Y: np.ndarray):
        delta2 = self.d_loss_fn(Y, self.Z2)
        dW2 = self.A1.T @ delta2
        db2 = np.sum(delta2, axis=0, keepdims=True)

        delta1 = (delta2 @ self.W2.T) * self.d_activation(self.Z1)
        dW1 = X.T @ delta1
        db1 = np.sum(delta1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def step(self, lr: float, grads):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def compute_loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        Yhat = self.forward(X)
        return float(self.loss_fn(Y, Yhat))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def fit(self, X: np.ndarray, Y: np.ndarray, lr=0.01, epochs=1000, verbose=False):
        for ep in range(epochs):
            Yhat = self.forward(X)
            grads = self.backward(X, Y)
            self.step(lr, grads)
            loss = self.loss_fn(Y, Yhat)
            if verbose and ep % max(1, epochs // 10) == 0:
                print(f"epoch={ep:4d}  loss={loss:.6f}")


# ------------------------- TEST -------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=8, suppress=True)

    n_in, h, n_out = 2, 2, 1
    net = MLP(n_in, h, n_out, seed=0)

    # testowe parametry
    net.W1 = np.array([[0.10, -0.20],
                       [0.05,  0.03]], dtype=float)
    net.b1 = np.array([[0.01, -0.02]], dtype=float)
    net.W2 = np.array([[0.20],
                       [-0.10]], dtype=float)
    net.b2 = np.array([[0.05]], dtype=float)

    X = np.array([[ 1.0, 0.0],
                  [ 0.5, 1.0],
                  [-1.0, 2.0]], dtype=float)
    Y = np.array([[0.2],
                  [0.0],
                  [0.1]], dtype=float)

    print("Loss before:", net.compute_loss(X, Y))
    grads = net.backward(X, Y)
    net.step(0.1, grads)
    print("Loss  after:", net.compute_loss(X, Y))