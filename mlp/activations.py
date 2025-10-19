from typing import Union
import numpy as np

ArrayLike = Union[float, np.ndarray]

def sigmoid(x: ArrayLike) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x: ArrayLike) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)

def identity(x: ArrayLike) -> np.ndarray:
    return np.asarray(x)

def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x_shift = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(x_shift)
    return exps / np.sum(exps, axis=axis, keepdims=True)

if __name__ == "__main__":
    x = np.array([-2.0, 0.0, 2.0])
    y = sigmoid(x)
    dy = d_sigmoid(x)

    print("Input:", x)
    print("Sigmoid:", y)
    print("Expected sigmoid:", [0.11920292, 0.5, 0.88079708])
    print("Difference:", y - np.array([0.11920292, 0.5, 0.88079708]))

    print("\nDerivative:", dy)
    print("Expected dSigmoid:", [0.10499359, 0.25, 0.10499359])
    print("Difference:", dy - np.array([0.10499359, 0.25, 0.10499359]))

    X = np.array([[1.0, 2.0, 3.0]])
    print("\nSoftmax row sums (should be 1):", softmax(X).sum(axis=1))
