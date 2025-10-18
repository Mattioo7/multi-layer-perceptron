from typing import Union
import numpy as np

ArrayLike = Union[float, np.ndarray]

def sigmoid(x: ArrayLike) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x: ArrayLike) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)

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