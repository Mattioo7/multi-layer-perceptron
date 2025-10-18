import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_pred - y_true) ** 2)

def d_mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.array:
    return 2 * (y_pred - y_true) / y_true.size

if __name__ == "__main__":
    import numpy as np

    # Test A: 1D
    y_true = np.array([1.0, 0.0, 1.0])
    y_pred = np.array([0.8, 0.2, 0.9])

    loss = mse(y_true, y_pred)
    grad = d_mse(y_true, y_pred)

    print("=== Test A (1D) ===")
    print("MSE:", loss, "Expected: 0.03")
    print("Gradient:", grad)
    print("Expected grad:", [-0.13333333, 0.13333333, -0.06666667])

    # Test B: 2D
    y_true = np.array([[1.0, 2.0],
                       [3.0, 4.0]])
    y_pred = np.array([[1.1, 1.9],
                       [2.8, 4.2]])

    loss = mse(y_true, y_pred)
    grad = d_mse(y_true, y_pred)

    print("\n=== Test B (2D) ===")
    print("MSE:", loss, "Expected: 0.025")
    print("Gradient:\n", grad)
    print("Expected grad:\n", [[0.05, -0.05], [-0.10, 0.10]])
