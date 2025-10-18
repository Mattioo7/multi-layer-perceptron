from __future__ import annotations
from collections.abc import Callable

import numpy as np

from .activations import sigmoid, d_sigmoid
from .losses import mse, d_mse

class MLP:
    layer_sizes: list[int]
    n_layers: int
    learning_rate: float

    activation: Callable[[np.ndarray], np.ndarray]
    d_activation: Callable[[np.ndarray], np.ndarray]
    loss_fn: Callable[[np.ndarray, np.ndarray], float]
    d_loss_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]

    W: list[np.ndarray]
    b: list[np.ndarray]

    Z: list[np.ndarray] | None
    A: list[np.ndarray] | None

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "sigmoid",
        loss: str = "mse",
        learning_rate: float = 1e-2,
        seed: int | None = None,
    ):
        # --- Walidacja i podstawowe parametry ---
        assert len(layer_sizes) >= 2, "Podaj co najmniej [n_in, n_out]"
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.learning_rate = float(learning_rate)

        if seed is not None:
            np.random.seed(seed)

        # --- Inicjalizacja list i buforów ---
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        self.Z: list[np.ndarray] | None = None
        self.A: list[np.ndarray] | None = None

        # --- Wybór funkcji aktywacji ---
        if activation == "sigmoid":
            self.activation = sigmoid
            self.d_activation = d_sigmoid
        else:
            raise ValueError(f"Nieznana funkcja aktywacji: {activation}")

        # --- Wybór funkcji straty ---
        if loss == "mse":
            self.loss_fn = mse
            self.d_loss_fn = d_mse
        else:
            raise ValueError(f"Nieznana funkcja straty: {loss}")

        # --- Inicjalizacja wag i biasów ---
        for l in range(self.n_layers):
            n_in, n_out = layer_sizes[l], layer_sizes[l + 1]
            W_l = np.random.randn(n_in, n_out) * 0.01
            b_l = np.zeros((1, n_out), dtype=float)
            self.W.append(W_l)
            self.b.append(b_l)

    def forward(self, X: np.ndarray) -> np.ndarray:
        assert X.ndim == 2 and X.shape[1] == self.layer_sizes[0], (
            f"X ma kształt {X.shape}, oczekiwano (m, {self.layer_sizes[0]})"
        )

        Z: list[np.ndarray] = [np.empty((0, 0))]  # placeholder dla indeksu 0
        A: list[np.ndarray] = [X]

        for l in range(self.n_layers - 1):
            z = A[l] @ self.W[l] + self.b[l]
            a = self.activation(z)
            Z.append(z)
            A.append(a)

        zL = A[self.n_layers - 1] @ self.W[self.n_layers - 1] + self.b[self.n_layers - 1]
        Y_pred = zL  # liniowe wyjście

        Z.append(zL)
        self.Z = Z
        self.A = A
        return Y_pred

    def backward(self, X: np.ndarray, Y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        assert self.Z is not None and self.A is not None, "Najpierw wywołaj forward(X)"
        Z, A, L = self.Z, self.A, self.n_layers

        dW: list[np.ndarray] = [np.zeros_like(Wl) for Wl in self.W]
        db: list[np.ndarray] = [np.zeros_like(bl) for bl in self.b]

        Y_pred = Z[L]
        delta_next = self.d_loss_fn(Y, Y_pred)

        dW[L - 1] = A[L - 1].T @ delta_next
        db[L - 1] = np.sum(delta_next, axis=0, keepdims=True)

        for l in range(L - 2, -1, -1):
            delta_l = (delta_next @ self.W[l + 1].T) * self.d_activation(Z[l + 1])
            dW[l] = A[l].T @ delta_l
            db[l] = np.sum(delta_l, axis=0, keepdims=True)
            delta_next = delta_l

        return dW, db

    def step(self, learning_rate: float | None, grads: tuple[list[np.ndarray], list[np.ndarray]]) -> None:
        eta = float(learning_rate) if learning_rate is not None else self.learning_rate
        dW, db = grads
        for l in range(self.n_layers):
            self.W[l] -= eta * dW[l]
            self.b[l] -= eta * db[l]

    def compute_loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        Y_pred = self.forward(X)
        return float(self.loss_fn(Y, Y_pred))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        learning_rate: float | None = None,
        epochs: int = 1000,
        verbose: bool = False,
    ) -> list[float]:
        history: list[float] = []
        for ep in range(epochs):
            Y_pred = self.forward(X)
            grads = self.backward(X, Y)
            self.step(learning_rate, grads)
            loss = float(self.loss_fn(Y, Y_pred))
            history.append(loss)
            if verbose and ep % max(1, epochs // 10) == 0:
                cur_learning_rate = float(learning_rate) if learning_rate is not None else self.learning_rate
                print(f"epoch={ep:4d}  loss={loss:.8f}  learning_rate={cur_learning_rate}")
        return history


# ------------------------- TEST -------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=8, suppress=True)

    net = MLP(layer_sizes=[2, 16, 1], learning_rate=0.01, seed=0)

    rng = np.random.default_rng(1)
    X = rng.normal(size=(256, 2))
    Y = (2 * X[:, :1] - 3 * X[:, 1:2]) + 0.05 * rng.normal(size=(256, 1))

    print("Loss (start):", net.compute_loss(X, Y))
    hist = net.fit(X, Y, epochs=100000, verbose=True)
    print("Loss (end):  ", hist[-1])
