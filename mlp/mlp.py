from __future__ import annotations
from collections.abc import Callable
from typing import Literal

import numpy as np

from .activations import sigmoid, d_sigmoid, identity, softmax
from .losses import (
    mse, d_mse,
    binary_cross_entropy, d_binary_cross_entropy,
    cross_entropy, d_cross_entropy
)

TaskType = Literal["regression", "binary", "multiclass"]

class MLP:
    layer_sizes: list[int]
    n_layers: int
    learning_rate: float
    task: TaskType

    # hidden layers
    activation: Callable[[np.ndarray], np.ndarray]
    d_activation: Callable[[np.ndarray], np.ndarray]

    # output layer
    out_activation: Callable[[np.ndarray], np.ndarray]
    d_out_activation: Callable[[np.ndarray], np.ndarray]

    # loss (na wyjściu po aktywacji)
    loss_fn: Callable[[np.ndarray, np.ndarray], float]
    d_loss_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]

    W: list[np.ndarray]
    b: list[np.ndarray]

    Z: list[np.ndarray] | None
    A: list[np.ndarray] | None

    def __init__(
        self,
        layer_sizes: list[int],
        task: TaskType,
        activation: str,
        learning_rate: float = 1e-2,
        seed: int | None = None,
    ):
        assert len(layer_sizes) >= 2, "Podaj co najmniej [n_in, n_out]"
        assert task in ("regression", "binary", "multiclass"), \
            f"Nieprawidłowy task: {task}. Dozwolone: 'regression', 'binary', 'multiclass'"
        assert activation in ("sigmoid",), \
            f"Nieprawidłowa funkcja aktywacji: {activation}. Dozwolone: 'sigmoid'"

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.learning_rate = float(learning_rate)
        self.task = task

        if seed is not None:
            np.random.seed(seed)

        # --- Inicjalizacja list i buforów ---
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        self.Z: list[np.ndarray] | None = None
        self.A: list[np.ndarray] | None = None

        # --- Wybór funkcji aktywacji w warstwach ukrytych ---
        if activation == "sigmoid":
            self.activation = sigmoid
            self.d_activation = d_sigmoid
        else:
            raise ValueError(f"Nieznana funkcja aktywacji: {activation}")

        # --- Wybór funkcji aktywacji warstwy wyjściowej oraz funkcja straty ---
        if task == "regression":
            self.out_activation = identity
            self.d_out_activation = identity
            self.loss_fn = mse
            self.d_loss_fn = d_mse
        elif task == "binary":
            self.out_activation = sigmoid
            self.d_out_activation = identity
            self.loss_fn = binary_cross_entropy
            self.d_loss_fn = d_binary_cross_entropy
        elif task == "multiclass":
            self.out_activation = softmax
            self.d_out_activation = identity
            self.loss_fn = cross_entropy
            self.d_loss_fn = d_cross_entropy
        else:
            raise ValueError(f"Nieznane zadanie: {task}")

        # --- Inicjalizacja wag ---
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

        Z: list[np.ndarray] = [np.empty((0, 0))]
        A: list[np.ndarray] = [X]

        # hidden layers
        for l in range(self.n_layers - 1):
            z = A[l] @ self.W[l] + self.b[l]
            a = self.activation(z)
            Z.append(z)
            A.append(a)

        # output layer
        zL = A[self.n_layers - 1] @ self.W[self.n_layers - 1] + self.b[self.n_layers - 1]
        aL = self.out_activation(zL)

        Z.append(zL)
        A.append(aL)

        self.Z, self.A = Z, A
        return aL  # zawsze zwracamy po aktywacji wyjścia

    def backward(self, X: np.ndarray, Y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        assert self.Z is not None and self.A is not None, "Najpierw wywołaj forward(X)"
        Z, A, L = self.Z, self.A, self.n_layers

        dW = [np.zeros_like(Wl) for Wl in self.W]
        db = [np.zeros_like(bl) for bl in self.b]

        Y_hat = A[L]  # po aktywacji wyjścia
        # Dla (softmax+CE) i (sigmoid+BCE) d_loss już odpowiada dL/dz,
        # bo d_out_activation = identity. Dla regresji z identity też OK.
        delta_next = self.d_loss_fn(Y, Y_hat) * 1.0  # * d_out_activation(Z[L]) == identity

        # gradient dla warstwy wyjściowej
        dW[L - 1] = A[L - 1].T @ delta_next
        db[L - 1] = np.sum(delta_next, axis=0, keepdims=True)

        # ukryte warstwy
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
        Y_hat = self.forward(X)
        return float(self.loss_fn(Y, Y_hat))

    def predict(self, X: np.ndarray, return_proba: bool = False) -> np.ndarray:
        Y_hat = self.forward(X)
        if self.task == "regression":
            return Y_hat
        if self.task == "binary":
            if return_proba:
                return Y_hat
            return (Y_hat >= 0.5).astype(int)
        # multiclass
        if return_proba:
            return Y_hat
        return np.argmax(Y_hat, axis=1).reshape(-1, 1)

    # pomocnicze: konwersja etykiet 1D -> one-hot dla multiclass
    @staticmethod
    def _to_one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        one_hot = np.zeros((y.shape[0], n_classes), dtype=float)
        one_hot[np.arange(y.shape[0]), y.astype(int)] = 1.0
        return one_hot

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        learning_rate: float | None = None,
        epochs: int = 1000,
        verbose: bool = False,
        one_hot_if_needed: bool = True,
    ) -> list[float]:
        # dopasuj kształt Y do zadania
        if self.task == "binary":
            # oczekujemy (m,1) z wartościami 0/1
            Y = Y.reshape(-1, 1)
        elif self.task == "multiclass":
            # oczekujemy one-hot o wymiarze (m, n_classes); jeśli y ma kształt (m,) lub (m,1) z indeksami klas,
            # to twórz one-hot zgodnie z rozmiarem wyjścia.
            if one_hot_if_needed and (Y.ndim == 1 or (Y.ndim == 2 and Y.shape[1] == 1)):
                Y = self._to_one_hot(Y, self.layer_sizes[-1])

        history: list[float] = []
        for ep in range(epochs):
            Y_hat = self.forward(X)
            grads = self.backward(X, Y)
            self.step(learning_rate, grads)
            loss = float(self.loss_fn(Y, Y_hat))
            history.append(loss)
            if verbose and ep % max(1, epochs // 10) == 0:
                cur_lr = float(learning_rate) if learning_rate is not None else self.learning_rate
                print(f"epoch={ep:4d}  loss={loss:.8f}  lr={cur_lr}")
        return history


# ------------------------- PRZYKŁADY -------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    # === Regresja ===
    net_r = MLP(layer_sizes=[2, 16, 1], task="regression", learning_rate=0.01, seed=0)
    rng = np.random.default_rng(1)
    Xr = rng.normal(size=(512, 2))
    Yr = (2 * Xr[:, :1] - 3 * Xr[:, 1:2]) + 0.05 * rng.normal(size=(512, 1))
    print("Reg loss start:", net_r.compute_loss(Xr, Yr))
    hist_r = net_r.fit(Xr, Yr, epochs=2000)
    print("Reg loss end:  ", hist_r[-1])
    print("\n")

    # === Binary ===
    net_b = MLP(layer_sizes=[2, 8, 1], task="binary", learning_rate=0.05, seed=0)
    Xb = rng.normal(size=(400, 2))
    yb = ((Xb[:, 0] * Xb[:, 1]) > 0).astype(int).reshape(-1, 1)  # XOR-like znak iloczynu
    print("Bin loss start:", net_b.compute_loss(Xb, yb))
    hist_b = net_b.fit(Xb, yb, epochs=1000)
    print("Bin loss end:  ", hist_b[-1])
    preds_b = net_b.predict(Xb)
    print("Bin acc ~:", (preds_b.ravel() == yb.ravel()).mean())
    print("\n")

    # === Multiclass (K=3) ===
    net_m = MLP(layer_sizes=[2, 16, 3], task="multiclass", learning_rate=0.05, seed=0)
    Xm = rng.normal(size=(450, 2))
    ym_idx = (Xm[:, 0] > 0).astype(int) + (Xm[:, 1] > 0).astype(int)  # klasy 0/1/2
    print("MC loss start:", net_m.compute_loss(Xm, net_m._to_one_hot(ym_idx, 3)))
    hist_m = net_m.fit(Xm, ym_idx.reshape(-1, 1), epochs=1500)  # one-hot zrobi się automatycznie
    print("MC loss end:  ", hist_m[-1])
    preds_m = net_m.predict(Xm)
    print("MC acc ~:", (preds_m.ravel() == ym_idx).mean())
