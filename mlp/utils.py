import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# === DATA HANDLING ===
def load_dataset(path: str):
    """
    Load a CSV dataset. Assumes all columns except the last are features (X)
    and the last column is the target (y).

    Also normalizes binary labels:
      - {1, 2} -> {0, 1}
      - {-1, 1} -> {0, 1}
    """
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df.iloc[:, -1].to_numpy()

    # --- Normalize binary labels ---
    unique_vals = np.unique(y)
    if set(unique_vals) == {1, 2}:
        y = y - 1
    elif set(unique_vals) == {-1, 1}:
        y = (y + 1) // 2

    return X, y


def train_test_split(X, y, test_ratio=0.3, seed=42):
    """
    Simple manual train/test split using NumPy (no sklearn).
    """
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    test_size = int(len(X) * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# === METRICS ===
def accuracy(y_true, y_pred):
    """
    Accuracy for classification tasks.
    """
    return np.mean(y_true == y_pred)


def mse(y_true, y_pred):
    """
    Mean squared error for regression tasks.
    """
    return np.mean((y_true - y_pred) ** 2)


# === PLOTTING ===
def plot_loss(losses, title="Training loss"):
    """
    Plot training loss over epochs.
    """
    plt.plot(losses, label="Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_predictions(y_true, y_pred, title="Predictions vs True"):
    """
    Scatter plot for regression results.
    """
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel("True values")
    plt.ylabel("Predictions")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """
    Plots the decision boundary for a trained classifier in a 2D feature space.
    """
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Predict on the meshgrid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=20, edgecolor='k', cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()


def plot_weight_evolution(weight_history, title="Weight Evolution"):
    """
    Plots the evolution of the L2 norm of weights for each layer over epochs.
    """
    # weight_history is a list of lists, where each inner list contains the L2 norm of weights for a layer at a given epoch.
    weight_history_np = np.array(weight_history)
    num_layers = weight_history_np.shape[1]

    for i in range(num_layers):
        plt.plot(weight_history_np[:, i], label=f'Layer {i+1} Weights')

    plt.title(title)
    plt.xlabel("Epoka")
    plt.ylabel("L2 Norm of Weights")
    plt.grid(True)
    plt.legend()
    plt.show()
