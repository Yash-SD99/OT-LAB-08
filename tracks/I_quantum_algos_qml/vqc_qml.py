"""Variational Quantum Classifier (VQC) vs Classical MLP (Track I).

Implements:
1. Pure NumPy 2D non-linear concentric circles / moons dataset generator.
2. Variational Quantum Classifier (VQC):
   - Quantum ZZ-feature map for non-linear data embedding.
   - Parameterized ansatz with Ry single-qubit rotations and CNOT entanglers.
   - Exact analytical Parameter-Shift Rule for gradients:
     d<Z>/d theta = 0.5 * [<Z>(theta + pi/2) - <Z>(theta - pi/2)]
   - Adam optimization.
3. Classical Multi-Layer Perceptron (MLP):
   - Parameter-matched architecture (2 -> 3 -> 1, tanh activations).
   - Exact backpropagation and Adam optimizer.
4. Evaluation: Decision boundary generation, accuracy, loss trajectories, and parameter efficiency.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from tracks.I_quantum_algos_qml.quantum_circuits import AdvancedQuantumCircuit


# ==============================================================================
# 1. Dataset Generation
# ==============================================================================

def generate_concentric_circles(
    n_samples: int = 160,
    noise: float = 0.08,
    factor: float = 0.45,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates 2D concentric circles dataset with binary labels y in {-1, +1}.
    Features are scaled into [0, pi] for quantum angle encoding.
    """
    rng = np.random.default_rng(seed)
    n_inner = n_samples // 2
    n_outer = n_samples - n_inner

    # Outer circle: radius ~ 1.0, label +1
    theta_out = rng.uniform(0, 2 * np.pi, size=n_outer)
    r_out = 1.0 + rng.normal(0, noise, size=n_outer)
    x_out = np.column_stack([r_out * np.cos(theta_out), r_out * np.sin(theta_out)])
    y_out = np.ones(n_outer, dtype=np.float64)

    # Inner circle: radius ~ factor, label -1
    theta_in = rng.uniform(0, 2 * np.pi, size=n_inner)
    r_in = factor + rng.normal(0, noise, size=n_inner)
    x_in = np.column_stack([r_in * np.cos(theta_in), r_in * np.sin(theta_in)])
    y_in = -np.ones(n_inner, dtype=np.float64)

    X = np.vstack([x_out, x_in])
    y = np.concatenate([y_out, y_in])

    # Shuffle
    perm = rng.permutation(n_samples)
    X = X[perm]
    y = y[perm]

    # Normalize X to [0.1 * pi, 0.9 * pi] to avoid phase wrap-around
    x_min, x_max = X.min(axis=0), X.max(axis=0)
    X_norm = (X - x_min) / (x_max - x_min + 1e-8)
    X_scaled = 0.1 * np.pi + 0.8 * np.pi * X_norm

    return X_scaled, y


# ==============================================================================
# 2. Variational Quantum Classifier (VQC)
# ==============================================================================

class VariationalQuantumClassifier:
    """2-qubit Variational Quantum Classifier with ZZ-feature map and Parameter-Shift gradients."""

    def __init__(self, n_layers: int = 2, seed: int = 42):
        self.n_qubits = 2
        self.n_layers = n_layers
        # 4 parameters per layer: 2 for pre-CNOT Ry, 2 for post-CNOT Ry
        self.n_params = n_layers * 4
        rng = np.random.default_rng(seed)
        # Initialize variational parameters uniformly in [-pi, pi]
        self.theta = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=self.n_params)

    def _circuit_forward(self, x: np.ndarray, theta: np.ndarray) -> float:
        """Executes quantum circuit for single 2D feature vector x under parameters theta.
        Returns expectation value <Z_0> in [-1, 1].
        """
        qc = AdvancedQuantumCircuit(self.n_qubits)

        # 1. Quantum Feature Map: Angle encoding + ZZ entangling
        # Hadamards to create superposition
        qc.h(0).h(1)
        # Single-qubit rotations
        qc.rz(0, 2.0 * float(x[0]))
        qc.rz(1, 2.0 * float(x[1]))
        # Non-linear ZZ interaction
        zz_phase = 2.0 * (np.pi - float(x[0])) * (np.pi - float(x[1]))
        qc.rzz(0, 1, zz_phase)

        # 2. Parameterized Variational Ansatz
        for l in range(self.n_layers):
            base_idx = l * 4
            qc.ry(0, float(theta[base_idx + 0]))
            qc.ry(1, float(theta[base_idx + 1]))
            qc.cnot(0, 1)
            qc.ry(0, float(theta[base_idx + 2]))
            qc.ry(1, float(theta[base_idx + 3]))

        # 3. Measurement: Pauli-Z expectation on qubit 0
        return qc.expectation_z(0)

    def predict_single(self, x: np.ndarray) -> float:
        """Returns continuous model output in [-1, 1]."""
        return self._circuit_forward(x, self.theta)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns continuous predictions for batch X."""
        return np.array([self._circuit_forward(xi, self.theta) for xi in X])

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Returns discrete class predictions in {-1, +1}."""
        preds = self.predict(X)
        return np.where(preds >= 0.0, 1.0, -1.0)

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean Squared Error loss: (1/N) * sum((y_hat - y)^2)."""
        preds = self.predict(X)
        return float(np.mean((preds - y)**2))

    def parameter_shift_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Computes exact analytical gradient of MSE loss w.r.t theta using Parameter-Shift Rule:
        d<Z>/d theta_k = 0.5 * (<Z>(theta_k + pi/2) - <Z>(theta_k - pi/2))
        """
        N = len(X)
        grad = np.zeros(self.n_params, dtype=np.float64)
        shift = np.pi / 2.0

        # Compute baseline predictions y_hat
        y_hat = self.predict(X)
        residuals = y_hat - y  # shape (N,)

        for k in range(self.n_params):
            # Forward shift: theta_k + pi/2
            theta_plus = self.theta.copy()
            theta_plus[k] += shift
            y_plus = np.array([self._circuit_forward(xi, theta_plus) for xi in X])

            # Backward shift: theta_k - pi/2
            theta_minus = self.theta.copy()
            theta_minus[k] -= shift
            y_minus = np.array([self._circuit_forward(xi, theta_minus) for xi in X])

            # Parameter-shift gradient for expectation value
            dy_dtheta = 0.5 * (y_plus - y_minus)

            # Chain rule: d/d theta (1/N sum (y_hat - y)^2) = 2/N sum (y_hat - y) * dy/d theta
            grad[k] = (2.0 / N) * np.sum(residuals * dy_dtheta)

        return grad

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        lr: float = 0.08,
        batch_size: int = 32,
        seed: int = 42
    ) -> List[float]:
        """Trains VQC using Adam optimizer and parameter-shift gradients."""
        loss_history = []
        rng = np.random.default_rng(seed)
        N = len(X_train)

        # Adam state
        m = np.zeros(self.n_params)
        v = np.zeros(self.n_params)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0

        for epoch in range(1, epochs + 1):
            perm = rng.permutation(N)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            for b in range(0, N, batch_size):
                t += 1
                xb = X_shuffled[b:b + batch_size]
                yb = y_shuffled[b:b + batch_size]

                g = self.parameter_shift_gradient(xb, yb)
                m = beta1 * m + (1.0 - beta1) * g
                v = beta2 * v + (1.0 - beta2) * (g**2)

                m_hat = m / (1.0 - beta1**t)
                v_hat = v / (1.0 - beta2**t)

                self.theta -= lr * m_hat / (np.sqrt(v_hat) + eps)

            loss = self.compute_loss(X_train, y_train)
            loss_history.append(loss)

        return loss_history


# ==============================================================================
# 3. Classical Multi-Layer Perceptron (MLP)
# ==============================================================================

class ClassicalMLP:
    """Parameter-matched classical feed-forward neural network (2 -> 3 -> 1) with tanh activations."""

    def __init__(self, hidden_dim: int = 3, seed: int = 42):
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)
        # Xavier initialization
        self.W1 = rng.uniform(-np.sqrt(6.0 / (2 + hidden_dim)), np.sqrt(6.0 / (2 + hidden_dim)), size=(2, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.uniform(-np.sqrt(6.0 / (hidden_dim + 1)), np.sqrt(6.0 / (hidden_dim + 1)), size=(hidden_dim, 1))
        self.b2 = np.zeros(1)
        # Total parameters: 2*3 + 3 + 3*1 + 1 = 13 parameters
        self.n_params = (2 * hidden_dim) + hidden_dim + (hidden_dim * 1) + 1

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass: returns (y_hat, h, z1)."""
        z1 = X @ self.W1 + self.b1
        h = np.tanh(z1)
        z2 = h @ self.W2 + self.b2
        y_hat = np.tanh(z2).ravel()
        return y_hat, h, z1

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_hat, _, _ = self.forward(X)
        return y_hat

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        return np.where(preds >= 0.0, 1.0, -1.0)

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean((preds - y)**2))

    def backward(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Analytical backpropagation gradients for MSE loss."""
        N = len(X)
        y_hat, h, z1 = self.forward(X)
        residuals = (y_hat - y).reshape(-1, 1)  # (N, 1)

        # Output layer tanh derivative: d y_hat / d z2 = 1 - y_hat^2
        d_z2 = residuals * (1.0 - (y_hat.reshape(-1, 1)**2)) * (2.0 / N)  # (N, 1)
        d_W2 = h.T @ d_z2  # (hidden_dim, 1)
        d_b2 = np.sum(d_z2, axis=0)  # (1,)

        # Hidden layer tanh derivative: d h / d z1 = 1 - h^2
        d_h = d_z2 @ self.W2.T  # (N, hidden_dim)
        d_z1 = d_h * (1.0 - (h**2))  # (N, hidden_dim)
        d_W1 = X.T @ d_z1  # (2, hidden_dim)
        d_b1 = np.sum(d_z1, axis=0)  # (hidden_dim,)

        return d_W1, d_b1, d_W2, d_b2

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        lr: float = 0.08,
        batch_size: int = 32,
        seed: int = 42
    ) -> List[float]:
        """Trains MLP using Adam optimizer."""
        loss_history = []
        rng = np.random.default_rng(seed)
        N = len(X_train)

        # Adam moment vectors
        mW1, vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        mb1, vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        mW2, vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        mb2, vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0

        for epoch in range(1, epochs + 1):
            perm = rng.permutation(N)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            for b in range(0, N, batch_size):
                t += 1
                xb = X_shuffled[b:b + batch_size]
                yb = y_shuffled[b:b + batch_size]

                gW1, gb1, gW2, gb2 = self.backward(xb, yb)

                # Update W1
                mW1 = beta1 * mW1 + (1.0 - beta1) * gW1
                vW1 = beta2 * vW1 + (1.0 - beta2) * (gW1**2)
                self.W1 -= lr * (mW1 / (1.0 - beta1**t)) / (np.sqrt(vW1 / (1.0 - beta2**t)) + eps)

                # Update b1
                mb1 = beta1 * mb1 + (1.0 - beta1) * gb1
                vb1 = beta2 * vb1 + (1.0 - beta2) * (gb1**2)
                self.b1 -= lr * (mb1 / (1.0 - beta1**t)) / (np.sqrt(vb1 / (1.0 - beta2**t)) + eps)

                # Update W2
                mW2 = beta1 * mW2 + (1.0 - beta1) * gW2
                vW2 = beta2 * vW2 + (1.0 - beta2) * (gW2**2)
                self.W2 -= lr * (mW2 / (1.0 - beta1**t)) / (np.sqrt(vW2 / (1.0 - beta2**t)) + eps)

                # Update b2
                mb2 = beta1 * mb2 + (1.0 - beta1) * gb2
                vb2 = beta2 * vb2 + (1.0 - beta2) * (gb2**2)
                self.b2 -= lr * (mb2 / (1.0 - beta1**t)) / (np.sqrt(vb2 / (1.0 - beta2**t)) + eps)

            loss = self.compute_loss(X_train, y_train)
            loss_history.append(loss)

        return loss_history
