"""Pure NumPy Neural Network & Backpropagation Implementation (Track A).

Separates the machine learning model and gradient computation (backpropagation)
from the optimizer that updates the parameters. Includes analytical chain-rule
derivatives and rigorous finite-difference gradient verification.
"""

from typing import Dict, Tuple
import numpy as np


class SimpleMLP:
    """Multi-Layer Perceptron in pure NumPy with L2 Regularization.
    
    Architecture:
      Input (D_in) -> Linear -> Tanh Activation -> Linear -> Output (D_out)
    Loss:
      Mean Squared Error (MSE) + 0.5 * reg_lambda * sum(||W_l||_F^2)
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, reg_lambda: float = 1e-4, seed: int = 42):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.reg_lambda = float(reg_lambda)
        
        # Xavier / Glorot initialization
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / (in_dim + hidden_dim)), size=(in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / (hidden_dim + out_dim)), size=(hidden_dim, out_dim))
        self.b2 = np.zeros(out_dim, dtype=np.float64)
        
        # Cache for forward pass
        self.X_cache: Optional[np.ndarray] = None
        self.z1_cache: Optional[np.ndarray] = None
        self.a1_cache: Optional[np.ndarray] = None
        self.z2_cache: Optional[np.ndarray] = None

    def get_flat_params(self) -> np.ndarray:
        """Flattens all weight matrices and biases into a single 1D vector."""
        return np.concatenate([self.W1.ravel(), self.b1.ravel(), self.W2.ravel(), self.b2.ravel()])

    def set_flat_params(self, theta: np.ndarray) -> None:
        """Restores model parameter matrices from a 1D vector."""
        idx = 0
        w1_size = self.in_dim * self.hidden_dim
        self.W1 = theta[idx : idx + w1_size].reshape((self.in_dim, self.hidden_dim))
        idx += w1_size
        
        b1_size = self.hidden_dim
        self.b1 = theta[idx : idx + b1_size].reshape(self.hidden_dim)
        idx += b1_size
        
        w2_size = self.hidden_dim * self.out_dim
        self.W2 = theta[idx : idx + w2_size].reshape((self.hidden_dim, self.out_dim))
        idx += w2_size
        
        b2_size = self.out_dim
        self.b2 = theta[idx : idx + b2_size].reshape(self.out_dim)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: z1 = X W1 + b1, a1 = tanh(z1), z2 = a1 W2 + b2."""
        self.X_cache = np.asarray(X, dtype=np.float64)
        self.z1_cache = self.X_cache @ self.W1 + self.b1
        self.a1_cache = np.tanh(self.z1_cache)
        self.z2_cache = self.a1_cache @ self.W2 + self.b2
        return self.z2_cache

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Computes MSE loss plus L2 regularization."""
        y_pred = self.forward(X)
        N = X.shape[0]
        mse = float(np.mean(np.sum((y_pred - y)**2, axis=1)))
        reg = 0.5 * self.reg_lambda * float(np.sum(self.W1**2) + np.sum(self.W2**2))
        return mse + reg

    def backward(self, y: np.ndarray) -> np.ndarray:
        """Backpropagation: Computes analytical gradients dL/dW1, dL/db1, dL/dW2, dL/db2.
        
        Returns flat gradient vector matching get_flat_params().
        """
        N = self.X_cache.shape[0]
        y_pred = self.z2_cache
        
        # dLoss / dz2 for MSE: 2 * (y_pred - y) / N
        dL_dz2 = (2.0 / N) * (y_pred - y)
        
        # Layer 2 gradients
        dL_dW2 = (self.a1_cache.T @ dL_dz2) + self.reg_lambda * self.W2
        dL_db2 = np.sum(dL_dz2, axis=0)
        
        # Backprop through tanh: dtanh(z1) = 1 - tanh^2(z1) = 1 - a1^2
        dL_da1 = dL_dz2 @ self.W2.T
        dL_dz1 = dL_da1 * (1.0 - self.a1_cache**2)
        
        # Layer 1 gradients
        dL_dW1 = (self.X_cache.T @ dL_dz1) + self.reg_lambda * self.W1
        dL_db1 = np.sum(dL_dz1, axis=0)
        
        return np.concatenate([dL_dW1.ravel(), dL_db1.ravel(), dL_dW2.ravel(), dL_db2.ravel()])

    def per_sample_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Computes gradients individually for each sample i in 1..N.
        Used for Empirical Fisher matrix computation: F = 1/N sum_i (g_i g_i^T).
        """
        N = X.shape[0]
        grads = []
        for i in range(N):
            x_i = X[i:i+1]
            y_i = y[i:i+1]
            # Forward single sample
            _ = self.forward(x_i)
            # Analytical grad for single sample (without N-averaging in loss)
            dL_dz2 = 2.0 * (self.z2_cache - y_i)
            dL_dW2 = (self.a1_cache.T @ dL_dz2) + self.reg_lambda * self.W2
            dL_db2 = np.sum(dL_dz2, axis=0)
            dL_da1 = dL_dz2 @ self.W2.T
            dL_dz1 = dL_da1 * (1.0 - self.a1_cache**2)
            dL_dW1 = (x_i.T @ dL_dz1) + self.reg_lambda * self.W1
            dL_db1 = np.sum(dL_dz1, axis=0)
            g_i = np.concatenate([dL_dW1.ravel(), dL_db1.ravel(), dL_dW2.ravel(), dL_db2.ravel()])
            grads.append(g_i)
        return np.array(grads)

    def numerical_gradient_check(self, X: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
        """Verifies analytical backprop gradients against two-sided finite differences.
        
        Returns the maximum relative error across all parameters:
        max_i (|g_analytical,i - g_numerical,i| / (|g_analytical,i| + |g_numerical,i| + eps)).
        """
        # Analytical gradients
        self.loss(X, y)
        g_analytical = self.backward(y)
        
        # Numerical gradients
        theta_orig = self.get_flat_params()
        g_numerical = np.zeros_like(theta_orig)
        
        for i in range(len(theta_orig)):
            theta_pos = theta_orig.copy()
            theta_pos[i] += eps
            self.set_flat_params(theta_pos)
            loss_pos = self.loss(X, y)
            
            theta_neg = theta_orig.copy()
            theta_neg[i] -= eps
            self.set_flat_params(theta_neg)
            loss_neg = self.loss(X, y)
            
            g_numerical[i] = (loss_pos - loss_neg) / (2.0 * eps)
            
        # Restore original parameters
        self.set_flat_params(theta_orig)
        
        diff = np.abs(g_analytical - g_numerical)
        scale = np.maximum(np.abs(g_analytical) + np.abs(g_numerical), 1e-8)
        rel_errors = diff / scale
        return float(np.max(rel_errors))
