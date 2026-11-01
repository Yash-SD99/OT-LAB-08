"""Pure NumPy Implementation of 1st and 2nd Order Continuous Optimizers (Track A).

Includes:
- First-Order: Gradient Descent, SGD / Mini-batch, Momentum, NAG, AdaGrad, RMSProp, Adam.
- Second-Order / Curvature: Newton-Raphson, Gauss-Newton with Levenberg-Marquardt damping.
- Curvature Approximations: Empirical Fisher, Diagonal Curvature, Low-Rank Curvature, Hessian-Vector Products.
"""

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


class Optimizer:
    """Base class for continuous optimizers."""
    def __init__(self, lr: float = 0.01):
        self.lr = float(lr)
        self.iterations = 0

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GradientDescent(Optimizer):
    """Standard Batch Gradient Descent: x_{t+1} = x_t - lr * grad."""
    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.iterations += 1
        return x - self.lr * grad


class Momentum(Optimizer):
    """Polyak Momentum (Heavy-ball):
    v_{t+1} = gamma * v_t + lr * grad
    x_{t+1} = x_t - v_{t+1}
    """
    def __init__(self, lr: float = 0.01, gamma: float = 0.9):
        super().__init__(lr)
        self.gamma = float(gamma)
        self.velocity: Optional[np.ndarray] = None

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.iterations += 1
        if self.velocity is None:
            self.velocity = np.zeros_like(x)
        self.velocity = self.gamma * self.velocity + self.lr * grad
        return x - self.velocity


class NAG(Optimizer):
    """Nesterov Accelerated Gradient:
    v_{t+1} = gamma * v_t + lr * grad(x_t - gamma * v_t)
    x_{t+1} = x_t - v_{t+1}
    """
    def __init__(self, lr: float = 0.01, gamma: float = 0.9):
        super().__init__(lr)
        self.gamma = float(gamma)
        self.velocity: Optional[np.ndarray] = None

    def lookahead(self, x: np.ndarray) -> np.ndarray:
        """Computes the lookahead position where gradient should be evaluated."""
        if self.velocity is None:
            return x.copy()
        return x - self.gamma * self.velocity

    def step(self, x: np.ndarray, lookahead_grad: np.ndarray) -> np.ndarray:
        self.iterations += 1
        if self.velocity is None:
            self.velocity = np.zeros_like(x)
        self.velocity = self.gamma * self.velocity + self.lr * lookahead_grad
        return x - self.velocity


class AdaGrad(Optimizer):
    """AdaGrad (Adaptive Subgradient):
    G_{t+1} = G_t + grad^2
    x_{t+1} = x_t - (lr / sqrt(G_{t+1} + eps)) * grad
    """
    def __init__(self, lr: float = 0.1, eps: float = 1e-8):
        super().__init__(lr)
        self.eps = float(eps)
        self.G: Optional[np.ndarray] = None

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.iterations += 1
        if self.G is None:
            self.G = np.zeros_like(x)
        self.G += grad**2
        adjusted_lr = self.lr / (np.sqrt(self.G) + self.eps)
        return x - adjusted_lr * grad


class RMSProp(Optimizer):
    """RMSProp (Root Mean Square Propagation):
    v_{t+1} = beta * v_t + (1 - beta) * grad^2
    x_{t+1} = x_t - (lr / sqrt(v_{t+1} + eps)) * grad
    """
    def __init__(self, lr: float = 0.01, beta: float = 0.9, eps: float = 1e-8):
        super().__init__(lr)
        self.beta = float(beta)
        self.eps = float(eps)
        self.v: Optional[np.ndarray] = None

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.iterations += 1
        if self.v is None:
            self.v = np.zeros_like(x)
        self.v = self.beta * self.v + (1.0 - self.beta) * (grad**2)
        adjusted_lr = self.lr / (np.sqrt(self.v) + self.eps)
        return x - adjusted_lr * grad


class Adam(Optimizer):
    """Adam (Adaptive Moment Estimation with bias correction):
    m_{t+1} = beta1 * m_t + (1 - beta1) * grad
    v_{t+1} = beta2 * v_t + (1 - beta2) * grad^2
    m_hat = m_{t+1} / (1 - beta1^{t+1})
    v_hat = v_{t+1} / (1 - beta2^{t+1})
    x_{t+1} = x_t - (lr / sqrt(v_hat + eps)) * m_hat
    """
    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.iterations += 1
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad**2)
        
        # Bias corrections
        m_hat = self.m / (1.0 - self.beta1**self.iterations)
        v_hat = self.v / (1.0 - self.beta2**self.iterations)
        
        return x - (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat


class NewtonRaphson:
    """Second-order Newton-Raphson optimizer with Levenberg damping for non-convex regions:
    step = (H + lambda * I)^{-1} grad
    x_{t+1} = x_t - lr * step
    """
    def __init__(self, lr: float = 1.0, damping: float = 1e-4):
        self.lr = float(lr)
        self.damping = float(damping)
        self.iterations = 0

    def step(self, x: np.ndarray, grad: np.ndarray, hessian: np.ndarray) -> np.ndarray:
        self.iterations += 1
        dim = x.shape[0]
        H_reg = hessian + self.damping * np.eye(dim)
        
        # Ensure positive-definiteness via eigenvalue clamping if needed
        eigvals, eigvecs = np.linalg.eigh(H_reg)
        min_eig = eigvals[0]
        if min_eig < 1e-6:
            shift = (1e-6 - min_eig) + self.damping
            H_reg += shift * np.eye(dim)

        delta = np.linalg.solve(H_reg, grad)
        return x - self.lr * delta


class GaussNewton:
    """Gauss-Newton solver for non-linear least squares objectives:
    r(x) in R^M, J(x) in R^{M x N}.
    Step solves normal equations: (J^T J + lambda * I) delta = J^T r.
    x_{t+1} = x_t - lr * delta.
    """
    def __init__(self, lr: float = 1.0, damping: float = 1e-3):
        self.lr = float(lr)
        self.damping = float(damping)
        self.iterations = 0

    def step(self, x: np.ndarray, residuals: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
        self.iterations += 1
        JT_J = jacobian.T @ jacobian
        JT_r = jacobian.T @ residuals
        dim = x.shape[0]
        damped_matrix = JT_J + self.damping * np.eye(dim)
        delta = np.linalg.solve(damped_matrix, JT_r)
        return x - self.lr * delta


# =========================================================================
# Curvature Approximations & Hessian-Vector Products
# =========================================================================

def empirical_fisher(per_sample_grads: np.ndarray) -> np.ndarray:
    """Computes the Empirical Fisher Information Matrix:
    F = (1 / N) sum_{i=1}^N (g_i g_i^T)
    where per_sample_grads has shape (N, D).
    """
    N, D = per_sample_grads.shape
    return (per_sample_grads.T @ per_sample_grads) / float(N)


def diagonal_curvature(matrix: np.ndarray) -> np.ndarray:
    """Extracts diagonal elements of Hessian or Fisher matrix."""
    return np.diag(matrix)


def low_rank_curvature(matrix: np.ndarray, rank: int = 2) -> np.ndarray:
    """Computes a rank-k truncated SVD approximation of curvature matrix:
    F_k = U_k S_k V_k^T.
    """
    U, s, Vt = np.linalg.svd(matrix)
    k = min(rank, len(s))
    return (U[:, :k] * s[:k]) @ Vt[:k, :]


def hessian_vector_product_fd(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    v: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """Approximates Hessian-vector product (H v) via central finite differences:
    H v approx (grad(x + eps * v) - grad(x - eps * v)) / (2 * eps).
    Avoids forming or storing the full D x D Hessian matrix.
    """
    x_pos = x + eps * v
    x_neg = x - eps * v
    grad_pos = grad_fn(x_pos)
    grad_neg = grad_fn(x_neg)
    return (grad_pos - grad_neg) / (2.0 * eps)
