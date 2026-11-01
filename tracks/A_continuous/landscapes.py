"""Optimization Landscapes and Machine Learning Objectives (Track A Foundations).

Covers:
1. Strongly Convex Quadratic: f(x) = 1/2 x^T A x - b^T x
2. Non-Convex Saddle Point: f(x, y) = x^2 - y^2 (and Monkey Saddle: f(x, y) = x^3 - 3xy^2)
3. Non-Convex Rosenbrock (Banana) Valley: f(x1, x2) = (1 - x1)^2 + 100(x2 - x1^2)^2
4. Non-Convex Multi-modal Rastrigin (Local vs Global Minima)
5. Noisy Objective Wrapper (Stochastic evaluations)
"""

import numpy as np


class StronglyConvexQuadratic:
    """Strongly convex quadratic objective: f(x) = 0.5 * x^T A x - b^T x.
    
    A must be symmetric positive-definite.
    Global minimum is at x* = A^{-1} b with f(x*) = -0.5 * b^T A^{-1} b.
    """
    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.A = np.asarray(A, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        eigvals = np.linalg.eigvalsh(self.A)
        if np.any(eigvals <= 0):
            raise ValueError("Matrix A must be strictly positive definite.")
        self.cond_number = eigvals[-1] / eigvals[0]
        self.x_opt = np.linalg.solve(self.A, self.b)
        self.f_opt = float(0.5 * (self.x_opt @ self.A @ self.x_opt) - (self.b @ self.x_opt))

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        return float(0.5 * x @ self.A @ x - self.b @ x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return self.A @ x - self.b

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return self.A.copy()


class SaddlePointLandscape:
    """2D hyperbolic saddle point: f(x, y) = x^2 - y^2.
    
    Critical point is at (0, 0), where gradient is [0, 0] and
    Hessian is diag(2, -2) (indefinite). Positive curvature along x,
    negative curvature along y.
    """
    def value(self, x: np.ndarray) -> float:
        return float(x[0]**2 - x[1]**2)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.array([2.0 * x[0], -2.0 * x[1]], dtype=np.float64)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array([[2.0, 0.0], [0.0, -2.0]], dtype=np.float64)


class RosenbrockLandscape:
    """Rosenbrock non-convex banana valley:
    f(x1, x2) = (1 - x1)^2 + 100 * (x2 - x1^2)^2.
    
    Global minimum is at (1.0, 1.0) with f = 0.
    Can also be expressed as non-linear least squares:
    r(x) = [1 - x1, 10*(x2 - x1^2)]^T, with f(x) = 0.5 * ||r(x)||^2 * 2.
    """
    def value(self, x: np.ndarray) -> float:
        return float((1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        g0 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0]**2)
        g1 = 200.0 * (x[1] - x[0]**2)
        return np.array([g0, g1], dtype=np.float64)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        h00 = 2.0 - 400.0 * x[1] + 1200.0 * x[0]**2
        h01 = -400.0 * x[0]
        h10 = -400.0 * x[0]
        h11 = 200.0
        return np.array([[h00, h01], [h10, h11]], dtype=np.float64)

    def residuals(self, x: np.ndarray) -> np.ndarray:
        """Residual vector for Gauss-Newton solver: r(x)."""
        return np.array([1.0 - x[0], 10.0 * (x[1] - x[0]**2)], dtype=np.float64)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian J(x) = dr/dx of residuals."""
        return np.array([
            [-1.0, 0.0],
            [-20.0 * x[0], 10.0]
        ], dtype=np.float64)


class RastriginLandscape:
    """Multi-modal non-convex Rastrigin function.
    f(x) = 10*d + sum_{i=1}^d [x_i^2 - 10*cos(2*pi*x_i)].
    
    Contains a dense grid of local minima with one global minimum at x* = 0, f(x*) = 0.
    """
    def __init__(self, dim: int = 2):
        self.dim = dim

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        return float(10.0 * self.dim + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        diag_h = 2.0 + 40.0 * (np.pi**2) * np.cos(2.0 * np.pi * x)
        return np.diag(diag_h)


class NoisyObjectiveWrapper:
    """Wraps an objective function with additive Gaussian noise on value and gradient."""
    def __init__(self, base_objective, val_noise_std: float = 0.1, grad_noise_std: float = 0.05, seed: int = 42):
        self.base = base_objective
        self.val_noise_std = val_noise_std
        self.grad_noise_std = grad_noise_std
        self.rng = np.random.default_rng(seed)

    def value(self, x: np.ndarray) -> float:
        clean = self.base.value(x)
        noise = self.rng.normal(0.0, self.val_noise_std)
        return clean + noise

    def gradient(self, x: np.ndarray) -> np.ndarray:
        clean = self.base.gradient(x)
        noise = self.rng.normal(0.0, self.grad_noise_std, size=clean.shape)
        return clean + noise

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return self.base.hessian(x)
