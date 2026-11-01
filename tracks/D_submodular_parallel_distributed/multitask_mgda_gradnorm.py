"""Multi-Task Optimization: Equal Weighting vs MGDA vs GradNorm (Track D).

Implements:
1. Two-Task Shared Representation Linear Model with conflicting task gradients
2. Baseline: Uniform Equal Task Weighting (L = 0.5 * L1 + 0.5 * L2)
3. MGDA (Multiple Gradient Descent Algorithm):
   Finds minimum-norm vector in convex hull of task gradients:
   min_{alpha in [0, 1]} ||alpha * g1 + (1 - alpha) * g2||^2
4. GradNorm: Dynamically balances gradient norms based on relative task training rates
5. Multi-Seed Pareto Front Analysis across 15 seeds
"""

import time
from typing import Dict, List, Tuple
import numpy as np


class TwoTaskModel:
    """A two-task linear model with a shared representation layer and task-specific heads.
    Input x in R^D -> Shared representation h = W_shared x in R^H
    Task 1: y1_hat = w1^T h
    Task 2: y2_hat = w2^T h
    """
    def __init__(self, in_dim: int = 6, hidden_dim: int = 4, seed: int = 42):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)
        
        self.W_shared = rng.normal(0.0, 0.5, size=(in_dim, hidden_dim))
        self.w1 = rng.normal(0.0, 0.5, size=hidden_dim)
        self.w2 = rng.normal(0.0, 0.5, size=hidden_dim)

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([self.W_shared.ravel(), self.w1.ravel(), self.w2.ravel()])

    def set_flat_params(self, theta: np.ndarray) -> None:
        idx = 0
        w_size = self.in_dim * self.hidden_dim
        self.W_shared = theta[idx : idx + w_size].reshape((self.in_dim, self.hidden_dim))
        idx += w_size
        self.w1 = theta[idx : idx + self.hidden_dim].copy()
        idx += self.hidden_dim
        self.w2 = theta[idx : idx + self.hidden_dim].copy()

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (H_repr, y1_pred, y2_pred)."""
        H = X @ self.W_shared
        y1_pred = H @ self.w1
        y2_pred = H @ self.w2
        return H, y1_pred, y2_pred

    def compute_task_gradients(self, X: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Computes losses L1, L2 and analytical parameter gradients g1, g2."""
        N = X.shape[0]
        H, y1_pred, y2_pred = self.forward(X)
        
        # Losses
        loss1 = float(np.mean((y1_pred - y1)**2))
        loss2 = float(np.mean((y2_pred - y2)**2))
        
        # Gradients for Task 1
        dy1 = (2.0 / N) * (y1_pred - y1)
        dw1 = H.T @ dy1
        dW_shared_1 = X.T @ np.outer(dy1, self.w1)
        dw2_dummy = np.zeros_like(self.w2)
        g1 = np.concatenate([dW_shared_1.ravel(), dw1.ravel(), dw2_dummy.ravel()])
        
        # Gradients for Task 2
        dy2 = (2.0 / N) * (y2_pred - y2)
        dw2 = H.T @ dy2
        dW_shared_2 = X.T @ np.outer(dy2, self.w2)
        dw1_dummy = np.zeros_like(self.w1)
        g2 = np.concatenate([dW_shared_2.ravel(), dw1_dummy.ravel(), dw2.ravel()])
        
        return loss1, loss2, g1, g2


def solve_mgda_2task(g1: np.ndarray, g2: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Finds min-norm point in convex hull of 2 task gradients:
    min_{alpha in [0, 1]} ||alpha * g1 + (1 - alpha) * g2||^2.
    
    Closed-form analytical solution:
    alpha = (g2 - g1)^T g2 / ||g1 - g2||^2, clipped to [0, 1].
    """
    diff = g1 - g2
    denom = float(np.dot(diff, diff))
    if denom < 1e-12:
        return 0.5, 0.5, 0.5 * (g1 + g2)
        
    alpha = float(np.dot(g2 - g1, g2) / denom)
    alpha = max(0.0, min(1.0, alpha))
    beta = 1.0 - alpha
    g_mgda = alpha * g1 + beta * g2
    return alpha, beta, g_mgda


def train_equal_weighting(model: TwoTaskModel, X: np.ndarray, y1: np.ndarray, y2: np.ndarray, iters: int = 50, lr: float = 0.03) -> Dict:
    """Baseline: Equal task weighting (L = 0.5 * L1 + 0.5 * L2)."""
    t0 = time.perf_counter()
    l1_hist, l2_hist = [], []
    
    for _ in range(iters):
        l1, l2, g1, g2 = model.compute_task_gradients(X, y1, y2)
        l1_hist.append(l1)
        l2_hist.append(l2)
        
        joint_g = 0.5 * g1 + 0.5 * g2
        theta = model.get_flat_params() - lr * joint_g
        model.set_flat_params(theta)
        
    elapsed = time.perf_counter() - t0
    return {
        "method": "Equal Task Weighting (0.5 L1 + 0.5 L2)",
        "final_l1": float(l1_hist[-1]),
        "final_l2": float(l2_hist[-1]),
        "final_total_loss": float(l1_hist[-1] + l2_hist[-1]),
        "l1_history": l1_hist,
        "l2_history": l2_hist,
        "runtime_ms": elapsed * 1000.0
    }


def train_mgda(model: TwoTaskModel, X: np.ndarray, y1: np.ndarray, y2: np.ndarray, iters: int = 50, lr: float = 0.03) -> Dict:
    """MGDA (Multiple Gradient Descent Algorithm) with Frank-Wolfe Pareto updates."""
    t0 = time.perf_counter()
    l1_hist, l2_hist = [], []
    weights_history = []
    
    for _ in range(iters):
        l1, l2, g1, g2 = model.compute_task_gradients(X, y1, y2)
        l1_hist.append(l1)
        l2_hist.append(l2)
        
        alpha, beta, g_mgda = solve_mgda_2task(g1, g2)
        weights_history.append([alpha, beta])
        
        theta = model.get_flat_params() - lr * g_mgda
        model.set_flat_params(theta)
        
    elapsed = time.perf_counter() - t0
    return {
        "method": "MGDA (Pareto Stationary)",
        "final_l1": float(l1_hist[-1]),
        "final_l2": float(l2_hist[-1]),
        "final_total_loss": float(l1_hist[-1] + l2_hist[-1]),
        "l1_history": l1_hist,
        "l2_history": l2_hist,
        "weights_history": weights_history,
        "runtime_ms": elapsed * 1000.0
    }


def train_gradnorm(
    model: TwoTaskModel,
    X: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    iters: int = 50,
    lr: float = 0.03,
    alpha_grad: float = 1.5
) -> Dict:
    """GradNorm: Dynamically balances gradient norms by relative task loss ratios."""
    t0 = time.perf_counter()
    l1_hist, l2_hist = [], []
    task_weights = np.array([1.0, 1.0], dtype=np.float64)
    
    init_l1, init_l2, _, _ = model.compute_task_gradients(X, y1, y2)
    init_losses = np.array([init_l1, init_l2])
    
    for _ in range(iters):
        l1, l2, g1, g2 = model.compute_task_gradients(X, y1, y2)
        l1_hist.append(l1)
        l2_hist.append(l2)
        
        curr_losses = np.array([l1, l2])
        loss_ratios = curr_losses / np.maximum(init_losses, 1e-8)
        mean_ratio = np.mean(loss_ratios)
        inv_rates = loss_ratios / max(mean_ratio, 1e-8)
        
        norm_g1 = np.linalg.norm(g1)
        norm_g2 = np.linalg.norm(g2)
        norms = np.array([norm_g1, norm_g2])
        mean_norm = np.mean(norms)
        
        target_norms = mean_norm * (inv_rates**alpha_grad)
        task_weights = np.clip(task_weights - 0.05 * (task_weights * norms - target_norms), 0.1, 5.0)
        task_weights = 2.0 * task_weights / np.sum(task_weights)
        
        joint_g = task_weights[0] * g1 + task_weights[1] * g2
        theta = model.get_flat_params() - lr * joint_g
        model.set_flat_params(theta)
        
    elapsed = time.perf_counter() - t0
    return {
        "method": "GradNorm (Dynamic Balancing)",
        "final_l1": float(l1_hist[-1]),
        "final_l2": float(l2_hist[-1]),
        "final_total_loss": float(l1_hist[-1] + l2_hist[-1]),
        "final_task_weights": task_weights.tolist(),
        "l1_history": l1_hist,
        "l2_history": l2_hist,
        "runtime_ms": elapsed * 1000.0
    }


def run_multitask_pareto_benchmarks(n_seeds: int = 15) -> Dict:
    """Evaluates Equal Weighting vs MGDA vs GradNorm over 15 seeds."""
    methods = ["equal_weighting", "mgda", "gradnorm"]
    results = {m: {"l1": [], "l2": [], "total": [], "runtime": []} for m in methods}
    
    for s in range(n_seeds):
        rng = np.random.default_rng(700 + s * 17)
        N = 100
        D = 6
        X = rng.normal(0.0, 1.0, size=(N, D))
        # Conflicting regression targets
        y1 = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.1 * rng.normal(0, 1, size=N)
        y2 = -2.5 * X[:, 0] + 3.0 * X[:, 2] + 0.1 * rng.normal(0, 1, size=N)
        
        model_init = TwoTaskModel(in_dim=D, hidden_dim=4, seed=42 + s)
        theta_0 = model_init.get_flat_params()
        
        # 1. Equal
        m_eq = TwoTaskModel(in_dim=D, hidden_dim=4, seed=42 + s)
        m_eq.set_flat_params(theta_0.copy())
        r_eq = train_equal_weighting(m_eq, X, y1, y2, iters=50, lr=0.03)
        results["equal_weighting"]["l1"].append(r_eq["final_l1"])
        results["equal_weighting"]["l2"].append(r_eq["final_l2"])
        results["equal_weighting"]["total"].append(r_eq["final_total_loss"])
        results["equal_weighting"]["runtime"].append(r_eq["runtime_ms"])
        
        # 2. MGDA
        m_mgda = TwoTaskModel(in_dim=D, hidden_dim=4, seed=42 + s)
        m_mgda.set_flat_params(theta_0.copy())
        r_mgda = train_mgda(m_mgda, X, y1, y2, iters=50, lr=0.03)
        results["mgda"]["l1"].append(r_mgda["final_l1"])
        results["mgda"]["l2"].append(r_mgda["final_l2"])
        results["mgda"]["total"].append(r_mgda["final_total_loss"])
        results["mgda"]["runtime"].append(r_mgda["runtime_ms"])
        
        # 3. GradNorm
        m_gn = TwoTaskModel(in_dim=D, hidden_dim=4, seed=42 + s)
        m_gn.set_flat_params(theta_0.copy())
        r_gn = train_gradnorm(m_gn, X, y1, y2, iters=50, lr=0.03)
        results["gradnorm"]["l1"].append(r_gn["final_l1"])
        results["gradnorm"]["l2"].append(r_gn["final_l2"])
        results["gradnorm"]["total"].append(r_gn["final_total_loss"])
        results["gradnorm"]["runtime"].append(r_gn["runtime_ms"])
        
    summary = {}
    for m in methods:
        summary[m] = {
            "mean_l1": float(np.mean(results[m]["l1"])),
            "std_l1": float(np.std(results[m]["l1"])),
            "mean_l2": float(np.mean(results[m]["l2"])),
            "std_l2": float(np.std(results[m]["l2"])),
            "mean_total_loss": float(np.mean(results[m]["total"])),
            "std_total_loss": float(np.std(results[m]["total"])),
            "mean_runtime_ms": float(np.mean(results[m]["runtime"])),
            "std_runtime_ms": float(np.std(results[m]["runtime"])),
            "all_l1": results[m]["l1"],
            "all_l2": results[m]["l2"]
        }
    return summary
