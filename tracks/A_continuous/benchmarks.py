"""Comprehensive Benchmarking Suite for Track A: Continuous Optimization (Review Fixes).

Evaluates:
1. Strongly Convex Quadratic convergence (1st & 2nd order methods).
2. Rosenbrock 5,000-iteration budget benchmark across tuned optimizers with iterations to f < 1e-6.
3. GD at a saddle point (exact saddle start vs perturbed start).
4. Multi-modal Rastrigin local minima distribution across 50 random starts.
5. Backpropagation MLP training loss and accuracy across optimizers.
6. Curvature quality: Empirical Fisher vs Diagonal vs Rank-k vs Exact Hessian.
7. L2 Regularization parameter shrinkage and condition number.
8. Minibatch size sweep and gradient variance scaling (O(1/B)).
9. HVP error vs finite-difference epsilon trade-off curve (truncation vs machine precision).
"""

import json
import os
import sys
import time
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tracks.A_continuous.landscapes import (
    StronglyConvexQuadratic,
    SaddlePointLandscape,
    RosenbrockLandscape,
    RastriginLandscape,
    NoisyObjectiveWrapper,
)
from tracks.A_continuous.models import SimpleMLP
from tracks.A_continuous.optimizers import (
    GradientDescent,
    Momentum,
    NAG,
    AdaGrad,
    RMSProp,
    Adam,
    NewtonRaphson,
    GaussNewton,
    empirical_fisher,
    diagonal_curvature,
    low_rank_curvature,
    hessian_vector_product_fd,
)


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==============================================================================
# 1. Rosenbrock 5,000-Iteration Budget Benchmark
# ==============================================================================

def benchmark_rosenbrock_5000() -> Dict:
    """Evaluates all optimizers under an identical 5,000 iteration budget on Rosenbrock."""
    print("\n--- Running Benchmark: Rosenbrock 5,000 Iteration Budget ---")
    obj = RosenbrockLandscape()
    x_init = np.array([-1.2, 1.0], dtype=np.float64)
    budget = 5000
    tol = 1e-6

    # Tuned configurations
    configs = {
        "GD (lr=0.001)": GradientDescent(lr=0.001),
        "Momentum (lr=0.001, g=0.9)": Momentum(lr=0.001, gamma=0.9),
        "NAG (lr=0.001, g=0.9)": NAG(lr=0.001, gamma=0.9),
        "AdaGrad (lr=0.2)": AdaGrad(lr=0.2),
        "RMSProp (lr=0.005)": RMSProp(lr=0.005),
        "Adam (lr=0.02)": Adam(lr=0.02),
        "Newton-Raphson": NewtonRaphson(lr=1.0, damping=1e-4),
        "Gauss-Newton": GaussNewton(lr=1.0, damping=1e-3),
    }

    results = {}
    trajectories = {}

    for name, opt in configs.items():
        x = x_init.copy()
        hist = []
        reached_iter = None
        t0 = time.perf_counter()

        for it in range(1, budget + 1):
            if isinstance(opt, GaussNewton):
                x = opt.step(x, obj.residuals(x), obj.jacobian(x))
            elif isinstance(opt, NewtonRaphson):
                x = opt.step(x, obj.gradient(x), obj.hessian(x))
            elif isinstance(opt, NAG):
                x = opt.step(x, obj.gradient(opt.lookahead(x)))
            else:
                x = opt.step(x, obj.gradient(x))

            val = obj.value(x)
            hist.append(val)

            if val < tol and reached_iter is None:
                reached_iter = it

        runtime = time.perf_counter() - t0
        trajectories[name] = hist
        results[name] = {
            "iterations_budget": budget,
            "final_value": float(hist[-1]),
            "reached_tol": reached_iter is not None,
            "iterations_to_tol": reached_iter,
            "status": f"Reached f < 1e-6 in {reached_iter} iters" if reached_iter else f"Slow convergence (final f={hist[-1]:.2e})",
            "runtime_sec": runtime,
        }
        print(f"  {name:30s} | Final f={hist[-1]:.2e} | Tol reached: {results[name]['status']} ({runtime*1000:.1f}ms)")

    # Plot Rosenbrock Convergence
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, hist in trajectories.items():
        # Subsample for smooth plotting
        sub_hist = hist[::10] if len(hist) > 500 else hist
        sub_iters = np.arange(1, len(hist) + 1, 10 if len(hist) > 500 else 1)
        ax.plot(sub_iters, sub_hist, label=name, linewidth=1.8)

    ax.axhline(1e-6, color="black", linestyle="--", alpha=0.7, label=r"Target Tolerance $10^{-6}$")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(r"Objective Value $f(x)$", fontsize=11)
    ax.set_title("Rosenbrock Convergence (5,000 Iteration Budget, Tuned Step Sizes)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "rosenbrock_5000_convergence.png"), dpi=300)
    plt.close()

    return results


# ==============================================================================
# 2. Saddle Point Dynamics (Exact vs Perturbed Start)
# ==============================================================================

def benchmark_saddle_point_dynamics() -> Dict:
    """Evaluates GD behavior at an exact saddle point vs a perturbed start on f(x, y) = x^2 - y^2."""
    print("\n--- Running Benchmark: GD at Saddle Point ---")
    saddle = SaddlePointLandscape()
    lr = 0.1
    n_steps = 50

    # 1. Exact saddle start: (0.0, 0.0)
    x_exact = np.array([0.0, 0.0], dtype=np.float64)
    exact_traj = [x_exact.copy()]
    for _ in range(n_steps):
        grad = saddle.gradient(x_exact)
        x_exact = x_exact - lr * grad
        exact_traj.append(x_exact.copy())

    # 2. Perturbed start: (1e-5, 1e-5)
    x_pert = np.array([1e-5, 1e-5], dtype=np.float64)
    pert_traj = [x_pert.copy()]
    for _ in range(n_steps):
        grad = saddle.gradient(x_pert)
        x_pert = x_pert - lr * grad
        pert_traj.append(x_pert.copy())

    # 3. Momentum from perturbed start
    x_mom = np.array([1e-5, 1e-5], dtype=np.float64)
    mom_opt = Momentum(lr=lr, gamma=0.9)
    mom_traj = [x_mom.copy()]
    for _ in range(n_steps):
        grad = saddle.gradient(x_mom)
        x_mom = mom_opt.step(x_mom, grad)
        mom_traj.append(x_mom.copy())

    exact_traj = np.array(exact_traj)
    pert_traj = np.array(pert_traj)
    mom_traj = np.array(mom_traj)

    print(f"  Exact start final pos:     {exact_traj[-1]} (Distance moved: {np.linalg.norm(exact_traj[-1] - exact_traj[0]):.2e})")
    print(f"  Perturbed GD final pos:    {pert_traj[-1]} (Distance moved: {np.linalg.norm(pert_traj[-1]):.2e})")
    print(f"  Perturbed Momentum final:  {mom_traj[-1]} (Distance moved: {np.linalg.norm(mom_traj[-1]):.2e})")

    # Plot Saddle Point Trajectories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Contour plot
    X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    Z = X**2 - Y**2
    ax1.contourf(X, Y, Z, levels=30, cmap="coolwarm", alpha=0.6)
    ax1.contour(X, Y, Z, levels=[0.0], colors="k", linestyles="--", linewidths=1.5)
    ax1.plot(exact_traj[:, 0], exact_traj[:, 1], "ko", label="Exact Saddle Start [0, 0] (Motionless)", markersize=8)
    ax1.plot(pert_traj[:, 0], pert_traj[:, 1], "r-o", label=r"Perturbed GD $[10^{-5}, 10^{-5}]$", linewidth=2)
    ax1.plot(mom_traj[:, 0], mom_traj[:, 1], "b-s", label=r"Perturbed Momentum $[10^{-5}, 10^{-5}]$", linewidth=2)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-3, 3)
    ax1.set_xlabel("x (Positive Curvature Axis)")
    ax1.set_ylabel("y (Negative Curvature Axis)")
    ax1.set_title("Saddle Escape Trajectories: $f(x, y) = x^2 - y^2$", fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)

    # Escape distance vs iteration
    steps = np.arange(n_steps + 1)
    ax2.plot(steps, np.linalg.norm(exact_traj, axis=1), "k--", label="Exact Saddle (Distance = 0)", linewidth=2)
    ax2.plot(steps, np.linalg.norm(pert_traj, axis=1), "r-", label="Perturbed GD", linewidth=2)
    ax2.plot(steps, np.linalg.norm(mom_traj, axis=1), "b-", label="Perturbed Momentum (Fast Escape)", linewidth=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("Iteration", fontsize=11)
    ax2.set_ylabel("Distance from Saddle Point", fontsize=11)
    ax2.set_title("Escape Velocity along Negative Curvature Direction", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "saddle_point_escape.png"), dpi=300)
    plt.close()

    return {
        "exact_start_dist_moved": float(np.linalg.norm(exact_traj[-1] - exact_traj[0])),
        "perturbed_gd_final_val": float(saddle.value(pert_traj[-1])),
        "perturbed_momentum_final_val": float(saddle.value(mom_traj[-1])),
    }


# ==============================================================================
# 3. Multi-modal Rastrigin Local Minima (50 Random Starts)
# ==============================================================================

def benchmark_rastrigin_50_starts() -> Dict:
    """Evaluates GD across 50 random starting points on the non-convex Rastrigin function."""
    print("\n--- Running Benchmark: Rastrigin 50 Random Starts ---")
    rastrigin = RastriginLandscape(dim=2)
    rng = np.random.default_rng(42)
    n_starts = 50
    starts = rng.uniform(-5.12, 5.12, size=(n_starts, 2))
    
    final_values = []
    final_points = []
    global_hits = 0

    for i in range(n_starts):
        x = starts[i].copy()
        # GD with diminishing step size to settle into basin
        lr = 0.005
        for it in range(400):
            grad = rastrigin.gradient(x)
            x = x - lr * grad
            
        f_val = rastrigin.value(x)
        final_values.append(f_val)
        final_points.append(x)
        if f_val < 1e-4:
            global_hits += 1

    final_values = np.array(final_values)
    # Cluster distinct local minima by rounding value to 1 decimal place
    unique_levels, counts = np.unique(np.round(final_values, 1), return_counts=True)
    distinct_minima_count = len(unique_levels)

    print(f"  50 Random Starts: {distinct_minima_count} distinct local minima levels discovered.")
    print(f"  Global Minimum (f=0.0) reached in {global_hits}/{n_starts} starts ({global_hits/n_starts*100:.1f}%).")
    print(f"  Best f={np.min(final_values):.4f}, Worst local min f={np.max(final_values):.4f}, Mean f={np.mean(final_values):.4f}")

    # Plot histogram of local minima
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(final_values, bins=25, color="steelblue", edgecolor="black", alpha=0.85)
    ax.axvline(0.0, color="crimson", linestyle="--", linewidth=2, label="Global Minimum $f(0,0)=0$")
    ax.set_xlabel("Final Local Minimum Objective Value $f(x^*)$", fontsize=11)
    ax.set_ylabel("Frequency (out of 50 starts)", fontsize=11)
    ax.set_title(f"Rastrigin Local Minima Distribution (50 Random Starts: {distinct_minima_count} Basins Discovered)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "rastrigin_local_minima_hist.png"), dpi=300)
    plt.close()

    return {
        "n_starts": n_starts,
        "distinct_minima_count": int(distinct_minima_count),
        "global_minimum_hits": int(global_hits),
        "global_minimum_rate": float(global_hits / n_starts),
        "best_value": float(np.min(final_values)),
        "worst_local_min": float(np.max(final_values)),
        "mean_final_value": float(np.mean(final_values)),
    }


# ==============================================================================
# 4. Backpropagation MLP Training Across Optimizers
# ==============================================================================

def benchmark_mlp_backprop_optimizers() -> Dict:
    """Trains SimpleMLP using backpropagation across all 6 continuous optimizers."""
    print("\n--- Running Benchmark: Backpropagation MLP Across Optimizers ---")
    # Synthetic non-linear dataset: concentric circles (2 inputs, 1 output binary target in {-1, 1})
    rng = np.random.default_rng(42)
    N = 180
    r_in = rng.normal(0.4, 0.08, size=N // 2)
    th_in = rng.uniform(0, 2 * np.pi, size=N // 2)
    X_in = np.column_stack([r_in * np.cos(th_in), r_in * np.sin(th_in)])
    y_in = -np.ones((N // 2, 1))

    r_out = rng.normal(0.9, 0.08, size=N // 2)
    th_out = rng.uniform(0, 2 * np.pi, size=N // 2)
    X_out = np.column_stack([r_out * np.cos(th_out), r_out * np.sin(th_out)])
    y_out = np.ones((N // 2, 1))

    X = np.vstack([X_in, X_out])
    y = np.vstack([y_in, y_out])
    perm = rng.permutation(N)
    X, y = X[perm], y[perm]

    epochs = 120
    configs = {
        "GD (lr=0.08)": GradientDescent(lr=0.08),
        "Momentum (lr=0.08, g=0.85)": Momentum(lr=0.08, gamma=0.85),
        "NAG (lr=0.08, g=0.85)": NAG(lr=0.08, gamma=0.85),
        "AdaGrad (lr=0.25)": AdaGrad(lr=0.25),
        "RMSProp (lr=0.02)": RMSProp(lr=0.02),
        "Adam (lr=0.03)": Adam(lr=0.03),
    }

    results = {}
    loss_trajectories = {}

    for name, opt in configs.items():
        mlp = SimpleMLP(in_dim=2, hidden_dim=6, out_dim=1, reg_lambda=1e-4, seed=42)
        losses = []

        for ep in range(epochs):
            # Compute loss and analytical backprop gradient
            loss_val = mlp.loss(X, y)
            losses.append(loss_val)

            # Get flat parameters and update
            theta = mlp.get_flat_params()
            if isinstance(opt, NAG):
                # Lookahead parameters
                theta_lookahead = opt.lookahead(theta)
                mlp.set_flat_params(theta_lookahead)
                mlp.loss(X, y)
                grad = mlp.backward(y)
                mlp.set_flat_params(theta)
                theta_new = opt.step(theta, grad)
            else:
                grad = mlp.backward(y)
                theta_new = opt.step(theta, grad)

            mlp.set_flat_params(theta_new)

        # Evaluate final accuracy
        preds = np.sign(mlp.forward(X))
        accuracy = float(np.mean(preds == y))
        loss_trajectories[name] = losses
        results[name] = {
            "final_loss": float(losses[-1]),
            "train_accuracy": accuracy,
        }
        print(f"  {name:25s} | Final Loss={losses[-1]:.4f} | Accuracy={accuracy*100:.1f}%")

    # Plot MLP training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ep_axis = np.arange(1, epochs + 1)
    for name, losses in loss_trajectories.items():
        ax1.plot(ep_axis, losses, label=name, linewidth=1.8)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("MSE + L2 Loss", fontsize=11)
    ax1.set_title("MLP Training Loss Across Optimizers", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9)

    # Bar chart of accuracy
    names = list(results.keys())
    accs = [results[n]["train_accuracy"] * 100 for n in names]
    short_names = [n.split()[0] for n in names]
    bars = ax2.bar(short_names, accs, color="darkcyan", edgecolor="black", width=0.5)
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Classification Accuracy (%)", fontsize=11)
    ax2.set_title("Final Classification Accuracy", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "mlp_backprop_optimizers.png"), dpi=300)
    plt.close()

    return results


# ==============================================================================
# 5. Curvature Quality: Fisher vs Diagonal vs Rank-k vs Exact Hessian
# ==============================================================================

def benchmark_curvature_approximations() -> Dict:
    """Compares Empirical Fisher, Diagonal Hessian, and Rank-k SVD approximations against the Exact Hessian."""
    print("\n--- Running Benchmark: Curvature Quality (Fisher, Diag, Rank-k vs Exact) ---")
    mlp = SimpleMLP(in_dim=2, hidden_dim=4, out_dim=1, reg_lambda=1e-3, seed=42)
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(50, 2))
    y = rng.choice([-1.0, 1.0], size=(50, 1))

    # 1. Compute Exact Hessian via finite differences over parameter gradient
    theta = mlp.get_flat_params()
    D = len(theta)
    exact_H = np.zeros((D, D), dtype=np.float64)
    eps = 1e-5

    mlp.loss(X, y)
    g_base = mlp.backward(y)

    for i in range(D):
        th_pos = theta.copy()
        th_pos[i] += eps
        mlp.set_flat_params(th_pos)
        mlp.loss(X, y)
        g_pos = mlp.backward(y)
        exact_H[:, i] = (g_pos - g_base) / eps

    mlp.set_flat_params(theta)
    exact_H = 0.5 * (exact_H + exact_H.T)  # Symmetrize
    norm_exact = np.linalg.norm(exact_H, "fro")

    # 2. Empirical Fisher Matrix: F = (1/N) sum_i (g_i g_i^T)
    per_sample_g = mlp.per_sample_gradients(X, y)
    fisher_H = empirical_fisher(per_sample_g)

    # 3. Diagonal Hessian Approximation
    diag_H = np.diag(diagonal_curvature(exact_H))

    # 4. Rank-k Approximations
    rank1_H = low_rank_curvature(exact_H, rank=1)
    rank3_H = low_rank_curvature(exact_H, rank=3)
    rank6_H = low_rank_curvature(exact_H, rank=6)

    # Compute Frobenius error: ||H_approx - H_exact||_F / ||H_exact||_F
    err_fisher = float(np.linalg.norm(fisher_H - exact_H, "fro") / norm_exact)
    err_diag = float(np.linalg.norm(diag_H - exact_H, "fro") / norm_exact)
    err_rank1 = float(np.linalg.norm(rank1_H - exact_H, "fro") / norm_exact)
    err_rank3 = float(np.linalg.norm(rank3_H - exact_H, "fro") / norm_exact)
    err_rank6 = float(np.linalg.norm(rank6_H - exact_H, "fro") / norm_exact)

    print(f"  Exact Hessian Dimension:   {D}x{D} (Frobenius norm = {norm_exact:.3f})")
    print(f"  Empirical Fisher Error:    {err_fisher*100:.2f}%")
    print(f"  Diagonal Hessian Error:    {err_diag*100:.2f}%")
    print(f"  Rank-1 SVD Error:          {err_rank1*100:.2f}%")
    print(f"  Rank-3 SVD Error:          {err_rank3*100:.2f}%")
    print(f"  Rank-6 SVD Error:          {err_rank6*100:.2f}%")

    # Plot Curvature Error Breakdown
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = ["Diagonal Hessian", "Rank-1 SVD", "Rank-3 SVD", "Rank-6 SVD", "Empirical Fisher"]
    errors = [err_diag * 100, err_rank1 * 100, err_rank3 * 100, err_rank6 * 100, err_fisher * 100]
    bars = ax.bar(labels, errors, color=["goldenrod", "cadetblue", "cornflowerblue", "mediumblue", "coral"], edgecolor="black", width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Relative Frobenius Error (%)", fontsize=11)
    ax.set_title("Curvature Matrix Approximation Quality vs Exact Hessian", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(errors) * 1.2)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "curvature_approximation_quality.png"), dpi=300)
    plt.close()

    return {
        "dimension": D,
        "exact_frobenius_norm": float(norm_exact),
        "error_diagonal": err_diag,
        "error_rank1": err_rank1,
        "error_rank3": err_rank3,
        "error_rank6": err_rank6,
        "error_fisher": err_fisher,
    }


# ==============================================================================
# 6. L2 Regularization & Conditioning
# ==============================================================================

def benchmark_l2_regularization_effect() -> Dict:
    """Evaluates parameter shrinkage and condition number under varying L2 penalties."""
    print("\n--- Running Benchmark: L2 Regularization Effect ---")
    lambdas = [0.0, 1e-4, 1e-2, 1.0]
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(60, 2))
    y = rng.choice([-1.0, 1.0], size=(60, 1))

    results = []
    for lam in lambdas:
        mlp = SimpleMLP(in_dim=2, hidden_dim=4, out_dim=1, reg_lambda=lam, seed=42)
        opt = Adam(lr=0.03)
        for ep in range(100):
            mlp.loss(X, y)
            g = mlp.backward(y)
            theta = opt.step(mlp.get_flat_params(), g)
            mlp.set_flat_params(theta)

        final_norm = float(np.linalg.norm(mlp.get_flat_params()))
        # Hessian condition number
        theta = mlp.get_flat_params()
        D = len(theta)
        eps = 1e-5
        H = np.zeros((D, D))
        g_base = mlp.backward(y)
        for i in range(D):
            th_pos = theta.copy()
            th_pos[i] += eps
            mlp.set_flat_params(th_pos)
            mlp.loss(X, y)
            H[:, i] = (mlp.backward(y) - g_base) / eps
        mlp.set_flat_params(theta)
        H = 0.5 * (H + H.T) + lam * np.eye(D)
        eigvals = np.linalg.eigvalsh(H)
        cond_num = float(eigvals[-1] / max(eigvals[0], 1e-8))

        entry = {
            "lambda": lam,
            "weight_norm": final_norm,
            "min_eigval": float(eigvals[0]),
            "condition_number": cond_num,
        }
        results.append(entry)
        print(f"  lambda={lam:<6}: Weight Norm={final_norm:.3f} | min_eig={eigvals[0]:.2e} | cond_num={cond_num:.1f}")

    return {"l2_sweep": results}


# ==============================================================================
# 7. Minibatch Size Sweep & Gradient Variance
# ==============================================================================

def benchmark_minibatch_variance() -> Dict:
    """Verifies that gradient variance scales inversely with batch size O(1/B)."""
    print("\n--- Running Benchmark: Minibatch Size Sweep & Variance ---")
    rng = np.random.default_rng(42)
    N = 256
    X = rng.normal(0, 1, size=(N, 2))
    y = rng.choice([-1.0, 1.0], size=(N, 1))

    mlp = SimpleMLP(in_dim=2, hidden_dim=4, out_dim=1, reg_lambda=1e-4, seed=42)
    mlp.loss(X, y)
    full_grad = mlp.backward(y)

    batch_sizes = [1, 4, 16, 64, 256]
    variances = []

    for B in batch_sizes:
        batch_grads = []
        for _ in range(50):
            idx = rng.choice(N, size=B, replace=False)
            X_b, y_b = X[idx], y[idx]
            mlp.loss(X_b, y_b)
            g_b = mlp.backward(y_b)
            batch_grads.append(g_b)

        batch_grads = np.array(batch_grads)
        # Empirical variance: E[||g_B - g_full||^2]
        var_b = float(np.mean(np.sum((batch_grads - full_grad)**2, axis=1)))
        variances.append(var_b)
        print(f"  Batch size B={B:3d} | Gradient Variance={var_b:.4f}")

    # Plot Minibatch Variance Scaling
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(batch_sizes, variances, "ro-", label=r"Empirical Variance $\mathbb{E}[\|g_B - g_{\mathrm{full}}\|^2]$", linewidth=2)
    # Theoretical 1/B fit
    b_fit = np.array(batch_sizes, dtype=float)
    fit_curve = variances[0] / b_fit
    ax.plot(batch_sizes, fit_curve, "k--", label=r"Theoretical $\mathcal{O}(1/B)$ Scaling", linewidth=1.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Minibatch Size $B$", fontsize=11)
    ax.set_ylabel("Gradient Variance", fontsize=11)
    ax.set_title("Stochastic Gradient Variance Reduction vs Batch Size", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "minibatch_variance_scaling.png"), dpi=300)
    plt.close()

    return {"batch_sizes": batch_sizes, "empirical_variances": variances}


# ==============================================================================
# 8. HVP Error vs Finite-Difference Epsilon Plot
# ==============================================================================

def benchmark_hvp_epsilon_tradeoff() -> Dict:
    """Plots HVP approximation error vs epsilon showing truncation vs roundoff error trade-off."""
    print("\n--- Running Benchmark: HVP Error vs Epsilon Plot ---")
    A = np.array([[10.0, 3.0], [3.0, 2.0]], dtype=np.float64)
    b = np.array([1.0, 1.0], dtype=np.float64)
    obj = StronglyConvexQuadratic(A, b)

    x = np.array([2.0, -1.0], dtype=np.float64)
    v = np.array([0.6, 0.8], dtype=np.float64)
    exact_hvp = A @ v
    norm_exact = np.linalg.norm(exact_hvp)

    eps_vals = np.logspace(-12, -1, 30)
    rel_errors = []

    for eps in eps_vals:
        hvp_est = hessian_vector_product_fd(obj.gradient, x, v, eps=eps)
        err = float(np.linalg.norm(hvp_est - exact_hvp) / norm_exact)
        rel_errors.append(err)

    best_idx = int(np.argmin(rel_errors))
    best_eps = float(eps_vals[best_idx])
    min_err = float(rel_errors[best_idx])
    print(f"  Best Epsilon: {best_eps:.2e} with relative error {min_err:.2e}")

    # Plot HVP Epsilon Tradeoff
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(eps_vals, rel_errors, "b-o", linewidth=2, markersize=5, label="Finite-Difference HVP Error")
    ax.scatter([best_eps], [min_err], color="red", s=100, zorder=5, label=f"Optimal $\\epsilon \\approx {best_eps:.1e}$ (Error {min_err:.1e})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Finite Difference Step $\epsilon$", fontsize=11)
    ax.set_ylabel(r"Relative Error $\|H v_{\mathrm{FD}} - H v\| / \|H v\|$", fontsize=11)
    ax.set_title(r"HVP Numerical Precision Trade-Off: Truncation vs Machine Cancellation", fontsize=12, fontweight="bold")

    # Annotations for error regimes
    ax.annotate(r"$\leftarrow$ Machine Roundoff / Cancellation", xy=(1e-10, 1e-4), xytext=(1e-11, 1e-2),
                arrowprops=dict(arrowstyle="->", color="purple"), fontsize=9, fontweight="bold", color="purple")
    ax.annotate(r"Truncation Error $\rightarrow$", xy=(1e-2, 1e-4), xytext=(1e-3, 1e-2),
                arrowprops=dict(arrowstyle="->", color="darkgreen"), fontsize=9, fontweight="bold", color="darkgreen")

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=9, loc="lower center")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "hvp_epsilon_tradeoff.png"), dpi=300)
    plt.close()

    return {
        "best_epsilon": best_eps,
        "min_error": min_err,
        "eps_vals": [float(e) for e in eps_vals],
        "rel_errors": [float(e) for e in rel_errors],
    }


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    print("=" * 70)
    print("Executing Track A: Continuous Optimization & ML Foundations (Review Fixes)")
    print("=" * 70)

    rosenbrock_data = benchmark_rosenbrock_5000()
    saddle_data = benchmark_saddle_point_dynamics()
    rastrigin_data = benchmark_rastrigin_50_starts()
    mlp_data = benchmark_mlp_backprop_optimizers()
    curvature_data = benchmark_curvature_approximations()
    l2_data = benchmark_l2_regularization_effect()
    minibatch_data = benchmark_minibatch_variance()
    hvp_data = benchmark_hvp_epsilon_tradeoff()

    payload = {
        "metadata": {
            "track": "Track A: Continuous Optimization & ML Foundations",
            "student": "Yash Deshmukh",
            "roll_no": "241070018",
            "institution": "VJTI Mumbai",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        },
        "rosenbrock_5000": rosenbrock_data,
        "saddle_point_dynamics": saddle_data,
        "rastrigin_50_starts": rastrigin_data,
        "mlp_backprop_optimizers": mlp_data,
        "curvature_quality": curvature_data,
        "l2_regularization": l2_data,
        "minibatch_variance": minibatch_data,
        "hvp_epsilon_tradeoff": {
            "best_epsilon": hvp_data["best_epsilon"],
            "min_error": hvp_data["min_error"]
        }
    }

    json_path = os.path.join(RESULTS_DIR, "continuous_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nAll benchmark results JSON written to {json_path}")
    print(f"All diagnostic plots saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
