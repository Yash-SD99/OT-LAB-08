"""Comprehensive Benchmarking Suite for Track A: Continuous Optimization.

Generates reproducible CSV/JSON benchmark data and visualization figures:
1. Convergence on Strongly Convex Quadratic (1st and 2nd order)
2. Convergence on Rosenbrock Valley (Gauss-Newton vs 1st order)
3. Saddle Point Trajectories (GD vs Momentum vs Adam)
4. Step-Size Sensitivity Grid Search
5. Noisy Objective Robustness (10 repeated runs, mean +/- std)
6. Hessian-Vector Product Runtime Scaling vs Full Hessian
"""

import json
import os
import time
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tracks.A_continuous.landscapes import (
    StronglyConvexQuadratic,
    SaddlePointLandscape,
    RosenbrockLandscape,
    NoisyObjectiveWrapper
)
from tracks.A_continuous.optimizers import (
    GradientDescent,
    Momentum,
    NAG,
    AdaGrad,
    RMSProp,
    Adam,
    NewtonRaphson,
    GaussNewton,
    hessian_vector_product_fd
)


def run_convex_quadratic_benchmarks(results_dir: str) -> Dict:
    """Benchmark 1: Convergence of all methods on ill-conditioned convex quadratic."""
    print("Running Benchmark 1: Strongly Convex Quadratic...")
    # Condition number ~25
    A = np.array([[20.0, 2.0], [2.0, 1.0]], dtype=np.float64)
    b = np.array([5.0, 2.0], dtype=np.float64)
    obj = StronglyConvexQuadratic(A, b)
    
    x_init = np.array([4.0, -3.0], dtype=np.float64)
    max_iters = 80
    
    configs = {
        "GD (lr=0.04)": GradientDescent(lr=0.04),
        "Momentum (lr=0.04, g=0.8)": Momentum(lr=0.04, gamma=0.8),
        "NAG (lr=0.04, g=0.8)": NAG(lr=0.04, gamma=0.8),
        "AdaGrad (lr=0.5)": AdaGrad(lr=0.5),
        "RMSProp (lr=0.1)": RMSProp(lr=0.1),
        "Adam (lr=0.2)": Adam(lr=0.2),
        "Newton-Raphson": NewtonRaphson(lr=1.0, damping=1e-5),
    }
    
    histories = {}
    metrics = {}
    
    for name, opt in configs.items():
        x = x_init.copy()
        loss_hist = []
        t0 = time.perf_counter()
        
        for it in range(max_iters):
            loss = obj.value(x) - obj.f_opt
            loss_hist.append(max(float(loss), 1e-16))
            
            if isinstance(opt, NewtonRaphson):
                g = obj.gradient(x)
                H = obj.hessian(x)
                x = opt.step(x, g, H)
            elif isinstance(opt, NAG):
                x_look = opt.lookahead(x)
                g_look = obj.gradient(x_look)
                x = opt.step(x, g_look)
            else:
                g = obj.gradient(x)
                x = opt.step(x, g)
                
        elapsed = time.perf_counter() - t0
        histories[name] = loss_hist
        metrics[name] = {
            "final_gap": loss_hist[-1],
            "runtime_ms": elapsed * 1000.0,
            "iterations": max_iters
        }
        
    # Plot convergence
    plt.figure(figsize=(8, 5))
    for name, hist in histories.items():
        plt.semilogy(hist, label=name, lw=2)
    plt.title("Convergence on Strongly Convex Quadratic (f(x) - f*)")
    plt.xlabel("Iteration")
    plt.ylabel("Optimality Gap log10(f - f*)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "convergence_convex.png"), dpi=200)
    plt.close()
    
    return metrics


def run_rosenbrock_benchmarks(results_dir: str) -> Dict:
    """Benchmark 2: Non-convex Rosenbrock Valley (Gauss-Newton vs First-Order)."""
    print("Running Benchmark 2: Rosenbrock Banana Valley...")
    obj = RosenbrockLandscape()
    x_init = np.array([-1.2, 1.0], dtype=np.float64)
    max_iters = 100
    
    configs = {
        "GD (lr=0.002)": GradientDescent(lr=0.002),
        "Momentum (lr=0.002, g=0.9)": Momentum(lr=0.002, gamma=0.9),
        "Adam (lr=0.05)": Adam(lr=0.05),
        "Gauss-Newton (damped)": GaussNewton(lr=1.0, damping=1e-2),
        "Newton-Raphson": NewtonRaphson(lr=1.0, damping=1e-3),
    }
    
    histories = {}
    metrics = {}
    
    for name, opt in configs.items():
        x = x_init.copy()
        loss_hist = []
        t0 = time.perf_counter()
        
        for it in range(max_iters):
            loss = obj.value(x)
            loss_hist.append(float(loss))
            
            if isinstance(opt, GaussNewton):
                r = obj.residuals(x)
                J = obj.jacobian(x)
                x = opt.step(x, r, J)
            elif isinstance(opt, NewtonRaphson):
                g = obj.gradient(x)
                H = obj.hessian(x)
                x = opt.step(x, g, H)
            else:
                g = obj.gradient(x)
                x = opt.step(x, g)
                
        elapsed = time.perf_counter() - t0
        histories[name] = loss_hist
        metrics[name] = {
            "final_loss": loss_hist[-1],
            "final_x": [float(x[0]), float(x[1])],
            "runtime_ms": elapsed * 1000.0
        }
        
    plt.figure(figsize=(8, 5))
    for name, hist in histories.items():
        plt.semilogy(hist, label=name, lw=2)
    plt.title("Convergence on Non-Convex Rosenbrock Valley")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value f(x)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "rosenbrock_comparison.png"), dpi=200)
    plt.close()
    
    return metrics


def run_step_size_sensitivity(results_dir: str) -> Dict:
    """Benchmark 3: Sensitivity of algorithms to step size (learning rate)."""
    print("Running Benchmark 3: Step-Size Sensitivity...")
    A = np.array([[10.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    b = np.array([2.0, 1.0], dtype=np.float64)
    obj = StronglyConvexQuadratic(A, b)
    
    lrs = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.18, 0.25]
    algorithms = ["GD", "Momentum (g=0.9)", "Adam"]
    results = {algo: [] for algo in algorithms}
    
    for lr in lrs:
        # GD
        x = np.array([3.0, 3.0])
        opt = GradientDescent(lr=lr)
        for _ in range(50):
            g = obj.gradient(x)
            x = opt.step(x, g)
        val = obj.value(x) - obj.f_opt
        results["GD"].append(min(float(val), 1e6) if np.isfinite(val) else 1e6)
        
        # Momentum
        x = np.array([3.0, 3.0])
        opt = Momentum(lr=lr, gamma=0.9)
        for _ in range(50):
            g = obj.gradient(x)
            x = opt.step(x, g)
        val = obj.value(x) - obj.f_opt
        results["Momentum (g=0.9)"].append(min(float(val), 1e6) if np.isfinite(val) else 1e6)
        
        # Adam
        x = np.array([3.0, 3.0])
        opt = Adam(lr=lr)
        for _ in range(50):
            g = obj.gradient(x)
            x = opt.step(x, g)
        val = obj.value(x) - obj.f_opt
        results["Adam"].append(min(float(val), 1e6) if np.isfinite(val) else 1e6)
        
    plt.figure(figsize=(8, 5))
    for algo, losses in results.items():
        plt.loglog(lrs, losses, 'o-', label=algo, lw=2)
    plt.title("Step-Size Sensitivity (Final Gap after 50 iters)")
    plt.xlabel("Learning Rate (eta)")
    plt.ylabel("Final Optimality Gap f(x) - f*")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "step_size_sensitivity.png"), dpi=200)
    plt.close()
    
    return {"lrs": lrs, "sensitivity": results}


def run_noisy_stochastic_evaluation(results_dir: str) -> Dict:
    """Benchmark 4: Stochastic Noisy Objective (10 repeated runs, mean +/- std)."""
    print("Running Benchmark 4: Stochastic Noisy Objective (10 runs)...")
    A = np.array([[6.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, 1.0], dtype=np.float64)
    base_obj = StronglyConvexQuadratic(A, b)
    
    methods = {
        "SGD": lambda: GradientDescent(lr=0.03),
        "Momentum": lambda: Momentum(lr=0.03, gamma=0.8),
        "Adam": lambda: Adam(lr=0.1),
    }
    
    num_seeds = 10
    noisy_stats = {}
    
    for name, init_fn in methods.items():
        final_gaps = []
        for seed in range(num_seeds):
            noisy_obj = NoisyObjectiveWrapper(base_obj, val_noise_std=0.05, grad_noise_std=0.1, seed=100 + seed)
            opt = init_fn()
            x = np.array([4.0, -2.0], dtype=np.float64)
            for _ in range(80):
                g = noisy_obj.gradient(x)
                x = opt.step(x, g)
            # Evaluate true clean objective
            gap = base_obj.value(x) - base_obj.f_opt
            final_gaps.append(float(gap))
            
        noisy_stats[name] = {
            "mean_gap": float(np.mean(final_gaps)),
            "std_gap": float(np.std(final_gaps)),
            "min_gap": float(np.min(final_gaps)),
            "max_gap": float(np.max(final_gaps)),
            "raw_runs": final_gaps
        }
        
    return noisy_stats


def run_hvp_scaling_benchmark(results_dir: str) -> Dict:
    """Benchmark 5: Scalability of Hessian-Vector Products vs Full Hessian Matrix."""
    print("Running Benchmark 5: HVP vs Full Hessian Runtime Scaling...")
    dims = [5, 15, 30, 60, 100]
    hvp_times = []
    hess_times = []
    
    for d in dims:
        np.random.seed(42)
        M = np.random.randn(d, d)
        A = M.T @ M + 0.1 * np.eye(d)
        b = np.random.randn(d)
        obj = StronglyConvexQuadratic(A, b)
        
        x = np.ones(d)
        v = np.ones(d) / np.sqrt(d)
        
        # Time full Hessian computation & matrix-vector product
        t0 = time.perf_counter()
        for _ in range(100):
            H = obj.hessian(x)
            _ = H @ v
        hess_time = (time.perf_counter() - t0) / 100.0
        hess_times.append(hess_time * 1e6) # microseconds
        
        # Time finite-difference HVP (never forms full D x D matrix)
        t0 = time.perf_counter()
        for _ in range(100):
            _ = hessian_vector_product_fd(obj.gradient, x, v, eps=1e-5)
        hvp_time = (time.perf_counter() - t0) / 100.0
        hvp_times.append(hvp_time * 1e6) # microseconds
        
    plt.figure(figsize=(8, 5))
    plt.plot(dims, hess_times, 's-', label="Full Hessian Formation O(D^2)", lw=2)
    plt.plot(dims, hvp_times, 'o-', label="Finite-Difference HVP O(D)", lw=2)
    plt.title("Curvature Computation Scaling: HVP vs Full Hessian")
    plt.xlabel("Problem Dimension D")
    plt.ylabel("Time per Operation (microseconds)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "hvp_scaling.png"), dpi=200)
    plt.close()
    
    return {"dims": dims, "hess_times_us": hess_times, "hvp_times_us": hvp_times}


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    all_results["benchmark_1_convex"] = run_convex_quadratic_benchmarks(results_dir)
    all_results["benchmark_2_rosenbrock"] = run_rosenbrock_benchmarks(results_dir)
    all_results["benchmark_3_sensitivity"] = run_step_size_sensitivity(results_dir)
    all_results["benchmark_4_noisy_stochastic"] = run_noisy_stochastic_evaluation(results_dir)
    all_results["benchmark_5_hvp_scaling"] = run_hvp_scaling_benchmark(results_dir)
    
    output_path = os.path.join(results_dir, "continuous_benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\nAll Track A benchmarks completed successfully!")
    print(f"Results and plots saved to: {results_dir}")


if __name__ == "__main__":
    main()
