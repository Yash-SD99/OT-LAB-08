"""Comprehensive Benchmarking Suite for Track D: Submodular, Parallel, and Distributed Optimization.

Generates reproducible CSV/JSON benchmark data and visualization figures:
1. Submodular Greedy vs Exact Optimal and Nemhauser-Wolsey Bound (1 - 1/e)
2. Parallel Speedup, Efficiency, and Amdahl/Gustafson Law Fitting (P in [1, 2, 4])
3. Work Stealing vs Static Scheduling under Heavy-Tailed Load Imbalance
4. Federated Learning (FedAvg with E in {1, 5, 10} vs Centralized SGD)
5. Multi-Task Optimization (Equal Weighting vs MGDA vs GradNorm)
"""

import json
import os
import time
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tracks.D_submodular_parallel_distributed.submodular_opt import (
    FacilityLocationSubmodular,
    verify_diminishing_returns,
    greedy_submodular_selection,
    exact_optimal_selection
)
from tracks.D_submodular_parallel_distributed.parallel_scheduling import (
    StaticScheduler,
    WorkStealingSimulator,
    fit_amdahl_gustafson
)
from tracks.D_submodular_parallel_distributed.federated_fedavg import (
    DistributedDataset,
    run_centralized_training,
    run_fedavg
)
from tracks.D_submodular_parallel_distributed.multitask_mgda_gradnorm import (
    TwoTaskModel,
    train_equal_weighting,
    train_mgda,
    train_gradnorm
)


def run_submodular_benchmarks(results_dir: str) -> Dict:
    """Benchmark 1: Submodular Greedy vs Exact Optimal across budget k."""
    print("Running Benchmark 1: Submodular Greedy vs Exact Optimal...")
    np.random.seed(42)
    N_cands = 14
    N_targets = 25
    candidates = np.random.uniform(-3, 3, size=(N_cands, 2))
    targets = np.random.uniform(-3, 3, size=(N_targets, 2))
    obj = FacilityLocationSubmodular(candidates, targets, sigma=1.5)
    
    audit = verify_diminishing_returns(obj, n_pairs_to_test=100, seed=42)
    
    budgets = [1, 2, 3, 4, 5]
    greedy_vals = []
    exact_vals = []
    ratios = []
    
    for k in budgets:
        res_g = greedy_submodular_selection(obj, k)
        res_e = exact_optimal_selection(obj, k)
        greedy_vals.append(res_g["final_value"])
        exact_vals.append(res_e["final_value"])
        ratios.append(float(res_g["final_value"] / res_e["final_value"]))
        
    # Plot Submodular Guarantee
    plt.figure(figsize=(8, 5))
    plt.plot(budgets, exact_vals, 's-', label="Exact Combinatorial Optimal f(S*)", lw=2, color="#1f77b4")
    plt.plot(budgets, greedy_vals, 'o-', label="Greedy Selection f(S_greedy)", lw=2, color="#2ca02c")
    theoretical_bound = [v * (1.0 - 1.0 / np.e) for v in exact_vals]
    plt.plot(budgets, theoretical_bound, 'r--', label="Nemhauser Bound (1 - 1/e) * f(S*)", lw=2)
    plt.xlabel("Budget / Selected Facilities (k)")
    plt.ylabel("Submodular Objective Value f(S)")
    plt.title("Submodular Maximization: Greedy vs Exact vs (1 - 1/e) Bound")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "submodular_guarantee.png"), dpi=200)
    plt.close()
    
    return {
        "diminishing_returns_audit": audit,
        "budgets": budgets,
        "greedy_values": greedy_vals,
        "exact_values": exact_vals,
        "approximation_ratios": ratios,
        "nemhauser_theoretical_lower_bound": float(1.0 - 1.0 / np.e)
    }


def run_parallel_scaling_benchmarks(results_dir: str) -> Dict:
    """Benchmark 2 & 3: Speedup, Amdahl's Law, and Work Stealing under load imbalance."""
    print("Running Benchmark 2 & 3: Parallel Speedup & Work Stealing...")
    # Heavy-tailed workload (16 tasks with power-law intensities)
    rng = np.random.default_rng(42)
    intensities = list(rng.pareto(a=1.5, size=24) + 0.2)
    
    p_workers = [1, 2, 4]
    static_runtimes = []
    stealing_runtimes = []
    static_imbalances = []
    stealing_imbalances = []
    
    for p in p_workers:
        # Static
        res_s = StaticScheduler(n_workers=p).run(intensities)
        static_runtimes.append(res_s["makespan_ms"])
        static_imbalances.append(res_s["load_imbalance_std_ms"])
        
        # Work Stealing
        res_ws = WorkStealingSimulator(n_workers=p).run(intensities, seed=42)
        stealing_runtimes.append(res_ws["makespan_ms"])
        stealing_imbalances.append(res_ws["load_imbalance_std_ms"])
        
    amdahl_fit = fit_amdahl_gustafson(p_workers, static_runtimes)
    
    # Plot Speedup & Amdahl Fit
    plt.figure(figsize=(8, 5))
    plt.plot(p_workers, amdahl_fit["measured_speedups"], 'bo-', label="Measured Speedup S(P)", lw=2)
    plt.plot(p_workers, amdahl_fit["amdahl_speedup_fit"], 'r--', label=f"Amdahl's Fit (serial fraction={amdahl_fit['fitted_serial_fraction']:.2f})", lw=2)
    plt.plot(p_workers, p_workers, 'k:', label="Ideal Linear Speedup S(P) = P", alpha=0.6)
    plt.xlabel("Worker Count (P)")
    plt.ylabel("Speedup S(P)")
    plt.xticks(p_workers)
    plt.title("Parallel Strong Scaling & Amdahl's Law Analysis")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "parallel_speedup_amdahl.png"), dpi=200)
    plt.close()
    
    return {
        "p_workers": p_workers,
        "static_runtimes_ms": static_runtimes,
        "work_stealing_runtimes_ms": stealing_runtimes,
        "static_imbalance_std_ms": static_imbalances,
        "work_stealing_imbalance_std_ms": stealing_imbalances,
        "amdahl_analysis": amdahl_fit
    }


def run_federated_benchmarks(results_dir: str) -> Dict:
    """Benchmark 4: FedAvg vs Centralized Optimization across local epochs E in {1, 5, 10}."""
    print("Running Benchmark 4: Federated Learning (FedAvg vs Centralized)...")
    dataset = DistributedDataset(n_clients=10, n_features=8, samples_per_client=50, seed=42)
    
    res_central = run_centralized_training(dataset, max_epochs=40, lr=0.05)
    res_fed_1 = run_fedavg(dataset, n_rounds=25, local_epochs=1, lr=0.05)
    res_fed_5 = run_fedavg(dataset, n_rounds=25, local_epochs=5, lr=0.05)
    res_fed_10 = run_fedavg(dataset, n_rounds=25, local_epochs=10, lr=0.05)
    
    # Plot FedAvg Convergence
    plt.figure(figsize=(8, 5))
    plt.plot(res_fed_1["loss_history"], label="FedAvg (E=1 local step)", lw=2)
    plt.plot(res_fed_5["loss_history"], label="FedAvg (E=5 local steps)", lw=2)
    plt.plot(res_fed_10["loss_history"], label="FedAvg (E=10 local steps)", lw=2)
    plt.axhline(res_central["final_loss"], color='k', ls='--', label=f"Centralized SGD (loss={res_central['final_loss']:.3f})")
    plt.xlabel("Communication Round")
    plt.ylabel("Global Validation Loss (MSE)")
    plt.title("Federated Learning: Convergence vs Local Epochs E")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fedavg_convergence.png"), dpi=200)
    plt.close()
    
    return {
        "centralized": res_central,
        "fedavg_E1": res_fed_1,
        "fedavg_E5": res_fed_5,
        "fedavg_E10": res_fed_10
    }


def run_multitask_benchmarks(results_dir: str) -> Dict:
    """Benchmark 5: Equal Task Weighting vs MGDA vs GradNorm on conflicting multi-task objective."""
    print("Running Benchmark 5: Multi-Task Optimization (MGDA vs GradNorm)...")
    np.random.seed(42)
    N = 100
    D = 6
    X = np.random.randn(N, D)
    # Conflicting task directions
    y1 = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.1 * np.random.randn(N)
    y2 = -2.5 * X[:, 0] + 3.0 * X[:, 2] + 0.1 * np.random.randn(N)
    
    # Run from identical initial weights
    model_init = TwoTaskModel(in_dim=D, hidden_dim=4, seed=42)
    theta_0 = model_init.get_flat_params()
    
    # 1. Equal
    m1 = TwoTaskModel(in_dim=D, hidden_dim=4, seed=42)
    m1.set_flat_params(theta_0.copy())
    res_eq = train_equal_weighting(m1, X, y1, y2, iters=50, lr=0.03)
    
    # 2. MGDA
    m2 = TwoTaskModel(in_dim=D, hidden_dim=4, seed=42)
    m2.set_flat_params(theta_0.copy())
    res_mgda = train_mgda(m2, X, y1, y2, iters=50, lr=0.03)
    
    # 3. GradNorm
    m3 = TwoTaskModel(in_dim=D, hidden_dim=4, seed=42)
    m3.set_flat_params(theta_0.copy())
    res_gradnorm = train_gradnorm(m3, X, y1, y2, iters=50, lr=0.03)
    
    # Plot Multi-Task Total Loss
    plt.figure(figsize=(8, 5))
    total_eq = [a + b for a, b in zip(res_eq["l1_history"], res_eq["l2_history"])]
    total_mgda = [a + b for a, b in zip(res_mgda["l1_history"], res_mgda["l2_history"])]
    total_gn = [a + b for a, b in zip(res_gradnorm["l1_history"], res_gradnorm["l2_history"])]
    
    plt.plot(total_eq, label=f"Equal Weighting (final={res_eq['final_total_loss']:.3f})", lw=2)
    plt.plot(total_mgda, label=f"MGDA Pareto (final={res_mgda['final_total_loss']:.3f})", lw=2)
    plt.plot(total_gn, label=f"GradNorm Dynamic (final={res_gradnorm['final_total_loss']:.3f})", lw=2)
    plt.xlabel("Iteration")
    plt.ylabel("Total Multi-Task Loss (L1 + L2)")
    plt.title("Multi-Task Optimization: Equal Weighting vs MGDA vs GradNorm")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "multitask_comparison.png"), dpi=200)
    plt.close()
    
    return {
        "equal_weighting": res_eq,
        "mgda": res_mgda,
        "gradnorm": res_gradnorm
    }


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    all_results["benchmark_1_submodular"] = run_submodular_benchmarks(results_dir)
    all_results["benchmark_2_3_parallel"] = run_parallel_scaling_benchmarks(results_dir)
    all_results["benchmark_4_federated"] = run_federated_benchmarks(results_dir)
    all_results["benchmark_5_multitask"] = run_multitask_benchmarks(results_dir)
    
    output_path = os.path.join(results_dir, "parallel_dist_benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\nAll Track D benchmarks completed successfully!")
    print(f"Results and plots saved to: {results_dir}")


if __name__ == "__main__":
    main()
