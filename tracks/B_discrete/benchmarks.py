"""Comprehensive Benchmarking Suite for Track B: Discrete Optimization.

Generates reproducible CSV/JSON benchmark data and visualization figures:
1. Runtime & Optimality Scaling: Greedy vs DP vs Custom B&B vs scipy.optimize.milp
2. Combinatorial Feature Selection: Exhaustive vs Greedy Forward Selection
3. Max-Cut QUBO: Exact Enumeration vs Simulated Annealing (Classical Baseline for Track C)
4. QUBO Constraint Penalty Multiplier Sensitivity (Feasibility Trade-Off)
"""

import json
import os
import time
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tracks.B_discrete.knapsack_solvers import (
    KnapsackInstance,
    solve_greedy,
    solve_dynamic_programming,
    solve_branch_and_bound,
    solve_ilp_scipy
)
from tracks.B_discrete.feature_selection import (
    FeatureSelectionProblem,
    exact_best_subset_selection,
    forward_stepwise_selection
)
from tracks.B_discrete.qubo_formulation import (
    build_maxcut_qubo,
    build_constrained_knapsack_qubo,
    qubo_energy
)
from tracks.B_discrete.classical_qubo_solver import (
    solve_qubo_exact,
    solve_qubo_simulated_annealing,
    analyze_minor_embedding_limits
)


def run_knapsack_scaling_benchmarks(results_dir: str) -> Dict:
    """Benchmark 1: Compares Greedy, DP, Custom B&B, and scipy.optimize.milp across N."""
    print("Running Benchmark 1: Knapsack Solvers Scaling...")
    sizes = [10, 15, 20, 25, 30, 40]
    
    runtimes = {
        "Greedy": [],
        "Dynamic Programming": [],
        "Custom Branch-and-Bound": [],
        "ILP (scipy.optimize.milp)": []
    }
    objectives = {k: [] for k in runtimes.keys()}
    pruning_stats = {"explored": [], "pruned": []}
    
    for n in sizes:
        # Deterministic synthetic instance
        rng = np.random.default_rng(42 + n)
        values = rng.integers(10, 100, size=n).astype(float)
        weights = rng.integers(5, 50, size=n).astype(float)
        capacity = float(np.sum(weights) * 0.4) # 40% capacity budget
        
        inst = KnapsackInstance(values, weights, capacity)
        
        # Greedy
        res_g = solve_greedy(inst)
        runtimes["Greedy"].append(res_g["runtime_ms"])
        objectives["Greedy"].append(res_g["objective"])
        
        # DP
        res_dp = solve_dynamic_programming(inst)
        runtimes["Dynamic Programming"].append(res_dp["runtime_ms"])
        objectives["Dynamic Programming"].append(res_dp["objective"])
        
        # Custom B&B
        res_bnb = solve_branch_and_bound(inst)
        runtimes["Custom Branch-and-Bound"].append(res_bnb["runtime_ms"])
        objectives["Custom Branch-and-Bound"].append(res_bnb["objective"])
        pruning_stats["explored"].append(res_bnb["nodes_explored"])
        pruning_stats["pruned"].append(res_bnb["nodes_pruned"])
        
        # SciPy MILP
        res_milp = solve_ilp_scipy(inst)
        runtimes["ILP (scipy.optimize.milp)"].append(res_milp["runtime_ms"])
        objectives["ILP (scipy.optimize.milp)"].append(res_milp["objective"])
        
    # Plot 1: Runtime Scaling
    plt.figure(figsize=(8, 5))
    for method, times in runtimes.items():
        plt.plot(sizes, times, 'o-', label=method, lw=2)
    plt.yscale("log")
    plt.title("Knapsack Solvers: Runtime Scaling vs Problem Size N")
    plt.xlabel("Number of Items N")
    plt.ylabel("Execution Time (ms, log scale)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "knapsack_runtime_scaling.png"), dpi=200)
    plt.close()
    
    # Plot 2: B&B Search Tree Pruning Efficiency
    plt.figure(figsize=(8, 5))
    total_nodes = [e + p for e, p in zip(pruning_stats["explored"], pruning_stats["pruned"])]
    plt.plot(sizes, pruning_stats["explored"], 's-', label="Nodes Explored (Visited)", color="#1f77b4", lw=2)
    plt.plot(sizes, pruning_stats["pruned"], '^-', label="Subtrees Pruned (Bound <= Best)", color="#d62728", lw=2)
    plt.plot(sizes, [2**min(n, 20) for n in sizes], 'k--', label="Unpruned Tree (2^N capped)", alpha=0.6)
    plt.yscale("log")
    plt.title("Custom Branch-and-Bound: Search Tree Pruning Efficiency")
    plt.xlabel("Number of Items N")
    plt.ylabel("Count of Nodes (log scale)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "bnb_pruning_efficiency.png"), dpi=200)
    plt.close()
    
    return {
        "sizes": sizes,
        "runtimes_ms": runtimes,
        "objectives": objectives,
        "pruning_stats": pruning_stats
    }


def run_feature_selection_benchmarks(results_dir: str) -> Dict:
    """Benchmark 2: Combinatorial Feature Selection (Exhaustive vs Forward Greedy)."""
    print("Running Benchmark 2: Combinatorial Feature Selection...")
    np.random.seed(42)
    N_samples = 80
    D_features = 10
    K_target = 3
    
    X = np.random.randn(N_samples, D_features)
    # Ground truth model: only features 1, 4, 7 are predictive
    true_beta = np.zeros(D_features)
    true_beta[[1, 4, 7]] = [3.5, -2.8, 1.9]
    y = X @ true_beta + 0.15 * np.random.randn(N_samples)
    
    problem = FeatureSelectionProblem(X, y, max_features=K_target)
    
    res_exact = exact_best_subset_selection(problem)
    res_greedy = forward_stepwise_selection(problem)
    
    return {
        "dataset_shape": [N_samples, D_features],
        "true_features": [1, 4, 7],
        "target_cardinality": K_target,
        "exact_selection": res_exact,
        "forward_greedy": res_greedy,
        "runtime_speedup": float(res_exact["runtime_ms"] / max(res_greedy["runtime_ms"], 1e-4)),
        "evaluations_ratio": float(res_exact["subsets_evaluated"] / max(res_greedy["subsets_evaluated"], 1))
    }


def run_maxcut_qubo_baseline(results_dir: str) -> Dict:
    """Benchmark 3: Max-Cut QUBO Classical Solvers (Exact vs Simulated Annealing).
    Serves as the classical benchmark for QAOA in Track C.
    """
    print("Running Benchmark 3: Max-Cut QUBO Classical Baseline...")
    # Graph sizes: 6, 8, 10 nodes (unweighted regular and random graphs)
    nodes = [6, 8, 10]
    results = {}
    
    for n in nodes:
        rng = np.random.default_rng(100 + n)
        # Create symmetric random graph adjacency matrix with ~40% density
        W = rng.binomial(1, 0.4, size=(n, n)).astype(float)
        W = np.triu(W, 1)
        W = W + W.T
        
        Q, J, h, C = build_maxcut_qubo(W)
        
        # Exact classical enumeration
        res_exact = solve_qubo_exact(Q)
        best_x_exact = np.array(res_exact["best_state"])
        max_cut_exact = 0.5 * float(np.sum(W * (best_x_exact[:, None] != best_x_exact[None, :])))
        
        # Classical Simulated Annealing (10 runs)
        sa_energies = []
        sa_cuts = []
        sa_times = []
        for seed in range(10):
            res_sa = solve_qubo_simulated_annealing(Q, initial_temp=15.0, seed=seed)
            best_x_sa = np.array(res_sa["best_state"])
            cut_sa = 0.5 * float(np.sum(W * (best_x_sa[:, None] != best_x_sa[None, :])))
            sa_energies.append(res_sa["min_energy"])
            sa_cuts.append(cut_sa)
            sa_times.append(res_sa["runtime_ms"])
            
        results[f"{n}_nodes"] = {
            "num_nodes": n,
            "edges": int(np.sum(W) / 2),
            "exact_max_cut": max_cut_exact,
            "exact_min_energy": res_exact["min_energy"],
            "exact_runtime_ms": res_exact["runtime_ms"],
            "sa_mean_cut": float(np.mean(sa_cuts)),
            "sa_std_cut": float(np.std(sa_cuts)),
            "sa_approximation_ratio": float(np.mean(sa_cuts) / max(max_cut_exact, 1e-9)),
            "sa_mean_runtime_ms": float(np.mean(sa_times))
        }
        
    return results


def run_qubo_penalty_tradeoff(results_dir: str) -> Dict:
    """Benchmark 4: Feasibility vs Objective Trade-Off across Penalty Multiplier lambda."""
    print("Running Benchmark 4: QUBO Constraint Penalty Multiplier Trade-Off...")
    values = np.array([40.0, 50.0, 70.0, 30.0, 60.0])
    weights = np.array([10.0, 15.0, 25.0, 8.0, 20.0])
    target_weight = 35.0
    
    lambdas = [0.01, 0.1, 0.5, 1.0, 2.5, 5.0, 15.0, 50.0]
    feasibility_results = []
    
    for lam in lambdas:
        Q, offset = build_constrained_knapsack_qubo(values, weights, target_weight, lam)
        res_exact = solve_qubo_exact(Q)
        x_opt = np.array(res_exact["best_state"])
        actual_wt = float(np.sum(weights * x_opt))
        actual_val = float(np.sum(values * x_opt))
        constraint_violation = abs(actual_wt - target_weight)
        
        feasibility_results.append({
            "lambda": lam,
            "selected_state": x_opt.tolist(),
            "actual_weight": actual_wt,
            "target_weight": target_weight,
            "actual_value": actual_val,
            "constraint_violation": constraint_violation,
            "is_feasible": bool(constraint_violation < 1e-4)
        })
        
    # Plot trade-off
    lams = [r["lambda"] for r in feasibility_results]
    violations = [r["constraint_violation"] for r in feasibility_results]
    vals = [r["actual_value"] for r in feasibility_results]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.semilogx(lams, violations, 'ro-', lw=2, label="Weight Violation |sum(w*x) - W_target|")
    ax1.set_xlabel("Penalty Multiplier (lambda, log scale)")
    ax1.set_ylabel("Constraint Violation", color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.grid(True, ls="--", alpha=0.5)
    
    ax2 = ax1.twinx()
    ax2.semilogx(lams, vals, 'bs-', lw=2, label="Pack Value sum(v*x)")
    ax2.set_ylabel("Pack Value", color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    plt.title("QUBO Constraint Enforcement: Penalty Multiplier Sensitivity")
    fig.tight_layout()
    plt.savefig(os.path.join(results_dir, "qubo_penalty_tradeoff.png"), dpi=200)
    plt.close()
    
    return {"penalty_sweep": feasibility_results}


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    all_results["benchmark_1_knapsack_scaling"] = run_knapsack_scaling_benchmarks(results_dir)
    all_results["benchmark_2_feature_selection"] = run_feature_selection_benchmarks(results_dir)
    all_results["benchmark_3_maxcut_qubo_baseline"] = run_maxcut_qubo_baseline(results_dir)
    all_results["benchmark_4_penalty_tradeoff"] = run_qubo_penalty_tradeoff(results_dir)
    all_results["minor_embedding_limits"] = analyze_minor_embedding_limits(n_logical_qubits=8)
    
    output_path = os.path.join(results_dir, "discrete_benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\nAll Track B benchmarks completed successfully!")
    print(f"Results and plots saved to: {results_dir}")


if __name__ == "__main__":
    main()
