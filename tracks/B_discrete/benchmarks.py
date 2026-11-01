"""Comprehensive Benchmarking Suite for Track B: Discrete Optimization (Review Fixes).

Evaluates:
1. Knapsack Solvers Scaling on Uncorrelated and Hard (Strongly Correlated) Instances:
   - DP, Custom B&B, scipy.optimize.milp, and Greedy.
   - Re-timed with 10 repeated runs (mean +/- std) using time.perf_counter().
   - Exact node counts explored and pruned.
   - Greedy optimality gap comparison on strongly correlated vs uncorrelated instances.
2. Combinatorial Feature Selection:
   - Separately reports evaluation count ratio (175 vs 27 = 6.48x) and measured wall-clock speedup.
3. Max-Cut QUBO:
   - Exact classical enumeration vs Simulated Annealing across 15 seeds (mean +/- std).
4. QUBO Penalty Multiplier Sensitivity (instance-specific bounds).
5. Minor-embedding limits analysis with academic citations.
"""

import json
import os
import sys
import time
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tracks.B_discrete.knapsack_solvers import (
    KnapsackInstance,
    solve_greedy,
    solve_dynamic_programming,
    solve_branch_and_bound,
    solve_ilp_scipy,
)
from tracks.B_discrete.feature_selection import (
    FeatureSelectionProblem,
    exact_best_subset_selection,
    forward_stepwise_selection,
)
from tracks.B_discrete.qubo_formulation import (
    build_maxcut_qubo,
    build_constrained_knapsack_qubo,
    qubo_energy,
)
from tracks.B_discrete.classical_qubo_solver import (
    solve_qubo_exact,
    solve_qubo_simulated_annealing,
    analyze_minor_embedding_limits,
)


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==============================================================================
# 1. Knapsack Solvers: Uncorrelated & Strongly Correlated Instances
# ==============================================================================

def run_knapsack_scaling_benchmarks() -> Dict:
    """Benchmark 1: Compares DP, B&B, ILP, and Greedy across N on both uncorrelated and strongly correlated instances."""
    print("\n--- Running Benchmark 1: Knapsack Solvers & Hard Instances ---")
    sizes = [10, 15, 20, 25, 30, 40]
    n_repeats = 10

    # Dictionaries to store results
    uncorrelated_results = []
    strongly_correlated_results = []

    for n in sizes:
        # A. Uncorrelated Instance: v_i in [10, 100], w_i in [10, 50]
        rng_u = np.random.default_rng(42 + n)
        w_u = rng_u.integers(10, 50, size=n).astype(float)
        v_u = rng_u.integers(10, 100, size=n).astype(float)
        cap_u = float(np.sum(w_u) * 0.4)
        inst_u = KnapsackInstance(v_u, w_u, cap_u)

        # B. Strongly Correlated Instance: v_i = w_i + 20 (harder LP relaxation gap)
        rng_s = np.random.default_rng(100 + n)
        w_s = rng_s.integers(10, 100, size=n).astype(float)
        v_s = w_s + 20.0
        cap_s = float(np.sum(w_s) * 0.4)
        inst_s = KnapsackInstance(v_s, w_s, cap_s)

        # Benchmark Uncorrelated
        res_g_u = solve_greedy(inst_u)
        res_dp_u = solve_dynamic_programming(inst_u)
        res_bnb_u = solve_branch_and_bound(inst_u)
        res_ilp_u = solve_ilp_scipy(inst_u)
        gap_u = float((res_dp_u["objective"] - res_g_u["objective"]) / max(res_dp_u["objective"], 1e-9) * 100.0)

        # Benchmark Strongly Correlated with 10 repeated runs for timing
        bnb_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _ = solve_branch_and_bound(inst_s)
            bnb_times.append((time.perf_counter() - t0) * 1000.0)

        dp_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _ = solve_dynamic_programming(inst_s)
            dp_times.append((time.perf_counter() - t0) * 1000.0)

        ilp_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _ = solve_ilp_scipy(inst_s)
            ilp_times.append((time.perf_counter() - t0) * 1000.0)

        res_g_s = solve_greedy(inst_s)
        res_dp_s = solve_dynamic_programming(inst_s)
        res_bnb_s = solve_branch_and_bound(inst_s)
        res_ilp_s = solve_ilp_scipy(inst_s)
        gap_s = float((res_dp_s["objective"] - res_g_s["objective"]) / max(res_dp_s["objective"], 1e-9) * 100.0)

        entry_s = {
            "n_items": n,
            "exact_objective": float(res_dp_s["objective"]),
            "greedy_objective": float(res_g_s["objective"]),
            "greedy_gap_percent": gap_s,
            "bnb_nodes_explored": res_bnb_s["nodes_explored"],
            "bnb_nodes_pruned": res_bnb_s["nodes_pruned"],
            "bnb_time_ms_mean": float(np.mean(bnb_times)),
            "bnb_time_ms_std": float(np.std(bnb_times)),
            "dp_time_ms_mean": float(np.mean(dp_times)),
            "dp_time_ms_std": float(np.std(dp_times)),
            "ilp_time_ms_mean": float(np.mean(ilp_times)),
            "ilp_time_ms_std": float(np.std(ilp_times)),
        }
        strongly_correlated_results.append(entry_s)

        entry_u = {
            "n_items": n,
            "exact_objective": float(res_dp_u["objective"]),
            "greedy_objective": float(res_g_u["objective"]),
            "greedy_gap_percent": gap_u,
            "bnb_nodes_explored": res_bnb_u["nodes_explored"],
            "bnb_nodes_pruned": res_bnb_u["nodes_pruned"],
        }
        uncorrelated_results.append(entry_u)

        print(f"  N={n:2d} (Strongly Correlated): Exact={res_dp_s['objective']:.1f} | Greedy Gap={gap_s:.2f}% | BnB Explored={res_bnb_s['nodes_explored']} (Pruned={res_bnb_s['nodes_pruned']}) | BnB={entry_s['bnb_time_ms_mean']:.2f}ms, ILP={entry_s['ilp_time_ms_mean']:.2f}ms")

    # Plot 1: Runtime Scaling on Strongly Correlated Instances
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    n_axis = [d["n_items"] for d in strongly_correlated_results]
    
    ax1.plot(n_axis, [d["bnb_time_ms_mean"] for d in strongly_correlated_results], "b-s", label="Custom B&B (mean +/- std)", linewidth=2)
    ax1.plot(n_axis, [d["dp_time_ms_mean"] for d in strongly_correlated_results], "r--o", label="Dynamic Programming O(N*W)", linewidth=2)
    ax1.plot(n_axis, [d["ilp_time_ms_mean"] for d in strongly_correlated_results], "g-.^", label="ILP (scipy.optimize.milp)", linewidth=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of Items $N$", fontsize=11)
    ax1.set_ylabel("Execution Time (ms, log scale)", fontsize=11)
    ax1.set_title("Hard (Strongly Correlated) Knapsack Scaling", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    # Plot 2: Greedy Optimality Gap Comparison
    gap_u_vals = [d["greedy_gap_percent"] for d in uncorrelated_results]
    gap_s_vals = [d["greedy_gap_percent"] for d in strongly_correlated_results]
    ax2.plot(n_axis, gap_s_vals, "r-s", label="Strongly Correlated ($v_i = w_i + 20$)", linewidth=2.2)
    ax2.plot(n_axis, gap_u_vals, "b--o", label="Uncorrelated Instances", linewidth=2.0)
    ax2.set_xlabel("Number of Items $N$", fontsize=11)
    ax2.set_ylabel("Greedy Suboptimality Gap (%)", fontsize=11)
    ax2.set_title("Greedy Optimality Gap: Hard vs Uncorrelated", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "knapsack_strongly_correlated_scaling.png"), dpi=300)
    plt.close()

    return {
        "strongly_correlated": strongly_correlated_results,
        "uncorrelated": uncorrelated_results
    }


# ==============================================================================
# 2. Combinatorial Feature Selection (Eval Counts vs Wall-Clock)
# ==============================================================================

def run_feature_selection_benchmarks() -> Dict:
    """Benchmark 2: Exhaustive vs Forward Stepwise Selection reporting eval counts and wall-clock time separately."""
    print("\n--- Running Benchmark 2: Feature Selection (Evals vs Wall-Clock) ---")
    rng = np.random.default_rng(42)
    N_samples = 80
    D_features = 10
    K_target = 3

    X = rng.normal(0, 1, size=(N_samples, D_features))
    true_beta = np.zeros(D_features)
    true_beta[[1, 4, 7]] = [3.5, -2.8, 1.9]
    y = X @ true_beta + 0.15 * rng.normal(0, 1, size=N_samples)

    problem = FeatureSelectionProblem(X, y, max_features=K_target)

    # 10 repeated runs for timing
    n_runs = 10
    ex_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        res_ex = exact_best_subset_selection(problem)
        ex_times.append((time.perf_counter() - t0) * 1000.0)

    fw_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        res_fw = forward_stepwise_selection(problem)
        fw_times.append((time.perf_counter() - t0) * 1000.0)

    eval_ex = res_ex["subsets_evaluated"]
    eval_fw = res_fw["subsets_evaluated"]
    eval_ratio = eval_ex / eval_fw

    mean_t_ex = float(np.mean(ex_times))
    std_t_ex = float(np.std(ex_times))
    mean_t_fw = float(np.mean(fw_times))
    std_t_fw = float(np.std(fw_times))
    wall_clock_speedup = mean_t_ex / mean_t_fw

    print(f"  Exhaustive Selection: {eval_ex} evaluations | Time: {mean_t_ex:.2f} ms (+/- {std_t_ex:.2f})")
    print(f"  Forward Stepwise:     {eval_fw} evaluations | Time: {mean_t_fw:.2f} ms (+/- {std_t_fw:.2f})")
    print(f"  Evaluation Count Reduction: {eval_ratio:.2f}x ({eval_ex} vs {eval_fw})")
    print(f"  Measured Wall-Clock Speedup: {wall_clock_speedup:.2f}x (correcting previous 8.4x claim)")

    return {
        "n_samples": N_samples,
        "n_features": D_features,
        "k_target": K_target,
        "exhaustive_evaluations": eval_ex,
        "forward_evaluations": eval_fw,
        "evaluation_count_ratio": float(eval_ratio),
        "exhaustive_time_ms_mean": mean_t_ex,
        "exhaustive_time_ms_std": std_t_ex,
        "forward_time_ms_mean": mean_t_fw,
        "forward_time_ms_std": std_t_fw,
        "wall_clock_speedup": float(wall_clock_speedup),
        "best_subset_exhaustive": list(res_ex["selected_indices"]),
        "best_subset_forward": list(res_fw["selected_indices"]),
    }


# ==============================================================================
# 3. Max-Cut QUBO & Simulated Annealing (10+ Seeds)
# ==============================================================================

def run_maxcut_sa_benchmarks() -> Dict:
    """Benchmark 3: Max-Cut QUBO exact enumeration vs Simulated Annealing across 15 seeds."""
    print("\n--- Running Benchmark 3: Max-Cut QUBO & Simulated Annealing (15 Seeds) ---")
    # Adjacency matrix for 4-node benchmark graph (same as Track C QAOA graph)
    W = np.array([
        [0.0, 3.0, 0.0, 2.0],
        [3.0, 0.0, 4.0, 0.0],
        [0.0, 4.0, 0.0, 5.0],
        [2.0, 0.0, 5.0, 0.0]
    ], dtype=np.float64)

    Q, J, h, offset = build_maxcut_qubo(W)
    res_exact = solve_qubo_exact(Q)
    exact_cut = -float(res_exact["min_energy"])

    # 15 random seeds for Simulated Annealing
    n_seeds = 15
    sa_cuts = []
    sa_energies = []
    sa_times = []

    for seed in range(n_seeds):
        res_sa = solve_qubo_simulated_annealing(Q, seed=seed)
        cut = -float(res_sa["min_energy"])
        sa_cuts.append(cut)
        sa_energies.append(float(res_sa["min_energy"]))
        sa_times.append(float(res_sa["runtime_ms"]))

    mean_cut = float(np.mean(sa_cuts))
    std_cut = float(np.std(sa_cuts))
    mean_energy = float(np.mean(sa_energies))
    std_energy = float(np.std(sa_energies))
    mean_time = float(np.mean(sa_times))
    std_time = float(np.std(sa_times))
    success_rate = float(np.mean(np.array(sa_cuts) == exact_cut))

    print(f"  Exact Max-Cut:   {exact_cut:.1f} (Energy = {res_exact['min_energy']:.1f})")
    print(f"  SA Cut (15 seeds): {mean_cut:.2f} +/- {std_cut:.2f} (Success Rate: {success_rate*100:.1f}%)")
    print(f"  SA Energy:         {mean_energy:.2f} +/- {std_energy:.2f}")
    print(f"  SA Runtime:        {mean_time:.2f} ms +/- {std_time:.2f}")

    return {
        "exact_max_cut": exact_cut,
        "exact_min_energy": float(res_exact["min_energy"]),
        "sa_n_seeds": n_seeds,
        "sa_mean_cut": mean_cut,
        "sa_std_cut": std_cut,
        "sa_mean_energy": mean_energy,
        "sa_std_energy": std_energy,
        "sa_mean_runtime_ms": mean_time,
        "sa_std_runtime_ms": std_time,
        "sa_success_rate": success_rate,
    }


# ==============================================================================
# 4. QUBO Penalty Multiplier Sensitivity (Instance-Specific)
# ==============================================================================

def run_penalty_multiplier_sensitivity() -> Dict:
    """Benchmark 4: Analyzes feasibility trade-off across penalty multipliers lambda."""
    print("\n--- Running Benchmark 4: Penalty Multiplier Sensitivity ---")
    values = np.array([12, 10, 7, 4], dtype=float)
    weights = np.array([4, 3, 2, 1], dtype=float)
    target_weight = 5.0

    multipliers = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    results = []

    for lam in multipliers:
        Q, offset = build_constrained_knapsack_qubo(values, weights, target_weight, penalty_multiplier=lam)
        res = solve_qubo_exact(Q)
        x = np.array(res["best_state"])
        total_wt = float(np.sum(weights * x))
        total_val = float(np.sum(values * x))
        feasible = (total_wt == target_weight)

        results.append({
            "lambda": lam,
            "best_state": res["best_state"],
            "total_weight": total_wt,
            "total_value": total_val,
            "is_feasible": feasible,
            "min_energy": float(res["min_energy"] + offset)
        })
        print(f"  lambda={lam:<5}: State={res['best_state']} | Weight={total_wt:.1f} (Target={target_weight}) | Feasible={feasible}")

    # Plot Penalty Sensitivity
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    lam_vals = [d["lambda"] for d in results]
    energies = [d["min_energy"] for d in results]
    feasibles = [d["is_feasible"] for d in results]

    colors = ["mediumseagreen" if f else "crimson" for f in feasibles]
    ax.scatter(lam_vals, energies, c=colors, s=100, zorder=5)
    ax.plot(lam_vals, energies, "k--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"Penalty Multiplier $\lambda$ (Instance-Specific)", fontsize=11)
    ax.set_ylabel("Total Hamiltonian Energy $H(x)$", fontsize=11)
    ax.set_title(r"QUBO Penalty Multiplier Sensitivity: Feasibility Transition ($\lambda \geq 5.0$)", fontsize=12, fontweight="bold")

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson", markersize=9, label="Infeasible (Violates Capacity)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="mediumseagreen", markersize=9, label="Feasible (Ground State Satisfies Constraint)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "qubo_penalty_sensitivity.png"), dpi=300)
    plt.close()

    return {"penalty_results": results}


# ==============================================================================
# 5. Minor Embedding Overhead Analysis
# ==============================================================================

def run_minor_embedding_analysis() -> Dict:
    """Benchmark 5: Minor-embedding physical qubit scaling for D-Wave Chimera vs Pegasus architectures."""
    print("\n--- Running Benchmark 5: Minor-Embedding Analysis ---")
    logical_qubit_range = [4, 8, 12, 16, 20, 24, 30, 36]
    embedding_data = []

    for N in logical_qubit_range:
        info = analyze_minor_embedding_limits(n_logical_qubits=N)
        embedding_data.append(info)

    return {"embedding_limits": embedding_data}


def main():
    print("=" * 70)
    print("Executing Track B: Discrete Optimization (Review Fixes)")
    print("=" * 70)

    knapsack_data = run_knapsack_scaling_benchmarks()
    feat_data = run_feature_selection_benchmarks()
    maxcut_data = run_maxcut_sa_benchmarks()
    penalty_data = run_penalty_multiplier_sensitivity()
    embedding_data = run_minor_embedding_analysis()

    payload = {
        "metadata": {
            "track": "Track B: Discrete Optimization",
            "student": "Yash Deshmukh",
            "roll_no": "241070018",
            "institution": "VJTI Mumbai",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        },
        "knapsack_scaling": knapsack_data,
        "feature_selection": feat_data,
        "maxcut_sa_15_seeds": maxcut_data,
        "penalty_sensitivity": penalty_data,
        "minor_embedding": embedding_data
    }

    json_path = os.path.join(RESULTS_DIR, "discrete_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nAll benchmark results JSON written to {json_path}")
    print(f"All diagnostic plots saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
