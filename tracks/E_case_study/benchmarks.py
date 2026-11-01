"""Benchmarking Suite for Non-Convex Portfolio Optimization (Track E: Main Case Study).

Evaluates:
1. Non-Convex Efficient Frontier Generation across risk tolerances lambda in [0.1, 2.0]
2. Cardinality Constraint Trade-Off across K in [3, 4, 5, 6]
3. Multi-Solver Comparison: Exact MIQP vs Greedy vs Simulated Annealing vs Genetic Algorithm
4. Comprehensive Strategy-Selection Table comparing computational trade-offs
"""

from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np

from tracks.E_case_study.portfolio_data import generate_portfolio_market_data, PortfolioDataset
from tracks.E_case_study.solvers import (
    MIQPSolver,
    GreedyPortfolioSolver,
    SimulatedAnnealingPortfolioSolver,
    GeneticAlgorithmPortfolioSolver,
    PortfolioSolution
)


# -----------------------------------------------------------------------------
# Strategy-Selection Decision Table (Amendment 7)
# -----------------------------------------------------------------------------
STRATEGY_SELECTION_TABLE = [
    {
        "method": "Exact MIQP (Branch-and-Bound)",
        "chapter6_category": "Mixed-Integer Quadratic Programming / B&B",
        "variable_types": "Mixed Binary (z) + Continuous (w)",
        "optimality_guarantee": "Global Mathematical Optimum",
        "scalability": "Exponential in Cardinality O(C(N, K)); practical for N <= 50, K <= 6",
        "runtime_characteristics": "High compute cost (seconds to minutes on large N)",
        "hardware_profile": "Single-thread or multi-core CPU branch-and-cut"
    },
    {
        "method": "Greedy Forward Selection",
        "chapter6_category": "Greedy Heuristic with QP Subspace Projection",
        "variable_types": "Iterative Binary Decision + Continuous QP",
        "optimality_guarantee": "Suboptimal (Greedy Approximation)",
        "scalability": "Polynomial O(K * N * QP_time); scales to N = 10,000+ assets",
        "runtime_characteristics": "Ultra-fast (< 50 ms), deterministic",
        "hardware_profile": "Standard lightweight CPU; ideal for real-time streaming"
    },
    {
        "method": "Simulated Annealing",
        "chapter6_category": "Stochastic Local Search Metaheuristic",
        "variable_types": "Mixed Discrete Neighborhood + Simplex Projection",
        "optimality_guarantee": "Asymptotic convergence in infinite time; heuristic in practice",
        "scalability": "Linear in iterations O(Iter * QP_time); tunable trade-off",
        "runtime_characteristics": "Moderate (100 - 300 ms); avoids local non-convex traps",
        "hardware_profile": "CPU-friendly; embarrassingly parallel across random restarts"
    },
    {
        "method": "Genetic Algorithm",
        "chapter6_category": "Population-Based Evolutionary Metaheuristic",
        "variable_types": "Binary Chromosome with Repair + Continuous Phenotype",
        "optimality_guarantee": "Heuristic (Explores multi-modal combinatorial fitness)",
        "scalability": "O(PopSize * Generations * QP_time); easily parallelizable",
        "runtime_characteristics": "Balanced exploration (1 - 2 seconds)",
        "hardware_profile": "Massively parallelizable across CPU cores or GPU threads"
    }
]


def run_portfolio_benchmarks() -> Dict[str, Any]:
    print("===================================================================")
    print(" Running Track E Main Case Study: Non-Convex Portfolio Optimization")
    print("===================================================================")

    data = generate_portfolio_market_data(seed=42)
    print(f"Generated Market Model: {len(data.asset_names)} assets across 5 economic sectors.")
    print(f"Constraints: w_min={data.min_weight}, w_max={data.max_weight}, c_fixed={data.fixed_trans_cost}, c_prop={data.prop_trans_cost}")

    # -------------------------------------------------------------------------
    # Experiment 1: Multi-Solver Comparison (K = 4, lambda = 1.0)
    # -------------------------------------------------------------------------
    print("\n--- Experiment 1: Solver Comparison (K = 4, lambda = 1.0) ---")
    solvers = {
        "MIQP": MIQPSolver(data, max_cardinality=4, lambda_risk=1.0),
        "Greedy": GreedyPortfolioSolver(data, max_cardinality=4, lambda_risk=1.0),
        "SimulatedAnnealing": SimulatedAnnealingPortfolioSolver(data, max_cardinality=4, lambda_risk=1.0, seed=42),
        "GeneticAlgorithm": GeneticAlgorithmPortfolioSolver(data, max_cardinality=4, lambda_risk=1.0, seed=42)
    }

    solver_results = {}
    for name, solver in solvers.items():
        sol = solver.solve()
        gap_pct = 0.0
        if "MIQP" in solver_results:
            miqp_obj = solver_results["MIQP"]["objective_value"]
            gap_pct = max(0.0, (sol.objective_value - miqp_obj) / abs(miqp_obj) * 100.0)

        solver_results[name] = {
            "method": sol.method_name,
            "objective_value": float(sol.objective_value),
            "expected_return": float(sol.expected_return),
            "portfolio_variance": float(sol.portfolio_variance),
            "portfolio_volatility": float(np.sqrt(sol.portfolio_variance)),
            "sharpe_ratio": float(sol.sharpe_ratio),
            "transaction_cost": float(sol.transaction_cost),
            "cardinality": sol.cardinality,
            "solve_time_ms": float(sol.solve_time_ms),
            "is_feasible": sol.is_feasible,
            "optimality_gap_pct": float(gap_pct),
            "active_assets": [data.asset_names[i] for i in range(len(data.asset_names)) if sol.active_mask[i] == 1],
            "weights": {data.asset_names[i]: float(sol.weights[i]) for i in range(len(data.asset_names)) if sol.active_mask[i] == 1}
        }
        print(f"  {name:<20}: Obj = {sol.objective_value:>8.4f} | Ret = {sol.expected_return*100:>5.1f}% | Vol = {np.sqrt(sol.portfolio_variance)*100:>5.1f}% | Sharpe = {sol.sharpe_ratio:>5.2f} | Time = {sol.solve_time_ms:>7.1f} ms | Feasible = {sol.is_feasible}")

    # -------------------------------------------------------------------------
    # Experiment 2: Non-Convex Efficient Frontier Sweep (lambda in [0.2, 2.0])
    # -------------------------------------------------------------------------
    print("\n--- Experiment 2: Constructing Efficient Frontiers across Risk Tolerances ---")
    lambdas = [0.2, 0.4, 0.7, 1.0, 1.5, 2.0]
    frontier_results = {name: [] for name in ["MIQP", "Greedy", "SimulatedAnnealing", "GeneticAlgorithm"]}

    for lam in lambdas:
        print(f"  Evaluating lambda = {lam:.1f}...")
        for name in frontier_results.keys():
            if name == "MIQP":
                s = MIQPSolver(data, max_cardinality=4, lambda_risk=lam).solve()
            elif name == "Greedy":
                s = GreedyPortfolioSolver(data, max_cardinality=4, lambda_risk=lam).solve()
            elif name == "SimulatedAnnealing":
                s = SimulatedAnnealingPortfolioSolver(data, max_cardinality=4, lambda_risk=lam, max_iterations=100, seed=int(lam*100)).solve()
            else:
                s = GeneticAlgorithmPortfolioSolver(data, max_cardinality=4, lambda_risk=lam, generations=20, seed=int(lam*100)).solve()

            frontier_results[name].append({
                "lambda": lam,
                "expected_return": float(s.expected_return),
                "volatility": float(np.sqrt(s.portfolio_variance)),
                "sharpe": float(s.sharpe_ratio),
                "objective": float(s.objective_value)
            })

    # -------------------------------------------------------------------------
    # Experiment 3: Cardinality Scaling Sweep (K in [3, 4, 5, 6])
    # -------------------------------------------------------------------------
    print("\n--- Experiment 3: Cardinality Constraint Scaling (K in [3, 4, 5, 6]) ---")
    cardinality_results = []
    for k in [3, 4, 5, 6]:
        t0 = time.perf_counter()
        sol_grd = GreedyPortfolioSolver(data, max_cardinality=k, lambda_risk=1.0).solve()
        sol_sa = SimulatedAnnealingPortfolioSolver(data, max_cardinality=k, lambda_risk=1.0, max_iterations=120, seed=42).solve()
        cardinality_results.append({
            "max_cardinality": k,
            "greedy_return": float(sol_grd.expected_return),
            "greedy_volatility": float(np.sqrt(sol_grd.portfolio_variance)),
            "greedy_sharpe": float(sol_grd.sharpe_ratio),
            "greedy_time_ms": float(sol_grd.solve_time_ms),
            "sa_return": float(sol_sa.expected_return),
            "sa_volatility": float(np.sqrt(sol_sa.portfolio_variance)),
            "sa_sharpe": float(sol_sa.sharpe_ratio),
            "sa_time_ms": float(sol_sa.solve_time_ms)
        })
        print(f"  K = {k}: Greedy Sharpe = {sol_grd.sharpe_ratio:.2f} (Time: {sol_grd.solve_time_ms:.1f} ms) | SA Sharpe = {sol_sa.sharpe_ratio:.2f} (Time: {sol_sa.solve_time_ms:.1f} ms)")

    master_results = {
        "strategy_selection_table": STRATEGY_SELECTION_TABLE,
        "solver_comparison_k4": solver_results,
        "efficient_frontiers": frontier_results,
        "cardinality_scaling": cardinality_results
    }

    # Save to JSON
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "portfolio_benchmark_results.json"), "w") as f:
        json.dump(master_results, f, indent=2)

    generate_portfolio_plots(master_results, out_dir)
    print("\nTrack E Benchmarks complete. Artifacts saved to tracks/E_case_study/results/")
    return master_results


def generate_portfolio_plots(data: Dict[str, Any], out_dir: str):
    """Generates comparative plots for Non-Convex Efficient Frontiers and Cardinality Trade-Offs."""
    frontiers = data["efficient_frontiers"]

    # Plot 1: Non-Convex Efficient Frontier
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    colors = {"MIQP": "#1f77b4", "Greedy": "#ff7f0e", "SimulatedAnnealing": "#2ca02c", "GeneticAlgorithm": "#d62728"}
    markers = {"MIQP": "o-", "Greedy": "s--", "SimulatedAnnealing": "^:", "GeneticAlgorithm": "d-."}

    for name, points in frontiers.items():
        vols = [p["volatility"] * 100.0 for p in points]
        rets = [p["expected_return"] * 100.0 for p in points]
        ax1.plot(vols, rets, markers[name], color=colors[name], lw=2, label=f"{name}")

    ax1.set_xlabel("Portfolio Annualized Volatility (%)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Expected Annual Return (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Track E: Non-Convex Efficient Frontiers (Cardinality K=4 & Transaction Costs)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "portfolio_efficient_frontiers.png"), dpi=300)
    plt.close()

    # Plot 2: Cardinality Scaling & Diversification Frontier
    card_data = data["cardinality_scaling"]
    k_vals = [d["max_cardinality"] for d in card_data]
    grd_sharpe = [d["greedy_sharpe"] for d in card_data]
    sa_sharpe = [d["sa_sharpe"] for d in card_data]
    grd_vol = [d["greedy_volatility"] * 100.0 for d in card_data]

    fig, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.plot(k_vals, grd_sharpe, "s-", color="#1f77b4", lw=2, label="Greedy Sharpe Ratio")
    ax2.plot(k_vals, sa_sharpe, "o--", color="#2ca02c", lw=2, label="Simulated Annealing Sharpe Ratio")

    ax2.set_xlabel("Maximum Cardinality Limit ($K$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Achieved Sharpe Ratio", fontsize=11, fontweight="bold")
    ax2.set_title("Track E: Cardinality Limit ($K$) vs Risk-Adjusted Return Trade-Off", fontsize=12, fontweight="bold")
    ax2.set_xticks(k_vals)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "portfolio_cardinality_tradeoff.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    run_portfolio_benchmarks()
