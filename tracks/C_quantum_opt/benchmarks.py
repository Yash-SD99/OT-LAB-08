"""Comprehensive Benchmarking Suite for Track C: Quantum Optimization.

Generates reproducible CSV/JSON benchmark data and visualization figures:
1. QAOA Max-Cut on Track B Benchmark Graph (Depths p=1,2,3 across 15 seeds vs Classical Exact & SA)
2. VQE on Transverse-Field Ising vs Exact Diagonalization
3. Equal-Budget Gradient Optimizer Comparison: Parameter-Shift vs SPSA vs Quantum Natural Gradient (15 seeds)
4. Shot Noise Scaling: Var(<H>) vs 1 / S under Born-rule multinomial sampling
5. Hardware Depolarizing Noise Degradation (QAOA depths p in {1,2,3} and VQE layers L in {1,2,3})
6. Barren Plateaus: Global vs Local Cost Gradient Variance Decay across Qubit Scale N in [2..7] (50 samples/point)
7. Quantum Neural Network (QNN) Binary Classifier Training
8. Quantum Annealing Minor-Embedding Limits (J_chain sweep, chain breaks vs ground state probability)
9. Simulator Cross-Validation against Analytical Kronecker Mechanics
"""

import json
import os
import time
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tracks.C_quantum_opt.quantum_simulator import QuantumCircuitSimulator
from tracks.C_quantum_opt.qaoa_maxcut import QAOAMaxCut
from tracks.C_quantum_opt.vqe_ising import VQEIsing
from tracks.C_quantum_opt.pqc_qnn import (
    ParameterizedCircuit,
    QuantumNeuralNetworkClassifier,
    optimize_equal_budget
)
from tracks.C_quantum_opt.noise_and_barren import (
    analyze_shot_noise_scaling,
    analyze_barren_plateaus,
    analyze_depolarizing_noise_sweep
)
from tracks.C_quantum_opt.quantum_annealing import MinorEmbeddingDemo
from tracks.C_quantum_opt.simulator_crosscheck import (
    crosscheck_against_analytical_kronecker,
    crosscheck_against_qiskit
)


# ==============================================================================
# 1. QAOA Max-Cut on Track B Benchmark Graph across 15 Seeds
# ==============================================================================

def run_qaoa_benchmarks(results_dir: str) -> Dict:
    """Benchmark 1: QAOA Max-Cut on Track B 4-Node Graph across depths p in [1, 2, 3] over 15 seeds."""
    print("Running Benchmark 1: QAOA Max-Cut across depths p=1,2,3 (15 seeds)...")
    
    # 4-node benchmark graph from Track B (Max-Cut = 14.0)
    # Edges: (0,1)=3, (0,3)=2, (1,2)=4, (2,3)=5
    W = np.array([
        [0.0, 3.0, 0.0, 2.0],
        [3.0, 0.0, 4.0, 0.0],
        [0.0, 4.0, 0.0, 5.0],
        [2.0, 0.0, 5.0, 0.0]
    ], dtype=np.float64)
    
    depths = [1, 2, 3]
    n_seeds = 15
    qaoa_summary = {}
    
    for p in depths:
        cuts = []
        approx_ratios = []
        opt_probs = []
        runtimes = []
        
        for seed in range(n_seeds):
            rng = np.random.default_rng(100 + seed * 10 + p)
            # Randomized initial parameters around linear ramp heuristic
            ramp_g = np.linspace(0.1, 0.8, p) + rng.uniform(-0.08, 0.08, size=p)
            ramp_b = np.linspace(0.8, 0.1, p) + rng.uniform(-0.08, 0.08, size=p)
            init_p = np.concatenate([ramp_g, ramp_b])
            
            qaoa = QAOAMaxCut(W, p_depth=p)
            res = qaoa.optimize(initial_params=init_p, max_iters=45, seed=100 + seed * 10 + p)
            cuts.append(res["final_expected_cut"])
            approx_ratios.append(res["approximation_ratio"])
            opt_probs.append(res["prob_optimal_ground_state"])
            runtimes.append(res["runtime_ms"])
            
        qaoa_summary[f"p_{p}"] = {
            "p_depth": p,
            "mean_expected_cut": float(np.mean(cuts)),
            "std_expected_cut": float(np.std(cuts)),
            "mean_approximation_ratio": float(np.mean(approx_ratios)),
            "std_approximation_ratio": float(np.std(approx_ratios)),
            "mean_prob_optimal": float(np.mean(opt_probs)),
            "std_prob_optimal": float(np.std(opt_probs)),
            "mean_runtime_ms": float(np.mean(runtimes)),
            "std_runtime_ms": float(np.std(runtimes)),
            "max_cut_ground_truth": 14.0,
            "optimal_bitstrings": ["0101", "1010"],
            "track_b_exact_cut": 14.0,
            "track_b_sa_mean_cut": 14.0,
            "track_b_sa_std_cut": 0.0
        }
        print(f"  QAOA (p={p}): Cut = {np.mean(cuts):.2f} +/- {np.std(cuts):.2f} | "
              f"Ratio = {np.mean(approx_ratios):.3f} +/- {np.std(approx_ratios):.3f} | "
              f"P(opt) = {np.mean(opt_probs)*100:.1f}% +/- {np.std(opt_probs)*100:.1f}%")
        
    # Plot QAOA Approximation Ratio and Optimal Bitstring Probability vs Depth
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ratios_mean = [qaoa_summary[f"p_{p}"]["mean_approximation_ratio"] for p in depths]
    ratios_std = [qaoa_summary[f"p_{p}"]["std_approximation_ratio"] for p in depths]
    probs_mean = [qaoa_summary[f"p_{p}"]["mean_prob_optimal"] for p in depths]
    probs_std = [qaoa_summary[f"p_{p}"]["std_prob_optimal"] for p in depths]
    
    ax1.errorbar(depths, ratios_mean, yerr=ratios_std, fmt='o-', color='#1f77b4', lw=2, capsize=5, label="Approximation Ratio <Cut>/14.0")
    ax1.set_xlabel("QAOA Circuit Depth p")
    ax1.set_ylabel("Approximation Ratio", color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_ylim(0.5, 1.05)
    ax1.set_xticks(depths)
    ax1.grid(True, ls="--", alpha=0.5)
    
    ax2 = ax1.twinx()
    ax2.errorbar(depths, probs_mean, yerr=probs_std, fmt='s--', color='#ff7f0e', lw=2, capsize=5, label="Prob. Optimal Bitstring (0101 + 1010)")
    ax2.set_ylabel("Optimal Ground State Probability", color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax2.set_ylim(0.0, 1.05)
    
    plt.title("QAOA Max-Cut on Track B Benchmark Graph (15 Seeds, Mean +/- Std)")
    fig.tight_layout()
    plt.savefig(os.path.join(results_dir, "qaoa_vs_depth.png"), dpi=200)
    plt.close()
    
    return qaoa_summary


# ==============================================================================
# 2. VQE on Transverse-Field Ising vs Exact Diagonalization
# ==============================================================================

def run_vqe_benchmarks(results_dir: str) -> Dict:
    """Benchmark 2: VQE Ground State Energy vs Exact Diagonalization."""
    print("Running Benchmark 2: VQE on Transverse-Field Ising Model...")
    qubit_configs = [3, 4, 5]
    vqe_results = {}
    
    for n in qubit_configs:
        vqe = VQEIsing(n_qubits=n, J=1.0, h=0.5, layers=2)
        res = vqe.optimize(max_iters=50, seed=42)
        vqe_results[f"{n}_qubits"] = res
        print(f"  VQE (N={n}): Exact E0 = {res['exact_ground_energy']:.6f} | VQE E = {res['vqe_ground_energy']:.6f} | "
              f"Error = {res['energy_error']:.2e}")
        
    # Plot VQE accuracy
    plt.figure(figsize=(8, 5))
    exacts = [vqe_results[f"{n}_qubits"]["exact_ground_energy"] for n in qubit_configs]
    vqes = [vqe_results[f"{n}_qubits"]["vqe_ground_energy"] for n in qubit_configs]
    
    x_indices = np.arange(len(qubit_configs))
    width = 0.35
    plt.bar(x_indices - width/2, exacts, width, label="Exact Diagonalization E_0", color="#1f77b4")
    plt.bar(x_indices + width/2, vqes, width, label="VQE Energy E_vqe (Layers=2)", color="#2ca02c")
    plt.xticks(x_indices, [f"N={n}" for n in qubit_configs])
    plt.ylabel("Ground State Energy")
    plt.title("VQE vs Exact Diagonalization (Transverse-Field Ising)")
    plt.grid(True, ls="--", alpha=0.5, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "vqe_convergence.png"), dpi=200)
    plt.close()
    
    return vqe_results


# ==============================================================================
# 3. Equal-Budget Optimizer Comparison (SPSA, Parameter-Shift, QNG) across 15 Seeds
# ==============================================================================

def run_equal_budget_optimizer_comparison(results_dir: str) -> Dict:
    """Benchmark 3: Equal-budget evaluation comparison (Budget = 300 circuit evaluations) over 15 seeds."""
    print("Running Benchmark 3: Equal-Budget Optimizer Comparison (Budget=300, 15 Seeds)...")
    n_qubits = 3
    pqc = ParameterizedCircuit(n_qubits=n_qubits, n_layers=1)
    
    budget = 300
    n_seeds = 15
    methods = ["spsa", "param_shift", "qng"]
    
    # Step-size sweeps to identify optimal learning rate for each
    lr_candidates = [0.05, 0.1, 0.2, 0.4]
    best_lrs = {}
    for m in methods:
        best_lr = lr_candidates[0]
        best_cost = float('inf')
        for lr in lr_candidates:
            res = optimize_equal_budget(pqc, method=m, budget=budget, lr=lr, seed=42)
            if res["final_cost"] < best_cost:
                best_cost = res["final_cost"]
                best_lr = lr
        best_lrs[m] = best_lr
        
    print(f"  Selected step sizes from sweep: SPSA lr={best_lrs['spsa']}, "
          f"PS lr={best_lrs['param_shift']}, QNG lr={best_lrs['qng']}")
    
    results_by_method = {}
    trajectories_by_method = {m: [] for m in methods}
    
    for m in methods:
        final_costs = []
        steps_list = []
        evals_list = []
        
        for s in range(n_seeds):
            rng = np.random.default_rng(200 + s * 13)
            init_p = rng.uniform(-np.pi, np.pi, size=pqc.num_params)
            res = optimize_equal_budget(
                pqc, method=m, budget=budget, lr=best_lrs[m], initial_params=init_p, seed=300 + s
            )
            final_costs.append(res["final_cost"])
            steps_list.append(res["steps_completed"])
            evals_list.append(res["total_evaluations"])
            trajectories_by_method[m].append((res["history_evals"], res["history_costs"]))
            
        results_by_method[m] = {
            "method": m,
            "budget": budget,
            "optimal_lr": best_lrs[m],
            "n_seeds": n_seeds,
            "mean_final_cost": float(np.mean(final_costs)),
            "std_final_cost": float(np.std(final_costs)),
            "mean_steps_completed": float(np.mean(steps_list)),
            "mean_total_evaluations": float(np.mean(evals_list))
        }
        print(f"  {m.upper():12s}: Final Cost = {np.mean(final_costs):.4f} +/- {np.std(final_costs):.4f} | "
              f"Steps Completed = {np.mean(steps_list):.1f} | Evals = {np.mean(evals_list):.0f}")
        
    # Plot cost trajectories vs Circuit Evaluations (interpolated mean +/- std)
    plt.figure(figsize=(8, 5))
    eval_grid = np.linspace(1, budget, 60)
    colors = {"spsa": "#ff7f0e", "param_shift": "#1f77b4", "qng": "#2ca02c"}
    labels = {
        "spsa": f"SPSA (2 evals/step, Final: {results_by_method['spsa']['mean_final_cost']:.2f})",
        "param_shift": f"Parameter-Shift GD (2D evals/step, Final: {results_by_method['param_shift']['mean_final_cost']:.2f})",
        "qng": f"Quantum Natural Gradient (Metric-scaled, Final: {results_by_method['qng']['mean_final_cost']:.2f})"
    }
    
    for m in methods:
        interp_costs = []
        for evs, costs in trajectories_by_method[m]:
            interp_c = np.interp(eval_grid, evs, costs)
            interp_costs.append(interp_c)
        mean_c = np.mean(interp_costs, axis=0)
        std_c = np.std(interp_costs, axis=0)
        plt.plot(eval_grid, mean_c, color=colors[m], lw=2, label=labels[m])
        plt.fill_between(eval_grid, mean_c - std_c, mean_c + std_c, color=colors[m], alpha=0.15)
        
    plt.axhline(-3.0, color='k', ls='--', alpha=0.6, label="Exact Ground State Energy (-3.0)")
    plt.xlabel("Circuit Evaluations (Strict Equal Budget = 300)")
    plt.ylabel("Cost Function <H>")
    plt.title("Quantum Optimizer Comparison at Equal Circuit Evaluation Budget (15 Seeds)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "gradient_methods_comparison.png"), dpi=200)
    plt.close()
    
    return results_by_method


# ==============================================================================
# 4, 5, 6. Noise, Barren Plateaus, QNN, and Annealing
# ==============================================================================

def run_noise_barren_qnn_annealing_benchmarks(results_dir: str) -> Dict:
    """Benchmark 4, 5, 6, 7, 8: Shot noise, Depolarizing noise, Barren plateaus, QNN, Annealing."""
    # 4. Shot Noise Scaling
    print("Running Benchmark 4: Shot Noise Scaling (Multinomial Born-Rule Sampling)...")
    shot_res = analyze_shot_noise_scaling(n_qubits=3, shots_list=[50, 100, 250, 500, 1000, 2500, 5000])
    
    plt.figure(figsize=(8, 5))
    plt.loglog(shot_res["shots_list"], shot_res["measured_variances"], 'o-', lw=2, label="Measured Variance Var(<H>)")
    plt.loglog(shot_res["shots_list"], shot_res["theoretical_scaling_fit"], 'k--', label="Theoretical 1/S Multinomial Fit")
    plt.xlabel("Measurement Shots S (log scale)")
    plt.ylabel("Variance Var(<H>) (log scale)")
    plt.title("Finite Shot Measurement Noise: 1/S Multinomial Scaling")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "shot_noise_scaling.png"), dpi=200)
    plt.close()
    
    # 5. Depolarizing Hardware Noise Sweep
    print("Running Benchmark 5: Depolarizing Gate Noise Degradation (QAOA & VQE)...")
    W_4 = np.array([
        [0.0, 3.0, 0.0, 2.0],
        [3.0, 0.0, 4.0, 0.0],
        [0.0, 4.0, 0.0, 5.0],
        [2.0, 0.0, 5.0, 0.0]
    ], dtype=np.float64)
    noise_rates = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20]
    depol_res = analyze_depolarizing_noise_sweep(W_4, noise_rates=noise_rates, n_trials=10)
    
    plt.figure(figsize=(8, 5))
    for p in [1, 2, 3]:
        plt.plot(noise_rates, depol_res["qaoa_by_depth"][p], 'o-', lw=2, label=f"QAOA (Depth p={p})")
    plt.xlabel("Depolarizing Error Rate p_error")
    plt.ylabel("Max-Cut Approximation Ratio")
    plt.title("Hardware Depolarizing Noise Degradation across QAOA Depths")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "hardware_noise_degradation.png"), dpi=200)
    plt.close()
    
    # 6. Barren Plateaus: Global vs Local Observable
    print("Running Benchmark 6: Barren Plateaus (Global vs Local Observable, 50 samples/point)...")
    qubits = [2, 3, 4, 5, 6, 7]
    bp_res = analyze_barren_plateaus(qubits_range=qubits, n_samples=50)
    
    plt.figure(figsize=(8, 5))
    plt.semilogy(qubits, bp_res["global_gradient_variances"], 's-', color="#d62728", lw=2,
                 label=f"Global Cost H = Z^(ox N) (Decay rate: {bp_res['fitted_decay_rate_global']:.2f})")
    plt.semilogy(qubits, bp_res["local_gradient_variances"], 'o--', color="#1f77b4", lw=2,
                 label=f"Local Cost H = Z_0 (Decay rate: {bp_res['fitted_decay_rate_local']:.2f})")
    plt.xlabel("Number of Qubits N")
    plt.ylabel("Gradient Variance Var(d<H>/d theta) (log scale)")
    plt.title("Barren Plateau Phenomenon: Global vs Local Cost Variance Decay")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "barren_plateaus_decay.png"), dpi=200)
    plt.close()
    
    # 7. Quantum Neural Network (QNN) Classifier
    print("Running Benchmark 7: Parameterized Quantum Classifier (QNN on Non-Linear XOR Parity)...")
    # Generate non-linear XOR 2D dataset
    X = np.array([
        [-0.6, -0.6], [-0.5, -0.8], [-0.7, -0.4], [0.6, 0.6], [0.5, 0.8], [0.7, 0.4],
        [-0.6, 0.6], [-0.5, 0.8], [-0.7, 0.4], [0.6, -0.6], [0.5, -0.8], [0.7, -0.4]
    ], dtype=float)
    y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    
    qnn = QuantumNeuralNetworkClassifier(n_layers=2)
    qnn_res = qnn.fit(X, y, epochs=25, lr=0.5, seed=42)
    print(f"  QNN Training: Initial Acc = {qnn_res['initial_accuracy']*100:.1f}%, Final Acc = {qnn_res['final_accuracy']*100:.1f}%, "
          f"Final Loss = {qnn_res['final_loss']:.4f} ({qnn_res['runtime_ms']:.1f} ms)")
    
    # 8. Quantum Annealing Minor-Embedding Limits
    print("Running Benchmark 8: Quantum Annealing Minor-Embedding Limits...")
    anneal_demo = MinorEmbeddingDemo(seed=42)
    anneal_res = anneal_demo.evaluate_embedding_limits(n_trials=25)
    print(f"  Annealing Sweet Spot: Optimal J_chain = {anneal_res['optimal_j_chain']} "
          f"with Ground State Hit Rate = {max(anneal_res['ground_state_success_rates'])*100:.1f}%")
    
    # 9. Simulator Cross-Check
    print("Running Benchmark 9: Simulator Analytical Cross-Check...")
    kronecker_res = crosscheck_against_analytical_kronecker()
    qiskit_res = crosscheck_against_qiskit()
    
    return {
        "shot_noise": shot_res,
        "depolarizing_noise": depol_res,
        "barren_plateaus": bp_res,
        "qnn_classifier": qnn_res,
        "quantum_annealing_embedding": anneal_res,
        "kronecker_crosscheck": kronecker_res,
        "qiskit_crosscheck": qiskit_res
    }


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    all_results["benchmark_1_qaoa"] = run_qaoa_benchmarks(results_dir)
    all_results["benchmark_2_vqe"] = run_vqe_benchmarks(results_dir)
    all_results["benchmark_3_equal_budget_gradients"] = run_equal_budget_optimizer_comparison(results_dir)
    all_results["benchmarks_4_through_9"] = run_noise_barren_qnn_annealing_benchmarks(results_dir)
    
    output_path = os.path.join(results_dir, "quantum_benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\nAll Track C benchmarks completed successfully!")
    print(f"Results and plots saved to: {results_dir}")


if __name__ == "__main__":
    main()
