"""Comprehensive Benchmarking Suite for Track C: Quantum Optimization.

Generates reproducible CSV/JSON benchmark data and visualization figures:
1. QAOA Max-Cut on Track B Graphs (Depth p=1,2,3 vs Classical Baselines)
2. VQE on Transverse-Field Ising vs Exact Diagonalization
3. Quantum Gradient Strategies: Parameter-Shift vs SPSA vs Quantum Natural Gradient
4. Shot Noise Scaling: Var(<H>) vs 1 / S
5. Hardware Depolarizing Noise Degradation
6. Barren Plateaus: Gradient Variance Decay across Qubit Scale N in [2..7]
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
from tracks.C_quantum_opt.pqc_qnn import ParameterizedCircuit
from tracks.C_quantum_opt.noise_and_barren import (
    analyze_shot_noise_scaling,
    analyze_barren_plateaus
)


def run_qaoa_benchmarks(results_dir: str) -> Dict:
    """Benchmark 1: QAOA Max-Cut vs Classical Baselines across depths p in [1, 2, 3]."""
    print("Running Benchmark 1: QAOA Max-Cut across depths p=1,2,3...")
    # 6-node and 8-node graphs from Track B
    rng_6 = np.random.default_rng(106)
    W_6 = rng_6.binomial(1, 0.4, size=(6, 6)).astype(float)
    W_6 = np.triu(W_6, 1)
    W_6 = W_6 + W_6.T
    
    rng_8 = np.random.default_rng(108)
    W_8 = rng_8.binomial(1, 0.4, size=(8, 8)).astype(float)
    W_8 = np.triu(W_8, 1)
    W_8 = W_8 + W_8.T
    
    graphs = {"6_nodes": W_6, "8_nodes": W_8}
    qaoa_results = {}
    
    depths = [1, 2, 3]
    
    for g_name, W in graphs.items():
        qaoa_results[g_name] = {}
        for p in depths:
            qaoa = QAOAMaxCut(W, p_depth=p)
            res = qaoa.optimize(max_iters=40, seed=42 + p)
            qaoa_results[g_name][f"p_{p}"] = res
            
    # Plot QAOA Approximation Ratio vs Depth p
    plt.figure(figsize=(8, 5))
    for g_name in graphs.keys():
        ratios = [qaoa_results[g_name][f"p_{p}"]["approximation_ratio"] for p in depths]
        plt.plot(depths, ratios, 'o-', lw=2, label=f"Graph ({g_name})")
        
    plt.axhline(1.0, color='k', ls='--', alpha=0.5, label="Classical Exact (Track B)")
    plt.title("QAOA Max-Cut: Approximation Ratio vs Circuit Depth p")
    plt.xlabel("QAOA Depth p")
    plt.ylabel("Approximation Ratio (Expected Cut / Max Cut)")
    plt.xticks(depths)
    plt.ylim(0.5, 1.05)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "qaoa_vs_depth.png"), dpi=200)
    plt.close()
    
    return qaoa_results


def run_vqe_benchmarks(results_dir: str) -> Dict:
    """Benchmark 2: VQE Ground State Energy vs Exact Diagonalization."""
    print("Running Benchmark 2: VQE on Transverse-Field Ising Model...")
    qubit_configs = [3, 4, 5]
    vqe_results = {}
    
    errors = []
    runtimes = []
    
    for n in qubit_configs:
        vqe = VQEIsing(n_qubits=n, J=1.0, h=0.5, layers=2)
        res = vqe.optimize(max_iters=50, seed=42)
        vqe_results[f"{n}_qubits"] = res
        errors.append(res["energy_error"])
        runtimes.append(res["runtime_ms"])
        
    # Plot VQE accuracy
    plt.figure(figsize=(8, 5))
    exacts = [vqe_results[f"{n}_qubits"]["exact_ground_energy"] for n in qubit_configs]
    vqes = [vqe_results[f"{n}_qubits"]["vqe_ground_energy"] for n in qubit_configs]
    
    x_indices = np.arange(len(qubit_configs))
    width = 0.35
    plt.bar(x_indices - width/2, exacts, width, label="Exact Diagonalization E_0", color="#1f77b4")
    plt.bar(x_indices + width/2, vqes, width, label="VQE Energy E_vqe", color="#2ca02c")
    plt.xticks(x_indices, [f"N={n}" for n in qubit_configs])
    plt.ylabel("Ground State Energy")
    plt.title("VQE vs Exact Diagonalization (Transverse-Field Ising)")
    plt.grid(True, ls="--", alpha=0.5, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "vqe_convergence.png"), dpi=200)
    plt.close()
    
    return vqe_results


def run_gradient_methods_comparison(results_dir: str) -> Dict:
    """Benchmark 3: Parameter-Shift vs SPSA vs Quantum Natural Gradient."""
    print("Running Benchmark 3: Parameter-Shift vs SPSA vs QNG...")
    n_qubits = 3
    pqc = ParameterizedCircuit(n_qubits=n_qubits, n_layers=1)
    
    np.random.seed(42)
    theta_0 = np.random.uniform(-np.pi, np.pi, size=pqc.num_params)
    iters = 30
    
    # 1. Parameter-Shift Gradient Descent
    theta_ps = theta_0.copy()
    hist_ps = []
    evals_ps = 0
    lr_ps = 0.1
    for _ in range(iters):
        cost = pqc.cost(theta_ps)
        hist_ps.append(cost)
        g = pqc.parameter_shift_gradient(theta_ps)
        theta_ps -= lr_ps * g
        evals_ps += 2 * pqc.num_params + 1
        
    # 2. SPSA (Only 2 evaluations per iteration)
    theta_spsa = theta_0.copy()
    hist_spsa = []
    evals_spsa = 0
    a = 0.2
    c = 0.1
    for it in range(iters):
        cost = pqc.cost(theta_spsa)
        hist_spsa.append(cost)
        a_k = a / ((it + 1)**0.602)
        c_k = c / ((it + 1)**0.101)
        g = pqc.spsa_gradient(theta_spsa, c_k=c_k, seed=100 + it)
        theta_spsa -= a_k * g
        evals_spsa += 3
        
    # 3. Quantum Natural Gradient (QNG)
    theta_qng = theta_0.copy()
    hist_qng = []
    evals_qng = 0
    lr_qng = 0.15
    for _ in range(iters):
        cost = pqc.cost(theta_qng)
        hist_qng.append(cost)
        step = pqc.quantum_natural_gradient(theta_qng, damping=1e-2)
        theta_qng -= lr_qng * step
        evals_qng += 2 * pqc.num_params + pqc.num_params + 1
        
    plt.figure(figsize=(8, 5))
    plt.plot(hist_ps, 'o-', label="Parameter-Shift GD (Exact)", lw=2)
    plt.plot(hist_spsa, 's-', label="SPSA (Stochastic 2-eval)", lw=2)
    plt.plot(hist_qng, '^-', label="Quantum Natural Gradient (Metric-preconditioned)", lw=2)
    plt.xlabel("Iteration")
    plt.ylabel("Cost Function <H>")
    plt.title("Quantum Optimization: Gradient Methods Comparison")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "gradient_methods_comparison.png"), dpi=200)
    plt.close()
    
    return {
        "iterations": iters,
        "final_cost_ps": hist_ps[-1],
        "evaluations_ps": evals_ps,
        "final_cost_spsa": hist_spsa[-1],
        "evaluations_spsa": evals_spsa,
        "final_cost_qng": hist_qng[-1],
        "evaluations_qng": evals_qng
    }


def run_noise_and_barren_benchmarks(results_dir: str) -> Dict:
    """Benchmark 4, 5, 6: Shot Noise, Hardware Depolarizing Noise, and Barren Plateaus."""
    print("Running Benchmark 4 & 5: Shot Noise & Hardware Depolarizing Noise...")
    # Shot noise
    shot_res = analyze_shot_noise_scaling(n_qubits=3, shots_list=[50, 100, 250, 500, 1000, 2500, 5000])
    
    plt.figure(figsize=(8, 5))
    plt.loglog(shot_res["shots_list"], shot_res["measured_variances"], 'o-', lw=2, label="Measured Variance Var(<H>)")
    plt.loglog(shot_res["shots_list"], shot_res["theoretical_scaling_fit"], 'k--', label="Theoretical 1/S Scaling Fit")
    plt.xlabel("Measurement Shots S (log scale)")
    plt.ylabel("Variance Var(<H>) (log scale)")
    plt.title("Finite Shot Measurement Noise: 1/S Scaling")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "shot_noise_scaling.png"), dpi=200)
    plt.close()
    
    # Depolarizing Hardware Noise
    print("Running Benchmark 5: Depolarizing Noise Degradation...")
    W_test = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ], dtype=float)
    noise_rates = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20]
    noisy_ratios = []
    
    for p_err in noise_rates:
        qaoa = QAOAMaxCut(W_test, p_depth=2)
        trial_ratios = []
        for trial in range(5):
            res = qaoa.optimize(max_iters=30, p_depolarizing=p_err, seed=42 + trial * 10)
            trial_ratios.append(res["approximation_ratio"])
        noisy_ratios.append(float(np.mean(trial_ratios)))
        
    plt.figure(figsize=(8, 5))
    plt.plot(noise_rates, noisy_ratios, 'ro-', lw=2)
    plt.xlabel("Depolarizing Gate Error Rate p_err")
    plt.ylabel("QAOA Max-Cut Approximation Ratio")
    plt.title("Hardware Noise Degradation: Approximation Ratio vs Error Rate")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "hardware_noise_degradation.png"), dpi=200)
    plt.close()
    
    # Barren Plateaus
    print("Running Benchmark 6: Barren Plateaus Gradient Variance Decay...")
    qubits = [2, 3, 4, 5, 6, 7]
    bp_res = analyze_barren_plateaus(qubits_range=qubits, n_samples=30)
    
    plt.figure(figsize=(8, 5))
    plt.semilogy(qubits, bp_res["gradient_variances"], 's-', color="#d62728", lw=2, label="Sample Gradient Variance Var(d<H>/d theta)")
    plt.xlabel("Number of Qubits N")
    plt.ylabel("Gradient Variance Var(d<H>/d theta) (log scale)")
    plt.title("Barren Plateau Phenomenon: Exponential Concentration of Gradients")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "barren_plateaus_decay.png"), dpi=200)
    plt.close()
    
    return {
        "shot_noise": shot_res,
        "depolarizing_noise": {"error_rates": noise_rates, "ratios": noisy_ratios},
        "barren_plateaus": bp_res
    }


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    all_results["benchmark_1_qaoa"] = run_qaoa_benchmarks(results_dir)
    all_results["benchmark_2_vqe"] = run_vqe_benchmarks(results_dir)
    all_results["benchmark_3_gradients"] = run_gradient_methods_comparison(results_dir)
    all_results["benchmark_4_5_6_noise_barren"] = run_noise_and_barren_benchmarks(results_dir)
    
    output_path = os.path.join(results_dir, "quantum_benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\nAll Track C benchmarks completed successfully!")
    print(f"Results and plots saved to: {results_dir}")


if __name__ == "__main__":
    main()
