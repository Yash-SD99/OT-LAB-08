"""Empirical Analysis of Finite Shots, Hardware Noise, and Barren Plateaus (Track C).

Investigates:
1. Finite-shot measurement variance: Var(<H>) propto 1 / S (multinomial/binomial sampling).
2. Depolarizing hardware noise degradation: QAOA (depth p in {1,2,3}) and VQE (layers L in {1,2,3}) vs error rate.
3. Barren Plateau phenomenon:
   - Evaluated across scale N in [2..7] qubits with 50 samples per point.
   - Global observable (H = Z^{ox n}) vs Local observable (H = Z_0).
   - Log-scale variance vs qubits with fitted exponential decay rates.
   - Accurate circuit architecture description: Random alternating layered hardware-efficient ansatz
     (forming a pseudo-random 2-design, NOT Haar-distributed random unitaries).
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from tracks.C_quantum_opt.quantum_simulator import QuantumCircuitSimulator
from tracks.C_quantum_opt.pqc_qnn import ParameterizedCircuit


def analyze_shot_noise_scaling(
    n_qubits: int = 3,
    shots_list: Optional[List[int]] = None,
    n_trials: int = 25
) -> Dict:
    """Demonstrates that expectation value variance scales as 1 / S under Born-rule multinomial sampling."""
    if shots_list is None:
        shots_list = [50, 100, 250, 500, 1000, 2500, 5000]
        
    pqc = ParameterizedCircuit(n_qubits=n_qubits, n_layers=1)
    np.random.seed(42)
    theta = np.random.uniform(-np.pi, np.pi, size=pqc.num_params)
    
    exact_val = pqc.cost(theta) # Noiseless expectation value
    measured_variances = []
    
    for s in shots_list:
        trial_estimates = []
        for trial in range(n_trials):
            val_s = pqc.cost(theta, n_shots=s, seed=1000 + trial * 10 + s)
            trial_estimates.append(val_s)
        var_s = float(np.var(trial_estimates))
        measured_variances.append(var_s)
        
    # Fit theoretical 1/S slope
    theoretical_fit = [float(measured_variances[0] * (shots_list[0] / s)) for s in shots_list]
    
    return {
        "n_qubits": n_qubits,
        "exact_expectation": float(exact_val),
        "shots_list": shots_list,
        "measured_variances": measured_variances,
        "theoretical_scaling_fit": theoretical_fit,
        "sampling_distribution": "multinomial/binomial Born-rule projection",
        "n_trials": n_trials
    }


def analyze_barren_plateaus(
    qubits_range: Optional[List[int]] = None,
    n_samples: int = 50
) -> Dict:
    """Measures gradient variance Var(d<H>/d theta_1) across qubit counts N in [2..7].
    
    Compares:
    1. Global Cost: H_global = Z^{ox N} (projection onto all qubits).
       Theory: Exponential barren plateau Var ~ O(2^{-N}) at depth L >= 1.
    2. Local Cost: H_local = Z_0 (single-qubit observable).
       Theory: Polynomial decay or flat variance for shallow circuits.
       
    Ansatz: Hardware-efficient alternating Ry-Rz rotation layers with entangling CNOT ring.
    Note: Circuit is a pseudo-random 2-design, NOT Haar-distributed random unitaries.
    """
    if qubits_range is None:
        qubits_range = [2, 3, 4, 5, 6, 7]
        
    global_variances = []
    local_variances = []
    
    shift = np.pi / 2.0
    
    for n in qubits_range:
        dim = 2**n
        # Global Hamiltonian: Z^{ox n}
        H_global_diag = np.zeros(dim, dtype=float)
        for idx in range(dim):
            parity = bin(idx).count('1') % 2
            H_global_diag[idx] = -1.0 if parity == 1 else 1.0
            
        # Local Hamiltonian: Z_0
        H_local_diag = np.zeros(dim, dtype=float)
        for idx in range(dim):
            bit0 = (idx >> (n - 1)) & 1
            H_local_diag[idx] = 1.0 if bit0 == 0 else -1.0
            
        pqc = ParameterizedCircuit(n_qubits=n, n_layers=n)
        
        sample_grads_global = []
        sample_grads_local = []
        
        for s in range(n_samples):
            rng = np.random.default_rng(4000 + n * 200 + s)
            theta = rng.uniform(-np.pi, np.pi, size=pqc.num_params)
            
            # Parameter-shift on first parameter theta_0
            p_plus = theta.copy()
            p_plus[0] += shift
            sim_plus = pqc.forward(p_plus)
            
            p_minus = theta.copy()
            p_minus[0] -= shift
            sim_minus = pqc.forward(p_minus)
            
            # Global gradient
            e_g_plus = sim_plus.expectation_diagonal(H_global_diag)
            e_g_minus = sim_minus.expectation_diagonal(H_global_diag)
            grad_g = (e_g_plus - e_g_minus) / 2.0
            sample_grads_global.append(float(grad_g))
            
            # Local gradient
            e_l_plus = sim_plus.expectation_diagonal(H_local_diag)
            e_l_minus = sim_minus.expectation_diagonal(H_local_diag)
            grad_l = (e_l_plus - e_l_minus) / 2.0
            sample_grads_local.append(float(grad_l))
            
        global_variances.append(float(np.var(sample_grads_global)))
        local_variances.append(float(np.var(sample_grads_local)))
        
    # Fit exponential decay rate: log(Var) = -alpha * N + c
    log_v_global = np.log(np.array(global_variances) + 1e-15)
    log_v_local = np.log(np.array(local_variances) + 1e-15)
    q_arr = np.array(qubits_range, dtype=float)
    
    slope_global, intercept_global = np.polyfit(q_arr, log_v_global, 1)
    slope_local, intercept_local = np.polyfit(q_arr, log_v_local, 1)
    
    return {
        "qubits_range": qubits_range,
        "n_samples_per_qubit": n_samples,
        "global_gradient_variances": global_variances,
        "local_gradient_variances": local_variances,
        "fitted_decay_rate_global": float(-slope_global),
        "fitted_decay_rate_local": float(-slope_local),
        "ansatz_description": "Hardware-efficient alternating Ry-Rz with CNOT ring (pseudo-random 2-design, NOT Haar-distributed)"
    }


def analyze_depolarizing_noise_sweep(
    qaoa_graph: np.ndarray,
    noise_rates: Optional[List[float]] = None,
    n_trials: int = 10
) -> Dict:
    """Evaluates QAOA Max-Cut and VQE Ising degradation across depolarizing rates and depths."""
    from tracks.C_quantum_opt.qaoa_maxcut import QAOAMaxCut
    from tracks.C_quantum_opt.vqe_ising import VQEIsing
    
    if noise_rates is None:
        noise_rates = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20]
        
    qaoa_by_depth = {1: [], 2: [], 3: []}
    for p in [1, 2, 3]:
        for rate in noise_rates:
            trial_ratios = []
            for trial in range(n_trials):
                qaoa = QAOAMaxCut(qaoa_graph, p_depth=p)
                res = qaoa.optimize(max_iters=25, p_depolarizing=rate, seed=5000 + p * 100 + trial)
                trial_ratios.append(res["approximation_ratio"])
            qaoa_by_depth[p].append(float(np.mean(trial_ratios)))
            
    vqe_by_layers = {1: [], 2: [], 3: []}
    for L in [1, 2, 3]:
        vqe = VQEIsing(n_qubits=4, J=1.0, h=0.5, layers=L)
        for rate in noise_rates:
            trial_errors = []
            for trial in range(n_trials):
                # Apply depolarizing noise to ansatz state evaluation
                ansatz = vqe.build_ansatz(np.ones(vqe.num_params) * 0.5)
                ansatz.apply_depolarizing_noise(rate)
                noisy_energy = float(np.real(np.vdot(ansatz.state, vqe.H_full @ ansatz.state)))
                trial_errors.append(abs(noisy_energy - vqe.exact_ground_energy))
            vqe_by_layers[L].append(float(np.mean(trial_errors)))
            
    return {
        "noise_rates": noise_rates,
        "qaoa_by_depth": qaoa_by_depth,
        "vqe_by_layers": vqe_by_layers,
        "n_trials": n_trials
    }
