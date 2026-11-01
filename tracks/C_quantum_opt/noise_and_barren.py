"""Empirical Analysis of Finite Shots, Hardware Noise, and Barren Plateaus (Track C).

Investigates:
1. Finite-shot measurement variance: Var(<H>) propto 1 / S
2. Depolarizing hardware noise degradation of QAOA / VQE
3. Barren Plateau phenomenon: Exponential decay of gradient variance Var(d<H>/d theta) ~ O(2^{-N})
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tracks.C_quantum_opt.quantum_simulator import QuantumCircuitSimulator
from tracks.C_quantum_opt.pqc_qnn import ParameterizedCircuit


def analyze_shot_noise_scaling(n_qubits: int = 4, shots_list: Optional[List[int]] = None, n_trials: int = 20) -> Dict:
    """Demonstrates that expectation value variance scales as 1 / S (Shot Noise Limit)."""
    if shots_list is None:
        shots_list = [50, 100, 250, 500, 1000, 2500, 5000]
        
    pqc = ParameterizedCircuit(n_qubits=n_qubits, n_layers=1)
    # Fixed parameter point
    np.random.seed(42)
    theta = np.random.uniform(-np.pi, np.pi, size=pqc.num_params)
    
    exact_val = pqc.cost(theta) # Noiseless exact expectation
    measured_variances = []
    
    for s in shots_list:
        trial_estimates = []
        for trial in range(n_trials):
            val_s = pqc.cost(theta, n_shots=s, seed=1000 + trial * 10 + s)
            trial_estimates.append(val_s)
        var_s = float(np.var(trial_estimates))
        measured_variances.append(var_s)
        
    return {
        "n_qubits": n_qubits,
        "exact_expectation": float(exact_val),
        "shots_list": shots_list,
        "measured_variances": measured_variances,
        "theoretical_scaling_fit": [float(measured_variances[0] * (shots_list[0] / s)) for s in shots_list]
    }


def analyze_barren_plateaus(qubits_range: Optional[List[int]] = None, n_samples: int = 25) -> Dict:
    """Measures gradient variance Var(d<H>/d theta_1) across qubit counts N in [2..8].
    
    According to McClean et al. (2018), for deep random parameterized circuits,
    the gradient variance decays exponentially with system size:
    Var[d<H>/d theta] ~ O(2^{-N}).
    """
    if qubits_range is None:
        qubits_range = [2, 3, 4, 5, 6, 7]
        
    gradient_variances = []
    gradient_means = []
    
    for n in qubits_range:
        # Layer depth = n ensures 2-design expressibility
        pqc = ParameterizedCircuit(n_qubits=n, n_layers=n)
        sample_grads = []
        
        for sample_idx in range(n_samples):
            rng = np.random.default_rng(2000 + n * 100 + sample_idx)
            theta_rand = rng.uniform(-np.pi, np.pi, size=pqc.num_params)
            
            # Compute partial derivative with respect to first parameter theta_0
            shift = np.pi / 2.0
            p_plus = theta_rand.copy()
            p_plus[0] += shift
            p_minus = theta_rand.copy()
            p_minus[0] -= shift
            
            grad_0 = (pqc.cost(p_plus) - pqc.cost(p_minus)) / 2.0
            sample_grads.append(float(grad_0))
            
        var_grad = float(np.var(sample_grads))
        mean_grad = float(np.mean(sample_grads))
        gradient_variances.append(var_grad)
        gradient_means.append(mean_grad)
        
    return {
        "qubits_range": qubits_range,
        "gradient_variances": gradient_variances,
        "gradient_means": gradient_means,
        "n_samples_per_qubit": n_samples
    }
