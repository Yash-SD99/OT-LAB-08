"""Quantum Approximate Optimization Algorithm (QAOA) for Max-Cut (Track C).

Solves the Track B Max-Cut QUBO using QAOA with depth p = 1, 2, 3:
1. Cost Hamiltonian: H_C = sum_{(i,j) in E} (W_ij / 2) * (I - Z_i Z_j)
2. Mixer Hamiltonian: H_M = sum_i X_i
3. Parameterized Ansatz: |psi(gamma, beta)> = prod_{l=1}^p exp(-i * beta_l * H_M) exp(-i * gamma_l * H_C) |+>^N
4. Classical optimization of variational parameters (gamma, beta)
5. Approximation ratio comparison against Track B classical exact & simulated annealing baselines
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from tracks.C_quantum_opt.quantum_simulator import (
    QuantumCircuitSimulator,
    HADAMARD,
    rx_matrix
)


class QAOAMaxCut:
    """QAOA solver for Max-Cut on undirected graph with adjacency matrix W."""
    def __init__(self, adjacency_matrix: np.ndarray, p_depth: int = 1):
        self.W = np.asarray(adjacency_matrix, dtype=np.float64)
        self.n = self.W.shape[0]
        self.p = int(p_depth)
        self.dim = 2**self.n
        
        # Precompute diagonal Cost Hamiltonian: H_C(x) = -Cut(x) for all 2^N states
        self.cut_values = np.zeros(self.dim, dtype=np.float64)
        for idx in range(self.dim):
            # Extract binary bitstring
            bits = np.array([(idx >> (self.n - 1 - i)) & 1 for i in range(self.n)])
            # Cut value: sum_{i < j} W_ij * (bits[i] ^ bits[j])
            cut = 0.5 * np.sum(self.W * (bits[:, None] != bits[None, :]))
            self.cut_values[idx] = float(cut)
            
        self.max_cut = float(np.max(self.cut_values))
        # We minimize H_C = -Cut(x)
        self.H_C_diag = -self.cut_values

    def build_state(
        self,
        gammas: np.ndarray,
        betas: np.ndarray,
        p_depolarizing: float = 0.0,
        seed: Optional[int] = None
    ) -> QuantumCircuitSimulator:
        """Constructs the QAOA state |psi(gamma, beta)>."""
        sim = QuantumCircuitSimulator(self.n, seed=seed)
        
        # Step 1: Initial state |+>^N = H^N |0>^N
        for q in range(self.n):
            sim.h(q)
            
        # Step 2: Alternate p layers of Cost and Mixer unitaries
        for l in range(self.p):
            gamma = float(gammas[l])
            beta = float(betas[l])
            
            # Problem Unitary: exp(-i * gamma * H_C)
            phase_factors = np.exp(-1j * gamma * self.H_C_diag)
            sim.state = sim.state * phase_factors
            
            if p_depolarizing > 0.0:
                sim.apply_depolarizing_noise(p_depolarizing)
                
            # Mixer Unitary: exp(-i * beta * H_M) = prod_i Rx(2 * beta)
            for q in range(self.n):
                sim.rx(q, 2.0 * beta)
                
            if p_depolarizing > 0.0:
                sim.apply_depolarizing_noise(p_depolarizing)
                
        return sim

    def cost_function(
        self,
        params: np.ndarray,
        n_shots: Optional[int] = None,
        p_depolarizing: float = 0.0,
        seed: Optional[int] = None
    ) -> float:
        """Evaluates expected energy <H_C> = - <Cut>."""
        gammas = params[:self.p]
        betas = params[self.p:]
        sim = self.build_state(gammas, betas, p_depolarizing=p_depolarizing, seed=seed)
        energy = sim.expectation_diagonal(self.H_C_diag, n_shots=n_shots, seed=seed)
        return float(energy)

    def optimize(
        self,
        initial_params: Optional[np.ndarray] = None,
        max_iters: int = 50,
        n_shots: Optional[int] = None,
        p_depolarizing: float = 0.0,
        seed: int = 42
    ) -> Dict:
        """Optimizes variational parameters (gamma, beta) to maximize expected cut."""
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        
        if initial_params is None:
            # Linear ramp heuristic: gamma in [0, pi], beta in [pi/4, 0]
            gammas = np.linspace(0.1, 0.8, self.p)
            betas = np.linspace(0.8, 0.1, self.p)
            initial_params = np.concatenate([gammas, betas])
            
        history = []
        
        def obj_fn(p_arr):
            val = self.cost_function(p_arr, n_shots=n_shots, p_depolarizing=p_depolarizing, seed=seed)
            history.append(float(val))
            return val
            
        # Optimize using COBYLA (derivative-free bounded simplex)
        res = minimize(
            obj_fn,
            initial_params,
            method="COBYLA",
            options={"maxiter": max_iters, "rhobeg": 0.2}
        )
        
        elapsed = time.perf_counter() - t0
        best_params = res.x
        best_gammas = best_params[:self.p]
        best_betas = best_params[self.p:]
        
        # Evaluate final state properties with actual noise level
        eval_sim = self.build_state(best_gammas, best_betas, p_depolarizing=p_depolarizing, seed=seed)
        final_expected_energy = eval_sim.expectation_diagonal(self.H_C_diag)
        final_expected_cut = -final_expected_energy
        approx_ratio = final_expected_cut / max(self.max_cut, 1e-9)
        
        # Probability of measuring the exact optimal cut bitstrings
        probs = eval_sim.probabilities()
        optimal_indices = np.where(np.isclose(self.cut_values, self.max_cut))[0]
        prob_optimal = float(np.sum(probs[optimal_indices]))
        
        return {
            "method": f"QAOA (p={self.p})",
            "p_depth": self.p,
            "optimal_gammas": best_gammas.tolist(),
            "optimal_betas": best_betas.tolist(),
            "final_expected_cut": float(final_expected_cut),
            "max_cut_ground_truth": self.max_cut,
            "approximation_ratio": float(approx_ratio),
            "prob_optimal_ground_state": prob_optimal,
            "runtime_ms": elapsed * 1000.0,
            "iterations": len(history),
            "is_simulation": True,
            "label": "noise-model simulation, not real hardware"
        }
