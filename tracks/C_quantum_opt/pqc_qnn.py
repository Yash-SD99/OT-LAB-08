"""Parameterized Quantum Circuits & Quantum Gradient Methods (Track C).

Implements:
1. Exact Parameter-Shift Rule: d<H>/d theta_k = (<H>_{theta + pi/2} - <H>_{theta - pi/2}) / 2
2. SPSA (Simultaneous Perturbation Stochastic Approximation) requiring only 2 evaluations
3. Quantum Natural Gradient (QNG) using the Fubini-Study Metric Tensor g(theta)
4. Parameterized Quantum Circuit (PQC) Ansatz
"""

import time
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from tracks.C_quantum_opt.quantum_simulator import (
    QuantumCircuitSimulator,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    PAULI_I
)


class ParameterizedCircuit:
    """N-qubit Parameterized Quantum Circuit with single-qubit rotations and entangling gates."""
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        self.n = n_qubits
        self.n_layers = n_layers
        self.dim = 2**n_qubits
        # Each layer has: Ry and Rz on all qubits -> 2 * n_qubits per layer
        # plus final Ry layer -> 2 * n_qubits * n_layers + n_qubits
        self.num_params = 2 * self.n * self.n_layers + self.n
        
        # Simple observable: sum of Z on all qubits: H = sum_i Z_i
        self.H_diag = np.zeros(self.dim, dtype=np.float64)
        for idx in range(self.dim):
            # Eigenvalue of Z_i is (-1)^bit
            val = 0.0
            for i in range(self.n):
                bit = (idx >> (self.n - 1 - i)) & 1
                val += 1.0 if bit == 0 else -1.0
            self.H_diag[idx] = val

    def forward(self, params: np.ndarray) -> QuantumCircuitSimulator:
        """Executes circuit for parameters theta."""
        sim = QuantumCircuitSimulator(self.n)
        p_idx = 0
        
        for l in range(self.n_layers):
            # Ry rotations
            for q in range(self.n):
                sim.ry(q, float(params[p_idx]))
                p_idx += 1
            # Rz rotations
            for q in range(self.n):
                sim.rz(q, float(params[p_idx]))
                p_idx += 1
            # Entangling CNOT ring
            for q in range(self.n):
                sim.cnot(q, (q + 1) % self.n)
                
        # Final Ry layer
        for q in range(self.n):
            sim.ry(q, float(params[p_idx]))
            p_idx += 1
            
        return sim

    def cost(self, params: np.ndarray, n_shots: Optional[int] = None, seed: Optional[int] = None) -> float:
        """Evaluates expectation value <psi(params)| H |psi(params)>."""
        sim = self.forward(params)
        return sim.expectation_diagonal(self.H_diag, n_shots=n_shots, seed=seed)

    def parameter_shift_gradient(self, params: np.ndarray, n_shots: Optional[int] = None, seed: Optional[int] = None) -> np.ndarray:
        """Computes exact analytical gradient via the Parameter-Shift Rule:
        d<H>/d theta_k = (<H>(theta + pi/2 * e_k) - <H>(theta - pi/2 * e_k)) / 2.
        Requires 2 * D circuit evaluations.
        """
        grads = np.zeros(self.num_params, dtype=np.float64)
        shift = np.pi / 2.0
        
        for k in range(self.num_params):
            p_plus = params.copy()
            p_plus[k] += shift
            e_plus = self.cost(p_plus, n_shots=n_shots, seed=seed)
            
            p_minus = params.copy()
            p_minus[k] -= shift
            e_minus = self.cost(p_minus, n_shots=n_shots, seed=seed)
            
            grads[k] = (e_plus - e_minus) / 2.0
            
        return grads

    def spsa_gradient(
        self,
        params: np.ndarray,
        c_k: float = 0.1,
        n_shots: Optional[int] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Simultaneous Perturbation Stochastic Approximation (SPSA).
        Estimates gradient across all D dimensions using only 2 function evaluations:
        g_k = (<H>(theta + c*Delta) - <H>(theta - c*Delta)) / (2 * c * Delta_k)
        where Delta_k in {-1, +1} are Rademacher random perturbations.
        """
        rng = np.random.default_rng(seed)
        delta = rng.choice([-1.0, 1.0], size=self.num_params)
        
        p_plus = params + c_k * delta
        p_minus = params - c_k * delta
        
        e_plus = self.cost(p_plus, n_shots=n_shots, seed=seed)
        e_minus = self.cost(p_minus, n_shots=n_shots, seed=seed)
        
        return (e_plus - e_minus) / (2.0 * c_k * delta)

    def fubini_study_metric_tensor(self, params: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """Computes the Fubini-Study metric tensor g_ij(theta) for Quantum Natural Gradient.
        
        g_ij = Re[ <d_i psi | d_j psi> - <d_i psi | psi> <psi | d_j psi> ]
        Approximated via finite differences of state vector wavefunctions.
        """
        sim_0 = self.forward(params)
        psi_0 = sim_0.state
        
        d_psi = np.zeros((self.num_params, self.dim), dtype=np.complex128)
        for i in range(self.num_params):
            p_step = params.copy()
            p_step[i] += eps
            sim_i = self.forward(p_step)
            d_psi[i] = (sim_i.state - psi_0) / eps
            
        # Overlaps
        inner_d_psi_0 = d_psi @ np.conj(psi_0) # shape (D,)
        
        # Metric tensor: G_ij = Re[ <d_i psi | d_j psi> - (<d_i psi | psi>)(<psi | d_j psi>) ]
        gram_matrix = d_psi @ np.conj(d_psi.T) # shape (D, D)
        outer_prod = np.outer(inner_d_psi_0, np.conj(inner_d_psi_0))
        
        metric = np.real(gram_matrix - outer_prod)
        # Symmetrize and regularize
        metric = 0.5 * (metric + metric.T)
        return metric

    def quantum_natural_gradient(self, params: np.ndarray, damping: float = 1e-3) -> np.ndarray:
        """Quantum Natural Gradient step:
        step = (g(theta) + lambda * I)^{-1} * grad_{param_shift}
        """
        grad = self.parameter_shift_gradient(params)
        g_tensor = self.fubini_study_metric_tensor(params)
        damped_g = g_tensor + damping * np.eye(self.num_params)
        natural_grad = np.linalg.solve(damped_g, grad)
        return natural_grad
