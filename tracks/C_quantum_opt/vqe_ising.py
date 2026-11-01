"""Variational Quantum Eigensolver (VQE) on Transverse-Field Ising Hamiltonian (Track C).

Finds the ground state energy of:
H = -J * sum_{i=0}^{N-2} Z_i Z_{i+1} - h * sum_{i=0}^{N-1} X_i
using a hardware-efficient parameterized quantum circuit and compares against
exact classical diagonalization.
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from tracks.C_quantum_opt.quantum_simulator import (
    QuantumCircuitSimulator,
    PAULI_X,
    PAULI_Z,
    PAULI_I
)


def construct_ising_full_hamiltonian(n_qubits: int, J: float = 1.0, h: float = 0.5) -> np.ndarray:
    """Constructs the full 2^N x 2^N matrix of the Transverse-Field Ising Model for exact diagonalization."""
    dim = 2**n_qubits
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Term 1: -J * sum_{i=0}^{N-2} Z_i Z_{i+1}
    for i in range(n_qubits - 1):
        op_list = [PAULI_I] * n_qubits
        op_list[i] = PAULI_Z
        op_list[i + 1] = PAULI_Z
        term = op_list[0]
        for op in op_list[1:]:
            term = np.kron(term, op)
        H -= J * term
        
    # Term 2: -h * sum_{i=0}^{N-1} X_i
    for i in range(n_qubits):
        op_list = [PAULI_I] * n_qubits
        op_list[i] = PAULI_X
        term = op_list[0]
        for op in op_list[1:]:
            term = np.kron(term, op)
        H -= h * term
        
    return H


class VQEIsing:
    """Hardware-efficient VQE for Transverse-Field Ising Model."""
    def __init__(self, n_qubits: int = 4, J: float = 1.0, h: float = 0.5, layers: int = 2):
        self.n = n_qubits
        self.J = float(J)
        self.h = float(h)
        self.layers = int(layers)
        self.dim = 2**n_qubits
        
        # Number of parameters: n_qubits * (layers + 1) for Ry rotations
        self.num_params = self.n * (self.layers + 1)
        
        # Exact diagonalization baseline
        self.H_full = construct_ising_full_hamiltonian(self.n, self.J, self.h)
        self.eigvals = np.linalg.eigvalsh(self.H_full)
        self.exact_ground_energy = float(self.eigvals[0])

    def build_ansatz(self, params: np.ndarray) -> QuantumCircuitSimulator:
        """Constructs hardware-efficient ansatz:
        Layer 0: Ry(theta_{i,0}) on all qubits
        For each layer l:
          Entangling CNOT linear chain (0->1, 1->2, ...)
          Ry(theta_{i,l}) on all qubits
        """
        sim = QuantumCircuitSimulator(self.n)
        param_idx = 0
        
        # Initial rotation layer
        for q in range(self.n):
            sim.ry(q, float(params[param_idx]))
            param_idx += 1
            
        for l in range(self.layers):
            # Entangling linear CNOT chain
            for q in range(self.n - 1):
                sim.cnot(q, q + 1)
                
            # Rotation layer
            for q in range(self.n):
                sim.ry(q, float(params[param_idx]))
                param_idx += 1
                
        return sim

    def energy(self, params: np.ndarray, n_shots: Optional[int] = None, seed: Optional[int] = None) -> float:
        """Computes expected energy <psi(theta)| H |psi(theta)>."""
        sim = self.build_ansatz(params)
        psi = sim.state
        
        if n_shots is None:
            # Exact expectation value: psi^dagger H psi
            e_val = float(np.real(np.vdot(psi, self.H_full @ psi)))
            return e_val
        else:
            # Finite shot simulation with binomial variance
            rng = np.random.default_rng(seed)
            clean_e = float(np.real(np.vdot(psi, self.H_full @ psi)))
            # Standard error scale ~ 1 / sqrt(shots)
            variance = float(np.real(np.vdot(psi, (self.H_full @ self.H_full) @ psi)) - clean_e**2)
            shot_noise = rng.normal(0.0, np.sqrt(max(variance, 1e-8) / n_shots))
            return clean_e + shot_noise

    def optimize(self, initial_params: Optional[np.ndarray] = None, max_iters: int = 60, seed: int = 42) -> Dict:
        """Optimizes ansatz parameters to find the minimum eigenvalue."""
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        
        if initial_params is None:
            initial_params = rng.uniform(-np.pi, np.pi, size=self.num_params)
            
        history = []
        def obj(p):
            val = self.energy(p)
            history.append(val)
            return val
            
        res = minimize(obj, initial_params, method="BFGS", options={"maxiter": max_iters})
        elapsed = time.perf_counter() - t0
        
        best_energy = float(res.fun)
        energy_error = abs(best_energy - self.exact_ground_energy)
        
        return {
            "method": f"VQE (layers={self.layers})",
            "n_qubits": self.n,
            "num_parameters": self.num_params,
            "vqe_ground_energy": best_energy,
            "exact_ground_energy": self.exact_ground_energy,
            "energy_error": float(energy_error),
            "relative_error": float(energy_error / abs(self.exact_ground_energy)),
            "runtime_ms": elapsed * 1000.0,
            "iterations": len(history),
            "is_simulation": True,
            "label": "noise-model simulation, not real hardware"
        }
