"""Classical QUBO & Simulated Annealing Solvers (Baseline for Track C).

Includes:
1. Exact Enumeration Solver (O(2^N) optimal baseline for small graphs)
2. Classical Simulated Annealing Solver (thermal analog of Quantum Annealing)
3. Minor Embedding Overhead Analysis for Annealing Hardware (Chimera / Pegasus topologies)
"""

import itertools
import time
from typing import Dict, List, Optional, Tuple
import numpy as np


def solve_qubo_exact(Q: np.ndarray) -> Dict:
    """Solves QUBO exactly via exhaustive enumeration over all 2^N binary states."""
    t0 = time.perf_counter()
    N = Q.shape[0]
    if N > 20:
        raise ValueError(f"Exact enumeration intractable for N={N} > 20.")
        
    best_x = None
    best_energy = float("inf")
    
    # Iterate through all binary bitstrings
    for bits in itertools.product([0, 1], repeat=N):
        x = np.array(bits, dtype=np.float64)
        energy = float(x @ Q @ x)
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()
            
    elapsed = time.perf_counter() - t0
    return {
        "method": "Exact Classical Enumeration",
        "best_state": best_x.astype(int).tolist() if best_x is not None else [],
        "min_energy": best_energy,
        "runtime_ms": elapsed * 1000.0,
        "states_evaluated": 2**N
    }


def solve_qubo_simulated_annealing(
    Q: np.ndarray,
    initial_temp: float = 10.0,
    cooling_rate: float = 0.95,
    min_temp: float = 1e-4,
    steps_per_temp: int = 50,
    seed: int = 42
) -> Dict:
    """Solves QUBO via Classical Simulated Annealing (Thermal Annealer Baseline).
    
    Uses O(N) local energy delta computation for single-spin flips:
    Delta E = (1 - 2*x_i) * (Q_ii + sum_{j != i} (Q_ij + Q_ji) * x_j).
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    N = Q.shape[0]
    
    # Symmetrize Q for local delta evaluations
    Q_sym = 0.5 * (Q + Q.T)
    
    # Initialize random binary state
    x = rng.integers(0, 2, size=N).astype(np.float64)
    current_energy = float(x @ Q @ x)
    
    best_x = x.copy()
    best_energy = current_energy
    
    temp = float(initial_temp)
    total_evals = 0
    energy_history = []
    
    while temp > min_temp:
        for _ in range(steps_per_temp):
            total_evals += 1
            # Pick a random variable to flip
            i = rng.integers(0, N)
            
            # Efficient O(N) delta computation:
            # If x_i = 0, new x_i = 1 (dx = +1). If x_i = 1, new x_i = 0 (dx = -1).
            dx = 1.0 - 2.0 * x[i]
            delta_E = dx * (Q_sym[i, i] + 2.0 * np.dot(Q_sym[i, :], x) - 2.0 * Q_sym[i, i] * x[i])
            
            # Metropolis acceptance criterion
            if delta_E < 0.0 or rng.random() < np.exp(-delta_E / max(temp, 1e-12)):
                x[i] = 1.0 - x[i]
                current_energy += delta_E
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = x.copy()
                    
        energy_history.append(best_energy)
        temp *= cooling_rate
        
    elapsed = time.perf_counter() - t0
    return {
        "method": "Simulated Annealing (Classical Annealer)",
        "best_state": best_x.astype(int).tolist(),
        "min_energy": best_energy,
        "runtime_ms": elapsed * 1000.0,
        "total_evals": total_evals,
        "energy_history": energy_history
    }


def analyze_minor_embedding_limits(n_logical_qubits: int, graph_density: float = 1.0) -> Dict:
    """Analyzes the minor embedding overhead for mapping a dense QUBO onto hardware annealers.
    
    NOTE: The formulas below are theoretical upper-bound approximations for complete graph embeddings.
    Citations:
      1. Choi, V. (2008). 'Minor-embedding in adiabatic quantum computation: I. The parameter setting problem',
         Quantum Information Processing, 7(5), 193-209.
      2. D-Wave Systems (2020). 'The D-Wave 5000Q Advantage Processor: Pegasus Architecture Whitepaper'.
    
    D-Wave Chimera topology (degree 6) and Pegasus topology (degree 15) cannot natively
    embed fully connected K_N graphs. Fully connected logical graphs require 'chains'
    of physical qubits where each logical variable is represented by multiple ferromagnetic-coupled
    physical qubits.
    
    Approximate complete graph scaling:
      Chimera (K_4,4 unit cells): Physical qubits ~ ceil(N^2 / 4)
      Pegasus (degree 15):        Physical qubits ~ ceil(N^2 / 12)
    """
    N = n_logical_qubits
    chimera_physical = int(np.ceil((N**2) / 4.0)) if N > 4 else N
    pegasus_physical = int(np.ceil((N**2) / 12.0)) if N > 12 else N
    
    return {
        "logical_qubits": N,
        "graph_density": graph_density,
        "chimera_estimated_physical_qubits": chimera_physical,
        "pegasus_estimated_physical_qubits": pegasus_physical,
        "qubit_expansion_factor_chimera": float(chimera_physical / max(N, 1)),
        "qubit_expansion_factor_pegasus": float(pegasus_physical / max(N, 1)),
        "limiting_factors": [
            "Ferromagnetic chain breaks (destroys ground-state solution if coupling J_chain is too weak)",
            "Precision loss due to digital-to-analog converter (DAC) quantization of couplings",
            "Energy scale attenuation: adding penalty couplings compresses the dynamic range of problem Hamiltonian"
        ]
    }
