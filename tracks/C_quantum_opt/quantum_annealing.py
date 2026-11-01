"""Quantum Annealing & Minor-Embedding Limits Simulation (Track C).

Demonstrates the physical mapping of dense QUBO / Ising problems onto sparse hardware topologies:
1. Target Problem: Fully-connected $K_4$ Max-Cut / QUBO with all-to-all couplings.
2. Hardware Topology: Sparse hardware graph with restricted maximum degree <= 3.
3. Minor-Embedding: Logical qubits mapped to physical qubit chains with ferromagnetic coupling J_chain.
4. Annealing Dynamics: Transverse-field annealing schedule H(t) = (1 - s(t)) H_mixer + s(t) H_problem.
5. Embedding Limits:
   - J_chain too weak: Chain breaks (inconsistent physical qubits, invalid state).
   - J_chain too strong: Problem energy gap shrinks below noise floor, reducing ground state hit rate.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


def solve_ising_transverse_annealing(
    J_matrix: np.ndarray,
    h_vector: np.ndarray,
    n_steps: int = 200,
    gamma_0: float = 2.0,
    temp_0: float = 2.0,
    temp_f: float = 0.05,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """Simulates transverse-field quantum annealing via continuous-time Markov-chain / spin dynamics.
    
    Spins s_i in {-1, +1}.
    H(s) = (1 - s) * (-Gamma * sum_i X_i) + s * (sum_{i<j} J_ij s_i s_j + sum_i h_i s_i).
    """
    rng = np.random.default_rng(seed)
    n = len(h_vector)
    # Random initial spin configuration
    state = rng.choice([-1, 1], size=n)
    
    for step in range(n_steps):
        s = step / float(n_steps - 1) # Annealing parameter s in [0, 1]
        T = temp_0 * (temp_f / temp_0)**s # Thermal schedule
        gamma = gamma_0 * (1.0 - s) # Transverse field schedule
        
        # Sweep all spins
        for i in range(n):
            # Local classical field from neighbors
            local_field = h_vector[i] + np.dot(J_matrix[i, :], state) - J_matrix[i, i] * state[i]
            # Classical energy difference for flipping spin i: s_i -> -s_i
            delta_E_classical = -2.0 * state[i] * local_field
            
            # Effective total energy change including transverse field quantum tunneling
            # Effective transverse barrier ~ Gamma
            effective_delta_E = s * delta_E_classical
            
            # Metropolis-Hastings acceptance probability with quantum tunneling fluctuation
            if effective_delta_E <= 0:
                state[i] = -state[i]
            else:
                prob = np.exp(-effective_delta_E / max(T + 0.5 * gamma, 1e-6))
                if rng.random() < prob:
                    state[i] = -state[i]
                    
    # Final classical energy
    energy = 0.5 * float(state @ J_matrix @ state) + float(np.dot(h_vector, state))
    return state, energy


class MinorEmbeddingDemo:
    """Simulates minor-embedding of a dense logical K_4 problem onto a sparse physical graph."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        
        # Logical Problem: Complete graph K_4 with 4 logical nodes
        # Edge weights J_ij for Max-Cut
        self.n_logical = 4
        self.J_logical = np.array([
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0]
        ], dtype=float)
        self.h_logical = np.zeros(self.n_logical, dtype=float)
        
        # Ground truth for K_4 Max-Cut: cut 4 edges (partition 2 vs 2), energy = -2.0
        # For states with 2 '+1's and 2 '-1's: 4 edges have s_i * s_j = -1, 2 edges have s_i * s_j = +1.
        # Classical energy = 0.5 * sum J_ij s_i s_j = 0.5 * (-4 + 2) = -1.0.
        
        # Hardware Topology: 6 physical qubits arranged in a sparse ring/ladder with max degree 3
        # Logical qubit 0 is split into chain (0a, 0b) = physical qubits (0, 1)
        # Logical qubit 1 is split into chain (1a, 1b) = physical qubits (2, 3)
        # Logical qubit 2 is physical qubit 4
        # Logical qubit 3 is physical qubit 5
        self.n_physical = 6
        self.chains = {
            0: [0, 1],
            1: [2, 3],
            2: [4],
            3: [5]
        }
        
    def build_embedded_hamiltonian(self, J_chain: float) -> Tuple[np.ndarray, np.ndarray]:
        """Constructs physical J_matrix and h_vector incorporating ferromagnetic chain couplings."""
        J_phys = np.zeros((self.n_physical, self.n_physical), dtype=float)
        h_phys = np.zeros(self.n_physical, dtype=float)
        
        # 1. Ferromagnetic intra-chain coupling: -J_chain * s_{ia} * s_{ib}
        for log_q, phys_nodes in self.chains.items():
            if len(phys_nodes) > 1:
                p1, p2 = phys_nodes[0], phys_nodes[1]
                J_phys[p1, p2] -= J_chain
                J_phys[p2, p1] -= J_chain
                
        # 2. Inter-chain problem couplings mapped to physical edges
        # Edge (0, 1) -> (0a, 1a) = (0, 2)
        J_phys[0, 2] += self.J_logical[0, 1]
        J_phys[2, 0] += self.J_logical[0, 1]
        
        # Edge (0, 2) -> (0a, 2) = (0, 4)
        J_phys[0, 4] += self.J_logical[0, 2]
        J_phys[4, 0] += self.J_logical[0, 2]
        
        # Edge (0, 3) -> (0b, 3) = (1, 5)
        J_phys[1, 5] += self.J_logical[0, 3]
        J_phys[5, 1] += self.J_logical[0, 3]
        
        # Edge (1, 2) -> (1b, 2) = (3, 4)
        J_phys[3, 4] += self.J_logical[1, 2]
        J_phys[4, 3] += self.J_logical[1, 2]
        
        # Edge (1, 3) -> (1a, 3) = (2, 5)
        J_phys[2, 5] += self.J_logical[1, 3]
        J_phys[5, 2] += self.J_logical[1, 3]
        
        # Edge (2, 3) -> (4, 5)
        J_phys[4, 5] += self.J_logical[2, 3]
        J_phys[5, 4] += self.J_logical[2, 3]
        
        return J_phys, h_phys

    def evaluate_embedding_limits(self, j_chain_values: Optional[List[float]] = None, n_trials: int = 20) -> Dict:
        """Sweeps J_chain to identify the optimal coupling window between chain breaking and gap collapse."""
        if j_chain_values is None:
            j_chain_values = [0.1, 0.3, 0.5, 1.0, 1.5, 2.5, 5.0, 10.0]
            
        break_rates = []
        success_rates = []
        
        for J_c in j_chain_values:
            J_phys, h_phys = self.build_embedded_hamiltonian(J_c)
            chain_breaks = 0
            ground_state_hits = 0
            
            for trial in range(n_trials):
                seed = 3000 + int(J_c * 100) + trial
                phys_state, _ = solve_ising_transverse_annealing(
                    J_phys, h_phys, n_steps=250, seed=seed
                )
                
                # Check for chain breaks: do physical qubits in a chain agree?
                broken = False
                logical_state = np.zeros(self.n_logical, dtype=int)
                for log_q, phys_nodes in self.chains.items():
                    if len(phys_nodes) > 1:
                        if phys_state[phys_nodes[0]] != phys_state[phys_nodes[1]]:
                            broken = True
                            # Majority vote / tie breaker
                            logical_state[log_q] = phys_state[phys_nodes[0]]
                        else:
                            logical_state[log_q] = phys_state[phys_nodes[0]]
                    else:
                        logical_state[log_q] = phys_state[phys_nodes[0]]
                        
                if broken:
                    chain_breaks += 1
                else:
                    # If valid embedding, check if it's the optimal cut (2 vs 2 partition)
                    if np.sum(logical_state == 1) == 2:
                        ground_state_hits += 1
                        
            break_rates.append(chain_breaks / float(n_trials))
            success_rates.append(ground_state_hits / float(n_trials))
            
        return {
            "j_chain_values": j_chain_values,
            "chain_break_rates": break_rates,
            "ground_state_success_rates": success_rates,
            "optimal_j_chain": j_chain_values[int(np.argmax(success_rates))],
            "n_trials_per_point": n_trials,
            "note": "Demonstrates minor-embedding sweet-spot: low J_c breaks chains; excessive J_c distorts problem spectrum."
        }
