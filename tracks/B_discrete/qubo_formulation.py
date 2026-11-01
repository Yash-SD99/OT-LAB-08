"""QUBO and Ising Formulations with Binary-to-Spin Conversion & Penalty Terms (Track B).

Provides:
1. General QUBO energy evaluation: E(x) = x^T Q x
2. Binary-to-Spin Conversion (x in {0, 1} <-> s in {-1, +1}) and Ising Hamiltonian extraction:
   H(s) = sum_{i<j} J_ij s_i s_j + sum_i h_i s_i + C
3. Max-Cut QUBO and Ising formulation on graph adjacency matrix W
4. Constrained Knapsack/Resource QUBO with quadratic penalty terms:
   Q(x) = -sum_i v_i x_i + lambda * (sum_i w_i x_i - W_target)^2
"""

from typing import Dict, Tuple
import numpy as np


def qubo_energy(Q: np.ndarray, x: np.ndarray) -> float:
    """Evaluates QUBO energy: E(x) = x^T Q x."""
    x = np.asarray(x, dtype=np.float64)
    return float(x @ Q @ x)


def ising_energy(J: np.ndarray, h: np.ndarray, C: float, s: np.ndarray) -> float:
    """Evaluates Ising Hamiltonian energy: H(s) = 0.5 * s^T J s + h^T s + C."""
    s = np.asarray(s, dtype=np.float64)
    return float(0.5 * s @ J @ s + h @ s + C)


def binary_to_spin(x: np.ndarray) -> np.ndarray:
    """Maps binary x in {0, 1} to spin s in {-1, +1}: s = 2x - 1."""
    return 2.0 * np.asarray(x, dtype=np.float64) - 1.0


def spin_to_binary(s: np.ndarray) -> np.ndarray:
    """Maps spin s in {-1, +1} to binary x in {0, 1}: x = (s + 1) / 2."""
    return (np.asarray(s, dtype=np.float64) + 1.0) / 2.0


def qubo_to_ising(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Converts a QUBO matrix Q to Ising parameters (J, h, C).
    
    Substitute x_i = (s_i + 1) / 2 into x^T Q x:
    x^T Q x = sum_i Q_ii (s_i+1)/2 + sum_{i != j} Q_ij (s_i+1)(s_j+1)/4
    
    Returns:
      J: Couplings matrix (N x N, symmetric, zero diagonal)
      h: Local magnetic fields vector (N,)
      C: Energy constant offset (float)
    """
    Q = np.asarray(Q, dtype=np.float64)
    N = Q.shape[0]
    
    # Symmetrize off-diagonals
    Q_sym = 0.5 * (Q + Q.T)
    
    J = np.zeros((N, N), dtype=np.float64)
    h = np.zeros(N, dtype=np.float64)
    C = 0.0
    
    for i in range(N):
        # Diagonal term contribution: Q_ii * (s_i + 1) / 2 = 0.5 * Q_ii * s_i + 0.5 * Q_ii
        h[i] += 0.5 * Q_sym[i, i]
        C += 0.5 * Q_sym[i, i]
        
        for j in range(i + 1, N):
            q_sum = Q_sym[i, j] + Q_sym[j, i]
            # (Q_ij + Q_ji) * (s_i + 1)(s_j + 1) / 4 = q_sum * (s_i s_j + s_i + s_j + 1) / 4
            J[i, j] = 0.25 * q_sum
            J[j, i] = 0.25 * q_sum
            h[i] += 0.25 * q_sum
            h[j] += 0.25 * q_sum
            C += 0.25 * q_sum
            
    return J, h, float(C)


def build_maxcut_qubo(adjacency_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Builds QUBO and Ising models for Max-Cut on an undirected graph.
    
    Max-Cut Objective:
      Maximize Cut(x) = sum_{(i,j) in E} W_ij * (x_i + x_j - 2 x_i x_j)
      Equivalent to minimizing QUBO:
      min x^T Q x where:
      Q_ii = -sum_j W_ij
      Q_ij = 2 * W_ij (for i != j)
      
    Returns:
      Q: QUBO matrix
      J: Ising coupling matrix (J_ij = W_ij / 2)
      h: Ising field vector (all zeros for Max-Cut)
      offset: Constant offset between Cut and Ising energy
    """
    W = np.asarray(adjacency_matrix, dtype=np.float64)
    N = W.shape[0]
    
    # QUBO matrix formulation
    Q = np.zeros((N, N), dtype=np.float64)
    degrees = np.sum(W, axis=1)
    
    for i in range(N):
        Q[i, i] = -degrees[i]
        for j in range(N):
            if i != j:
                Q[i, j] = W[i, j]
                
    # Ising formulation: Cut(s) = sum_{i<j} W_ij * (1 - s_i s_j) / 2
    # Min -Cut(s) = sum_{i<j} (W_ij / 2) * s_i s_j - (1/4) * sum_{i,j} W_ij
    J = 0.5 * W.copy()
    np.fill_diagonal(J, 0.0)
    h = np.zeros(N, dtype=np.float64)
    C = -0.25 * float(np.sum(W))
    
    return Q, J, h, C


def build_constrained_knapsack_qubo(
    values: np.ndarray,
    weights: np.ndarray,
    target_weight: float,
    penalty_multiplier: float
) -> Tuple[np.ndarray, float]:
    """Formulates target-weight knapsack as a QUBO with quadratic penalty terms:
    
    min -sum_i v_i x_i + penalty * (sum_i w_i x_i - target_weight)^2
    
    Using binary property x_i^2 = x_i:
    (sum_i w_i x_i - W_t)^2 = sum_i (w_i^2 - 2 W_t w_i) x_i + 2 sum_{i < j} w_i w_j x_i x_j + W_t^2
    
    NOTE: The penalty multiplier required for feasibility (e.g. lambda >= 1.5 max|c_ij|)
    is strictly instance-specific. The required magnitude depends on the problem spectrum,
    constraint matrix condition, and the gap between feasible and infeasible configurations.
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    W_t = float(target_weight)
    lam = float(penalty_multiplier)
    N = len(v)
    
    Q = np.zeros((N, N), dtype=np.float64)
    
    # Linear terms (diagonal)
    for i in range(N):
        Q[i, i] = -v[i] + lam * (w[i]**2 - 2.0 * W_t * w[i])
        
    # Quadratic cross terms (off-diagonal)
    for i in range(N):
        for j in range(i + 1, N):
            Q[i, j] = 2.0 * lam * w[i] * w[j]
            Q[j, i] = 0.0 # Upper-triangular standard form
            
    constant_offset = lam * (W_t**2)
    return Q, constant_offset
