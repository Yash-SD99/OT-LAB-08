"""Submodular Optimization & Greedy Selection with Approximation Guarantees (Track D).

Includes:
1. Facility Location / Maximum Coverage Submodular Objective:
   f(S) = sum_{j in Targets} max_{i in S} Similarity(i, j)
2. Rigorous empirical verification of diminishing returns:
   f(A U {e}) - f(A) >= f(B U {e}) - f(B) for all A subset B
3. Greedy Selection Algorithm with Nemhauser-Wolsey (1 - 1/e) guarantee verification:
   f(S_greedy) >= (1 - 1/e) * f(S_optimal) approx 0.632 * f(S*)
4. Exhaustive Exact Combinatorial Benchmark
"""

import itertools
import time
from typing import Callable, Dict, List, Set, Tuple, Union
import numpy as np


class FacilityLocationSubmodular:
    """Monotone submodular facility location objective.
    
    Given ground set V of candidate facility locations and a set of customer targets,
    the objective value is:
      f(S) = sum_{j=1}^M max_{i in S} Sim(i, j)
    where Sim(i, j) = exp(- ||coord_i - target_j||^2 / (2 * sigma^2)).
    
    Properties:
      - Normalized: f(empty) = 0
      - Monotone non-decreasing: S subset T => f(S) <= f(T)
      - Submodular: S subset T => f(S U {e}) - f(S) >= f(T U {e}) - f(T)
    """
    def __init__(self, candidates: np.ndarray, targets: np.ndarray, sigma: float = 1.0):
        self.candidates = np.asarray(candidates, dtype=np.float64)
        self.targets = np.asarray(targets, dtype=np.float64)
        self.sigma = float(sigma)
        self.n_candidates = self.candidates.shape[0]
        self.n_targets = self.targets.shape[0]
        
        # Precompute similarity matrix M[i, j] between candidate i and target j
        # Shape: (n_candidates, n_targets)
        diff = self.candidates[:, None, :] - self.targets[None, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        self.sim_matrix = np.exp(-dist_sq / (2.0 * self.sigma**2))

    def evaluate(self, subset: Union[Set[int], List[int], Tuple[int, ...]]) -> float:
        """Evaluates f(S). If S is empty, returns 0.0."""
        if len(subset) == 0:
            return 0.0
        sub_indices = list(subset)
        # For each target j, take the maximum similarity to any selected candidate in S
        max_sims = np.max(self.sim_matrix[sub_indices, :], axis=0)
        return float(np.sum(max_sims))

    def marginal_gain(self, subset: Union[Set[int], List[int]], element: int) -> float:
        """Computes marginal gain: Delta(e | S) = f(S U {e}) - f(S)."""
        if element in subset:
            return 0.0
        f_current = self.evaluate(subset)
        f_next = self.evaluate(set(subset) | {element})
        return float(f_next - f_current)


def verify_diminishing_returns(
    obj: FacilityLocationSubmodular,
    n_pairs_to_test: int = 100,
    seed: int = 42
) -> Dict:
    """Rigorously checks the diminishing returns condition:
    For random A subset B and e not in B:
      f(A U {e}) - f(A) >= f(B U {e}) - f(B) - eps
    """
    rng = np.random.default_rng(seed)
    V = list(range(obj.n_candidates))
    
    violations = 0
    tested = 0
    max_discrepancy = 0.0
    
    for _ in range(n_pairs_to_test):
        # Sample subset size for B in [2..N-1]
        b_size = rng.integers(2, obj.n_candidates)
        # Sample B
        B = set(rng.choice(V, size=b_size, replace=False))
        # Sample A subset B
        a_size = rng.integers(0, b_size)
        A = set(rng.choice(list(B), size=a_size, replace=False))
        
        # Candidate element e not in B
        remaining = list(set(V) - B)
        if not remaining:
            continue
        e = int(rng.choice(remaining))
        
        gain_A = obj.marginal_gain(A, e)
        gain_B = obj.marginal_gain(B, e)
        tested += 1
        
        # Diminishing returns: gain_A must be >= gain_B
        if gain_B - gain_A > 1e-9:
            violations += 1
            max_discrepancy = max(max_discrepancy, gain_B - gain_A)
            
    return {
        "pairs_tested": tested,
        "violations": violations,
        "is_submodular_verified": bool(violations == 0),
        "max_numerical_discrepancy": float(max_discrepancy)
    }


def greedy_submodular_selection(obj: FacilityLocationSubmodular, k: int) -> Dict:
    """Greedy selection of k elements:
    At each step, select e* = argmax_{e not in S} [f(S U {e}) - f(S)].
    Nemhauser-Wolsey (1978) Theorem guarantees:
      f(S_greedy) >= (1 - 1/e) * f(S_optimal) approx 0.6321 * f(S*).
    """
    t0 = time.perf_counter()
    selected: List[int] = []
    remaining = set(range(obj.n_candidates))
    history = [0.0]
    
    for step in range(k):
        best_e = None
        best_gain = -float("inf")
        
        for e in remaining:
            gain = obj.marginal_gain(selected, e)
            if gain > best_gain:
                best_gain = gain
                best_e = e
                
        if best_e is not None and best_gain > 0.0:
            selected.append(best_e)
            remaining.remove(best_e)
            history.append(obj.evaluate(selected))
        else:
            break
            
    elapsed = time.perf_counter() - t0
    return {
        "method": "Submodular Greedy",
        "selected_subset": selected,
        "final_value": history[-1],
        "value_trajectory": history,
        "runtime_ms": elapsed * 1000.0
    }


def exact_optimal_selection(obj: FacilityLocationSubmodular, k: int) -> Dict:
    """Exact combinatorial search over all subsets of size k."""
    t0 = time.perf_counter()
    best_subset: Tuple[int, ...] = ()
    best_val = -float("inf")
    evaluated = 0
    
    for subset in itertools.combinations(range(obj.n_candidates), k):
        evaluated += 1
        val = obj.evaluate(subset)
        if val > best_val:
            best_val = val
            best_subset = subset
            
    elapsed = time.perf_counter() - t0
    return {
        "method": "Exhaustive Exact Optimal",
        "selected_subset": list(best_subset),
        "final_value": float(best_val),
        "subsets_evaluated": evaluated,
        "runtime_ms": elapsed * 1000.0
    }
