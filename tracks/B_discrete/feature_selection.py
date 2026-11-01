"""Combinatorial Feature Selection under Cardinality Constraints (Track B).

Demonstrates:
1. Exact Exhaustive Subset Selection
2. Forward Stepwise Greedy Selection
3. Backward Elimination Greedy Selection
4. Formulation of Feature Selection as a Quadratic Binary Problem (QUBO)
"""

import itertools
import time
from typing import Dict, List, Tuple
import numpy as np


class FeatureSelectionProblem:
    """Combinatorial feature subset selection for linear regression."""
    def __init__(self, X: np.ndarray, y: np.ndarray, max_features: int):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.n_samples, self.n_features = self.X.shape
        self.k = int(max_features)
        
    def evaluate_subset(self, subset: Tuple[int, ...]) -> Tuple[float, float]:
        """Fits OLS on the subset and returns (MSE, R2)."""
        if len(subset) == 0:
            y_pred = np.mean(self.y)
            mse = float(np.mean((self.y - y_pred)**2))
            return mse, 0.0
            
        X_sub = self.X[:, subset]
        # Regularized pseudo-inverse for numerical stability
        XT_X = X_sub.T @ X_sub + 1e-8 * np.eye(len(subset))
        beta = np.linalg.solve(XT_X, X_sub.T @ self.y)
        y_pred = X_sub @ beta
        
        mse = float(np.mean((self.y - y_pred)**2))
        total_var = float(np.sum((self.y - np.mean(self.y))**2))
        res_var = float(np.sum((self.y - y_pred)**2))
        r2 = 1.0 - (res_var / max(total_var, 1e-12))
        return mse, float(r2)


def exact_best_subset_selection(problem: FeatureSelectionProblem) -> Dict:
    """Exact combinatorial search: examines all subsets of size <= K."""
    t0 = time.perf_counter()
    best_subset: Tuple[int, ...] = ()
    best_mse = float("inf")
    best_r2 = -float("inf")
    evaluated_count = 0
    
    for size in range(1, problem.k + 1):
        for subset in itertools.combinations(range(problem.n_features), size):
            evaluated_count += 1
            mse, r2 = problem.evaluate_subset(subset)
            if mse < best_mse:
                best_mse = mse
                best_r2 = r2
                best_subset = subset
                
    elapsed = time.perf_counter() - t0
    mask = np.zeros(problem.n_features, dtype=int)
    mask[list(best_subset)] = 1
    
    return {
        "method": "Exact Exhaustive Subset Selection",
        "selected_indices": list(best_subset),
        "selected_mask": mask.tolist(),
        "mse": best_mse,
        "r2": best_r2,
        "subsets_evaluated": evaluated_count,
        "runtime_ms": elapsed * 1000.0,
        "is_exact": True
    }


def forward_stepwise_selection(problem: FeatureSelectionProblem) -> Dict:
    """Greedy forward selection: iteratively adds the feature yielding highest R2 increase."""
    t0 = time.perf_counter()
    current_subset: List[int] = []
    remaining = set(range(problem.n_features))
    evaluated_count = 0
    
    for step in range(problem.k):
        best_candidate = None
        best_mse = float("inf")
        best_r2 = -float("inf")
        
        for feat in sorted(remaining):
            evaluated_count += 1
            cand_subset = tuple(current_subset + [feat])
            mse, r2 = problem.evaluate_subset(cand_subset)
            if mse < best_mse:
                best_mse = mse
                best_r2 = r2
                best_candidate = feat
                
        if best_candidate is not None:
            current_subset.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break
            
    elapsed = time.perf_counter() - t0
    final_mse, final_r2 = problem.evaluate_subset(tuple(current_subset))
    mask = np.zeros(problem.n_features, dtype=int)
    mask[current_subset] = 1
    
    return {
        "method": "Forward Greedy Selection",
        "selected_indices": current_subset,
        "selected_mask": mask.tolist(),
        "mse": final_mse,
        "r2": final_r2,
        "subsets_evaluated": evaluated_count,
        "runtime_ms": elapsed * 1000.0,
        "is_exact": False
    }
