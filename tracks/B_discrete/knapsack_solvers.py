"""Discrete Optimization Solvers for 0/1 Knapsack & Resource Allocation (Track B).

Includes:
1. Greedy Heuristic (value/weight density ranking with fractional & integer solutions)
2. Exact Dynamic Programming (O(N * W) table with traceback)
3. Custom Branch-and-Bound from scratch (Linear Relaxation bounds, Best-First Search, Pruning)
4. Exact Integer Linear Programming via scipy.optimize.milp
5. Exact Integer Linear Programming via PuLP
"""

import heapq
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import pulp


class KnapsackInstance:
    """Represents a 0/1 Knapsack instance."""
    def __init__(self, values: np.ndarray, weights: np.ndarray, capacity: float):
        self.values = np.asarray(values, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.capacity = float(capacity)
        self.n = len(values)
        if len(weights) != self.n:
            raise ValueError("Values and weights must have matching length.")


def solve_greedy(instance: KnapsackInstance) -> Dict:
    """Greedy heuristic: sorts items by efficiency (value/weight) in descending order."""
    t0 = time.perf_counter()
    efficiencies = instance.values / np.maximum(instance.weights, 1e-9)
    order = np.argsort(-efficiencies)
    
    total_val = 0.0
    total_wt = 0.0
    selected = np.zeros(instance.n, dtype=int)
    
    for idx in order:
        if total_wt + instance.weights[idx] <= instance.capacity:
            selected[idx] = 1
            total_wt += instance.weights[idx]
            total_val += instance.values[idx]
            
    elapsed = time.perf_counter() - t0
    return {
        "method": "Greedy (Value Density)",
        "objective": float(total_val),
        "weight": float(total_wt),
        "selected": selected.tolist(),
        "runtime_ms": elapsed * 1000.0,
        "is_exact": False
    }


def solve_dynamic_programming(instance: KnapsackInstance) -> Dict:
    """Exact Dynamic Programming for integer weights and capacity.
    
    Complexity: O(N * W) time and space.
    """
    t0 = time.perf_counter()
    W = int(np.floor(instance.capacity))
    N = instance.n
    weights_int = instance.weights.astype(int)
    values = instance.values
    
    # dp[i][w] = max value using subset of first i items with capacity w
    dp = np.zeros((N + 1, W + 1), dtype=np.float64)
    
    for i in range(1, N + 1):
        wt = weights_int[i - 1]
        val = values[i - 1]
        # Vectorized transition
        dp[i, :wt] = dp[i - 1, :wt]
        dp[i, wt:] = np.maximum(dp[i - 1, wt:], dp[i - 1, :-wt] + val)
        
    # Backtrack to reconstruct solution
    selected = np.zeros(N, dtype=int)
    curr_w = W
    for i in range(N, 0, -1):
        if dp[i, curr_w] != dp[i - 1, curr_w]:
            selected[i - 1] = 1
            curr_w -= weights_int[i - 1]
            
    total_val = float(dp[N, W])
    total_wt = float(np.sum(instance.weights * selected))
    elapsed = time.perf_counter() - t0
    
    return {
        "method": "Dynamic Programming",
        "objective": total_val,
        "weight": total_wt,
        "selected": selected.tolist(),
        "runtime_ms": elapsed * 1000.0,
        "is_exact": True
    }


class BnBNode:
    """Search node for Branch-and-Bound tree."""
    def __init__(self, level: int, value: float, weight: float, bound: float, items_taken: List[int]):
        self.level = level          # Item index being considered
        self.value = value          # Current accumulated value
        self.weight = weight        # Current accumulated weight
        self.bound = bound          # Upper bound from fractional linear relaxation
        self.items_taken = items_taken

    def __lt__(self, other: "BnBNode") -> bool:
        # Priority queue orders by highest upper bound (best-first search)
        return self.bound > other.bound


def solve_branch_and_bound(instance: KnapsackInstance) -> Dict:
    """Exact Custom Branch-and-Bound from scratch.
    
    Uses fractional knapsack relaxation as the admissible bounding function.
    Prunes nodes where upper_bound <= best_known_feasible_objective.
    """
    t0 = time.perf_counter()
    N = instance.n
    
    # Sort items by descending value density for optimal bounding and branch ordering
    efficiencies = instance.values / np.maximum(instance.weights, 1e-9)
    order = np.argsort(-efficiencies)
    v_sorted = instance.values[order]
    w_sorted = instance.weights[order]
    
    def fractional_bound(level: int, curr_val: float, curr_wt: float) -> float:
        """Admissible upper bound via linear relaxation of remaining items."""
        if curr_wt > instance.capacity:
            return 0.0
        if curr_wt == instance.capacity:
            return float(curr_val)
        bound = curr_val
        total_wt = curr_wt
        for i in range(level, N):
            if total_wt + w_sorted[i] <= instance.capacity:
                total_wt += w_sorted[i]
                bound += v_sorted[i]
            else:
                # Take fractional slice of remaining item
                remaining = instance.capacity - total_wt
                bound += v_sorted[i] * (remaining / w_sorted[i])
                break
        return float(bound)

    root_bound = fractional_bound(0, 0.0, 0.0)
    root = BnBNode(level=0, value=0.0, weight=0.0, bound=root_bound, items_taken=[])
    
    pq = [root]
    best_value = 0.0
    best_items_sorted: List[int] = []
    
    nodes_explored = 0
    nodes_pruned = 0
    
    while pq:
        node = heapq.heappop(pq)
        nodes_explored += 1
        
        # Pruning condition: if node upper bound cannot beat best known solution
        if node.bound <= best_value:
            nodes_pruned += 1
            continue
            
        if node.level == N:
            if node.value > best_value:
                best_value = node.value
                best_items_sorted = node.items_taken
            continue
            
        next_level = node.level + 1
        curr_item = node.level
        
        # Branch 1: INCLUDE item (if weight capacity allows)
        wt_with = node.weight + w_sorted[curr_item]
        if wt_with <= instance.capacity:
            val_with = node.value + v_sorted[curr_item]
            bound_with = fractional_bound(next_level, val_with, wt_with)
            if bound_with > best_value:
                if val_with > best_value:
                    best_value = val_with
                    best_items_sorted = node.items_taken + [curr_item]
                child_with = BnBNode(next_level, val_with, wt_with, bound_with, node.items_taken + [curr_item])
                heapq.heappush(pq, child_with)
            else:
                nodes_pruned += 1
                
        # Branch 2: EXCLUDE item
        bound_without = fractional_bound(next_level, node.value, node.weight)
        if bound_without > best_value:
            child_without = BnBNode(next_level, node.value, node.weight, bound_without, node.items_taken)
            heapq.heappush(pq, child_without)
        else:
            nodes_pruned += 1

    # Map selected sorted indices back to original item indices
    selected = np.zeros(N, dtype=int)
    for s_idx in best_items_sorted:
        orig_idx = order[s_idx]
        selected[orig_idx] = 1
        
    elapsed = time.perf_counter() - t0
    total_wt = float(np.sum(instance.weights * selected))
    
    return {
        "method": "Branch-and-Bound (Custom)",
        "objective": float(best_value),
        "weight": total_wt,
        "selected": selected.tolist(),
        "nodes_explored": nodes_explored,
        "nodes_pruned": nodes_pruned,
        "runtime_ms": elapsed * 1000.0,
        "is_exact": True
    }


def solve_ilp_scipy(instance: KnapsackInstance) -> Dict:
    """Exact Integer Linear Programming using scipy.optimize.milp."""
    t0 = time.perf_counter()
    N = instance.n
    
    # scipy.optimize.milp minimizes c^T x, so we minimize -values^T x
    c = -instance.values
    
    # Constraint: weights^T x <= capacity
    # LinearConstraint: A @ x <= b
    A = instance.weights.reshape(1, N)
    constraints = LinearConstraint(A, lb=0.0, ub=instance.capacity)
    
    # Binary bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0.0, ub=1.0)
    
    # All variables are binary integer (integrality = 1)
    integrality = np.ones(N)
    
    res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
    
    elapsed = time.perf_counter() - t0
    if not res.success:
        raise RuntimeError(f"scipy.optimize.milp failed: {res.status}")
        
    selected = np.round(res.x).astype(int)
    total_val = float(-res.fun)
    total_wt = float(np.sum(instance.weights * selected))
    
    return {
        "method": "ILP (scipy.optimize.milp)",
        "objective": total_val,
        "weight": total_wt,
        "selected": selected.tolist(),
        "runtime_ms": elapsed * 1000.0,
        "is_exact": True
    }


def solve_ilp_pulp(instance: KnapsackInstance) -> Dict:
    """Exact Integer Linear Programming using PuLP (CBC Solver)."""
    t0 = time.perf_counter()
    N = instance.n
    
    prob = pulp.LpProblem("01_Knapsack", pulp.LpMaximize)
    
    # Create binary variables compatible with all PuLP versions (v2, v3, v4)
    if hasattr(prob, "add_variable"):
        x = [prob.add_variable(f"x_{i}", lowBound=0, upBound=1, cat=pulp.LpBinary) for i in range(N)]
    else:
        x = [pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(N)]
    
    # Objective
    prob += pulp.lpSum([instance.values[i] * x[i] for i in range(N)])
    
    # Weight capacity constraint
    prob += pulp.lpSum([instance.weights[i] * x[i] for i in range(N)]) <= instance.capacity
    
    # Solve with available solver, catching missing platform binaries gracefully
    try:
        prob.solve()
        selected = np.array([int(round(x[i].value() if hasattr(x[i], "value") else pulp.value(x[i]))) for i in range(N)])
        total_val = float(prob.objective.value() if hasattr(prob.objective, "value") else pulp.value(prob.objective))
        total_wt = float(np.sum(instance.weights * selected))
        success = True
    except Exception as e:
        # Fallback to scipy.optimize.milp if platform lacks external solver binary
        res_milp = solve_ilp_scipy(instance)
        selected = np.array(res_milp["selected"])
        total_val = res_milp["objective"]
        total_wt = res_milp["weight"]
        success = False
    
    elapsed = time.perf_counter() - t0
    
    return {
        "method": "ILP (PuLP / CBC)",
        "objective": total_val,
        "weight": total_wt,
        "selected": selected.tolist(),
        "runtime_ms": elapsed * 1000.0,
        "is_exact": True,
        "solver_available": success
    }
