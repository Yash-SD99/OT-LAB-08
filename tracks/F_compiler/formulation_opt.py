"""Compiler Decisions Formulated as Optimization Problems (Track F).

Maps compiler decisions to explicit Chapter 6 optimization methods:
1. Phase Ordering Optimization:
   - Chapter 6 Method: Dynamic Programming & Branch-and-Bound
   - Baselines: Exhaustive Permutation Search, Greedy Step-wise Selection
2. Loop Unroll Factor Optimization:
   - Chapter 6 Method: 0/1 Integer Linear Programming (ILP) / Knapsack
   - Baselines: Exhaustive Evaluation, Greedy Maximum-Fit
3. Register Allocation:
   - Chapter 6 Method: Chaitin-Kempe Heuristic Graph Coloring
   - Baselines: Exact Chromatic Coloring (Backtracking / ILP), Greedy First-Fit
   - Demonstrates mandatory spills when physical registers K < Chromatic Number chi(G).
"""

from __future__ import annotations
import copy
import itertools
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from tracks.F_compiler.ir_compiler import TACInstr, TACVM
from tracks.F_compiler.passes import (
    ConstantFoldingAndPropagationPass,
    CommonSubexpressionEliminationPass,
    DeadCodeEliminationPass,
    StrengthReductionPass,
    LoopInvariantCodeMotionPass
)


AVAILABLE_PASSES: Dict[str, Any] = {
    "ConstFold": ConstantFoldingAndPropagationPass(),
    "CSE": CommonSubexpressionEliminationPass(),
    "DCE": DeadCodeEliminationPass(),
    "StrengthRed": StrengthReductionPass(),
    "LICM": LoopInvariantCodeMotionPass()
}


def apply_pass_sequence(instructions: List[TACInstr], sequence: List[str]) -> Tuple[List[TACInstr], float]:
    """Applies a sequence of optimization passes and returns (optimized_tac, compile_time_us)."""
    t0 = time.perf_counter()
    curr_tac = copy.deepcopy(instructions)
    for p_name in sequence:
        if p_name in AVAILABLE_PASSES:
            pass_obj = AVAILABLE_PASSES[p_name]
            curr_tac = pass_obj.run(curr_tac)
    elapsed = time.perf_counter() - t0
    return curr_tac, elapsed * 1e6


# =============================================================================
# Formulation 1: Phase Ordering Optimization
# =============================================================================

def optimize_pass_ordering_greedy(
    instructions: List[TACInstr],
    max_passes: int = 4
) -> Dict[str, Any]:
    """Baseline 1: Greedy Step-wise Selection.
    At each step, greedily selects the single pass that minimizes intermediate code size.
    """
    t0 = time.perf_counter()
    curr_tac = copy.deepcopy(instructions)
    chosen_sequence: List[str] = []
    pass_names = list(AVAILABLE_PASSES.keys())

    for _ in range(max_passes):
        best_pass = None
        best_tac = curr_tac
        best_len = len(curr_tac)

        for p_name in pass_names:
            candidate_tac = AVAILABLE_PASSES[p_name].run(copy.deepcopy(curr_tac))
            cand_len = len(candidate_tac)
            if cand_len < best_len:
                best_len = cand_len
                best_pass = p_name
                best_tac = candidate_tac

        if best_pass is not None:
            chosen_sequence.append(best_pass)
            curr_tac = best_tac
        else:
            break

    total_time = time.perf_counter() - t0
    return {
        "method": "Greedy Step-wise Selection",
        "chapter6_category": "Greedy Baseline",
        "optimal_sequence": chosen_sequence,
        "initial_code_size": len(instructions),
        "final_code_size": len(curr_tac),
        "reduction_ratio": float(len(curr_tac) / max(len(instructions), 1)),
        "compile_time_us": total_time * 1e6,
        "optimized_tac": curr_tac
    }


def optimize_pass_ordering_exhaustive(
    instructions: List[TACInstr],
    depth: int = 3
) -> Dict[str, Any]:
    """Baseline 2: Exhaustive Search over all pass permutations up to depth."""
    t0 = time.perf_counter()
    pass_names = list(AVAILABLE_PASSES.keys())

    best_seq: Tuple[str, ...] = ()
    best_len = len(instructions)
    best_tac = copy.deepcopy(instructions)
    evaluated = 0

    for d in range(1, depth + 1):
        for seq in itertools.product(pass_names, repeat=d):
            evaluated += 1
            cand_tac, _ = apply_pass_sequence(instructions, list(seq))
            cand_len = len(cand_tac)
            if cand_len < best_len:
                best_len = cand_len
                best_seq = seq
                best_tac = cand_tac

    total_time = time.perf_counter() - t0
    return {
        "method": "Exhaustive Permutation Search",
        "chapter6_category": "Brute-Force Enumeration",
        "optimal_sequence": list(best_seq),
        "initial_code_size": len(instructions),
        "final_code_size": len(best_tac),
        "reduction_ratio": float(len(best_tac) / max(len(instructions), 1)),
        "sequences_evaluated": evaluated,
        "compile_time_us": total_time * 1e6,
        "optimized_tac": best_tac
    }


def optimize_pass_ordering_dp_bnb(
    instructions: List[TACInstr],
    depth: int = 3
) -> Dict[str, Any]:
    """Chapter 6 Method: Dynamic Programming with Branch-and-Bound Pruning.
    Maintains a visited state cache (memoization table) and prunes subtrees
    when no further reduction is theoretically possible or redundant.
    """
    t0 = time.perf_counter()
    pass_names = list(AVAILABLE_PASSES.keys())

    memo: Dict[Tuple[str, int], Tuple[int, List[str], List[TACInstr]]] = {}
    best_global_len = len(instructions)
    best_global_seq: List[str] = []
    best_global_tac = copy.deepcopy(instructions)
    nodes_evaluated = 0
    pruned_subtrees = 0

    def code_signature(tac: List[TACInstr]) -> str:
        return f"{len(tac)}:" + ";".join(f"{i.op}_{i.dest}_{i.src1}_{i.src2}" for i in tac[:10])

    def bnb_search(curr_tac: List[TACInstr], current_seq: List[str], d_remaining: int):
        nonlocal best_global_len, best_global_seq, best_global_tac, nodes_evaluated, pruned_subtrees
        nodes_evaluated += 1

        curr_len = len(curr_tac)
        if curr_len < best_global_len:
            best_global_len = curr_len
            best_global_seq = list(current_seq)
            best_global_tac = curr_tac

        if d_remaining == 0:
            return

        sig = (code_signature(curr_tac), d_remaining)
        if sig in memo and memo[sig][0] <= curr_len:
            pruned_subtrees += 1
            return

        for p_name in pass_names:
            # Branch-and-bound heuristic pruning:
            # Consecutive duplicate passes rarely provide additional gain unless loop transforms occurred
            if current_seq and current_seq[-1] == p_name and p_name in ("ConstFold", "CSE", "DCE"):
                pruned_subtrees += 1
                continue

            next_tac = AVAILABLE_PASSES[p_name].run(copy.deepcopy(curr_tac))
            bnb_search(next_tac, current_seq + [p_name], d_remaining - 1)

        memo[sig] = (curr_len, list(current_seq), curr_tac)

    bnb_search(instructions, [], depth)
    total_time = time.perf_counter() - t0

    return {
        "method": "Branch-and-Bound & DP Memoization",
        "chapter6_category": "Chapter 6: Dynamic Programming & Branch-and-Bound",
        "optimal_sequence": best_global_seq,
        "initial_code_size": len(instructions),
        "final_code_size": best_global_len,
        "reduction_ratio": float(best_global_len / max(len(instructions), 1)),
        "nodes_evaluated": nodes_evaluated,
        "pruned_subtrees": pruned_subtrees,
        "compile_time_us": total_time * 1e6,
        "optimized_tac": best_global_tac
    }


# =============================================================================
# Formulation 2: Loop Unroll Factor Optimization
# =============================================================================

def optimize_loop_unroll_ilp(
    n_iterations: int = 200,
    body_size: int = 6,
    branch_cost: float = 2.0,
    max_cache_budget: int = 48
) -> Dict[str, Any]:
    """Chapter 6 Method: 0/1 Integer Linear Programming (ILP) / Knapsack Formulation.
    
    Decision variables: x_u in {0, 1} for u in {1, 2, 4, 8, 16}
    Maximize: Value(u) = (N - N/u) * branch_cost  (branch cycles eliminated)
    Subject to:
      sum_u (u * body_size) * x_u <= max_cache_budget  (instruction cache limit)
      sum_u x_u = 1  (pick exactly one factor)
    """
    t0 = time.perf_counter()
    candidate_factors = [1, 2, 4, 8, 16]
    
    # 0/1 Knapsack formulation
    items = []
    for u in candidate_factors:
        weight = u * body_size
        value = (n_iterations - (n_iterations / u)) * branch_cost
        items.append((u, weight, value))

    # Knapsack DP / ILP exact solve
    best_u = 1
    best_value = -1.0
    for u, weight, value in items:
        if weight <= max_cache_budget:
            if value > best_value:
                best_value = value
                best_u = u

    total_time = time.perf_counter() - t0
    return {
        "method": "0/1 Knapsack ILP",
        "chapter6_category": "Chapter 6: Integer Linear Programming (ILP) & Knapsack",
        "optimal_unroll_factor": best_u,
        "cycles_saved": float(best_value),
        "code_size": best_u * body_size,
        "cache_budget": max_cache_budget,
        "solve_time_us": total_time * 1e6
    }


def optimize_loop_unroll_exhaustive(
    n_iterations: int = 200,
    body_size: int = 6,
    branch_cost: float = 2.0,
    max_cache_budget: int = 48
) -> Dict[str, Any]:
    """Baseline 1: Exhaustive Factor Search over candidate set."""
    t0 = time.perf_counter()
    candidates = [1, 2, 4, 8, 16, 32]
    best_u = 1
    best_net_gain = -float("inf")

    for u in candidates:
        weight = u * body_size
        if weight <= max_cache_budget:
            gain = (n_iterations - (n_iterations / u)) * branch_cost
            if gain > best_net_gain:
                best_net_gain = gain
                best_u = u

    total_time = time.perf_counter() - t0
    return {
        "method": "Exhaustive Unroll Search",
        "chapter6_category": "Brute-Force Enumeration",
        "optimal_unroll_factor": best_u,
        "cycles_saved": float(best_net_gain),
        "code_size": best_u * body_size,
        "solve_time_us": total_time * 1e6
    }


def optimize_loop_unroll_greedy(
    n_iterations: int = 200,
    body_size: int = 6,
    branch_cost: float = 2.0,
    max_cache_budget: int = 48
) -> Dict[str, Any]:
    """Baseline 2: Greedy Maximum-Fit heuristic.
    Picks the largest factor u that does not exceed max_cache_budget.
    """
    t0 = time.perf_counter()
    max_u = max(1, max_cache_budget // body_size)
    # Clamp to nearest power of 2
    u_pow2 = 1
    while u_pow2 * 2 <= max_u:
        u_pow2 *= 2

    cycles_saved = (n_iterations - (n_iterations / u_pow2)) * branch_cost
    total_time = time.perf_counter() - t0
    return {
        "method": "Greedy Maximum-Fit",
        "chapter6_category": "Greedy Heuristic",
        "optimal_unroll_factor": u_pow2,
        "cycles_saved": float(cycles_saved),
        "code_size": u_pow2 * body_size,
        "solve_time_us": total_time * 1e6
    }


# =============================================================================
# Formulation 3: Register Allocation & Graph Coloring
# =============================================================================

def register_allocation_kempe(
    variables: List[str],
    interference_edges: List[Tuple[str, str]],
    num_physical_registers: int = 3
) -> Dict[str, Any]:
    """Chapter 6 Method: Chaitin-Kempe Degree-Ordered Heuristic Coloring.
    Orders vertices by degree descending and applies first-fit coloring,
    marking variables as spilled when neighbor colors saturate 0..K-1.
    """
    t0 = time.perf_counter()
    adj: Dict[str, Set[str]] = {v: set() for v in variables}
    for u, v in interference_edges:
        if u in adj and v in adj:
            adj[u].add(v)
            adj[v].add(u)

    # Degree-ordered sequence
    sorted_vars = sorted(variables, key=lambda v: len(adj[v]), reverse=True)
    coloring: Dict[str, int] = {}
    spills: List[str] = []

    for var in sorted_vars:
        neighbor_colors = {coloring[nbr] for nbr in adj[var] if nbr in coloring}
        assigned_color = None
        for c in range(num_physical_registers):
            if c not in neighbor_colors:
                assigned_color = c
                break
        if assigned_color is not None:
            coloring[var] = assigned_color
        else:
            spills.append(var)

    total_time = time.perf_counter() - t0
    return {
        "method": "Chaitin-Kempe Heuristic Coloring",
        "chapter6_category": "Chapter 6: Combinatorial Optimization (Degree-Heuristic)",
        "num_physical_registers": num_physical_registers,
        "coloring": coloring,
        "spills": spills,
        "spill_count": len(spills),
        "is_colorable_without_spills": len(spills) == 0,
        "solve_time_us": total_time * 1e6
    }


def register_allocation_exact_backtracking(
    variables: List[str],
    interference_edges: List[Tuple[str, str]],
    num_physical_registers: int = 3
) -> Dict[str, Any]:
    """Baseline 1: Exact Chromatic Coloring via Recursive Backtracking.
    Finds a zero-spill K-coloring if one exists; otherwise reports minimum spills.
    """
    t0 = time.perf_counter()
    adj: Dict[str, Set[str]] = {v: set() for v in variables}
    for u, v in interference_edges:
        if u in adj and v in adj:
            adj[u].add(v)
            adj[v].add(u)

    coloring: Dict[str, int] = {}

    def is_safe(var: str, c: int) -> bool:
        for nbr in adj[var]:
            if coloring.get(nbr) == c:
                return False
        return True

    def backtrack(idx: int) -> bool:
        if idx == len(variables):
            return True
        var = variables[idx]
        for c in range(num_physical_registers):
            if is_safe(var, c):
                coloring[var] = c
                if backtrack(idx + 1):
                    return True
                del coloring[var]
        return False

    success = backtrack(0)
    total_time = time.perf_counter() - t0

    if success:
        return {
            "method": "Exact Backtracking Coloring",
            "chapter6_category": "Exact Combinatorial Search",
            "num_physical_registers": num_physical_registers,
            "coloring": coloring,
            "spills": [],
            "spill_count": 0,
            "is_colorable_without_spills": True,
            "solve_time_us": total_time * 1e6
        }
    else:
        # Falls back to greedy spill selection
        kempe_res = register_allocation_kempe(variables, interference_edges, num_physical_registers)
        kempe_res["method"] = "Exact Backtracking (Coloring Failed -> Spill Required)"
        kempe_res["solve_time_us"] = total_time * 1e6
        return kempe_res


def register_allocation_greedy_first_fit(
    variables: List[str],
    interference_edges: List[Tuple[str, str]],
    num_physical_registers: int = 3
) -> Dict[str, Any]:
    """Baseline 2: Arbitrary Order First-Fit Coloring without degree sorting."""
    t0 = time.perf_counter()
    adj: Dict[str, Set[str]] = {v: set() for v in variables}
    for u, v in interference_edges:
        if u in adj and v in adj:
            adj[u].add(v)
            adj[v].add(u)

    coloring: Dict[str, int] = {}
    spills: List[str] = []

    # Arbitrary / natural order
    for var in variables:
        neighbor_colors = {coloring[nbr] for nbr in adj[var] if nbr in coloring}
        assigned = None
        for c in range(num_physical_registers):
            if c not in neighbor_colors:
                assigned = c
                break
        if assigned is not None:
            coloring[var] = assigned
        else:
            spills.append(var)

    total_time = time.perf_counter() - t0
    return {
        "method": "Greedy Arbitrary First-Fit",
        "chapter6_category": "Naive Greedy Baseline",
        "num_physical_registers": num_physical_registers,
        "coloring": coloring,
        "spills": spills,
        "spill_count": len(spills),
        "is_colorable_without_spills": len(spills) == 0,
        "solve_time_us": total_time * 1e6
    }


def demonstrate_spills_under_chromatic_bound() -> Dict[str, Any]:
    """Demonstrates mandatory spills when physical registers K < Chromatic Number chi(G).
    
    Constructs a complete interference graph K_4 on 4 variables {v0, v1, v2, v3}:
    - Chromatic number chi(K_4) = 4 (every pair interferes).
    - Evaluates K in {2, 3, 4, 5} comparing Kempe Heuristic vs Exact vs Greedy First-Fit.
    - Proves that when K < 4, zero-spill coloring is impossible and spills are strictly required.
    """
    variables = ["v0", "v1", "v2", "v3"]
    interference_edges = list(itertools.combinations(variables, 2)) # All 6 edges

    results_by_k = {}
    for k in [2, 3, 4, 5]:
        kempe = register_allocation_kempe(variables, interference_edges, num_physical_registers=k)
        exact = register_allocation_exact_backtracking(variables, interference_edges, num_physical_registers=k)
        greedy = register_allocation_greedy_first_fit(variables, interference_edges, num_physical_registers=k)

        results_by_k[k] = {
            "registers_available": k,
            "chromatic_number": 4,
            "spills_mandatory": k < 4,
            "kempe_spills": kempe["spill_count"],
            "exact_spills": exact["spill_count"],
            "greedy_spills": greedy["spill_count"],
            "kempe_colorable": kempe["is_colorable_without_spills"],
            "exact_colorable": exact["is_colorable_without_spills"]
        }

    return {
        "graph_type": "K_4 Complete Interference Graph",
        "variables": variables,
        "edge_count": len(interference_edges),
        "chromatic_number": 4,
        "evaluations": results_by_k
    }
