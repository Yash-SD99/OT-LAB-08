"""Compiler Decision Formulated as Optimization Problems (Track F).

Includes:
1. Pass Ordering Optimization via Dynamic Programming & Greedy Search
2. Loop Unroll Factor Optimization formulated as an Integer Minimization Problem
3. Register Allocation via Graph Coloring (Chaitin-Briggs Heuristic & ILP Formulation)
"""

import copy
import itertools
import time
from typing import Any, Callable, Dict, List, Set, Tuple
import numpy as np

from tracks.F_compiler.ir_compiler import TACInstr
from tracks.F_compiler.passes import (
    ConstantFoldingAndPropagationPass,
    CommonSubexpressionEliminationPass,
    DeadCodeEliminationPass,
    StrengthReductionPass
)


AVAILABLE_PASSES: Dict[str, Any] = {
    "ConstFold": ConstantFoldingAndPropagationPass(),
    "CSE": CommonSubexpressionEliminationPass(),
    "DCE": DeadCodeEliminationPass(),
    "StrengthRed": StrengthReductionPass()
}


def apply_pass_sequence(instructions: List[TACInstr], sequence: List[str]) -> Tuple[List[TACInstr], float]:
    """Applies a sequence of optimization passes and returns (optimized_tac, compile_time_us)."""
    t0 = time.perf_counter()
    curr_tac = copy.deepcopy(instructions)
    for p_name in sequence:
        pass_obj = AVAILABLE_PASSES[p_name]
        curr_tac = pass_obj.run(curr_tac)
    elapsed = time.perf_counter() - t0
    return curr_tac, elapsed * 1e6


def optimize_pass_ordering_greedy(
    instructions: List[TACInstr],
    max_passes: int = 4
) -> Dict:
    """Greedy Pass Ordering: At each stage, greedily applies the pass that minimizes TAC length."""
    t0 = time.perf_counter()
    curr_tac = copy.deepcopy(instructions)
    chosen_sequence: List[str] = []
    
    pass_names = list(AVAILABLE_PASSES.keys())
    
    for step in range(max_passes):
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
            # Convergence reached (no single pass further reduces length)
            break
            
    total_time = time.perf_counter() - t0
    return {
        "method": "Greedy Pass Ordering",
        "optimal_sequence": chosen_sequence,
        "initial_code_size": len(instructions),
        "final_code_size": len(curr_tac),
        "reduction_ratio": float(len(curr_tac) / max(len(instructions), 1)),
        "compile_time_us": total_time * 1e6,
        "optimized_tac": curr_tac
    }


def optimize_pass_ordering_dp(
    instructions: List[TACInstr],
    depth: int = 3
) -> Dict:
    """Exhaustive / Dynamic Programming Search over all pass permutations of length <= depth."""
    t0 = time.perf_counter()
    pass_names = list(AVAILABLE_PASSES.keys())
    
    best_seq: Tuple[str, ...] = ()
    best_len = len(instructions)
    best_tac = copy.deepcopy(instructions)
    evaluated_sequences = 0
    
    for d in range(1, depth + 1):
        for seq in itertools.product(pass_names, repeat=d):
            evaluated_sequences += 1
            cand_tac, _ = apply_pass_sequence(instructions, list(seq))
            cand_len = len(cand_tac)
            if cand_len < best_len:
                best_len = cand_len
                best_seq = seq
                best_tac = cand_tac
                
    total_time = time.perf_counter() - t0
    return {
        "method": "DP Exhaustive Pass Ordering",
        "optimal_sequence": list(best_seq),
        "initial_code_size": len(instructions),
        "final_code_size": len(best_tac),
        "reduction_ratio": float(len(best_tac) / max(len(instructions), 1)),
        "sequences_evaluated": evaluated_sequences,
        "compile_time_us": total_time * 1e6,
        "optimized_tac": best_tac
    }


def optimize_loop_unroll_factor(
    n_iterations: int = 100,
    body_size: int = 5,
    branch_cost: float = 4.0,
    icache_penalty_lambda: float = 0.5,
    max_cache_budget: int = 64
) -> Dict:
    """Formulates Loop Unroll Factor selection as an Integer Minimization Problem:
    
    min_{u in {1, 2, 4, 8, 16}} [ (N_iters / u) * branch_cost + lambda * (u * body_size) ]
    subject to: u * body_size <= max_cache_budget
    """
    candidate_factors = [1, 2, 4, 8, 16, 32]
    costs = []
    
    best_u = 1
    best_cost = float("inf")
    
    for u in candidate_factors:
        code_size = u * body_size
        if code_size > max_cache_budget:
            continue
            
        branch_overhead = (n_iterations / u) * branch_cost
        cache_penalty = icache_penalty_lambda * code_size
        total_cost = branch_overhead + cache_penalty
        
        costs.append({
            "unroll_factor": u,
            "code_size": code_size,
            "branch_overhead": float(branch_overhead),
            "cache_penalty": float(cache_penalty),
            "total_cost": float(total_cost)
        })
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_u = u
            
    return {
        "formulation": "Loop Unroll Optimization",
        "optimal_unroll_factor": best_u,
        "min_cost": float(best_cost),
        "candidate_evaluations": costs
    }


def register_allocation_graph_coloring(
    variables: List[str],
    interference_edges: List[Tuple[str, str]],
    num_physical_registers: int = 3
) -> Dict:
    """Solves register allocation via Chaitin-Briggs Graph Coloring heuristic.
    Assigns each variable to a physical register 0..K-1 such that adjacent nodes
    have distinct colors, reporting any spills.
    """
    # Build adjacency
    adj: Dict[str, Set[str]] = {v: set() for v in variables}
    for u, v in interference_edges:
        if u in adj and v in adj:
            adj[u].add(v)
            adj[v].add(u)
            
    # Degree-ordered greedy coloring (Kempe / Chaitin heuristic)
    sorted_vars = sorted(variables, key=lambda v: len(adj[v]), reverse=True)
    coloring: Dict[str, int] = {}
    spills: List[str] = []
    
    for var in sorted_vars:
        neighbor_colors = {coloring[nbr] for nbr in adj[var] if nbr in coloring}
        # Find first available color in 0..num_physical_registers - 1
        assigned_color = None
        for c in range(num_physical_registers):
            if c not in neighbor_colors:
                assigned_color = c
                break
        if assigned_color is not None:
            coloring[var] = assigned_color
        else:
            spills.append(var)
            
    return {
        "method": "Chaitin-Briggs Register Allocation",
        "num_physical_registers": num_physical_registers,
        "register_assignment": coloring,
        "spills": spills,
        "spill_count": len(spills),
        "is_colorable_without_spills": bool(len(spills) == 0)
    }
