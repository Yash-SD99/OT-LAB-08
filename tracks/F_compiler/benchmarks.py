"""Comprehensive Benchmarking Suite for Track F: Compiler Optimization.

Measures & Compares:
1. Instruction Count (Code Size) across Optimization Levels: O0 vs O1 vs O2 vs O3
2. VM Execution Time & Cycles
3. Compile Time Overhead vs Runtime Speedup Trade-Off
4. Pass Ordering Optimization (Greedy vs DP)
5. Loop Unrolling Optimization Curve (Overhead vs I-Cache Expansion)
"""

import copy
import json
import os
import time
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tracks.F_compiler.ir_compiler import TACInstr, TACVM
from tracks.F_compiler.passes import (
    ConstantFoldingAndPropagationPass,
    CommonSubexpressionEliminationPass,
    DeadCodeEliminationPass,
    StrengthReductionPass
)
from tracks.F_compiler.formulation_opt import (
    optimize_pass_ordering_greedy,
    optimize_pass_ordering_dp,
    optimize_loop_unroll_factor,
    register_allocation_graph_coloring
)


def create_benchmark_program_1() -> List[TACInstr]:
    """Program 1: Arithmetic Pipeline with Constant Redundancies, Dead Code, and Power-of-2 Multiplies."""
    return [
        TACInstr("ASSIGN", "base", 16),
        TACInstr("ASSIGN", "offset", 4),
        TACInstr("ADD", "t1", "base", "offset"),      # 16 + 4 = 20
        TACInstr("MUL", "t_dead", "t1", 500),         # Dead code
        TACInstr("ADD", "t2", "base", "offset"),      # Common Subexpression (t1)
        TACInstr("MUL", "t3", "t2", 8),               # Strength reduction -> t2 << 3
        TACInstr("ADD", "t4", "t3", 0),               # Algebraic identity -> t3
        TACInstr("DIV", "t5", "t4", 1),               # Algebraic identity -> t4
        TACInstr("ASSIGN", "result", "t5"),
        TACInstr("PRINT", None, "result"),
        TACInstr("RETURN", None, "result")
    ]


def create_benchmark_program_2() -> List[TACInstr]:
    """Program 2: Loop Accumulation with Invariant Computations."""
    return [
        TACInstr("ASSIGN", "sum", 0),
        TACInstr("ASSIGN", "i", 0),
        TACInstr("ASSIGN", "n", 50),
        TACInstr("ASSIGN", "factor", 4),              # Loop invariant
        TACInstr("MUL", "inv_step", "factor", 2),     # Invariant folded to 8
        TACInstr("LABEL", "LOOP_START", None),
        TACInstr("SUB", "t_cond", "i", "n"),          # cond = i - n
        TACInstr("JUMP_IF_ZERO", "LOOP_END", "t_cond"),
        TACInstr("ADD", "t_term", "i", "inv_step"),   # term = i + 8
        TACInstr("ADD", "sum", "sum", "t_term"),      # sum += term
        TACInstr("ADD", "i", "i", 1),                 # i += 1
        TACInstr("JUMP", "LOOP_START", None),
        TACInstr("LABEL", "LOOP_END", None),
        TACInstr("PRINT", None, "sum"),
        TACInstr("RETURN", None, "sum")
    ]


def run_compiler_level_benchmarks(results_dir: str) -> Dict:
    """Benchmark 1: Evaluates O0, O1, O2, O3 optimization levels across code size, runtime, and compile time."""
    print("Running Benchmark 1: Compiler Optimization Levels (O0, O1, O2, O3)...")
    prog1 = create_benchmark_program_1()
    prog2 = create_benchmark_program_2()
    
    programs = {"ArithmeticPipeline": prog1, "LoopAccumulator": prog2}
    level_results = {}
    
    for p_name, raw_tac in programs.items():
        level_results[p_name] = {}
        
        # O0: Unoptimized baseline
        t0 = time.perf_counter()
        vm_o0 = TACVM(raw_tac)
        res_o0 = vm_o0.run()
        c_time_o0 = (time.perf_counter() - t0) * 1e6
        
        level_results[p_name]["O0"] = {
            "code_size": len(raw_tac),
            "instructions_executed": res_o0["instructions_executed"],
            "compile_time_us": 0.0,
            "runtime_us": res_o0["runtime_us"],
            "output": res_o0["output"]
        }
        
        # O1: Constant Folding + DCE
        t0 = time.perf_counter()
        tac_o1 = ConstantFoldingAndPropagationPass().run(raw_tac)
        tac_o1 = DeadCodeEliminationPass().run(tac_o1)
        c_time_o1 = (time.perf_counter() - t0) * 1e6
        
        vm_o1 = TACVM(tac_o1)
        res_o1 = vm_o1.run()
        level_results[p_name]["O1"] = {
            "code_size": len(tac_o1),
            "instructions_executed": res_o1["instructions_executed"],
            "compile_time_us": c_time_o1,
            "runtime_us": res_o1["runtime_us"],
            "output": res_o1["output"]
        }
        
        # O2: ConstFold + CSE + Strength Reduction + DCE
        t0 = time.perf_counter()
        tac_o2 = ConstantFoldingAndPropagationPass().run(raw_tac)
        tac_o2 = CommonSubexpressionEliminationPass().run(tac_o2)
        tac_o2 = StrengthReductionPass().run(tac_o2)
        tac_o2 = DeadCodeEliminationPass().run(tac_o2)
        c_time_o2 = (time.perf_counter() - t0) * 1e6
        
        vm_o2 = TACVM(tac_o2)
        res_o2 = vm_o2.run()
        level_results[p_name]["O2"] = {
            "code_size": len(tac_o2),
            "instructions_executed": res_o2["instructions_executed"],
            "compile_time_us": c_time_o2,
            "runtime_us": res_o2["runtime_us"],
            "output": res_o2["output"]
        }
        
        # O3: Formulated DP Optimal Pass Ordering
        dp_res = optimize_pass_ordering_dp(raw_tac, depth=3)
        tac_o3 = dp_res["optimized_tac"]
        vm_o3 = TACVM(tac_o3)
        res_o3 = vm_o3.run()
        
        level_results[p_name]["O3"] = {
            "code_size": len(tac_o3),
            "instructions_executed": res_o3["instructions_executed"],
            "compile_time_us": dp_res["compile_time_us"],
            "runtime_us": res_o3["runtime_us"],
            "optimal_pass_sequence": dp_res["optimal_sequence"],
            "output": res_o3["output"]
        }
        
        # Verify strict output equivalence!
        for lvl in ["O1", "O2", "O3"]:
            assert level_results[p_name][lvl]["output"] == level_results[p_name]["O0"]["output"], \
                f"Output mismatch in {p_name} at {lvl}!"
                
    # Plot Code Size & Execution Speedup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    levels = ["O0", "O1", "O2", "O3"]
    sizes_p1 = [level_results["ArithmeticPipeline"][lvl]["code_size"] for lvl in levels]
    sizes_p2 = [level_results["LoopAccumulator"][lvl]["code_size"] for lvl in levels]
    
    x = np.arange(len(levels))
    width = 0.35
    ax1.bar(x - width/2, sizes_p1, width, label="ArithmeticPipeline", color="#1f77b4")
    ax1.bar(x + width/2, sizes_p2, width, label="LoopAccumulator", color="#ff7f0e")
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels)
    ax1.set_ylabel("Code Size (TAC Instructions)")
    ax1.set_title("Instruction Count Reduction")
    ax1.grid(True, ls="--", alpha=0.5, axis='y')
    ax1.legend()
    
    # Runtime comparison
    cycles_p1 = [level_results["ArithmeticPipeline"][lvl]["instructions_executed"] for lvl in levels]
    cycles_p2 = [level_results["LoopAccumulator"][lvl]["instructions_executed"] for lvl in levels]
    ax2.bar(x - width/2, cycles_p1, width, label="ArithmeticPipeline", color="#1f77b4")
    ax2.bar(x + width/2, cycles_p2, width, label="LoopAccumulator", color="#ff7f0e")
    ax2.set_xticks(x)
    ax2.set_xticklabels(levels)
    ax2.set_ylabel("VM Instructions Executed")
    ax2.set_title("Execution Dynamic Instruction Count")
    ax2.grid(True, ls="--", alpha=0.5, axis='y')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "compiler_instruction_reduction.png"), dpi=200)
    plt.close()
    
    return level_results


def run_formulation_benchmarks(results_dir: str) -> Dict:
    """Benchmark 2: Loop Unroll Factor and Register Allocation Optimization Formulations."""
    print("Running Benchmark 2: Compiler Formulation Decisions (Unrolling & Pass Ordering)...")
    unroll_res = optimize_loop_unroll_factor(
        n_iterations=100,
        body_size=5,
        branch_cost=4.0,
        icache_penalty_lambda=0.4,
        max_cache_budget=100
    )
    
    # Plot Loop Unrolling trade-off
    factors = [c["unroll_factor"] for c in unroll_res["candidate_evaluations"]]
    branch_costs = [c["branch_overhead"] for c in unroll_res["candidate_evaluations"]]
    cache_penalties = [c["cache_penalty"] for c in unroll_res["candidate_evaluations"]]
    total_costs = [c["total_cost"] for c in unroll_res["candidate_evaluations"]]
    
    plt.figure(figsize=(8, 5))
    plt.plot(factors, branch_costs, 'ro--', label="Branch Overhead (N/u * C_br)", lw=2)
    plt.plot(factors, cache_penalties, 'bs--', label="I-Cache Expansion (u * B * lambda)", lw=2)
    plt.plot(factors, total_costs, 'g^-', label="Total Cost Function J(u)", lw=2.5)
    plt.axvline(unroll_res["optimal_unroll_factor"], color='k', ls=':', label=f"Optimal Unroll Factor u*={unroll_res['optimal_unroll_factor']}")
    plt.xlabel("Unroll Factor u")
    plt.ylabel("Cost Metric")
    plt.title("Loop Unroll Factor Optimization: Branch vs I-Cache Trade-Off")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "loop_unroll_tradeoff.png"), dpi=200)
    plt.close()
    
    # Register Allocation
    reg_res = register_allocation_graph_coloring(
        variables=["v1", "v2", "v3", "v4", "v5", "v6"],
        interference_edges=[
            ("v1", "v2"), ("v2", "v3"), ("v3", "v4"),
            ("v4", "v5"), ("v5", "v6"), ("v1", "v6"),
            ("v1", "v4"), ("v2", "v5")
        ],
        num_physical_registers=3
    )
    
    return {
        "loop_unroll_optimization": unroll_res,
        "register_allocation": reg_res
    }


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    all_results["optimization_levels"] = run_compiler_level_benchmarks(results_dir)
    all_results["formulations"] = run_formulation_benchmarks(results_dir)
    
    output_path = os.path.join(results_dir, "compiler_benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\nAll Track F benchmarks completed successfully!")
    print(f"Results and plots saved to: {results_dir}")


if __name__ == "__main__":
    main()
