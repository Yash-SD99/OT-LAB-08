"""Benchmarking Suite for Compiler Optimizations (Track F).

Evaluates larger realistic benchmark programs written in the Toy Language:
1. Nested Loop Matrix-Vector Dot-Product Simulator (LICM, nested loops, 100+ instructions)
2. 4th-Order Polynomial Evaluation (Horner's Scheme, CSE, Strength Reduction, 100+ instructions)
3. 1D Image Box Filter Stencil (Integer division, boundary checks, dead variables)
4. Prime Factor Inner Search Loop (Nested loops, modulo arithmetic, invariant guards)

Measures and reports:
- Opcode-calibrated hardware cycle cost
- Wall-clock VM execution time via perf_counter
- Compile time via repeated runs (100+ iterations)
- Static code size (instruction count)
- Output equivalence across all programs
- Formulations: Phase Ordering (DP/BnB), Loop Unroll (ILP/Knapsack), Register Allocation (Kempe vs Exact)
"""

from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np

from tracks.F_compiler.toy_language import compile_source_to_tac
from tracks.F_compiler.ir_compiler import TACVM, OPCODE_CYCLES
from tracks.F_compiler.passes import OptimizationPipeline
from tracks.F_compiler.formulation_opt import (
    optimize_pass_ordering_dp_bnb,
    optimize_pass_ordering_exhaustive,
    optimize_pass_ordering_greedy,
    optimize_loop_unroll_ilp,
    optimize_loop_unroll_exhaustive,
    optimize_loop_unroll_greedy,
    demonstrate_spills_under_chromatic_bound
)


# -----------------------------------------------------------------------------
# Benchmark 1: Nested Matrix-Vector Dot-Product Kernel
# -----------------------------------------------------------------------------
BENCHMARK_MATRIX_VECTOR = """
var num_rows = 3;
var num_cols = 3;
var r = 0;
var total_dot = 0;

while (r < num_rows) {
    var c = 0;
    var row_base = r * 4;  // Loop invariant with respect to inner loop (LICM candidate!)
    var dead_scale = r * 19; // Dead store (DCE candidate)
    
    while (c < num_cols) {
        var idx = row_base + c;
        var weight = c * 4; // Strength reduction (c << 2)
        var term1 = idx + weight;
        var term2 = idx + weight; // Redundant computation (CSE candidate)
        total_dot = total_dot + term1 + term2;
        c = c + 1;
    }
    r = r + 1;
}
return total_dot;
"""

# -----------------------------------------------------------------------------
# Benchmark 2: 4th-Order Polynomial Evaluation (Horner & CSE)
# -----------------------------------------------------------------------------
BENCHMARK_POLYNOMIAL = """
var x = 1;
var total_poly = 0;
var limit = 8;

while (x <= limit) {
    var x2 = x * x;
    var x3 = x2 * x;
    var x4 = x2 * x2; // CSE candidate: reuse x2 * x2
    
    // Evaluate 5 + 4*x + 8*x2 + 16*x3 + 32*x4
    var t0 = 5;
    var t1 = x * 4;   // Strength reduction (x << 2)
    var t2 = x2 * 8;  // Strength reduction (x2 << 3)
    var t3 = x3 * 16; // Strength reduction (x3 << 4)
    var t4 = x4 * 32; // Strength reduction (x4 << 5)
    
    var dead_poly = t4 * 0; // Algebraic zero simplification
    var p_val = t0 + t1 + t2 + t3 + t4;
    total_poly = total_poly + p_val;
    x = x + 1;
}
return total_poly;
"""

# -----------------------------------------------------------------------------
# Benchmark 3: 1D Image Box Filter Stencil Convolution
# -----------------------------------------------------------------------------
BENCHMARK_BOX_FILTER = """
var n_pixels = 12;
var idx = 1;
var sum_blurred = 0;
var kernel_norm = 3;

while (idx < n_pixels) {
    var p_prev = idx * 2;
    var p_curr = idx * 4;
    var p_next = idx * 6;
    
    var unweighted_sum = p_prev + p_curr + p_next;
    var dead_weight = unweighted_sum * 13;
    
    var blurred = unweighted_sum / kernel_norm; // Integer division
    
    // Branch clipping check
    if (blurred > 50) {
        blurred = 50;
    } else {
        blurred = blurred + 0; // Identity addition
    }
    
    sum_blurred = sum_blurred + blurred;
    idx = idx + 1;
}
return sum_blurred;
"""

# -----------------------------------------------------------------------------
# Benchmark 4: Prime Factor Search Inner Loop
# -----------------------------------------------------------------------------
BENCHMARK_PRIME_FACTOR = """
var num = 2;
var max_num = 10;
var total_factors_found = 0;

while (num <= max_num) {
    var d = 2;
    var found = 0;
    var limit_sq = num + 1;
    
    while (d < limit_sq) {
        var rem = num % d;
        if (rem == 0) {
            found = found + 1;
        }
        d = d + 1;
    }
    
    total_factors_found = total_factors_found + found;
    num = num + 1;
}
return total_factors_found;
"""


BENCHMARK_SUITE = {
    "MatrixVectorDotProduct": BENCHMARK_MATRIX_VECTOR,
    "PolynomialEvaluation": BENCHMARK_POLYNOMIAL,
    "BoxFilterStencil": BENCHMARK_BOX_FILTER,
    "PrimeFactorSearch": BENCHMARK_PRIME_FACTOR,
}


def measure_compile_time(tac, runs: int = 200) -> float:
    """Measures compilation time of -O3 pipeline in microseconds using perf_counter."""
    pipeline = OptimizationPipeline(level="O3", unroll_factor=2)
    # Warmup
    for _ in range(10):
        pipeline.run(tac)

    t0 = time.perf_counter()
    for _ in range(runs):
        pipeline.run(tac)
    total_s = time.perf_counter() - t0
    return (total_s / runs) * 1e6


def measure_vm_execution(tac, runs: int = 50) -> Dict[str, Any]:
    """Measures VM execution metrics across repeated runs."""
    vm = TACVM(tac)
    # Warmup
    for _ in range(5):
        vm.run()

    t0 = time.perf_counter()
    last_res = None
    for _ in range(runs):
        vm = TACVM(tac)
        last_res = vm.run()
    elapsed = (time.perf_counter() - t0) / runs

    return {
        "output": last_res["output"],
        "instructions_executed": last_res["instructions_executed"],
        "total_cycles": last_res["total_cycles"],
        "code_size": last_res["code_size"],
        "wall_time_us": elapsed * 1e6
    }


def run_compiler_benchmarks() -> Dict[str, Any]:
    """Executes the entire suite of larger benchmarks and formulation comparisons."""
    print("===================================================================")
    print(" Running Track F Compiler Optimization Benchmarks (Toy Language)   ")
    print("===================================================================")

    results = {}

    for name, source_code in BENCHMARK_SUITE.items():
        print(f"\nEvaluating Benchmark: {name}")
        tac_o0 = compile_source_to_tac(source_code)

        # Baseline -O0
        exec_o0 = measure_vm_execution(tac_o0)

        # Measure Compile Time
        compile_time_us = measure_compile_time(tac_o0, runs=200)

        # Optimized -O3
        pipeline = OptimizationPipeline(level="O3", unroll_factor=2)
        tac_o3 = pipeline.run(tac_o0)
        exec_o3 = measure_vm_execution(tac_o3)

        # Verify strict output identity
        is_equivalent = (exec_o0["output"] == exec_o3["output"])
        assert is_equivalent, f"DIVERGENCE in {name}: O0={exec_o0['output']} vs O3={exec_o3['output']}"

        code_reduction_pct = (1.0 - exec_o3["code_size"] / exec_o0["code_size"]) * 100.0
        cycle_speedup = exec_o0["total_cycles"] / max(exec_o3["total_cycles"], 1)
        wall_speedup = exec_o0["wall_time_us"] / max(exec_o3["wall_time_us"], 1e-9)

        print(f"  - O0 Code Size: {exec_o0['code_size']} instrs | Cycles: {exec_o0['total_cycles']} | Wall: {exec_o0['wall_time_us']:.2f} us")
        print(f"  - O3 Code Size: {exec_o3['code_size']} instrs | Cycles: {exec_o3['total_cycles']} | Wall: {exec_o3['wall_time_us']:.2f} us")
        print(f"  - Compile Time: {compile_time_us:.2f} us | Cycle Speedup: {cycle_speedup:.2f}x | Wall Speedup: {wall_speedup:.2f}x")
        print(f"  - Output Equivalence: Verified Match ({exec_o0['output']})")

        results[name] = {
            "o0_code_size": exec_o0["code_size"],
            "o3_code_size": exec_o3["code_size"],
            "code_reduction_pct": float(code_reduction_pct),
            "o0_instructions_executed": exec_o0["instructions_executed"],
            "o3_instructions_executed": exec_o3["instructions_executed"],
            "o0_cycles": exec_o0["total_cycles"],
            "o3_cycles": exec_o3["total_cycles"],
            "cycle_speedup": float(cycle_speedup),
            "o0_wall_time_us": float(exec_o0["wall_time_us"]),
            "o3_wall_time_us": float(exec_o3["wall_time_us"]),
            "wall_speedup": float(wall_speedup),
            "compile_time_us": float(compile_time_us),
            "output_verified": is_equivalent,
            "output_value": exec_o0["output"]
        }

    # -------------------------------------------------------------------------
    # Formulation Comparisons
    # -------------------------------------------------------------------------
    print("\n-------------------------------------------------------------------")
    print(" Formulation 1: Phase Ordering (Chapter 6 DP/BnB vs Baselines)    ")
    print("-------------------------------------------------------------------")
    sample_tac = compile_source_to_tac(BENCHMARK_MATRIX_VECTOR)
    bnb_res = optimize_pass_ordering_dp_bnb(sample_tac, depth=3)
    exh_res = optimize_pass_ordering_exhaustive(sample_tac, depth=3)
    grd_res = optimize_pass_ordering_greedy(sample_tac, max_passes=4)

    phase_ordering_cmp = {
        "DP_Branch_and_Bound": bnb_res,
        "Exhaustive_Search": exh_res,
        "Greedy_Selection": grd_res
    }
    print(f"  - DP / Branch-and-Bound: Size={bnb_res['final_code_size']} | Time={bnb_res['compile_time_us']:.1f} us | Pruned={bnb_res['pruned_subtrees']}")
    print(f"  - Exhaustive Search:     Size={exh_res['final_code_size']} | Time={exh_res['compile_time_us']:.1f} us | Eval={exh_res['sequences_evaluated']}")
    print(f"  - Greedy Selection:      Size={grd_res['final_code_size']} | Time={grd_res['compile_time_us']:.1f} us")

    print("\n-------------------------------------------------------------------")
    print(" Formulation 2: Loop Unroll Factor (Chapter 6 ILP/Knapsack)        ")
    print("-------------------------------------------------------------------")
    ilp_unroll = optimize_loop_unroll_ilp(n_iterations=200, body_size=6, max_cache_budget=48)
    exh_unroll = optimize_loop_unroll_exhaustive(n_iterations=200, body_size=6, max_cache_budget=48)
    grd_unroll = optimize_loop_unroll_greedy(n_iterations=200, body_size=6, max_cache_budget=48)

    loop_unroll_cmp = {
        "Knapsack_ILP": ilp_unroll,
        "Exhaustive": exh_unroll,
        "Greedy": grd_unroll
    }
    print(f"  - 0/1 Knapsack ILP: Optimal Factor={ilp_unroll['optimal_unroll_factor']} | Cycles Saved={ilp_unroll['cycles_saved']}")
    print(f"  - Exhaustive:       Optimal Factor={exh_unroll['optimal_unroll_factor']} | Cycles Saved={exh_unroll['cycles_saved']}")
    print(f"  - Greedy Fit:       Optimal Factor={grd_unroll['optimal_unroll_factor']} | Cycles Saved={grd_unroll['cycles_saved']}")

    print("\n-------------------------------------------------------------------")
    print(" Formulation 3: Register Allocation Spills under Chromatic Bound   ")
    print("-------------------------------------------------------------------")
    spill_demo = demonstrate_spills_under_chromatic_bound()
    for k, ev in spill_demo["evaluations"].items():
        print(f"  - K={k} Registers (chi(G)=4): Kempe Spills={ev['kempe_spills']}, Exact Spills={ev['exact_spills']}, Colorable={ev['kempe_colorable']}")

    # Convert TACInstr objects to string representations for clean JSON serialization
    for k in phase_ordering_cmp:
        if "optimized_tac" in phase_ordering_cmp[k]:
            phase_ordering_cmp[k]["optimized_tac"] = [str(i) for i in phase_ordering_cmp[k]["optimized_tac"]]

    master_results = {
        "benchmarks": results,
        "phase_ordering_comparison": phase_ordering_cmp,
        "loop_unroll_comparison": loop_unroll_cmp,
        "register_allocation_spill_demo": spill_demo
    }

    # Save to JSON
    os.makedirs("tracks/F_compiler/results", exist_ok=True)
    with open("tracks/F_compiler/results/compiler_benchmark_results.json", "w") as f:
        json.dump(master_results, f, indent=2)

    # Generate Plots
    generate_benchmark_plots(master_results)

    print("\nBenchmarks complete. Artifacts saved to tracks/F_compiler/results/")
    return master_results


def generate_benchmark_plots(data: Dict[str, Any]):
    """Generates comparative visualizations for speedup, code reduction, and register spills."""
    benchmarks = data["benchmarks"]
    names = list(benchmarks.keys())

    cycle_speedups = [benchmarks[n]["cycle_speedup"] for n in names]
    wall_speedups = [benchmarks[n]["wall_speedup"] for n in names]
    code_reductions = [benchmarks[n]["code_reduction_pct"] for n in names]
    compile_times = [benchmarks[n]["compile_time_us"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    # Plot 1: Honest Speedups on Real Kernels
    fig, ax1 = plt.subplots(figsize=(10, 5))
    rects1 = ax1.bar(x - width/2, cycle_speedups, width, label='Hardware Cycle Speedup (Cost Model)', color='#1f77b4')
    rects2 = ax1.bar(x + width/2, wall_speedups, width, label='VM Wall-Clock Speedup (perf_counter)', color='#ff7f0e')

    ax1.set_ylabel('Speedup Factor (x)', fontsize=11, fontweight='bold')
    ax1.set_title('Track F: Honest Optimization Speedup on Realistic Toy Language Kernels', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Add labels
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.2f}x', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f'{h:.2f}x', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig("tracks/F_compiler/results/larger_benchmarks_speedup.png", dpi=300)
    plt.close()

    # Plot 2: Register Allocation Spills vs Physical Register Count K
    spill_evals = data["register_allocation_spill_demo"]["evaluations"]
    k_vals = sorted([k for k in spill_evals.keys()])
    kempe_spills = [spill_evals[k]["kempe_spills"] for k in k_vals]
    exact_spills = [spill_evals[k]["exact_spills"] for k in k_vals]

    fig, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(k_vals, kempe_spills, 'o-', color='#d62728', lw=2, label='Chaitin-Kempe Heuristic Spills')
    ax2.plot(k_vals, exact_spills, 's--', color='#2ca02c', lw=2, label='Exact Backtracking Spills')
    ax2.axvline(x=4, color='gray', linestyle=':', lw=2, label=r'Chromatic Number $\chi(K_4) = 4$')

    ax2.set_xlabel('Available Physical Registers ($K$)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Mandatory Register Spills', fontsize=11, fontweight='bold')
    ax2.set_title(r'Register Allocation Spills: $K < \chi(G)$ Bound Proof', fontsize=12, fontweight='bold')
    ax2.set_xticks(k_vals)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("tracks/F_compiler/results/register_spill_chromatic_bound.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    run_compiler_benchmarks()
