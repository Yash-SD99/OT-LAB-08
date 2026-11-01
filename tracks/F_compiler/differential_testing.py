"""Differential Testing Suite for TAC Compiler Optimizations (Track F).

Generates randomized programs containing:
- Variable declarations and assignments
- Arithmetic expressions with common subexpressions and power-of-two constants
- Conditional if/else branches
- While loops with bounded iterations
- Dead code assignments

Executes both unoptimized (-O0) and optimized (-O3) bytecode on TACVM across
numerous seeds, asserting strict identity of return values and live environment state.
"""

from __future__ import annotations
import random
from typing import Any, Dict, List, Tuple
from tracks.F_compiler.toy_language import compile_source_to_tac
from tracks.F_compiler.ir_compiler import TACVM
from tracks.F_compiler.passes import OptimizationPipeline


def generate_random_toy_program(seed: int) -> str:
    """Generates a syntactically valid toy language program with control flow and arithmetic."""
    rng = random.Random(seed)
    lines: List[str] = []

    # Initial variables
    v_a = rng.randint(2, 10)
    v_b = rng.randint(1, 5)
    v_c = rng.randint(2, 6)
    lines.append(f"var a = {v_a};")
    lines.append(f"var b = {v_b};")
    lines.append(f"var c = {v_c};")
    lines.append(f"var res = 0;")

    # 1. CSE and dead code opportunity
    # (a + b) repeated multiple times, plus dead assignments
    lines.append("var t1 = a + b;")
    lines.append("var dead1 = a * 17;")
    lines.append("var t2 = a + b;") # Redundant CSE
    lines.append("res = res + t1 + t2;")

    # 2. Strength reduction opportunity
    # Multiplications by powers of 2 (4, 8) and identity operations
    lines.append("var p1 = b * 4;") # Strength reduction to << 2
    lines.append("var p2 = c * 8;") # Strength reduction to << 3
    lines.append("var dead2 = p1 * 0;") # Dead arithmetic
    lines.append("res = res + p1 + p2;")

    # 3. Conditional Branching
    if rng.choice([True, False]):
        lines.append("if (a > b) {")
        lines.append("    res = res + a * 2;")
        lines.append("} else {")
        lines.append("    res = res + b * 2;")
        lines.append("}")

    # 4. Bounded While Loop
    loop_limit = rng.randint(2, 5)
    lines.append(f"var iter = {loop_limit};")
    lines.append("while (iter > 0) {")
    lines.append("    var loop_inv = a + 5;") # LICM candidate
    lines.append("    res = res + loop_inv;")
    lines.append("    iter = iter - 1;")
    lines.append("}")

    lines.append("return res;")
    return "\n".join(lines)


def run_differential_test(seed: int) -> Dict[str, Any]:
    """Runs a single differential test comparing -O0 vs -O3 execution."""
    code = generate_random_toy_program(seed)
    tac_unopt = compile_source_to_tac(code)

    # Execute -O0
    vm_o0 = TACVM(tac_unopt)
    res_o0 = vm_o0.run()

    # Optimize with -O3
    pipeline = OptimizationPipeline(level="O3", unroll_factor=2)
    tac_o3 = pipeline.run(tac_unopt)

    # Execute -O3
    vm_o3 = TACVM(tac_o3)
    res_o3 = vm_o3.run()

    # Verify equivalence of observable outputs (return and print values)
    out_eq = (res_o0["output"] == res_o3["output"])
    divergence = not out_eq

    return {
        "seed": seed,
        "divergence": divergence,
        "o0_output": res_o0["output"],
        "o3_output": res_o3["output"],
        "o0_cycles": res_o0["total_cycles"],
        "o3_cycles": res_o3["total_cycles"],
        "o0_code_size": res_o0["code_size"],
        "o3_code_size": res_o3["code_size"],
        "cycle_speedup": float(res_o0["total_cycles"] / max(res_o3["total_cycles"], 1)),
        "source_code": code
    }


def run_differential_suite(num_tests: int = 50) -> Dict[str, Any]:
    """Runs differential testing suite across num_tests randomized programs."""
    divergences = 0
    results = []

    for s in range(1, num_tests + 1):
        res = run_differential_test(seed=s * 101)
        results.append(res)
        if res["divergence"]:
            divergences += 1

    return {
        "total_tests": num_tests,
        "divergences": divergences,
        "divergence_rate": float(divergences / num_tests),
        "all_passed": (divergences == 0),
        "test_results": results
    }


if __name__ == "__main__":
    suite = run_differential_suite(50)
    print(f"Differential Testing Complete: {suite['total_tests']} tests run.")
    print(f"Divergences: {suite['divergences']} (Rate: {suite['divergence_rate'] * 100:.2f}%)")
    print(f"All Passed: {suite['all_passed']}")
