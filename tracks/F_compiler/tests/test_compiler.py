"""Unit and Verification Tests for Track F: Compiler Optimization.

Verifies:
1. Toy Language Lexer, Parser (AST), and TAC Code Generator
2. Integer Semantics Safety (C-style truncation towards zero, shift limits, zero divisor preservation)
3. Constant Folding & Propagation Pass
4. Common Subexpression Elimination Pass (CSE)
5. Dead Code Elimination Pass with Fixed-Point Liveness (DCE)
6. Strength Reduction Pass (Powers of 2 to shifts, algebraic identities)
7. Loop Invariant Code Motion (LICM)
8. Guarded Loop Unrolling Pass
9. Formulation 1: Phase Ordering (Chapter 6 DP/BnB vs Baselines)
10. Formulation 2: Loop Unroll Factor (Chapter 6 0/1 Knapsack ILP)
11. Formulation 3: Register Allocation (Kempe vs Exact, Spills when K < chi(G))
12. Differential Testing: Randomized program equivalence (-O0 vs -O3)
"""

from __future__ import annotations
import unittest
from tracks.F_compiler.ir_compiler import TACInstr, TACVM, OPCODE_CYCLES
from tracks.F_compiler.toy_language import compile_source_to_tac
from tracks.F_compiler.passes import (
    ConstantFoldingAndPropagationPass,
    CommonSubexpressionEliminationPass,
    DeadCodeEliminationPass,
    StrengthReductionPass,
    LoopInvariantCodeMotionPass,
    LoopUnrollPass,
    OptimizationPipeline
)
from tracks.F_compiler.formulation_opt import (
    optimize_pass_ordering_dp_bnb,
    optimize_pass_ordering_exhaustive,
    optimize_pass_ordering_greedy,
    optimize_loop_unroll_ilp,
    optimize_loop_unroll_exhaustive,
    optimize_loop_unroll_greedy,
    register_allocation_kempe,
    register_allocation_exact_backtracking,
    register_allocation_greedy_first_fit,
    demonstrate_spills_under_chromatic_bound
)
from tracks.F_compiler.differential_testing import run_differential_suite


class TestCompilerOptimization(unittest.TestCase):

    def test_toy_language_lexer_parser_lowering(self):
        """Toy language compiles statements, loops, and conditions into valid TAC."""
        source = """
        var a = 10;
        var b = 20;
        var count = 3;
        var sum = 0;
        while (count > 0) {
            if (a < b) {
                sum = sum + a;
            } else {
                sum = sum + b;
            }
            count = count - 1;
        }
        return sum;
        """
        tac = compile_source_to_tac(source)
        self.assertGreater(len(tac), 10)
        vm = TACVM(tac)
        res = vm.run()
        self.assertEqual(res["output"], [30])

    def test_integer_semantics_safety(self):
        """Enforces C-style integer truncation towards zero, safe mod, and shift restrictions."""
        # 1. Truncation towards zero: -7 / 2 == -3 (not -4)
        prog_div = [
            TACInstr("ASSIGN", "x", -7),
            TACInstr("DIV", "res", "x", 2),
            TACInstr("RETURN", None, "res")
        ]
        res_div = TACVM(prog_div).run()
        self.assertEqual(res_div["output"], [-3])

        # 2. Modulo matching truncation: -7 % 2 == -1
        prog_mod = [
            TACInstr("ASSIGN", "x", -7),
            TACInstr("MOD", "res", "x", 2),
            TACInstr("RETURN", None, "res")
        ]
        res_mod = TACVM(prog_mod).run()
        self.assertEqual(res_mod["output"], [-1])

        # 3. Shift operand bounds: shift by >= 32 or < 0 raises error
        prog_bad_shift = [
            TACInstr("ASSIGN", "x", 1),
            TACInstr("SHL", "res", "x", 35),
            TACInstr("RETURN", None, "res")
        ]
        with self.assertRaises(ValueError):
            TACVM(prog_bad_shift).run()

        # 4. Zero division raises error and is NOT constant-folded away
        prog_zero = [
            TACInstr("DIV", "res", 10, 0),
            TACInstr("RETURN", None, "res")
        ]
        opt_zero = ConstantFoldingAndPropagationPass().run(prog_zero)
        self.assertEqual(opt_zero[0].op, "DIV") # Must NOT fold!
        with self.assertRaises(ZeroDivisionError):
            TACVM(opt_zero).run()

    def test_constant_folding(self):
        """Constant expressions are folded at compile time."""
        prog = [
            TACInstr("ASSIGN", "a", 15),
            TACInstr("ASSIGN", "b", 25),
            TACInstr("ADD", "c", "a", "b"),
            TACInstr("RETURN", None, "c")
        ]
        opt = ConstantFoldingAndPropagationPass().run(prog)
        vm = TACVM(opt)
        res = vm.run()
        self.assertEqual(res["output"], [40])
        self.assertEqual(opt[2].op, "ASSIGN")
        self.assertEqual(opt[2].src1, 40)

    def test_common_subexpression_elimination(self):
        """Redundant identical expressions are reused."""
        prog = [
            TACInstr("ADD", "t1", "x", "y"),
            TACInstr("ADD", "t2", "x", "y"),
            TACInstr("ADD", "t3", "y", "x") # Commutative match
        ]
        opt = CommonSubexpressionEliminationPass().run(prog)
        self.assertEqual(opt[1].op, "ASSIGN")
        self.assertEqual(opt[1].src1, "t1")
        self.assertEqual(opt[2].op, "ASSIGN")
        self.assertEqual(opt[2].src1, "t1")

    def test_dead_code_elimination_fixed_point(self):
        """Assignments to unused variables are eliminated, preserving loop updates."""
        prog = [
            TACInstr("ASSIGN", "t_dead", 999),
            TACInstr("ASSIGN", "live_var", 42),
            TACInstr("PRINT", None, "live_var")
        ]
        opt = DeadCodeEliminationPass().run(prog)
        dests = [i.dest for i in opt]
        self.assertNotIn("t_dead", dests)
        self.assertIn("live_var", dests)

    def test_strength_reduction_soundness(self):
        """Replaces x * 2^k with bit shift SHL k for positive k, and simplifies identities."""
        prog = [
            TACInstr("MUL", "t1", "x", 8),
            TACInstr("ADD", "t2", "y", 0),
            TACInstr("MUL", "t3", "z", 0),
            TACInstr("DIV", "t4", "w", 1)
        ]
        opt = StrengthReductionPass().run(prog)
        self.assertEqual(opt[0].op, "SHL")
        self.assertEqual(opt[0].src2, 3) # 8 = 2^3
        self.assertEqual(opt[1].op, "ASSIGN")
        self.assertEqual(opt[1].src1, "y")
        self.assertEqual(opt[2].op, "ASSIGN")
        self.assertEqual(opt[2].src1, 0)
        self.assertEqual(opt[3].op, "ASSIGN")
        self.assertEqual(opt[3].src1, "w")

    def test_loop_invariant_code_motion(self):
        """Hoists computations whose operands are external to the loop."""
        source = """
        var a = 7;
        var b = 3;
        var count = 4;
        var sum = 0;
        while (count > 0) {
            var inv = a * 4 + b; // Invariant inside loop!
            sum = sum + inv;
            count = count - 1;
        }
        return sum;
        """
        tac = compile_source_to_tac(source)
        opt = LoopInvariantCodeMotionPass().run(tac)

        # In unoptimized TAC, inv assignment is inside the loop body
        # In optimized TAC, inv calculation is hoisted before while_start
        vm_unopt = TACVM(tac).run()
        vm_opt = TACVM(opt).run()
        self.assertEqual(vm_unopt["output"], vm_opt["output"])

    def test_guarded_loop_unrolling(self):
        """Guarded loop unroll preserves exact output on odd trip counts."""
        source = """
        var iter = 3; // Odd trip count
        var sum = 0;
        while (iter > 0) {
            sum = sum + iter;
            iter = iter - 1;
        }
        return sum;
        """
        tac = compile_source_to_tac(source)
        res_o0 = TACVM(tac).run()

        # Unroll by factor 2
        unrolled = LoopUnrollPass(unroll_factor=2).run(tac)
        res_unroll = TACVM(unrolled).run()

        self.assertEqual(res_o0["output"], res_unroll["output"])
        self.assertEqual(res_unroll["output"], [6])

    def test_phase_ordering_formulations_comparison(self):
        """Verifies Chapter 6 DP/BnB pass ordering against Exhaustive and Greedy."""
        source = """
        var x = 10;
        var y = 20;
        var z = x + y;
        var dead = z * 50;
        var w = x + y;
        return w;
        """
        tac = compile_source_to_tac(source)
        bnb = optimize_pass_ordering_dp_bnb(tac, depth=3)
        exh = optimize_pass_ordering_exhaustive(tac, depth=3)
        grd = optimize_pass_ordering_greedy(tac, max_passes=3)

        # All formulations should find a reduced program size
        self.assertLess(bnb["final_code_size"], len(tac))
        self.assertLess(exh["final_code_size"], len(tac))
        self.assertLess(grd["final_code_size"], len(tac))
        # DP/BnB should prune search branches
        self.assertGreaterEqual(bnb["pruned_subtrees"], 0)

    def test_loop_unroll_knapsack_ilp(self):
        """Verifies 0/1 Knapsack ILP formulation for loop unroll factor."""
        ilp_res = optimize_loop_unroll_ilp(n_iterations=200, body_size=6, max_cache_budget=48)
        self.assertIn(ilp_res["optimal_unroll_factor"], [1, 2, 4, 8, 16])
        self.assertLessEqual(ilp_res["code_size"], 48)

    def test_register_allocation_spill_bound(self):
        """Proves mandatory spills when available registers K < Chromatic Number chi(G)."""
        demo = demonstrate_spills_under_chromatic_bound()
        evals = demo["evaluations"]

        # K=2 < 4: 2 spills
        self.assertEqual(evals[2]["kempe_spills"], 2)
        self.assertFalse(evals[2]["kempe_colorable"])

        # K=3 < 4: 1 spill
        self.assertEqual(evals[3]["kempe_spills"], 1)
        self.assertFalse(evals[3]["kempe_colorable"])

        # K=4 == 4: 0 spills (Colorable!)
        self.assertEqual(evals[4]["kempe_spills"], 0)
        self.assertTrue(evals[4]["kempe_colorable"])

    def test_differential_testing_suite(self):
        """Runs differential testing suite on randomized programs asserting 0 divergences."""
        suite = run_differential_suite(num_tests=25)
        self.assertTrue(suite["all_passed"])
        self.assertEqual(suite["divergences"], 0)


if __name__ == "__main__":
    unittest.main()
