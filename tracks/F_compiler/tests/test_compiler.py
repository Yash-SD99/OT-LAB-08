"""Unit and Verification Tests for Track F: Compiler Optimization."""

import unittest
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


class TestCompilerOptimization(unittest.TestCase):

    def test_output_equivalence_across_optimizations(self):
        """Verifies that optimized TAC produces the EXACT same output as unoptimized TAC."""
        # Program:
        #   x = 10
        #   y = 20
        #   t1 = x + y      (30)
        #   dead = t1 * 100 (dead code)
        #   t2 = x + y      (redundant CSE)
        #   z = t2 * 2      (strength reduction -> z = t2 << 1)
        #   print z
        #   return z
        unoptimized = [
            TACInstr("ASSIGN", "x", 10),
            TACInstr("ASSIGN", "y", 20),
            TACInstr("ADD", "t1", "x", "y"),
            TACInstr("MUL", "dead", "t1", 100),
            TACInstr("ADD", "t2", "x", "y"),
            TACInstr("MUL", "z", "t2", 2),
            TACInstr("PRINT", None, "z"),
            TACInstr("RETURN", None, "z")
        ]
        
        # Run unoptimized
        vm_unopt = TACVM(unoptimized)
        res_unopt = vm_unopt.run()
        
        # Run through full optimization pipeline
        tac_opt = ConstantFoldingAndPropagationPass().run(unoptimized)
        tac_opt = CommonSubexpressionEliminationPass().run(tac_opt)
        tac_opt = StrengthReductionPass().run(tac_opt)
        tac_opt = DeadCodeEliminationPass().run(tac_opt)
        
        # Run optimized
        vm_opt = TACVM(tac_opt)
        res_opt = vm_opt.run()
        
        # 1. Output must be identical!
        self.assertEqual(res_unopt["output"], res_opt["output"])
        # 2. Both must output 60 ( (10+20) * 2 = 60 )
        self.assertEqual(res_opt["output"], [60, 60])
        # 3. Optimized instruction count must be strictly less than unoptimized
        self.assertLess(len(tac_opt), len(unoptimized))

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
        # The ADD should be folded to ASSIGN c = 40
        self.assertEqual(opt[2].op, "ASSIGN")
        self.assertEqual(opt[2].src1, 40)

    def test_common_subexpression_elimination(self):
        """Redundant identical expressions are reused."""
        prog = [
            TACInstr("ADD", "t1", "x", "y"),
            TACInstr("ADD", "t2", "x", "y"),
            TACInstr("ADD", "t3", "y", "x") # Commutative
        ]
        opt = CommonSubexpressionEliminationPass().run(prog)
        self.assertEqual(opt[1].op, "ASSIGN")
        self.assertEqual(opt[1].src1, "t1")
        self.assertEqual(opt[2].op, "ASSIGN")
        self.assertEqual(opt[2].src1, "t1")

    def test_dead_code_elimination(self):
        """Assignments to unused temporaries are eliminated."""
        prog = [
            TACInstr("ASSIGN", "t_dead", 999),
            TACInstr("ASSIGN", "live_var", 42),
            TACInstr("PRINT", None, "live_var")
        ]
        opt = DeadCodeEliminationPass().run(prog)
        dests = [i.dest for i in opt]
        self.assertNotIn("t_dead", dests)
        self.assertIn("live_var", dests)

    def test_strength_reduction(self):
        """Replaces x * 2 with bit shift SHL 1."""
        prog = [
            TACInstr("MUL", "t1", "x", 8),
            TACInstr("ADD", "t2", "y", 0),
            TACInstr("MUL", "t3", "z", 0)
        ]
        opt = StrengthReductionPass().run(prog)
        self.assertEqual(opt[0].op, "SHL")
        self.assertEqual(opt[0].src2, 3) # 8 = 2^3
        self.assertEqual(opt[1].op, "ASSIGN")
        self.assertEqual(opt[1].src1, "y")
        self.assertEqual(opt[2].op, "ASSIGN")
        self.assertEqual(opt[2].src1, 0)

    def test_loop_unroll_optimization(self):
        """Selects optimal loop unroll factor."""
        res = optimize_loop_unroll_factor(n_iterations=200, body_size=4, branch_cost=5.0)
        self.assertIn(res["optimal_unroll_factor"], [4, 8, 16])
        self.assertLess(res["min_cost"], 200 * 5.0)

    def test_register_allocation_graph_coloring(self):
        """Assigns registers without adjacent interference conflicts."""
        variables = ["r1", "r2", "r3", "r4"]
        # Triangle interference between r1, r2, r3; r4 only interferes with r1
        edges = [("r1", "r2"), ("r2", "r3"), ("r3", "r1"), ("r1", "r4")]
        res = register_allocation_graph_coloring(variables, edges, num_physical_registers=3)
        
        self.assertTrue(res["is_colorable_without_spills"])
        assignment = res["register_assignment"]
        for u, v in edges:
            self.assertNotEqual(assignment[u], assignment[v])


if __name__ == "__main__":
    unittest.main()
