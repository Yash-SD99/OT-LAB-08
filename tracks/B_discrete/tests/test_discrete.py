"""Unit and Verification Tests for Track B: Discrete Optimization."""

import unittest
import numpy as np
from tracks.B_discrete.knapsack_solvers import (
    KnapsackInstance,
    solve_greedy,
    solve_dynamic_programming,
    solve_branch_and_bound,
    solve_ilp_scipy,
    solve_ilp_pulp
)
from tracks.B_discrete.feature_selection import (
    FeatureSelectionProblem,
    exact_best_subset_selection,
    forward_stepwise_selection
)
from tracks.B_discrete.qubo_formulation import (
    qubo_energy,
    ising_energy,
    binary_to_spin,
    spin_to_binary,
    qubo_to_ising,
    build_maxcut_qubo,
    build_constrained_knapsack_qubo
)
from tracks.B_discrete.classical_qubo_solver import (
    solve_qubo_exact,
    solve_qubo_simulated_annealing,
    analyze_minor_embedding_limits
)


class TestDiscreteSolvers(unittest.TestCase):

    def test_knapsack_exact_solvers_equivalence(self):
        """All exact solvers (DP, Custom B&B, scipy.milp, PuLP) must yield identical optimal value."""
        values = np.array([60.0, 100.0, 120.0, 80.0, 30.0])
        weights = np.array([10.0, 20.0, 30.0, 15.0, 5.0])
        capacity = 50.0
        inst = KnapsackInstance(values, weights, capacity)
        
        res_dp = solve_dynamic_programming(inst)
        res_bnb = solve_branch_and_bound(inst)
        res_scipy = solve_ilp_scipy(inst)
        res_pulp = solve_ilp_pulp(inst)
        
        opt_val = res_dp["objective"]
        self.assertEqual(res_bnb["objective"], opt_val)
        self.assertEqual(res_scipy["objective"], opt_val)
        self.assertEqual(res_pulp["objective"], opt_val)
        
        # Verify feasibility
        for res in [res_dp, res_bnb, res_scipy, res_pulp]:
            self.assertLessEqual(res["weight"], capacity)

    def test_knapsack_greedy_feasibility_and_bound(self):
        """Greedy must be feasible and <= exact optimal."""
        values = np.array([10.0, 40.0, 30.0, 50.0])
        weights = np.array([5.0, 4.0, 6.0, 3.0])
        capacity = 10.0
        inst = KnapsackInstance(values, weights, capacity)
        
        res_greedy = solve_greedy(inst)
        res_exact = solve_branch_and_bound(inst)
        
        self.assertLessEqual(res_greedy["weight"], capacity)
        self.assertLessEqual(res_greedy["objective"], res_exact["objective"])

    def test_qubo_ising_conversion_identity(self):
        """Ising energy of s = 2x - 1 must match QUBO energy of x exactly."""
        np.random.seed(42)
        N = 5
        Q = np.random.randn(N, N)
        Q = 0.5 * (Q + Q.T) # Symmetric
        
        J, h, C = qubo_to_ising(Q)
        
        for _ in range(10):
            x = np.random.randint(0, 2, size=N).astype(float)
            s = binary_to_spin(x)
            
            e_qubo = qubo_energy(Q, x)
            e_ising = ising_energy(J, h, C, s)
            
            self.assertAlmostEqual(e_qubo, e_ising, places=6)

    def test_maxcut_qubo_on_c4_graph(self):
        """Max-Cut on 4-cycle graph C_4: max cut is 4, QUBO energy minimum matches."""
        # 4-cycle graph edges: (0,1), (1,2), (2,3), (3,0)
        W = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ], dtype=float)
        
        Q, J, h, C = build_maxcut_qubo(W)
        res_exact = solve_qubo_exact(Q)
        
        # For C4, bipartite partition [0, 1, 0, 1] cuts all 4 edges
        # Cut = 4.0. QUBO energy = sum W_ij (x_i + x_j - 2 x_i x_j) = -Cut
        best_x = np.array(res_exact["best_state"])
        cut_value = 0.5 * np.sum(W * (best_x[:, None] != best_x[None, :]))
        self.assertEqual(cut_value, 4.0)

    def test_feature_selection_cardinality(self):
        """Feature selection selects at most K features."""
        np.random.seed(42)
        X = np.random.randn(30, 8)
        # Ground truth: y depends heavily on features 0 and 2
        y = 3.0 * X[:, 0] - 2.0 * X[:, 2] + 0.1 * np.random.randn(30)
        prob = FeatureSelectionProblem(X, y, max_features=2)
        
        res_exact = exact_best_subset_selection(prob)
        res_greedy = forward_stepwise_selection(prob)
        
        self.assertLessEqual(len(res_exact["selected_indices"]), 2)
        self.assertLessEqual(len(res_greedy["selected_indices"]), 2)
        self.assertGreater(res_exact["r2"], 0.95)
        self.assertIn(0, res_exact["selected_indices"])
        self.assertIn(2, res_exact["selected_indices"])

    def test_simulated_annealing_qubo_convergence(self):
        """Simulated Annealing finds global optimum on small Max-Cut graph."""
        W = np.array([
            [0, 2, 0, 3],
            [2, 0, 4, 0],
            [0, 4, 0, 1],
            [3, 0, 1, 0]
        ], dtype=float)
        Q, _, _, _ = build_maxcut_qubo(W)
        
        res_exact = solve_qubo_exact(Q)
        res_sa = solve_qubo_simulated_annealing(Q, initial_temp=20.0, seed=123)
        
        self.assertAlmostEqual(res_sa["min_energy"], res_exact["min_energy"], places=4)

    def test_minor_embedding_limits(self):
        """Tests minor embedding physical qubit scaling report."""
        embed = analyze_minor_embedding_limits(n_logical_qubits=10)
        self.assertGreater(embed["chimera_estimated_physical_qubits"], 10)
        self.assertGreater(embed["qubit_expansion_factor_chimera"], 1.0)


if __name__ == "__main__":
    unittest.main()
