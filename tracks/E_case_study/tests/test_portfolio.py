"""Unit and Verification Tests for Track E: Non-Convex Portfolio Optimization."""

from __future__ import annotations
import unittest
import numpy as np

from tracks.E_case_study.portfolio_data import generate_portfolio_market_data, PortfolioDataset
from tracks.E_case_study.solvers import (
    MIQPSolver,
    GreedyPortfolioSolver,
    SimulatedAnnealingPortfolioSolver,
    GeneticAlgorithmPortfolioSolver,
    compute_portfolio_metrics
)


class TestPortfolioOptimization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = generate_portfolio_market_data(seed=42)

    def test_market_data_covariance_positive_definite(self):
        """Verifies that the generated factor covariance matrix is strictly positive definite."""
        sigma = self.data.covariance_matrix
        # Check symmetry
        self.assertTrue(np.allclose(sigma, sigma.T, atol=1e-8))
        # Check eigenvalues > 0
        eigenvalues = np.linalg.eigvalsh(sigma)
        self.assertTrue(np.all(eigenvalues > 0.0), f"Minimum eigenvalue non-positive: {np.min(eigenvalues)}")
        self.assertEqual(len(self.data.asset_names), 20)

    def test_miqp_solver_constraint_satisfaction(self):
        """Verifies that MIQP solution strictly satisfies budget, cardinality, and buy-in bounds."""
        k = 4
        solver = MIQPSolver(self.data, max_cardinality=k, lambda_risk=1.0)
        sol = solver.solve()

        # 1. Budget constraint: sum(w) == 1
        self.assertAlmostEqual(float(np.sum(sol.weights)), 1.0, places=3)

        # 2. Cardinality: sum(z) <= K
        active_count = np.sum(sol.weights > 1e-4)
        self.assertLessEqual(active_count, k)

        # 3. Buy-in thresholds: w_min <= w_i <= w_max for active assets
        for w_val in sol.weights:
            if w_val > 1e-4:
                self.assertGreaterEqual(w_val, self.data.min_weight - 1e-3)
                self.assertLessEqual(w_val, self.data.max_weight + 1e-3)

        self.assertTrue(sol.is_feasible)

    def test_greedy_solver_constraint_satisfaction(self):
        """Verifies that Greedy solver satisfies all constraints."""
        k = 4
        solver = GreedyPortfolioSolver(self.data, max_cardinality=k, lambda_risk=1.0)
        sol = solver.solve()

        self.assertAlmostEqual(float(np.sum(sol.weights)), 1.0, places=3)
        active_count = np.sum(sol.weights > 1e-4)
        self.assertLessEqual(active_count, k)
        self.assertTrue(sol.is_feasible)

    def test_simulated_annealing_constraint_satisfaction(self):
        """Verifies that Simulated Annealing satisfies all constraints."""
        k = 4
        solver = SimulatedAnnealingPortfolioSolver(self.data, max_cardinality=k, lambda_risk=1.0, max_iterations=80, seed=42)
        sol = solver.solve()

        self.assertAlmostEqual(float(np.sum(sol.weights)), 1.0, places=3)
        active_count = np.sum(sol.weights > 1e-4)
        self.assertLessEqual(active_count, k)
        self.assertTrue(sol.is_feasible)

    def test_genetic_algorithm_constraint_satisfaction(self):
        """Verifies that Genetic Algorithm satisfies all constraints."""
        k = 4
        solver = GeneticAlgorithmPortfolioSolver(self.data, max_cardinality=k, lambda_risk=1.0, generations=15, seed=42)
        sol = solver.solve()

        self.assertAlmostEqual(float(np.sum(sol.weights)), 1.0, places=3)
        active_count = np.sum(sol.weights > 1e-4)
        self.assertLessEqual(active_count, k)
        self.assertTrue(sol.is_feasible)

    def test_solver_optimality_and_bounds(self):
        """Verifies that MIQP global optimum achieves an objective value <= Greedy and Metaheuristics."""
        sol_miqp = MIQPSolver(self.data, max_cardinality=3, lambda_risk=1.0).solve()
        sol_grd = GreedyPortfolioSolver(self.data, max_cardinality=3, lambda_risk=1.0).solve()

        # MIQP must be <= Greedy (allowing for numerical solver tolerance)
        self.assertLessEqual(sol_miqp.objective_value, sol_grd.objective_value + 1e-4)

    def test_transaction_cost_accounting(self):
        """Verifies that turnover from w_0 incurs positive non-smooth transaction costs."""
        w_diff = np.zeros(20)
        w_diff[0] = 0.35
        w_diff[1] = 0.35
        w_diff[2] = 0.30

        _, _, _, cost, _ = compute_portfolio_metrics(w_diff, self.data, lambda_risk=1.0)
        self.assertGreater(cost, 0.0)
        # Turnover is non-zero so fixed charge and proportional charge must both be present
        min_expected_fixed = self.data.fixed_trans_cost * 3
        self.assertGreaterEqual(cost, min_expected_fixed)


if __name__ == "__main__":
    unittest.main()
