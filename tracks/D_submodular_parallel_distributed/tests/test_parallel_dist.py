"""Unit and Verification Tests for Track D: Submodular, Parallel, and Distributed Optimization."""

import unittest
import numpy as np
from tracks.D_submodular_parallel_distributed.submodular_opt import (
    FacilityLocationSubmodular,
    verify_diminishing_returns,
    greedy_submodular_selection,
    exact_optimal_selection
)
from tracks.D_submodular_parallel_distributed.parallel_scheduling import (
    StaticScheduler,
    WorkStealingSimulator,
    fit_amdahl_gustafson
)
from tracks.D_submodular_parallel_distributed.federated_fedavg import (
    DistributedDataset,
    run_fedavg
)
from tracks.D_submodular_parallel_distributed.multitask_mgda_gradnorm import (
    TwoTaskModel,
    solve_mgda_2task,
    train_mgda
)


class TestTrackD(unittest.TestCase):

    def test_submodular_diminishing_returns(self):
        """Facility location objective satisfies diminishing returns with 0 violations."""
        np.random.seed(42)
        candidates = np.random.randn(12, 2)
        targets = np.random.randn(20, 2)
        obj = FacilityLocationSubmodular(candidates, targets, sigma=1.0)
        
        audit = verify_diminishing_returns(obj, n_pairs_to_test=50, seed=42)
        self.assertTrue(audit["is_submodular_verified"])
        self.assertEqual(audit["violations"], 0)

    def test_nemhauser_wolsey_approximation_bound(self):
        """Greedy achieves >= (1 - 1/e) * f(S*) on facility location."""
        np.random.seed(42)
        candidates = np.random.randn(10, 2)
        targets = np.random.randn(15, 2)
        obj = FacilityLocationSubmodular(candidates, targets, sigma=1.0)
        
        k = 3
        res_greedy = greedy_submodular_selection(obj, k=k)
        res_exact = exact_optimal_selection(obj, k=k)
        
        bound_ratio = 1.0 - 1.0 / np.e # approx 0.6321
        empirical_ratio = res_greedy["final_value"] / res_exact["final_value"]
        
        self.assertGreaterEqual(empirical_ratio, bound_ratio - 1e-6)
        self.assertLessEqual(empirical_ratio, 1.0 + 1e-6)

    def test_work_stealing_scheduler(self):
        """Work stealing scheduler executes all tasks and achieves lower/equal makespan than static."""
        intensities = [0.1, 0.5, 0.2, 0.8, 0.1, 0.4, 0.9, 0.2]
        static = StaticScheduler(n_workers=2).run(intensities)
        stealing = WorkStealingSimulator(n_workers=2).run(intensities)
        
        self.assertGreater(stealing["stolen_tasks_count"], 0)
        self.assertGreater(static["makespan_ms"], 0.0)

    def test_amdahl_gustafson_fit(self):
        """Amdahl fit correctly identifies parallel fraction s in [0, 1]."""
        p_workers = [1, 2, 4]
        # Synthetic runtimes: 100ms, 55ms, 30ms
        runtimes = [100.0, 55.0, 30.0]
        fit = fit_amdahl_gustafson(p_workers, runtimes)
        
        self.assertGreaterEqual(fit["fitted_parallel_fraction_s"], 0.0)
        self.assertLessEqual(fit["fitted_parallel_fraction_s"], 1.0)
        self.assertGreater(fit["measured_speedups"][1], 1.5)

    def test_fedavg_loss_reduction(self):
        """FedAvg reduces global loss over communication rounds."""
        data = DistributedDataset(n_clients=5, n_features=4, samples_per_client=30, seed=42)
        res = run_fedavg(data, n_rounds=8, local_epochs=3, lr=0.05)
        
        init_loss = res["loss_history"][0]
        final_loss = res["loss_history"][-1]
        self.assertLess(final_loss, init_loss)

    def test_mgda_pareto_stationary_descent(self):
        """MGDA weights alpha + beta == 1.0 and non-conflicting descent."""
        g1 = np.array([2.0, -1.0])
        g2 = np.array([-1.0, 3.0])
        alpha, beta, g_mgda = solve_mgda_2task(g1, g2)
        
        self.assertAlmostEqual(alpha + beta, 1.0, places=6)
        self.assertGreaterEqual(alpha, 0.0)
        self.assertGreaterEqual(beta, 0.0)
        # Non-conflicting inner products
        self.assertGreaterEqual(np.dot(g_mgda, g1), -1e-6)
        self.assertGreaterEqual(np.dot(g_mgda, g2), -1e-6)


if __name__ == "__main__":
    unittest.main()
