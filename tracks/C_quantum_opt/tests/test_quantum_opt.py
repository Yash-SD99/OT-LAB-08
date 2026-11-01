"""Unit and Verification Tests for Track C: Quantum Optimization."""

import unittest
import numpy as np
from tracks.C_quantum_opt.quantum_simulator import (
    QuantumCircuitSimulator,
    HADAMARD,
    PAULI_X,
    PAULI_Z
)
from tracks.C_quantum_opt.qaoa_maxcut import QAOAMaxCut
from tracks.C_quantum_opt.vqe_ising import VQEIsing
from tracks.C_quantum_opt.pqc_qnn import (
    ParameterizedCircuit,
    QuantumNeuralNetworkClassifier,
    optimize_equal_budget
)
from tracks.C_quantum_opt.noise_and_barren import analyze_shot_noise_scaling
from tracks.C_quantum_opt.quantum_annealing import MinorEmbeddingDemo
from tracks.C_quantum_opt.simulator_crosscheck import (
    crosscheck_against_analytical_kronecker,
    crosscheck_against_qiskit
)


class TestQuantumOptimization(unittest.TestCase):

    def test_bell_state_generation(self):
        """Verifies Bell state creation (|00> + |11>) / sqrt(2) via H + CNOT."""
        sim = QuantumCircuitSimulator(2)
        sim.h(0)
        sim.cnot(0, 1)
        
        probs = sim.probabilities()
        np.testing.assert_allclose(probs, [0.5, 0.0, 0.0, 0.5], atol=1e-12)

    def test_parameter_shift_rule_tight_tolerance(self):
        """Parameter-shift rule must match numerical gradient to tight tolerance (< 1e-8)."""
        pqc = ParameterizedCircuit(n_qubits=2, n_layers=1)
        np.random.seed(42)
        theta = np.random.uniform(-np.pi, np.pi, size=pqc.num_params)
        
        # Analytical parameter-shift gradient
        grad_ps = pqc.parameter_shift_gradient(theta)
        
        # Central finite difference gradient with optimal eps=1e-7
        eps = 1e-7
        grad_fd = np.zeros_like(theta)
        for i in range(len(theta)):
            t_plus = theta.copy()
            t_plus[i] += eps
            t_minus = theta.copy()
            t_minus[i] -= eps
            grad_fd[i] = (pqc.cost(t_plus) - pqc.cost(t_minus)) / (2.0 * eps)
            
        # Assert maximum absolute error is strictly less than 1e-8
        max_err = float(np.max(np.abs(grad_ps - grad_fd)))
        self.assertLess(max_err, 1e-8)

    def test_vqe_ising_ground_state(self):
        """VQE finds ground state energy of 3-qubit Ising model within 3% of exact diagonalization."""
        vqe = VQEIsing(n_qubits=3, J=1.0, h=0.5, layers=2)
        res = vqe.optimize(max_iters=40, seed=42)
        
        # Assert VQE energy is greater than or equal to exact ground state (variational principle)
        self.assertGreaterEqual(res["vqe_ground_energy"], res["exact_ground_energy"] - 1e-6)
        # Assert relative error < 3%
        self.assertLess(res["relative_error"], 0.03)

    def test_qaoa_maxcut_benchmark_graph(self):
        """QAOA achieves approximation ratio >= 0.70 on the Track B 4-node benchmark graph (Max-Cut = 14.0)."""
        W = np.array([
            [0.0, 3.0, 0.0, 2.0],
            [3.0, 0.0, 4.0, 0.0],
            [0.0, 4.0, 0.0, 5.0],
            [2.0, 0.0, 5.0, 0.0]
        ], dtype=float)
        
        qaoa = QAOAMaxCut(W, p_depth=2)
        res = qaoa.optimize(max_iters=35, seed=42)
        
        self.assertGreater(res["approximation_ratio"], 0.70)
        self.assertEqual(res["max_cut_ground_truth"], 14.0)
        # Optimal bitstrings 0101 and 1010 have non-zero probability
        self.assertGreater(res["prob_optimal_ground_state"], 0.20)

    def test_shot_noise_variance_scaling(self):
        """Variance of multinomial expectation values must decrease as shots increase."""
        res = analyze_shot_noise_scaling(n_qubits=2, shots_list=[100, 1000], n_trials=15)
        var_100 = res["measured_variances"][0]
        var_1000 = res["measured_variances"][1]
        self.assertGreater(var_100, var_1000)

    def test_qnn_classifier_training(self):
        """Quantum Neural Network classifier successfully trains on 2D dataset."""
        X = np.array([
            [0.1, 0.1], [0.2, -0.1], [-0.1, 0.2], [-0.2, -0.2],
            [1.5, 1.5], [-1.5, 1.4], [1.4, -1.6], [-1.6, -1.5]
        ], dtype=float)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
        
        qnn = QuantumNeuralNetworkClassifier(n_layers=2)
        res = qnn.fit(X, y, epochs=25, lr=0.5, seed=42)
        self.assertGreaterEqual(res["final_accuracy"], 0.75)
        self.assertLess(res["final_loss"], res["initial_loss"])

    def test_kronecker_exact_crosscheck(self):
        """Simulator statevectors must match exact Kronecker tensor algebra to machine precision (< 1e-14)."""
        res = crosscheck_against_analytical_kronecker()
        for k, v in res.items():
            self.assertLess(v, 1e-14, f"Error in {k}: {v}")

    def test_quantum_annealing_embedding(self):
        """MinorEmbeddingDemo demonstrates chain breaks when J_chain is weak and identifies optimal coupling."""
        demo = MinorEmbeddingDemo(seed=42)
        res = demo.evaluate_embedding_limits(j_chain_values=[0.1, 1.5], n_trials=10)
        # Weak J_chain=0.1 should have higher chain break rate than J_chain=1.5
        self.assertGreater(res["chain_break_rates"][0], res["chain_break_rates"][1])


if __name__ == "__main__":
    unittest.main()
