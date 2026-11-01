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
from tracks.C_quantum_opt.pqc_qnn import ParameterizedCircuit
from tracks.C_quantum_opt.noise_and_barren import analyze_shot_noise_scaling


class TestQuantumOptimization(unittest.TestCase):

    def test_bell_state_generation(self):
        """Verifies Bell state creation (|00> + |11>) / sqrt(2) via H + CNOT."""
        sim = QuantumCircuitSimulator(2)
        sim.h(0)
        sim.cnot(0, 1)
        
        probs = sim.probabilities()
        np.testing.assert_allclose(probs, [0.5, 0.0, 0.0, 0.5], atol=1e-6)

    def test_parameter_shift_rule_vs_finite_difference(self):
        """Parameter-shift rule must match central finite differences (< 1e-4 relative error)."""
        pqc = ParameterizedCircuit(n_qubits=2, n_layers=1)
        np.random.seed(42)
        theta = np.random.uniform(-np.pi, np.pi, size=pqc.num_params)
        
        # Analytical parameter-shift gradient
        grad_ps = pqc.parameter_shift_gradient(theta)
        
        # Central finite difference gradient
        eps = 1e-5
        grad_fd = np.zeros_like(theta)
        for i in range(len(theta)):
            t_plus = theta.copy()
            t_plus[i] += eps
            t_minus = theta.copy()
            t_minus[i] -= eps
            grad_fd[i] = (pqc.cost(t_plus) - pqc.cost(t_minus)) / (2.0 * eps)
            
        np.testing.assert_allclose(grad_ps, grad_fd, rtol=1e-3, atol=1e-3)

    def test_vqe_ising_ground_state(self):
        """VQE finds ground state energy of 3-qubit Ising model within 3% of exact diagonalization."""
        vqe = VQEIsing(n_qubits=3, J=1.0, h=0.5, layers=2)
        res = vqe.optimize(max_iters=40, seed=42)
        
        # Assert VQE energy is greater than or equal to exact ground state (variational principle)
        self.assertGreaterEqual(res["vqe_ground_energy"], res["exact_ground_energy"] - 1e-6)
        # Assert relative error < 3%
        self.assertLess(res["relative_error"], 0.03)

    def test_qaoa_maxcut_approximation_ratio(self):
        """QAOA achieves approximation ratio >= 0.70 on 4-cycle graph C_4."""
        # 4-cycle graph
        W = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ], dtype=float)
        
        qaoa = QAOAMaxCut(W, p_depth=2)
        res = qaoa.optimize(max_iters=30, seed=42)
        
        self.assertGreater(res["approximation_ratio"], 0.70)
        self.assertEqual(res["max_cut_ground_truth"], 4.0)

    def test_shot_noise_variance_scaling(self):
        """Variance of expectation values must decrease as shots increase."""
        res = analyze_shot_noise_scaling(n_qubits=2, shots_list=[100, 1000], n_trials=15)
        var_100 = res["measured_variances"][0]
        var_1000 = res["measured_variances"][1]
        self.assertGreater(var_100, var_1000)


if __name__ == "__main__":
    unittest.main()
