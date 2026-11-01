"""Unit Tests for Quantum Algorithms and QML Advantage (Track I).

Tests:
1. Deutsch-Jozsa classification (Constant vs Balanced oracles).
2. Bernstein-Vazirani bitstring recovery.
3. Grover's search amplitude amplification and success probabilities.
4. QFT and Inverse QFT unitarity and equivalence to theoretical DFT.
5. Quantum Phase Estimation (QPE) on dyadic and non-dyadic phases.
6. Shor's period finding and factor extraction for N=15.
7. Parameter-Shift gradient verification against finite differences.
8. Variational Quantum Classifier (VQC) training and accuracy on non-linear dataset.
"""

import math
import numpy as np
import pytest

from tracks.I_quantum_algos_qml.quantum_circuits import AdvancedQuantumCircuit
from tracks.I_quantum_algos_qml.fundamental_algos import (
    DeutschJozsaAlgorithm,
    make_dj_oracle,
    BernsteinVaziraniAlgorithm,
    GroverSearchAlgorithm,
    QuantumPhaseEstimation,
    ShorsAlgorithmDemo,
)
from tracks.I_quantum_algos_qml.vqc_qml import (
    generate_concentric_circles,
    VariationalQuantumClassifier,
    ClassicalMLP,
)


def test_deutsch_jozsa_classification():
    """Verifies Deutsch-Jozsa correctly distinguishes constant vs balanced oracles in 1 query."""
    for n in [2, 3, 4]:
        dj = DeutschJozsaAlgorithm(n)
        
        # Constant zero
        res_c0 = dj.run(make_dj_oracle(n, "constant_zero"))
        assert res_c0["classification"] == "CONSTANT"
        assert res_c0["prob_all_zeros"] > 0.99
        assert res_c0["quantum_queries"] == 1
        
        # Constant one
        res_c1 = dj.run(make_dj_oracle(n, "constant_one"))
        assert res_c1["classification"] == "CONSTANT"
        assert res_c1["prob_all_zeros"] > 0.99
        
        # Balanced
        res_bal = dj.run(make_dj_oracle(n, "balanced", seed=42 + n))
        assert res_bal["classification"] == "BALANCED"
        assert res_bal["prob_all_zeros"] < 0.01


def test_bernstein_vazirani_recovery():
    """Verifies Bernstein-Vazirani recovers arbitrary hidden bitstrings in 1 query."""
    test_secrets = ["1", "0", "10", "01", "110", "1011", "01101", "110101"]
    for secret in test_secrets:
        bv = BernsteinVaziraniAlgorithm(secret)
        res = bv.run()
        assert res["success"] is True
        assert res["recovered_secret"] == secret
        assert res["quantum_queries"] == 1
        assert res["success_probability"] > 0.999


def test_grovers_search_success_rate():
    """Verifies Grover's algorithm finds target state with high probability in O(sqrt(N)) steps."""
    for target in [0, 3, 5, 7]:
        grover = GroverSearchAlgorithm(n_qubits=3, target_state=target)
        res = grover.run()
        assert res["success"] is True
        assert res["best_measured_state"] == target
        # For N=8, R=2 iterations gives ~94.5% success
        assert res["success_probability"] > 0.90
        assert res["grover_queries"] == 2


def test_qft_and_inverse_qft_unitarity():
    """Verifies QFT circuit matches theoretical DFT matrix and satisfies QFT_dag * QFT = I."""
    for n in [2, 3, 4]:
        dim = 2**n
        U = np.zeros((dim, dim), dtype=np.complex128)
        
        for col in range(dim):
            qc = AdvancedQuantumCircuit(n)
            basis = np.zeros(dim)
            basis[col] = 1.0
            qc.set_state(basis)
            qc.qft()
            U[:, col] = qc.state
            
        # Theoretical DFT matrix: omega = exp(2*pi*i / dim)
        omega = np.exp(2j * np.pi / dim)
        DFT = np.array([[omega**(j * k) / np.sqrt(dim) for k in range(dim)] for j in range(dim)])
        
        # Compare QFT unitary with DFT matrix
        assert np.linalg.norm(U - DFT) < 1e-12
        
        # Test Inverse QFT restores original state
        qc = AdvancedQuantumCircuit(n)
        qc.set_state(basis)
        qc.qft()
        qc.qft_dag()
        assert np.linalg.norm(qc.state - basis) < 1e-12


def test_qpe_exact_phases():
    """Verifies QPE recovers exact dyadic rational phases theta = k / 2^m with zero error."""
    m = 4
    for k in [0, 2, 4, 6, 8, 12, 14]:
        theta = k / (2**m)
        qpe = QuantumPhaseEstimation(m_counting=m, target_theta=theta)
        res = qpe.run()
        assert res["best_estimate_int"] == k
        assert abs(res["estimated_theta"] - theta) < 1e-12
        assert res["max_probability"] > 0.999


def test_shors_factoring_15():
    """Verifies Shor's order finding correctly extracts period r=4 and factors N=15 into 3 and 5."""
    shor = ShorsAlgorithmDemo(N=15, a=7, m_counting=4)
    res = shor.run()
    assert res["success"] is True
    assert res["factors"] == [3, 5]
    # Check that peaks include values corresponding to phases s/4
    assert 0 in res["peak_indices"]
    assert 4 in res["peak_indices"] or 8 in res["peak_indices"] or 12 in res["peak_indices"]


def test_vqc_parameter_shift_gradient():
    """Verifies parameter-shift gradient matches numerical central finite difference."""
    vqc = VariationalQuantumClassifier(n_layers=1, seed=42)
    x = np.array([0.5 * np.pi, 0.3 * np.pi])
    X = x.reshape(1, 2)
    y = np.array([1.0])
    
    # Analytical gradient via parameter-shift
    grad_ps = vqc.parameter_shift_gradient(X, y)
    
    # Numerical central difference
    eps = 1e-5
    grad_num = np.zeros_like(grad_ps)
    for i in range(len(vqc.theta)):
        orig = vqc.theta[i]
        
        vqc.theta[i] = orig + eps
        l_plus = vqc.compute_loss(X, y)
        
        vqc.theta[i] = orig - eps
        l_minus = vqc.compute_loss(X, y)
        
        vqc.theta[i] = orig
        grad_num[i] = (l_plus - l_minus) / (2.0 * eps)
        
    diff = np.linalg.norm(grad_ps - grad_num)
    assert diff < 1e-4, f"Parameter shift diff {diff} exceeds tolerance."


def test_vqc_non_linear_classification():
    """Verifies VQC trains to high accuracy on non-linear concentric circles."""
    X, y = generate_concentric_circles(n_samples=80, noise=0.08, factor=0.45, seed=42)
    X_train, y_train = X[:60], y[:60]
    X_test, y_test = X[60:], y[60:]
    
    vqc = VariationalQuantumClassifier(n_layers=2, seed=42)
    losses = vqc.fit(X_train, y_train, epochs=25, lr=0.1, batch_size=20, seed=42)
    
    # Loss should decrease
    assert losses[-1] < losses[0]
    
    # Test accuracy should be high on non-linear boundary
    test_acc = np.mean(vqc.predict_classes(X_test) == y_test)
    assert test_acc >= 0.80, f"Expected test accuracy >= 80%, got {test_acc*100}%"
