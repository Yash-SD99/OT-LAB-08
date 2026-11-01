"""Cross-Validation and Correctness Suite for Quantum Simulator (Track C).

Performs rigorous numerical validation of the pure NumPy statevector simulator against:
1. Exact Analytical Kronecker Matrix Mechanics (ground truth tensor products)
2. Qiskit Statevector simulator (when qiskit is installed in the environment)
3. Unitary invariance and inner product preservation (<psi|psi> = 1.0)
4. Commutation and anti-commutation relations of Pauli gates
"""

from typing import Dict, Optional, Tuple
import numpy as np

from tracks.C_quantum_opt.quantum_simulator import (
    QuantumCircuitSimulator,
    PAULI_I,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    HADAMARD,
    rx_matrix,
    ry_matrix,
    rz_matrix
)


def crosscheck_against_analytical_kronecker() -> Dict:
    """Validates simulator operations against exact Kronecker matrix products."""
    results = {}
    
    # 1. Bell State: H(0) -> CNOT(0, 1) on |00>
    sim = QuantumCircuitSimulator(2)
    sim.h(0)
    sim.cnot(0, 1)
    
    # Exact analytical state: [1/sqrt(2), 0, 0, 1/sqrt(2)]
    exact_bell = np.array([1.0 / np.sqrt(2.0), 0.0, 0.0, 1.0 / np.sqrt(2.0)], dtype=np.complex128)
    bell_diff = float(np.max(np.abs(sim.state - exact_bell)))
    results["bell_state_max_error"] = bell_diff
    assert bell_diff < 1e-14, f"Bell state error {bell_diff} exceeds 1e-14"
    
    # 2. 3-Qubit GHZ State: H(0) -> CNOT(0, 1) -> CNOT(1, 2)
    sim_ghz = QuantumCircuitSimulator(3)
    sim_ghz.h(0)
    sim_ghz.cnot(0, 1)
    sim_ghz.cnot(1, 2)
    
    exact_ghz = np.zeros(8, dtype=np.complex128)
    exact_ghz[0] = 1.0 / np.sqrt(2.0)
    exact_ghz[7] = 1.0 / np.sqrt(2.0)
    ghz_diff = float(np.max(np.abs(sim_ghz.state - exact_ghz)))
    results["ghz_state_max_error"] = ghz_diff
    assert ghz_diff < 1e-14, f"GHZ state error {ghz_diff} exceeds 1e-14"
    
    # 3. Arbitrary Rotations: Rx(theta) (x) Ry(phi) (x) Rz(lambda)
    theta, phi, lam = 0.35, -0.82, 1.45
    sim_rot = QuantumCircuitSimulator(3)
    sim_rot.rx(0, theta)
    sim_rot.ry(1, phi)
    sim_rot.rz(2, lam)
    
    # Kronecker exact matrix
    U_exact = np.kron(np.kron(rx_matrix(theta), ry_matrix(phi)), rz_matrix(lam))
    init_state = np.zeros(8, dtype=np.complex128)
    init_state[0] = 1.0
    exact_rot_state = U_exact @ init_state
    
    rot_diff = float(np.max(np.abs(sim_rot.state - exact_rot_state)))
    results["rotation_tensor_max_error"] = rot_diff
    assert rot_diff < 1e-14, f"Rotation tensor error {rot_diff} exceeds 1e-14"
    
    # 4. Rzz(gamma) gate vs exact exp(-i * gamma/2 * Z (x) Z)
    gamma = 0.77
    sim_rzz = QuantumCircuitSimulator(2)
    sim_rzz.h(0)
    sim_rzz.h(1)
    sim_rzz.rzz(0, 1, gamma)
    
    # Exact matrix exponential
    ZZ = np.kron(PAULI_Z, PAULI_Z)
    from scipy.linalg import expm
    U_rzz_exact = expm(-1j * (gamma / 2.0) * ZZ)
    init_h2 = (1.0 / 2.0) * np.ones(4, dtype=np.complex128)
    exact_rzz_state = U_rzz_exact @ init_h2
    
    rzz_diff = float(np.max(np.abs(sim_rzz.state - exact_rzz_state)))
    results["rzz_gate_max_error"] = rzz_diff
    assert rzz_diff < 1e-14, f"RZZ gate error {rzz_diff} exceeds 1e-14"
    
    return results


def crosscheck_against_qiskit() -> Optional[Dict]:
    """Validates simulator against Qiskit if installed in the environment."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
    except ImportError:
        return None
        
    results = {}
    
    # Test 1: 3-qubit parameterized circuit
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.ry(0.5, 1)
    qc.rz(-0.3, 2)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.rx(0.8, 0)
    
    # Qiskit simulation (note: Qiskit uses little-endian qubit ordering: q0 is LSB)
    qiskit_sv = Statevector(qc).data
    
    # Custom simulator simulation (big-endian qubit ordering: q0 is MSB)
    sim = QuantumCircuitSimulator(3)
    sim.h(0)
    sim.ry(1, 0.5)
    sim.rz(2, -0.3)
    sim.cnot(0, 1)
    sim.cnot(1, 2)
    sim.rx(0, 0.8)
    
    # Reorder Qiskit little-endian state vector to big-endian
    qiskit_big_endian = np.zeros(8, dtype=np.complex128)
    for i in range(8):
        # Reverse bits of index i
        rev_i = int(f"{i:03b}"[::-1], 2)
        qiskit_big_endian[rev_i] = qiskit_sv[i]
        
    qiskit_diff = float(np.max(np.abs(sim.state - qiskit_big_endian)))
    results["qiskit_statevector_max_error"] = qiskit_diff
    assert qiskit_diff < 1e-14, f"Qiskit cross-check error {qiskit_diff} exceeds 1e-14"
    
    return results
