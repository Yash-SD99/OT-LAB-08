"""Pure NumPy Quantum Circuit Simulator with Advanced Gate Operations (Track I).

Extends the vectorized state-vector simulator with:
- Single-qubit elementary and rotation gates (H, X, Y, Z, S, T, Rx, Ry, Rz)
- Controlled single-qubit gates (Controlled-Phase R_k, Controlled-U, CNOT, CZ)
- SWAP gates
- Multi-qubit phase oracles (inversion of arbitrary computational basis state |w>)
- Multi-qubit Grover diffusion operator (2|s><s| - I)
- Multi-qubit Quantum Fourier Transform (QFT) and Inverse QFT (QFT_dag)
- Expectation values and shot sampling
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


# Elementary single-qubit matrices
PAULI_I = np.eye(2, dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
HADAMARD = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
GATE_S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
GATE_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4.0)]], dtype=np.complex128)


def rx_matrix(theta: float) -> np.ndarray:
    """Rx(theta) = exp(-i * theta/2 * X)."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def ry_matrix(theta: float) -> np.ndarray:
    """Ry(theta) = exp(-i * theta/2 * Y)."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz_matrix(theta: float) -> np.ndarray:
    """Rz(theta) = exp(-i * theta/2 * Z)."""
    return np.array([
        [np.exp(-1j * theta / 2.0), 0.0],
        [0.0, np.exp(1j * theta / 2.0)]
    ], dtype=np.complex128)


def phase_shift_matrix(lam: float) -> np.ndarray:
    """Phase shift gate P(lam) = diag(1, exp(i * lam))."""
    return np.array([
        [1.0, 0.0],
        [0.0, np.exp(1j * lam)]
    ], dtype=np.complex128)


def rk_matrix(k: int) -> np.ndarray:
    """QFT Phase rotation gate R_k = diag(1, exp(2 * pi * i / 2^k))."""
    angle = 2.0 * np.pi / (2**k)
    return phase_shift_matrix(angle)


class AdvancedQuantumCircuit:
    """Vectorized statevector simulator for quantum algorithms and quantum ML."""
    
    def __init__(self, n_qubits: int, seed: Optional[int] = None):
        if n_qubits < 1 or n_qubits > 16:
            raise ValueError(f"Qubit count {n_qubits} out of range [1, 16] for statevector simulation.")
        self.n = n_qubits
        self.dim = 2**n_qubits
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(self.dim, dtype=np.complex128)
        self.state[0] = 1.0 + 0.0j

    def reset(self) -> "AdvancedQuantumCircuit":
        """Resets state to |00...0>."""
        self.state.fill(0.0)
        self.state[0] = 1.0 + 0.0j
        return self

    def set_state(self, state: np.ndarray) -> "AdvancedQuantumCircuit":
        """Sets internal state to specified vector, normalizing if necessary."""
        arr = np.asarray(state, dtype=np.complex128)
        if arr.shape != (self.dim,):
            raise ValueError(f"State shape {arr.shape} does not match dimension {self.dim}.")
        norm = np.linalg.norm(arr)
        if norm < 1e-12:
            raise ValueError("State vector has zero norm.")
        self.state = arr / norm
        return self

    def apply_single_qubit_gate(self, gate: np.ndarray, target: int) -> "AdvancedQuantumCircuit":
        """Applies single-qubit unitary gate (2x2) to target qubit."""
        if not (0 <= target < self.n):
            raise IndexError(f"Target qubit {target} out of range [0, {self.n-1}].")
        shape = [2] * self.n
        tensor = self.state.reshape(shape)
        tensor = np.tensordot(gate, tensor, axes=(1, target))
        tensor = np.moveaxis(tensor, 0, target)
        self.state = tensor.reshape(self.dim)
        return self

    def h(self, target: int) -> "AdvancedQuantumCircuit":
        """Hadamard gate."""
        return self.apply_single_qubit_gate(HADAMARD, target)

    def x(self, target: int) -> "AdvancedQuantumCircuit":
        """Pauli-X gate (NOT)."""
        return self.apply_single_qubit_gate(PAULI_X, target)

    def y(self, target: int) -> "AdvancedQuantumCircuit":
        """Pauli-Y gate."""
        return self.apply_single_qubit_gate(PAULI_Y, target)

    def z(self, target: int) -> "AdvancedQuantumCircuit":
        """Pauli-Z gate."""
        return self.apply_single_qubit_gate(PAULI_Z, target)

    def s(self, target: int) -> "AdvancedQuantumCircuit":
        """Phase gate S = diag(1, i)."""
        return self.apply_single_qubit_gate(GATE_S, target)

    def t(self, target: int) -> "AdvancedQuantumCircuit":
        """T gate = diag(1, exp(i*pi/4))."""
        return self.apply_single_qubit_gate(GATE_T, target)

    def rx(self, target: int, theta: float) -> "AdvancedQuantumCircuit":
        """Rx(theta) rotation gate."""
        return self.apply_single_qubit_gate(rx_matrix(theta), target)

    def ry(self, target: int, theta: float) -> "AdvancedQuantumCircuit":
        """Ry(theta) rotation gate."""
        return self.apply_single_qubit_gate(ry_matrix(theta), target)

    def rz(self, target: int, theta: float) -> "AdvancedQuantumCircuit":
        """Rz(theta) rotation gate."""
        return self.apply_single_qubit_gate(rz_matrix(theta), target)

    def phase(self, target: int, lam: float) -> "AdvancedQuantumCircuit":
        """Phase shift gate P(lam) = diag(1, exp(i*lam))."""
        return self.apply_single_qubit_gate(phase_shift_matrix(lam), target)

    def apply_controlled_gate(self, gate: np.ndarray, control: int, target: int) -> "AdvancedQuantumCircuit":
        """Applies arbitrary single-qubit gate to target conditioned on control qubit == 1."""
        if control == target:
            raise ValueError("Control and target qubits cannot be identical.")
        shape = [2] * self.n
        tensor = self.state.reshape(shape)

        slices_ctrl_1 = [slice(None)] * self.n
        slices_ctrl_1[control] = 1

        subtensor = tensor[tuple(slices_ctrl_1)]
        sub_target = target - 1 if target > control else target

        contracted = np.tensordot(gate, subtensor, axes=(1, sub_target))
        contracted = np.moveaxis(contracted, 0, sub_target)

        tensor[tuple(slices_ctrl_1)] = contracted
        self.state = tensor.reshape(self.dim)
        return self

    def cnot(self, control: int, target: int) -> "AdvancedQuantumCircuit":
        """Controlled-NOT (CX) gate."""
        return self.apply_controlled_gate(PAULI_X, control, target)

    def cz(self, control: int, target: int) -> "AdvancedQuantumCircuit":
        """Controlled-Z (CZ) gate."""
        return self.apply_controlled_gate(PAULI_Z, control, target)

    def crk(self, control: int, target: int, k: int) -> "AdvancedQuantumCircuit":
        """Controlled-R_k gate for QFT: applies R_k to target when control == 1."""
        return self.apply_controlled_gate(rk_matrix(k), control, target)

    def crk_dag(self, control: int, target: int, k: int) -> "AdvancedQuantumCircuit":
        r"""Controlled-R_k^\dagger gate for Inverse QFT: applies R_k^\dagger to target."""
        mat = np.conj(rk_matrix(k).T)
        return self.apply_controlled_gate(mat, control, target)

    def cphase(self, control: int, target: int, lam: float) -> "AdvancedQuantumCircuit":
        """Controlled phase shift gate: applies diag(1, exp(i*lam)) to target."""
        return self.apply_controlled_gate(phase_shift_matrix(lam), control, target)

    def swap(self, q1: int, q2: int) -> "AdvancedQuantumCircuit":
        """SWAP gate between q1 and q2 via 3 CNOT gates."""
        if q1 == q2:
            return self
        self.cnot(q1, q2)
        self.cnot(q2, q1)
        self.cnot(q1, q2)
        return self

    def rzz(self, q1: int, q2: int, gamma: float) -> "AdvancedQuantumCircuit":
        """Two-qubit Ising ZZ interaction: exp(-i * gamma/2 * Z (x) Z)."""
        self.cnot(q1, q2)
        self.rz(q2, gamma)
        self.cnot(q1, q2)
        return self

    def apply_phase_oracle(self, marked_state: int) -> "AdvancedQuantumCircuit":
        """Applies phase oracle O_w = I - 2|w><w|, inverting the phase of basis state |w>."""
        if not (0 <= marked_state < self.dim):
            raise IndexError(f"Marked state {marked_state} out of basis bounds [0, {self.dim-1}].")
        self.state[marked_state] *= -1.0
        return self

    def apply_grover_diffusion(self, qubits: Optional[List[int]] = None) -> "AdvancedQuantumCircuit":
        """Applies Grover diffusion operator D = 2|s><s| - I on specified qubits (default: all).
        D = H^{tensor n} (2|0><0| - I) H^{tensor n}.
        """
        active_qubits = list(range(self.n)) if qubits is None else qubits
        # 1. Apply H to all active qubits
        for q in active_qubits:
            self.h(q)
        
        # 2. Phase inversion of all-zeros state: 2|0><0| - I
        # Note: 2|0><0| - I = - (I - 2|0><0|)
        # In statevector representation, we can directly compute projection onto |0...0>
        # or equivalently: X on all -> multi-controlled Z -> X on all -> overall phase -1
        # For active_qubits == all qubits:
        if len(active_qubits) == self.n:
            # Reflection about average: |psi> -> 2 * mean(psi) - |psi>
            # Mathematically: D |psi> = 2 <s|psi> |s> - |psi>, where |s> = 1/sqrt(2^n) sum_x |x>
            # Since H^{tensor n} was applied: |s> transformed to |0>, so we reflect about |0>:
            # 2|0><0| - I applied to state: psi[0] -> psi[0], psi[k != 0] -> -psi[k]
            # Then apply H^{tensor n} again.
            self.state[1:] *= -1.0
        else:
            # Subsystem reflection
            mean_amp = np.mean(self.state)
            self.state = 2.0 * mean_amp - self.state
            
        # 3. Apply H to all active qubits
        for q in active_qubits:
            self.h(q)
        return self

    def qft(self, qubits: Optional[List[int]] = None) -> "AdvancedQuantumCircuit":
        """Applies Quantum Fourier Transform (QFT) on ordered list of qubits."""
        target_qubits = list(range(self.n)) if qubits is None else qubits
        m = len(target_qubits)
        for i in range(m):
            q_i = target_qubits[i]
            self.h(q_i)
            for j in range(i + 1, m):
                q_j = target_qubits[j]
                k = j - i + 1
                self.crk(q_j, q_i, k)
        # Reverse qubit order with SWAP gates
        for i in range(m // 2):
            self.swap(target_qubits[i], target_qubits[m - 1 - i])
        return self

    def qft_dag(self, qubits: Optional[List[int]] = None) -> "AdvancedQuantumCircuit":
        """Applies Inverse Quantum Fourier Transform (QFT^dagger)."""
        target_qubits = list(range(self.n)) if qubits is None else qubits
        m = len(target_qubits)
        # Reverse qubit order first
        for i in range(m // 2):
            self.swap(target_qubits[i], target_qubits[m - 1 - i])
        # Unwind rotations and Hadamards
        for i in reversed(range(m)):
            q_i = target_qubits[i]
            for j in reversed(range(i + 1, m)):
                q_j = target_qubits[j]
                k = j - i + 1
                self.crk_dag(q_j, q_i, k)
            self.h(q_i)
        return self

    def probabilities(self) -> np.ndarray:
        """Computes Born rule probability distribution p(x) = |psi(x)|^2."""
        return np.abs(self.state)**2

    def sample_shots(self, n_shots: int = 1000, seed: Optional[int] = None) -> np.ndarray:
        """Samples integer measurement outcomes from Born rule probabilities."""
        probs = self.probabilities()
        probs = np.clip(probs, 0.0, 1.0)
        probs /= np.sum(probs)
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        return rng.choice(self.dim, size=n_shots, p=probs)

    def expectation_z(self, target: int) -> float:
        """Computes exact expectation value <psi| Z_target |psi>."""
        if not (0 <= target < self.n):
            raise IndexError(f"Target qubit {target} out of range [0, {self.n-1}].")
        probs = self.probabilities()
        # In bitstring integer representation, qubit `target` corresponds to bit (self.n - 1 - target)
        # For bit value 0: eigenvalue +1; for bit value 1: eigenvalue -1
        bit_pos = self.n - 1 - target
        indices = np.arange(self.dim)
        bits = (indices >> bit_pos) & 1
        eigenvalues = 1.0 - 2.0 * bits  # 0 -> +1, 1 -> -1
        return float(np.sum(probs * eigenvalues))
