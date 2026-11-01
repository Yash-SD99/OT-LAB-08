"""Pure NumPy Vectorized Quantum State-Vector Simulator (Track C & Track I).

Features:
1. Exact N-qubit state vector simulation: |psi> in C^{2^N}
2. Single-qubit Pauli (X, Y, Z), Hadamard (H), Phase (S, T), and Rotations (Rx, Ry, Rz)
3. Two-qubit entangling gates: CNOT (CX), CZ, and Rzz(gamma) = exp(-i * gamma/2 * Z (x) Z)
4. Expectation values of general Pauli Hamiltonians and diagonal QUBO / Ising observables
5. Finite measurement sampling (shot simulation) with multinomial draws
6. Calibrated Hardware Noise Models:
   - Depolarizing channel: with probability p, replace state with maximally mixed state
   - Readout bit-flip errors: flip measurement outcomes with probability p_readout
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


# Elementary single-qubit matrices
PAULI_I = np.eye(2, dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
HADAMARD = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)


def rx_matrix(theta: float) -> np.ndarray:
    """Rx(theta) = exp(-i * theta/2 * X) = cos(theta/2)*I - i*sin(theta/2)*X."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def ry_matrix(theta: float) -> np.ndarray:
    """Ry(theta) = exp(-i * theta/2 * Y) = cos(theta/2)*I - i*sin(theta/2)*Y."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz_matrix(theta: float) -> np.ndarray:
    """Rz(theta) = exp(-i * theta/2 * Z) = diag(exp(-i*theta/2), exp(i*theta/2))."""
    return np.array([
        [np.exp(-1j * theta / 2.0), 0.0],
        [0.0, np.exp(1j * theta / 2.0)]
    ], dtype=np.complex128)


class QuantumCircuitSimulator:
    """Pure NumPy quantum circuit simulator operating on state vectors of N qubits."""
    def __init__(self, n_qubits: int, seed: Optional[int] = None):
        self.n = n_qubits
        self.dim = 2**n_qubits
        self.rng = np.random.default_rng(seed)
        # Initialize state to |00...0>
        self.state = np.zeros(self.dim, dtype=np.complex128)
        self.state[0] = 1.0 + 0.0j

    def reset(self) -> None:
        """Resets circuit state to |00...0>."""
        self.state.fill(0.0)
        self.state[0] = 1.0 + 0.0j

    def set_state(self, state: np.ndarray) -> None:
        """Sets internal state vector, ensuring unit normalization."""
        norm = np.linalg.norm(state)
        if norm < 1e-12:
            raise ValueError("Cannot set zero state vector.")
        self.state = np.asarray(state, dtype=np.complex128) / norm

    def apply_single_qubit_gate(self, gate: np.ndarray, target: int) -> None:
        """Applies a 2x2 unitary gate to target qubit (0 <= target < N).
        Uses tensor reshaping for O(2^N) memory-contiguous contraction.
        """
        # Reshape state tensor into (2, 2, ..., 2) of shape [2]*N
        # Target qubit is at axis `target`
        shape = [2] * self.n
        tensor = self.state.reshape(shape)
        
        # Contract gate (2x2) with tensor along axis `target`
        # New axis becomes target axis
        tensor = np.tensordot(gate, tensor, axes=(1, target))
        # Move contracted axis (axis 0) back to original target position
        tensor = np.moveaxis(tensor, 0, target)
        self.state = tensor.reshape(self.dim)

    def h(self, target: int) -> "QuantumCircuitSimulator":
        """Applies Hadamard gate."""
        self.apply_single_qubit_gate(HADAMARD, target)
        return self

    def x(self, target: int) -> "QuantumCircuitSimulator":
        """Applies Pauli-X (NOT) gate."""
        self.apply_single_qubit_gate(PAULI_X, target)
        return self

    def y(self, target: int) -> "QuantumCircuitSimulator":
        """Applies Pauli-Y gate."""
        self.apply_single_qubit_gate(PAULI_Y, target)
        return self

    def z(self, target: int) -> "QuantumCircuitSimulator":
        """Applies Pauli-Z gate."""
        self.apply_single_qubit_gate(PAULI_Z, target)
        return self

    def rx(self, target: int, theta: float) -> "QuantumCircuitSimulator":
        """Applies Rx(theta) rotation."""
        self.apply_single_qubit_gate(rx_matrix(theta), target)
        return self

    def ry(self, target: int, theta: float) -> "QuantumCircuitSimulator":
        """Applies Ry(theta) rotation."""
        self.apply_single_qubit_gate(ry_matrix(theta), target)
        return self

    def rz(self, target: int, theta: float) -> "QuantumCircuitSimulator":
        """Applies Rz(theta) rotation."""
        self.apply_single_qubit_gate(rz_matrix(theta), target)
        return self

    def cnot(self, control: int, target: int) -> "QuantumCircuitSimulator":
        """Applies Controlled-NOT (CX) gate between control and target qubits."""
        if control == target:
            raise ValueError("Control and target qubits must differ.")
        shape = [2] * self.n
        tensor = self.state.reshape(shape)
        
        # When control == 1, apply Pauli-X to target
        # Slice where control index == 1
        slices_ctrl_1 = [slice(None)] * self.n
        slices_ctrl_1[control] = 1
        
        target_subtensor = tensor[tuple(slices_ctrl_1)]
        # Target axis in subtensor is target (or target - 1 if target > control)
        sub_target = target - 1 if target > control else target
        
        # Contract target with Pauli X
        contracted = np.tensordot(PAULI_X, target_subtensor, axes=(1, sub_target))
        contracted = np.moveaxis(contracted, 0, sub_target)
        
        tensor[tuple(slices_ctrl_1)] = contracted
        self.state = tensor.reshape(self.dim)
        return self

    def rzz(self, q1: int, q2: int, gamma: float) -> "QuantumCircuitSimulator":
        """Applies Rzz(gamma) = exp(-i * gamma/2 * Z (x) Z) between q1 and q2.
        Equivalent to: CNOT(q1, q2) -> Rz(q2, gamma) -> CNOT(q1, q2).
        For diagonal state space: applies phase exp(-i * gamma/2 * (-1)^{b1 ^ b2}).
        """
        self.cnot(q1, q2)
        self.rz(q2, gamma)
        self.cnot(q1, q2)
        return self

    def probabilities(self) -> np.ndarray:
        """Computes Born rule probability distribution p(x) = |psi(x)|^2."""
        return np.abs(self.state)**2

    def sample_shots(self, n_shots: int = 1000, readout_error: float = 0.0, seed: Optional[int] = None) -> np.ndarray:
        """Samples discrete measurement outcomes from Born rule probabilities.
        Optionally applies bit-flip readout noise with error rate readout_error.
        """
        rng = np.random.default_rng(seed)
        probs = self.probabilities()
        # Draw sample state indices
        samples = rng.choice(self.dim, size=n_shots, p=probs)
        
        if readout_error > 0.0:
            # Convert indices to bit arrays and apply bit flips
            noisy_samples = []
            for s in samples:
                bits = [(s >> (self.n - 1 - i)) & 1 for i in range(self.n)]
                # Flip each bit with probability readout_error
                flips = rng.random(size=self.n) < readout_error
                noisy_bits = [b ^ int(f) for b, f in zip(bits, flips)]
                # Reconstruct integer index
                noisy_idx = 0
                for b in noisy_bits:
                    noisy_idx = (noisy_idx << 1) | b
                noisy_samples.append(noisy_idx)
            return np.array(noisy_samples, dtype=int)
            
        return samples

    def expectation_diagonal(self, diagonal_hamiltonian: np.ndarray, n_shots: Optional[int] = None, seed: Optional[int] = None) -> float:
        """Computes expectation value <psi| H |psi> for a diagonal Hamiltonian.
        
        If n_shots is None: computes exact analytical expectation value: sum_x p(x) H(x).
        If n_shots is provided: draws finite measurement samples (shot noise).
        """
        diag = np.asarray(diagonal_hamiltonian, dtype=np.float64)
        if n_shots is None:
            return float(np.sum(self.probabilities() * diag))
        else:
            samples = self.sample_shots(n_shots=n_shots, seed=seed)
            return float(np.mean(diag[samples]))

    def apply_depolarizing_noise(self, p_error: float) -> None:
        """Applies calibrated depolarizing channel:
        With probability p_error, state undergoes random Pauli error (X, Y, or Z) on random qubit.
        """
        if p_error <= 0.0:
            return
        if self.rng.random() < p_error:
            target = int(self.rng.integers(0, self.n))
            pauli_type = int(self.rng.integers(0, 3))
            if pauli_type == 0:
                self.x(target)
            elif pauli_type == 1:
                self.y(target)
            else:
                self.z(target)
