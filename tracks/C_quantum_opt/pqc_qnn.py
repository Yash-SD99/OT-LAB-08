"""Parameterized Quantum Circuits & Quantum Gradient Methods (Track C).

Implements:
1. Exact Parameter-Shift Rule: d<H>/d theta_k = (<H>_{theta + pi/2} - <H>_{theta - pi/2}) / 2
2. SPSA (Simultaneous Perturbation Stochastic Approximation) requiring only 2 evaluations
3. Quantum Natural Gradient (QNG) using the Fubini-Study Metric Tensor g(theta)
4. Parameterized Quantum Circuit (PQC) Ansatz
5. Quantum Neural Network (QNN) Binary Classifier with parameter-shift backpropagation
6. Equal-Budget Optimizer Benchmarking Suite across 10+ random seeds
"""

import time
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from tracks.C_quantum_opt.quantum_simulator import (
    QuantumCircuitSimulator,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    PAULI_I
)


class ParameterizedCircuit:
    """N-qubit Parameterized Quantum Circuit with single-qubit rotations and entangling gates."""
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        self.n = n_qubits
        self.n_layers = n_layers
        self.dim = 2**n_qubits
        # Each layer has: Ry and Rz on all qubits -> 2 * n_qubits per layer
        # plus final Ry layer -> 2 * n_qubits * n_layers + n_qubits
        self.num_params = 2 * self.n * self.n_layers + self.n
        
        # Simple observable: sum of Z on all qubits: H = sum_i Z_i
        self.H_diag = np.zeros(self.dim, dtype=np.float64)
        for idx in range(self.dim):
            # Eigenvalue of Z_i is (-1)^bit
            val = 0.0
            for i in range(self.n):
                bit = (idx >> (self.n - 1 - i)) & 1
                val += 1.0 if bit == 0 else -1.0
            self.H_diag[idx] = val

    def forward(self, params: np.ndarray) -> QuantumCircuitSimulator:
        """Executes circuit for parameters theta."""
        sim = QuantumCircuitSimulator(self.n)
        p_idx = 0
        
        for l in range(self.n_layers):
            # Ry rotations
            for q in range(self.n):
                sim.ry(q, float(params[p_idx]))
                p_idx += 1
            # Rz rotations
            for q in range(self.n):
                sim.rz(q, float(params[p_idx]))
                p_idx += 1
            # Entangling CNOT ring
            for q in range(self.n):
                sim.cnot(q, (q + 1) % self.n)
                
        # Final Ry layer
        for q in range(self.n):
            sim.ry(q, float(params[p_idx]))
            p_idx += 1
            
        return sim

    def cost(self, params: np.ndarray, n_shots: Optional[int] = None, seed: Optional[int] = None) -> float:
        """Evaluates expectation value <psi(params)| H |psi(params)>."""
        sim = self.forward(params)
        return sim.expectation_diagonal(self.H_diag, n_shots=n_shots, seed=seed)

    def parameter_shift_gradient(
        self,
        params: np.ndarray,
        n_shots: Optional[int] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Computes exact analytical gradient via the Parameter-Shift Rule:
        d<H>/d theta_k = (<H>(theta + pi/2 * e_k) - <H>(theta - pi/2 * e_k)) / 2.
        Requires 2 * D circuit evaluations.
        """
        grads = np.zeros(self.num_params, dtype=np.float64)
        shift = np.pi / 2.0
        
        for k in range(self.num_params):
            p_plus = params.copy()
            p_plus[k] += shift
            e_plus = self.cost(p_plus, n_shots=n_shots, seed=seed)
            
            p_minus = params.copy()
            p_minus[k] -= shift
            e_minus = self.cost(p_minus, n_shots=n_shots, seed=seed)
            
            grads[k] = (e_plus - e_minus) / 2.0
            
        return grads

    def spsa_gradient(
        self,
        params: np.ndarray,
        c_k: float = 0.1,
        n_shots: Optional[int] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Simultaneous Perturbation Stochastic Approximation (SPSA).
        Estimates gradient across all D dimensions using only 2 function evaluations:
        g_k = (<H>(theta + c*Delta) - <H>(theta - c*Delta)) / (2 * c * Delta_k)
        where Delta_k in {-1, +1} are Rademacher random perturbations.
        """
        rng = np.random.default_rng(seed)
        delta = rng.choice([-1.0, 1.0], size=self.num_params)
        
        p_plus = params + c_k * delta
        p_minus = params - c_k * delta
        
        e_plus = self.cost(p_plus, n_shots=n_shots, seed=seed)
        e_minus = self.cost(p_minus, n_shots=n_shots, seed=seed)
        
        return (e_plus - e_minus) / (2.0 * c_k * delta)

    def fubini_study_metric_tensor(self, params: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """Computes the Fubini-Study metric tensor g_ij(theta) for Quantum Natural Gradient.
        
        g_ij = Re[ <d_i psi | d_j psi> - <d_i psi | psi> <psi | d_j psi> ]
        Approximated via state vector differences.
        """
        sim_0 = self.forward(params)
        psi_0 = sim_0.state
        
        d_psi = np.zeros((self.num_params, self.dim), dtype=np.complex128)
        for i in range(self.num_params):
            p_step = params.copy()
            p_step[i] += eps
            sim_i = self.forward(p_step)
            d_psi[i] = (sim_i.state - psi_0) / eps
            
        inner_d_psi_0 = d_psi @ np.conj(psi_0) # shape (D,)
        gram_matrix = d_psi @ np.conj(d_psi.T) # shape (D, D)
        outer_prod = np.outer(inner_d_psi_0, np.conj(inner_d_psi_0))
        
        metric = np.real(gram_matrix - outer_prod)
        metric = 0.5 * (metric + metric.T)
        return metric

    def quantum_natural_gradient(self, params: np.ndarray, damping: float = 1e-3) -> np.ndarray:
        """Quantum Natural Gradient step:
        step = (g(theta) + lambda * I)^{-1} * grad_{param_shift}
        """
        grad = self.parameter_shift_gradient(params)
        g_tensor = self.fubini_study_metric_tensor(params)
        damped_g = g_tensor + damping * np.eye(self.num_params)
        natural_grad = np.linalg.solve(damped_g, grad)
        return natural_grad


def optimize_equal_budget(
    pqc: ParameterizedCircuit,
    method: str,
    budget: int = 300,
    lr: float = 0.1,
    initial_params: Optional[np.ndarray] = None,
    seed: int = 42
) -> Dict:
    """Optimizes PQC parameters under a strictly enforced circuit-evaluation budget."""
    if initial_params is None:
        rng = np.random.default_rng(seed)
        params = rng.uniform(-np.pi, np.pi, size=pqc.num_params)
    else:
        params = initial_params.copy()
        
    evaluations_used = 0
    history_evals = []
    history_costs = []
    
    # Initial evaluation
    current_cost = pqc.cost(params)
    evaluations_used += 1
    history_evals.append(evaluations_used)
    history_costs.append(float(current_cost))
    
    step = 0
    while evaluations_used < budget:
        if method == "spsa":
            # SPSA requires 2 evaluations per step
            if evaluations_used + 2 > budget:
                break
            a_k = lr / ((step + 1)**0.602)
            c_k = 0.1 / ((step + 1)**0.101)
            g = pqc.spsa_gradient(params, c_k=c_k, seed=seed + step * 17)
            evaluations_used += 2
            params -= a_k * g
            
        elif method == "param_shift":
            # Parameter shift requires 2 * num_params evaluations per step
            cost_per_step = 2 * pqc.num_params
            if evaluations_used + cost_per_step > budget:
                break
            g = pqc.parameter_shift_gradient(params)
            evaluations_used += cost_per_step
            params -= lr * g
            
        elif method == "qng":
            # QNG requires 2 * num_params (grad) + num_params (metric approximation) evals
            cost_per_step = 3 * pqc.num_params
            if evaluations_used + cost_per_step > budget:
                break
            step_dir = pqc.quantum_natural_gradient(params, damping=1e-2)
            evaluations_used += cost_per_step
            params -= lr * step_dir
        else:
            raise ValueError(f"Unknown method: {method}")
            
        current_cost = pqc.cost(params)
        evaluations_used += 1
        history_evals.append(evaluations_used)
        history_costs.append(float(current_cost))
        step += 1
        
    return {
        "method": method,
        "total_evaluations": evaluations_used,
        "steps_completed": step,
        "final_cost": float(history_costs[-1]),
        "history_evals": history_evals,
        "history_costs": history_costs,
        "final_params": params.tolist()
    }


# ==============================================================================
# Quantum Neural Network (QNN) Binary Classifier
# ==============================================================================

class QuantumNeuralNetworkClassifier:
    """Variational Quantum Classifier for 2D binary classification.
    
    Ansatz:
    1. Angle-Encoding Feature Map: Ry(x_0) (x) Ry(x_1)
    2. Variational Layers: Ry(theta) and Rz(theta) with CNOT entanglers
    3. Measurement: Pauli-Z expectation on qubit 0 as decision logit.
    """
    def __init__(self, n_layers: int = 2):
        self.n_qubits = 2
        self.n_layers = n_layers
        self.dim = 4
        # 4 parameters per layer (2 for Ry, 2 for Rz) + 2 final Ry = 4 * n_layers + 2
        self.num_params = 4 * n_layers + 2
        
        # Observable Z_0 = Z (x) I = diag(1, 1, -1, -1)
        self.Z0_diag = np.array([1.0, 1.0, -1.0, -1.0], dtype=float)

    def circuit(self, x: np.ndarray, params: np.ndarray) -> QuantumCircuitSimulator:
        """Builds state for input point x = [x0, x1] and parameters theta."""
        sim = QuantumCircuitSimulator(2)
        # Feature map
        sim.ry(0, float(x[0]))
        sim.ry(1, float(x[1]))
        
        p_idx = 0
        for l in range(self.n_layers):
            sim.ry(0, float(params[p_idx]))
            sim.ry(1, float(params[p_idx + 1]))
            p_idx += 2
            sim.rz(0, float(params[p_idx]))
            sim.rz(1, float(params[p_idx + 1]))
            p_idx += 2
            sim.cnot(0, 1)
            
        sim.ry(0, float(params[p_idx]))
        sim.ry(1, float(params[p_idx + 1]))
        return sim

    def predict_logit(self, x: np.ndarray, params: np.ndarray) -> float:
        """Evaluates <psi(x, theta)| Z_0 |psi(x, theta)> in [-1, +1]."""
        sim = self.circuit(x, params)
        return sim.expectation_diagonal(self.Z0_diag)

    def predict_proba(self, x: np.ndarray, params: np.ndarray) -> float:
        """Converts <Z_0> expectation into class-1 probability: p = (1 - <Z_0>) / 2 in [0, 1]."""
        logit = self.predict_logit(x, params)
        # logit = +1 -> |0> (class 0, prob 0.0); logit = -1 -> |1> (class 1, prob 1.0)
        return float(np.clip(0.5 * (1.0 - logit), 1e-7, 1.0 - 1e-7))

    def parameter_shift_sample_grad(self, x: np.ndarray, y: float, params: np.ndarray) -> np.ndarray:
        """Computes analytical gradient dL / d theta for a single sample using parameter shift."""
        grads = np.zeros(self.num_params, dtype=float)
        p_pred = self.predict_proba(x, params)
        # Loss: BCE L = - [y ln p + (1-y) ln (1-p)]
        # dL / dp = (p - y) / [p (1 - p)]
        # dp / d<Z0> = -0.5
        # dL / d<Z0> = (p - y) / [p (1 - p)] * (-0.5)
        dL_dz = (p_pred - y) / (p_pred * (1.0 - p_pred) + 1e-8) * (-0.5)
        
        shift = np.pi / 2.0
        for k in range(self.num_params):
            p_plus = params.copy()
            p_plus[k] += shift
            z_plus = self.predict_logit(x, p_plus)
            
            p_minus = params.copy()
            p_minus[k] -= shift
            z_minus = self.predict_logit(x, p_minus)
            
            dz_dtheta = (z_plus - z_minus) / 2.0
            grads[k] = dL_dz * dz_dtheta
            
        return grads

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 25,
        lr: float = 0.15,
        seed: int = 42
    ) -> Dict:
        """Trains the QNN classifier using parameter-shift batch gradient descent."""
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        params = rng.uniform(-np.pi, np.pi, size=self.num_params)
        n_samples = len(y)
        
        loss_history = []
        acc_history = []
        
        for ep in range(epochs):
            batch_grad = np.zeros(self.num_params, dtype=float)
            total_loss = 0.0
            correct = 0
            
            for i in range(n_samples):
                p_i = self.predict_proba(X[i], params)
                y_i = float(y[i])
                loss_i = -(y_i * np.log(p_i) + (1.0 - y_i) * np.log(1.0 - p_i))
                total_loss += loss_i
                
                pred_label = 1 if p_i >= 0.5 else 0
                if pred_label == int(y_i):
                    correct += 1
                    
                g_i = self.parameter_shift_sample_grad(X[i], y_i, params)
                batch_grad += g_i
                
            batch_grad /= float(n_samples)
            params -= lr * batch_grad
            
            mean_loss = total_loss / float(n_samples)
            acc = correct / float(n_samples)
            loss_history.append(float(mean_loss))
            acc_history.append(float(acc))
            
        elapsed = time.perf_counter() - t0
        return {
            "epochs": epochs,
            "final_loss": loss_history[-1],
            "final_accuracy": acc_history[-1],
            "initial_loss": loss_history[0],
            "initial_accuracy": acc_history[0],
            "loss_history": loss_history,
            "accuracy_history": acc_history,
            "runtime_ms": elapsed * 1000.0,
            "optimal_params": params.tolist()
        }
