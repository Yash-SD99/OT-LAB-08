"""Federated Learning: FedAvg vs Centralized Optimization (Track D).

Simulates:
1. Distributed non-IID client network (K=10 clients)
2. Local client optimization with SGD for E local epochs
3. Server Federated Averaging aggregation: theta_{t+1} = sum_k (n_k / N) * theta_k
4. Trade-off between local epochs E in {1, 5, 10}, communication rounds R, and convergence
"""

import copy
import time
from typing import Dict, List, Tuple
import numpy as np


class DistributedDataset:
    """Partitions a global dataset among K clients with controlled Non-IID heterogeneity."""
    def __init__(self, n_clients: int = 10, n_features: int = 8, samples_per_client: int = 50, seed: int = 42):
        self.K = n_clients
        self.D = n_features
        rng = np.random.default_rng(seed)
        
        # True global parameter
        self.true_theta = rng.normal(0.0, 1.0, size=self.D)
        
        self.client_data = []
        all_X = []
        all_y = []
        
        for k in range(self.K):
            # Client-specific feature shift (Non-IID)
            mean_shift = rng.normal(0.0, 0.5, size=self.D)
            X_k = rng.normal(mean_shift, 1.0, size=(samples_per_client, self.D))
            # Client-specific target with Gaussian noise
            y_k = X_k @ self.true_theta + rng.normal(0.0, 0.2, size=samples_per_client)
            
            self.client_data.append((X_k, y_k))
            all_X.append(X_k)
            all_y.append(y_k)
            
        self.X_central = np.vstack(all_X)
        self.y_central = np.concatenate(all_y)
        self.total_samples = len(self.y_central)
        
        # Validation set
        self.X_val = rng.normal(0.0, 1.0, size=(100, self.D))
        self.y_val = self.X_val @ self.true_theta + rng.normal(0.0, 0.2, size=100)

    def evaluate_loss(self, theta: np.ndarray) -> float:
        """Evaluates MSE on global validation set."""
        preds = self.X_val @ theta
        return float(np.mean((self.y_val - preds)**2))


def run_centralized_training(data: DistributedDataset, max_epochs: int = 50, lr: float = 0.05) -> Dict:
    """Centralized training baseline: trains on pooled global dataset."""
    t0 = time.perf_counter()
    theta = np.zeros(data.D, dtype=np.float64)
    loss_history = []
    
    N = data.total_samples
    for epoch in range(max_epochs):
        loss_val = data.evaluate_loss(theta)
        loss_history.append(loss_val)
        
        # Gradient on pooled dataset
        preds = data.X_central @ theta
        grad = (2.0 / N) * (data.X_central.T @ (preds - data.y_central))
        theta -= lr * grad
        
    elapsed = time.perf_counter() - t0
    return {
        "method": "Centralized SGD",
        "final_loss": loss_history[-1],
        "loss_history": loss_history,
        "runtime_ms": elapsed * 1000.0,
        "communication_rounds": 0
    }


def run_fedavg(
    data: DistributedDataset,
    n_rounds: int = 20,
    local_epochs: int = 5,
    lr: float = 0.05
) -> Dict:
    """Federated Averaging (FedAvg):
    At each communication round:
      1. Server broadcasts global theta_t to all K clients.
      2. Each client k performs local_epochs steps of SGD on local data (X_k, y_k).
      3. Server aggregates: theta_{t+1} = sum_k (n_k / N) * theta_k.
    """
    t0 = time.perf_counter()
    global_theta = np.zeros(data.D, dtype=np.float64)
    loss_history = []
    
    bytes_per_float = 8
    total_bytes_communicated = 0
    
    for r in range(n_rounds):
        loss_val = data.evaluate_loss(global_theta)
        loss_history.append(loss_val)
        
        local_models = []
        sample_weights = []
        
        # Broadcast & Local Training on each client
        for k in range(data.K):
            X_k, y_k = data.client_data[k]
            n_k = len(y_k)
            theta_k = global_theta.copy()
            
            # Local SGD steps
            for _ in range(local_epochs):
                preds_k = X_k @ theta_k
                grad_k = (2.0 / n_k) * (X_k.T @ (preds_k - y_k))
                theta_k -= lr * grad_k
                
            local_models.append(theta_k)
            sample_weights.append(n_k)
            
        # Communication overhead: K downloads + K uploads of D parameters
        total_bytes_communicated += 2 * data.K * data.D * bytes_per_float
        
        # Server Aggregation: weighted average
        aggregated_theta = np.zeros(data.D, dtype=np.float64)
        for theta_k, weight in zip(local_models, sample_weights):
            aggregated_theta += (weight / data.total_samples) * theta_k
            
        global_theta = aggregated_theta
        
    elapsed = time.perf_counter() - t0
    return {
        "method": f"FedAvg (E={local_epochs})",
        "local_epochs": local_epochs,
        "communication_rounds": n_rounds,
        "final_loss": loss_history[-1],
        "loss_history": loss_history,
        "total_bytes_communicated_kb": float(total_bytes_communicated / 1024.0),
        "runtime_ms": elapsed * 1000.0
    }
