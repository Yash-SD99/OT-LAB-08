"""Federated Learning: FedAvg vs Centralized Optimization (Track D).

Simulates:
1. Distributed non-IID client network (K=10 clients)
2. Local client optimization with SGD for E local epochs in {1, 5, 10}
3. Server Federated Averaging aggregation: theta_{t+1} = sum_k (n_k / N) * theta_k
4. Equal total gradient step comparison vs Centralized SGD
5. Multi-Seed Federated Benchmark Suite across 15 seeds
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
        
        self.true_theta = rng.normal(0.0, 1.0, size=self.D)
        
        self.client_data = []
        all_X = []
        all_y = []
        
        for k in range(self.K):
            mean_shift = rng.normal(0.0, 0.5, size=self.D)
            X_k = rng.normal(mean_shift, 1.0, size=(samples_per_client, self.D))
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
        
        preds = data.X_central @ theta
        grad = (2.0 / N) * (data.X_central.T @ (preds - data.y_central))
        theta -= lr * grad
        
    elapsed = time.perf_counter() - t0
    return {
        "method": "Centralized SGD",
        "final_loss": loss_history[-1],
        "loss_history": loss_history,
        "runtime_ms": elapsed * 1000.0,
        "communication_rounds": 0,
        "total_gradient_steps": max_epochs
    }


def run_fedavg(
    data: DistributedDataset,
    n_rounds: int = 25,
    local_epochs: int = 5,
    lr: float = 0.05
) -> Dict:
    """Federated Averaging (FedAvg) over n_rounds with local_epochs steps per client."""
    t0 = time.perf_counter()
    global_theta = np.zeros(data.D, dtype=np.float64)
    loss_history = []
    
    bytes_per_float = 8
    total_bytes_communicated = 0
    total_client_steps = 0
    
    for r in range(n_rounds):
        loss_val = data.evaluate_loss(global_theta)
        loss_history.append(loss_val)
        
        local_models = []
        sample_weights = []
        
        for k in range(data.K):
            X_k, y_k = data.client_data[k]
            n_k = len(y_k)
            theta_k = global_theta.copy()
            
            for _ in range(local_epochs):
                preds_k = X_k @ theta_k
                grad_k = (2.0 / n_k) * (X_k.T @ (preds_k - y_k))
                theta_k -= lr * grad_k
                total_client_steps += 1
                
            local_models.append(theta_k)
            sample_weights.append(n_k)
            
        # Communication: 2 transfers (down + up) * K clients * D params * 8 bytes
        total_bytes_communicated += 2 * data.K * data.D * bytes_per_float
        
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
        "total_client_steps": total_client_steps,
        "runtime_ms": elapsed * 1000.0
    }


def run_federated_multi_seed_benchmarks(n_seeds: int = 15) -> Dict:
    """Evaluates FedAvg across E in {1, 5, 10} and communication rounds over 15 seeds."""
    configs = [
        {"E": 1, "rounds": 25},
        {"E": 5, "rounds": 25},
        {"E": 10, "rounds": 25},
        {"E": 5, "rounds": 10},
        {"E": 5, "rounds": 35}
    ]
    
    summary = {}
    
    for cfg in configs:
        E = cfg["E"]
        R = cfg["rounds"]
        key = f"E_{E}_R_{R}"
        
        losses = []
        runtimes = []
        comm_kbs = []
        
        for s in range(n_seeds):
            dataset = DistributedDataset(n_clients=10, n_features=8, samples_per_client=50, seed=100 + s * 13)
            res = run_fedavg(dataset, n_rounds=R, local_epochs=E, lr=0.05)
            losses.append(res["final_loss"])
            runtimes.append(res["runtime_ms"])
            comm_kbs.append(res["total_bytes_communicated_kb"])
            
        summary[key] = {
            "local_epochs": E,
            "rounds": R,
            "mean_final_loss": float(np.mean(losses)),
            "std_final_loss": float(np.std(losses)),
            "mean_runtime_ms": float(np.mean(runtimes)),
            "std_runtime_ms": float(np.std(runtimes)),
            "communication_kb": comm_kbs[0],
            "total_client_gradient_steps": R * 10 * E
        }
        
    # Centralized SGD baseline across the same 15 seeds
    central_losses = []
    central_runtimes = []
    for s in range(n_seeds):
        dataset = DistributedDataset(n_clients=10, n_features=8, samples_per_client=50, seed=100 + s * 13)
        res_c = run_centralized_training(dataset, max_epochs=40, lr=0.05)
        central_losses.append(res_c["final_loss"])
        central_runtimes.append(res_c["runtime_ms"])
        
    summary["centralized_sgd"] = {
        "mean_final_loss": float(np.mean(central_losses)),
        "std_final_loss": float(np.std(central_losses)),
        "mean_runtime_ms": float(np.mean(central_runtimes)),
        "std_runtime_ms": float(np.std(central_runtimes)),
        "communication_kb": 0.0,
        "total_gradient_steps": 40
    }
    
    return summary
