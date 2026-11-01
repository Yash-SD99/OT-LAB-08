"""Parallel Program Optimization: Scheduling, Work Stealing, and Amdahl's Law (Track D).

Includes:
1. Static Partitioning vs Dynamic Chunking vs Work-Stealing Scheduling
2. Data Parallelism vs Pipeline Parallelism
3. Makespan & Worker Load Variance Analysis
4. Speedup S(P), Efficiency E(P), and Amdahl's / Gustafson's Law Fitting
"""

from collections import deque
import concurrent.futures
import multiprocessing
import os
import queue
import time
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


def synthetic_workload_task(task_id: int, workload_intensity: float) -> Tuple[int, float, float]:
    """A CPU-bound task that performs numerical floating-point operations.
    Returns (task_id, duration_seconds, result_checksum).
    """
    t0 = time.perf_counter()
    # Compute intensive iterations
    n_ops = int(workload_intensity * 100_000)
    acc = 0.0
    for i in range(1, n_ops + 1):
        acc += np.sin(i * 0.001) * np.cos(i * 0.002)
    elapsed = time.perf_counter() - t0
    return task_id, elapsed, float(acc)


class StaticScheduler:
    """Static block scheduling: pre-assigns fixed chunks of tasks to each worker."""
    def __init__(self, n_workers: int):
        self.n_workers = n_workers

    def run(self, task_intensities: List[float]) -> Dict:
        t0 = time.perf_counter()
        n_tasks = len(task_intensities)
        chunks = [[] for _ in range(self.n_workers)]
        for i, val in enumerate(task_intensities):
            chunks[i % self.n_workers].append((i, val))
            
        worker_times = [0.0] * self.n_workers
        
        def worker_fn(w_id: int, task_list: List[Tuple[int, float]]):
            start = time.perf_counter()
            results = []
            for t_id, intensity in task_list:
                results.append(synthetic_workload_task(t_id, intensity))
            duration = time.perf_counter() - start
            return w_id, duration, results

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(worker_fn, w, chunks[w]) for w in range(self.n_workers)]
            for fut in concurrent.futures.as_completed(futures):
                w_id, duration, _ = fut.result()
                worker_times[w_id] = duration
                
        total_time = time.perf_counter() - t0
        return {
            "scheduler": "Static Block",
            "makespan_ms": total_time * 1000.0,
            "worker_times_ms": [t * 1000.0 for t in worker_times],
            "load_imbalance_std_ms": float(np.std(worker_times) * 1000.0)
        }


class WorkStealingSimulator:
    """Work-Stealing Scheduler:
    Each worker maintains its own local double-ended queue.
    When a worker's deque is exhausted, it steals tasks from the tail of busy peers.
    """
    def __init__(self, n_workers: int):
        self.n_workers = n_workers

    def run(self, task_intensities: List[float], seed: int = 42) -> Dict:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        n_tasks = len(task_intensities)
        
        # Initialize deques (simulate random non-uniform initial arrival)
        deques = [deque() for _ in range(self.n_workers)]
        for i, intensity in enumerate(task_intensities):
            # Imbalanced initial distribution
            w_idx = int(rng.choice(self.n_workers, p=[0.7] + [0.3 / (self.n_workers - 1)] * (self.n_workers - 1)) if self.n_workers > 1 else 0)
            deques[w_idx].append((i, intensity))
            
        stolen_count = 0
        worker_busy_times = [0.0] * self.n_workers
        
        # Simulate work execution with stealing
        while any(len(d) > 0 for d in deques):
            for w in range(self.n_workers):
                if len(deques[w]) > 0:
                    t_id, intensity = deques[w].popleft()
                    _, dur, _ = synthetic_workload_task(t_id, intensity)
                    worker_busy_times[w] += dur
                else:
                    # Attempt work stealing from peer with largest deque
                    peers = [p for p in range(self.n_workers) if p != w and len(deques[p]) > 1]
                    if peers:
                        victim = max(peers, key=lambda p: len(deques[p]))
                        stolen_task = deques[victim].pop() # Steal from tail
                        stolen_count += 1
                        _, dur, _ = synthetic_workload_task(stolen_task[0], stolen_task[1])
                        worker_busy_times[w] += dur
                        
        makespan = max(worker_busy_times)
        return {
            "scheduler": "Work Stealing",
            "makespan_ms": makespan * 1000.0,
            "worker_times_ms": [t * 1000.0 for t in worker_busy_times],
            "load_imbalance_std_ms": float(np.std(worker_busy_times) * 1000.0),
            "stolen_tasks_count": stolen_count
        }


def fit_amdahl_gustafson(p_workers: List[int], runtimes: List[float]) -> Dict:
    """Computes measured speedups and fits Amdahl's Law and Gustafson's Law.
    
    Amdahl's Law (Strong Scaling):
      S(P) = 1 / ((1 - s) + s / P)
      where s in [0, 1] is the parallelizable fraction, (1 - s) is the serial bottleneck.
    Gustafson's Law (Weak Scaling):
      S(P) = P - (P - 1) * (1 - s)
    """
    t1 = runtimes[0]
    speedups = [float(t1 / t_p) for t_p in runtimes]
    efficiencies = [float(sp / p) for sp, p in zip(speedups, p_workers)]
    
    # Fit serial fraction (1 - s) using least squares on Amdahl's formula:
    # 1 / S(P) = (1 - s) + s * (1 / P) = (1 - s) * (1 - 1/P) + 1/P
    inv_speedups = np.array([1.0 / sp for sp in speedups])
    inv_p = np.array([1.0 / p for p in p_workers])
    
    # Linear fit: y = c0 + c1 * x where y = 1/S(P), x = 1/P
    # Amdahl: 1/S(P) = (1 - s) + s * (1/P)
    # So intercept = (1 - s) (serial fraction), slope = s (parallel fraction)
    A = np.vstack([np.ones_like(inv_p), inv_p]).T
    c0, s = np.linalg.lstsq(A, inv_speedups, rcond=None)[0]
    
    serial_fraction = max(float(1.0 - s), 0.0)
    parallel_fraction = min(float(s), 1.0)
    
    # Theoretical predictions
    amdahl_predictions = [float(1.0 / (serial_fraction + parallel_fraction / p)) for p in p_workers]
    gustafson_predictions = [float(p - (p - 1) * serial_fraction) for p in p_workers]
    
    return {
        "p_workers": p_workers,
        "measured_runtimes_ms": [float(r) for r in runtimes],
        "measured_speedups": speedups,
        "measured_efficiencies": efficiencies,
        "fitted_parallel_fraction_s": parallel_fraction,
        "fitted_serial_fraction": serial_fraction,
        "amdahl_speedup_fit": amdahl_predictions,
        "gustafson_speedup_fit": gustafson_predictions
    }
