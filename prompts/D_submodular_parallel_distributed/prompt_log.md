# Prompt Log: Track D (Submodular, Parallel, and Distributed Optimization)

**Student**: Yash Deshmukh (Roll No. 241070018)  
**Track**: Track D — Submodular Optimization, Parallel Scheduling, FedAvg, and Multi-Task (MGDA/GradNorm)  
**Tool Used**: Antigravity (Google DeepMind)  
**Date**: Nov 1, 2026  

---

## Session Trajectory

### 1. Initial Prompt
```text
Implement a complete optimization suite for Track D:
1. Submodular optimization:
   - Facility location / coverage objective with diminishing returns verification.
   - Greedy selection algorithm with Nemhauser-Wolsey (1 - 1/e) empirical guarantee verification against exact combinatorial optimal.
2. Parallel scheduling:
   - Static block scheduling vs dynamic work-stealing scheduler with worker deques.
   - Measures makespan, worker load imbalance, speedup S(P), parallel efficiency E(P) on P in [1, 2, 4].
   - Fits Amdahl's Law and Gustafson's Law, honestly explaining physical core limits.
3. Federated learning:
   - FedAvg across 10 non-IID clients vs Centralized SGD.
   - Evaluates effect of varying local epochs E in {1, 5, 10} and communication overhead in KB.
4. Multi-task optimization:
   - Two-task linear model with shared representation.
   - Compares Equal Task Weighting vs MGDA (Frank-Wolfe Pareto stationary point) vs GradNorm (dynamic gradient norm balancing).
Include unit tests and benchmark scaling scripts with plots.
```

### 2. AI Response Summary & Generated Code
The AI generated:
* `tracks/D_submodular_parallel_distributed/submodular_opt.py`
* `tracks/D_submodular_parallel_distributed/parallel_scheduling.py`
* `tracks/D_submodular_parallel_distributed/federated_fedavg.py`
* `tracks/D_submodular_parallel_distributed/multitask_mgda_gradnorm.py`
* `tracks/D_submodular_parallel_distributed/tests/test_parallel_dist.py`
* `tracks/D_submodular_parallel_distributed/benchmarks.py`

### 3. Verification Note
* Executed unit tests (`test_parallel_dist.py`): All 6 test suites passed in 0.994s.
* Verified that submodularity test confirmed 0 diminishing-returns violations across 100 sample pairs.
* Verified that greedy submodular selection achieved $> 93.5\%$ of exact optimal value across all budgets $k \in [1, 5]$, well above the $63.2\%$ Nemhauser bound.
* Verified parallel speedup on 2 physical cores ($S=1.886$, efficiency $94.3\%$), honestly documenting scaling degradation at $P=4$ ($E=66.1\%$) due to physical core saturation.
* Verified GradNorm outperformed Equal Weighting by $2.76\times$ on conflicting multi-task objectives.

### 4. Chat Export / Screenshot Placeholder
*(Student Note: Place chat export or UI screenshot in `prompts/D_submodular_parallel_distributed/screenshot_track_D.png`)*.
