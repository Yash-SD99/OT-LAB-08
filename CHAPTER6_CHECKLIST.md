# Chapter 6 Syllabus Coverage Checklist

This checklist tracks the implementation, experimental demonstration, and verification of every algorithm and technique taught in Chapter 6 (as specified in Section 5 of Assignment 8).

| Syllabus Area | Algorithm / Technique | Track | Implementation File | Verification Test | Result Artifact | Status |
|---|---|:---:|---|---|---|:---:|
| **Foundations of ML Optimization** | Parameters, loss functions & regularization | Track A | `tracks/A_continuous/models.py` | `tracks/A_continuous/tests/test_continuous.py` | `tracks/A_continuous/results/` | Verified & Completed |
| | Continuous vs discrete optimization formulation | Tracks A & B | `tracks/A_continuous/landscapes.py`, `tracks/B_discrete/knapsack_solvers.py` | Direct mathematical test | Benchmark tables | Verified & Completed |
| | Convex vs non-convex landscapes (saddle points, local minima) | Track A | `tracks/A_continuous/landscapes.py` | `tracks/A_continuous/tests/test_continuous.py` | Convergence curves | Verified & Completed |
| | Basic quantum computing concepts in optimization | Tracks C & I | `tracks/C_quantum_opt/pqc_qnn.py`, `tracks/I_quantum_algos_qml/circuits_from_scratch.py` | `tracks/C_quantum_opt/tests/test_quantum_opt.py` | Noise & state plots | Planned |
| **Continuous Optimization** | Gradient Descent (Batch GD) | Track A | `tracks/A_continuous/optimizers.py` | `tracks/A_continuous/tests/test_continuous.py` | Step-size sensitivity CSV | Verified & Completed |
| | Stochastic Gradient Descent (SGD) | Track A | `tracks/A_continuous/optimizers.py` | `tracks/A_continuous/tests/test_continuous.py` | Noisy objective CSV | Verified & Completed |
| | Mini-batch SGD | Track A | `tracks/A_continuous/optimizers.py` | `tracks/A_continuous/tests/test_continuous.py` | Batch-size comparison | Verified & Completed |
| | Momentum | Track A | `tracks/A_continuous/optimizers.py` | `tracks/A_continuous/tests/test_continuous.py` | Valley acceleration plot | Verified & Completed |
| | Nesterov Accelerated Gradient (NAG) | Track A | `tracks/A_continuous/optimizers.py` | `tracks/A_continuous/tests/test_continuous.py` | Convergence comparison | Verified & Completed |
| | AdaGrad | Track A | `tracks/A_continuous/optimizers.py` | `tracks/A_continuous/tests/test_continuous.py` | Learning rate decay curves | Verified & Completed |
| | RMSProp | Track A | `tracks/A_continuous/optimizers.py` | `tracks/A_continuous/tests/test_continuous.py` | Non-convex tracking | Verified & Completed |
| | Adam | Track A | `tracks/A_continuous/optimizers.py` | `tracks/A_continuous/tests/test_continuous.py` | Convergence plots | Verified & Completed |
| | Backpropagation (separate from optimizer) | Track A | `tracks/A_continuous/models.py` | Numerical gradient check ($<10^{-6}$) | Numerical diff report | Verified & Completed |
| | Newton's Method (2nd order) | Track A | `tracks/A_continuous/optimizers.py` | `tracks/A_continuous/tests/test_continuous.py` | Quadratic convergence table | Verified & Completed |
| | Gauss-Newton Method | Track A | `tracks/A_continuous/optimizers.py` | `tracks/A_continuous/tests/test_continuous.py` | Non-linear least squares | Verified & Completed |
| | Curvature approximations (Fisher, diagonal/low-rank) | Track A | `tracks/A_continuous/optimizers.py` | Curvature matrix audit | Eigenvalue spectrum | Verified & Completed |
| | Hessian-Vector Products (HVP) | Track A | `tracks/A_continuous/optimizers.py` | Finite-difference comparison | Runtime vs Exact Hessian | Verified & Completed |
| | Convergence (convex, strongly convex, non-convex) | Track A | `tracks/A_continuous/benchmarks.py` | Empirical rate validation | Rate plot $\log(f - f^*)$ | Verified & Completed |
| | Step-size and optimizer tuning | Track A | `tracks/A_continuous/benchmarks.py` | Grid search sensitivity analysis | Sensitivity heatmaps | Verified & Completed |
| | Noisy objectives & stochastic variance | Track A | `tracks/A_continuous/benchmarks.py` | Repeated runs (mean $\pm$ std) | Stochastic trajectory plot | Verified & Completed |
| **Discrete Optimization** | Greedy selection | Track B | `tracks/B_discrete/knapsack_solvers.py` | `tracks/B_discrete/tests/test_discrete.py` | Suboptimality gap table | Verified & Completed |
| | Integer Linear Programming (ILP) | Track B | `tracks/B_discrete/knapsack_solvers.py` | `scipy.optimize.milp` & PuLP solve | Exact vs heuristic | Verified & Completed |
| | Custom Branch-and-Bound (from scratch) | Track B | `tracks/B_discrete/knapsack_solvers.py` | `tracks/B_discrete/tests/test_discrete.py` | Node exploration tree log | Verified & Completed |
| | Dynamic Programming | Track B | `tracks/B_discrete/knapsack_solvers.py` | `tracks/B_discrete/tests/test_discrete.py` | Optimal cost matrix | Verified & Completed |
| | Combinatorial feature selection | Track B | `tracks/B_discrete/feature_selection.py` | Forward/backward selection test | AIC/BIC feature curve | Verified & Completed |
| | QUBO / Ising formulation (binary-to-spin) | Track B | `tracks/B_discrete/qubo_formulation.py` | Energy Hamiltonian check | QUBO matrix CSV | Verified & Completed |
| | Penalty terms for constraints (Max-Cut / Selection) | Track B | `tracks/B_discrete/qubo_formulation.py` | Feasibility audit under penalty | Constraint penalty plot | Verified & Completed |
| **Quantum Optimization** | QAOA (solving Track B Max-Cut QUBO) | Track C | `tracks/C_quantum_opt/qaoa_maxcut.py` | `tracks/C_quantum_opt/tests/test_quantum_opt.py` | Approximation ratio vs depth $p$ | Verified & Completed |
| | VQE (Variational Quantum Eigensolver) | Track C | `tracks/C_quantum_opt/vqe_ising.py` | `tracks/C_quantum_opt/tests/test_quantum_opt.py` | Ground state energy convergence | Verified & Completed |
| | Parameterized Quantum Circuits (PQCs / QNNs) | Track C | `tracks/C_quantum_opt/pqc_qnn.py` | Unitary trace test | Expressibility metric | Verified & Completed |
| | Parameter-shift rule | Track C | `tracks/C_quantum_opt/pqc_qnn.py` | Finite-difference gradient comparison | Exact gradient diff | Verified & Completed |
| | SPSA (Simultaneous Perturbation Stoch. Approx.) | Track C | `tracks/C_quantum_opt/pqc_qnn.py` | Convergence under stochastic shots | SPSA vs Parameter-shift | Verified & Completed |
| | Quantum Natural Gradient (QNG / Fubini-Study) | Track C | `tracks/C_quantum_opt/pqc_qnn.py` | Metric tensor positive-definiteness | QNG vs Standard Gradient | Verified & Completed |
| | Finite-shot measurement cost & hardware noise | Track C | `tracks/C_quantum_opt/noise_and_barren.py` | Noise model variance test | Error vs shot count ($10^2 - 10^4$) | Verified & Completed |
| | Barren Plateaus analysis | Track C | `tracks/C_quantum_opt/noise_and_barren.py` | Variance decay over qubits $n \in [2, 8]$ | Gradient variance curve $\mathrm{Var}(\partial E)$ | Verified & Completed |
| | Quantum Annealing embedding limits | Track C | `tracks/B_discrete/classical_qubo_solver.py` | Chimera/Pegasus graph mapping | Minor embedding overhead | Verified & Completed |
| **Submodular & Parallel Optimization** | Submodular diminishing returns verification | Track D | `tracks/D_submodular_parallel_distributed/submodular_greedy.py` | Marginal gain monotonicity test | Diminishing returns curve | Planned |
| | Greedy selection approximation guarantee ($1 - 1/e$) | Track D | `tracks/D_submodular_parallel_distributed/submodular_greedy.py` | Optimal exhaustive comparison | Empirical vs $(1 - 1/e)$ bound | Planned |
| | Makespan & Load balancing | Track D | `tracks/D_submodular_parallel_distributed/parallel_work_stealing.py` | Worker load variance test | Workload distribution chart | Planned |
| | Work stealing scheduler | Track D | `tracks/D_submodular_parallel_distributed/parallel_work_stealing.py` | Task completion queue audit | Stolen task count log | Planned |
| | Parallelism types (Data, Model, Pipeline) | Track D | `tracks/D_submodular_parallel_distributed/parallel_work_stealing.py` | Execution timeline breakdown | Throughput vs core count | Planned |
| | Speedup, Parallel Efficiency, Amdahl & Gustafson | Track D | `tracks/D_submodular_parallel_distributed/parallel_work_stealing.py` | Multi-core scaling ($P=1, 2, 4$) | Amdahl vs Measured fit | Planned |
| **Distributed & Multi-Task** | Federated Learning (FedAvg) | Track D | `tracks/D_submodular_parallel_distributed/federated_fedavg.py` | Local epochs $E \in \{1, 5, 10\}$ test | Non-IID accuracy curves | Planned |
| | MGDA (Multiple Gradient Descent Algorithm) | Track D | `tracks/D_submodular_parallel_distributed/multi_task_mgda.py` | Pareto-optimality / stationary test | Pareto front vs Equal weights | Planned |
| | GradNorm | Track D | `tracks/D_submodular_parallel_distributed/multi_task_mgda.py` | Gradient norm balance test | Dynamic task weight balance | Planned |
| **Domain-Specific Tracks** | Track E: Non-Convex Portfolio Optimization | Track E | `tracks/E_case_study/solvers.py` | Constraint feasibility check | Efficient frontier comparison | Planned |
| | Track F: Toy TAC Compiler (4 passes + optimization) | Track F | `tracks/F_compiler/passes.py` | Bytecode output equivalence | Code size & instruction delta | Planned |
| | Track G: xv6 OS Scheduler Optimization | Track G | `tracks/G_xv6/patches/` | Codespaces build & usertests plan | Marked "NOT YET MEASURED" | Planned |
| | Track H: SQLite Index Selection (Knapsack/ILP) | Track H | `tracks/H_dbms/index_selection_opt.py` | Query checksum verification | Latency & EXPLAIN ANALYZE | Planned |
| | Track I: Quantum Algorithms (DJ, BV, Grover, QFT, QPE) & QML | Track I | `tracks/I_quantum_algos_qml/circuits_from_scratch.py` | Correctness assertions on simulator | Grover vs Linear crossover | Planned |
| | Track J: Software Eng Optimization (>1,000 LOC) | Track J | `tracks/J_software_eng/formulated_optimizer.py` | Regression test suite | Profiler speedup & memory | Planned |
