# 10-Minute Viva Presentation & Demonstration Script
## Optimization Techniques: Lab Assignment 8 (The 14-Day Vibe-Coding Challenge)

* **Presenter**: Yash Deshmukh
* **Roll Number**: 241070018
* **Institution**: VJTI Mumbai
* **Course**: Optimization Techniques
* **Repository**: [github.com/Yash-SD99/OTLAB08](https://github.com/Yash-SD99/OTLAB08)

---

## Presentation Overview & Time Allocation

| Timestamp | Section | Core Focus | Supporting Artifacts / Commands |
|---|---|---|---|
| **00:00 - 01:00** | Introduction & Portfolio Architecture | Scope of 10 tracks, pure NumPy ethos, strict AI pair-programming disclosure | `README.md`, `CHAPTER6_CHECKLIST.md` |
| **01:00 - 02:30** | Track A: Continuous Optimization & Curvature | First vs second-order on Rosenbrock, Hessian approximations, HVP epsilon curve | `tracks/A_continuous/optimizers.py`, `hvp_error_vs_epsilon.png` |
| **02:30 - 04:00** | Track B & J: Combinatorial Optimization & Pruning | Strongly correlated knapsack B&B scaling, fuzzywuzzy admissible bounds (8.3x speedup) | `knapsack_solvers.py`, `formulated_optimizer.py` |
| **04:00 - 05:30** | Track C & I: Quantum Optimization & QML | QAOA Max-Cut depth scaling, equal-budget SPSA vs PS vs QNG, Barren Plateaus | `qaoa_maxcut.py`, `gradient_methods_comparison.png` |
| **05:30 - 07:00** | Track D: Parallel & Distributed Scalability | True OS multiprocessing scaling, FedAvg local epoch trade-off, MGDA Pareto analysis | `parallel_scheduling.py`, `multitask_comparison.png` |
| **07:00 - 08:30** | Track F & H: Compiler IR & Database Indexing | 4 TAC compiler passes + Chapter 6 formulations, SQLite 0/1 knapsack index selection | `passes.py`, `index_selection_opt.py` |
| **08:30 - 10:00** | AI Failures, Tool Comparison & Viva Defense | 10 authentic AI bugs caught by tests (`failures_log.md`), Antigravity vs Claude Code | `failures_log.md`, `tool_comparison.md` |

---

## Detailed Minute-by-Minute Script

### Minute 00:00 – 01:00: Introduction & Architectural Overview
> **Speaker**:
> "Good morning, respected professors and evaluators. Today I present our 14-day optimization portfolio covering ten comprehensive tracks: from continuous gradient dynamics and combinatorial branch-and-bound, to quantum variational algorithms, compiler optimization, and real-world software profiling.
>
> Every algorithm in this repository is implemented from foundational mathematical principles in pure NumPy—without relying on high-level solvers like PyTorch autograd or SciPy black boxes. We have maintained strict scientific integrity: every stochastic measurement is averaged over 10 to 15 random seeds with explicit standard deviations, every quantum simulation is transparently labeled as a noise-model simulation, and all 10 authentic AI coding bugs encountered during development were captured by unit tests and documented in `failures_log.md`."

* **Live Demo Action**:
  ```bash
  pytest tracks/
  # Output: 70 passed in ~10 seconds
  ```

---

### Minute 01:00 – 02:30: Track A — Continuous Optimization & Curvature Dynamics
> **Speaker**:
> "In Track A, we investigated continuous optimization on convex, saddle, and ill-conditioned surfaces. On the Rosenbrock banana valley, first-order gradient descent suffers from extreme cross-valley oscillations due to Hessian ill-conditioning ($\kappa \approx 2,500$), crawling along the valley floor.
>
> In contrast, our pure NumPy Gauss-Newton solver with Levenberg-Marquardt damping leverages the curvature tensor $(J^T J + \lambda I)^{-1} J^T r$ to reach $f < 10^{-6}$ in just **3 iterations** ($55.9\text{ ms}$), compared to **2,874 iterations** for Adam and over 5,000 for vanilla GD.
>
> Furthermore, we analyzed curvature approximations against the exact Hessian: rank-6 Cholesky captures curvature within $19.8\%$ error, while the empirical Fisher diverges with $139.5\%$ error away from local minima. Finally, our Hessian-Vector Product analysis confirms the classical numerical V-curve: truncation error dominates for $\epsilon > 10^{-4}$, while roundoff cancellation explodes below $10^{-8}$, isolating optimal stability at $\epsilon \approx 10^{-7}$."

* **Slide / Visual**: Show `tracks/A_continuous/results/hvp_error_vs_epsilon.png` and `rosenbrock_comparison.png`.

---

### Minute 02:30 – 04:00: Tracks B & J — Combinatorial Optimization & Pruning
> **Speaker**:
> "Moving to discrete optimization in Tracks B and J, we benchmarked Branch-and-Bound, Dynamic Programming, and Integer Linear Programming. On uncorrelated knapsack instances, greedy heuristics perform well ($\le 0.8\%$ gap). However, when we generate **strongly correlated instances** ($v_i = w_i + 20$), the greedy optimality gap deteriorates to **$11.5\%$**, and Branch-and-Bound node exploration explodes from 88 nodes at $N=10$ to **123,968 nodes** at $N=25$.
>
> In Track J, we applied this combinatorial pruning principle to `fuzzywuzzy` ($>1,000$ lines of code). Full Levenshtein distance matrices consume $94\%$ of top-K string matching runtime. We formulated two admissible bounding relaxations—length discrepancy $U_{\text{len}}$ and character multiset histogram intersection $U_{\text{char}}$—pruning **$91.4\%$ of candidate strings** before dynamic programming, achieving an **$8.3\times$ wall-clock speedup** with zero rank regressions."

* **Code Reference**: Open `tracks/J_software_eng/formulated_optimizer.py` lines 45–75 showing $U_{\text{len}}$ and $U_{\text{char}}$.

---

### Minute 04:00 – 05:30: Tracks C & I — Quantum Optimization & Variational Algorithms
> **Speaker**:
> "In Track C and Track I, we developed a pure NumPy vectorized state-vector simulator verified to machine precision ($< 1.24 \times 10^{-16}$) against analytical Kronecker tensor mechanics.
>
> For quantum optimization, we eliminated the misleading claim of '89% fewer evaluations for SPSA' by enforcing a **strictly equal circuit-evaluation budget of 300 evaluations** across 15 seeds. Parameter-Shift GD achieved the lowest energy ($-2.9058 \pm 0.3500$) in 15 steps, outperforming SPSA ($-2.8044$, 100 steps) and Quantum Natural Gradient ($-2.4868$, 10 steps), proving that QNG's metric tensor evaluation overhead hurts convergence under tight budgets.
>
> On the Track B 4-node Max-Cut graph, QAOA depth scaling from $p=1$ to $p=3$ increased optimal ground state probability from $50.2\%$ to **$67.8\%$**. Our barren plateau study confirmed exponential gradient variance concentration ($\alpha = 0.692$) across $N \in [2..7]$ qubits on a 2-design ansatz, while our quantum annealing demo identified the precise sweet spot at $J_{\text{chain}} = 1.5$ yielding $100\%$ valid ground states."

* **Slide / Visual**: Show `tracks/C_quantum_opt/results/gradient_methods_comparison.png` and `barren_plateaus_decay.png`.

---

### Minute 05:30 – 07:00: Track D — Parallel Scaling, Federated Learning & Multi-Task
> **Speaker**:
> "In Track D, we executed parallel scaling using independent OS processes (`multiprocessing.ProcessPoolExecutor`) to bypass the Python GIL. Benchmarking on identical heavy-tailed Pareto workloads over 15 seeds revealed an important physical insight: on Windows, process creation overhead ($\approx 100\text{ ms}$) limits strong scaling speedup to $1.18\times$ at $P=2$, correcting an earlier simulation artifact. Work-stealing queue simulation on the exact same task durations achieved a $1.97\times$ makespan reduction.
>
> In Federated Learning (FedAvg), $E=5$ local epochs matched Centralized SGD loss ($0.0410$ vs $0.0429$) within 25 communication rounds ($31.25\text{ KB}$ transfer).
> In Multi-Task Optimization, our 15-seed Pareto analysis revealed that MGDA strictly seeks Pareto-stationary points, becoming trapped when one task gradient vanishes ($L_2 = 10.33$). GradNorm dynamically rebalances gradient norms using inverse training rates, achieving the lowest joint loss of **$0.0230 \pm 0.0084$**."

* **Slide / Visual**: Show `tracks/D_submodular_parallel_distributed/results/multitask_comparison.png`.

---

### Minute 07:00 – 08:30: Tracks F & H — Compiler IR & Database Index Selection
> **Speaker**:
> "In Track F, we built a toy imperative compiler lowering high-level code to Three-Address Code (TAC), featuring Constant Folding, Strength Reduction, Common Subexpression Elimination (CSE), and Dead Code Elimination (DCE). All arithmetic strictly models 32-bit two's complement integer VM semantics. On a 100+ instruction nested loop benchmark, -O3 optimization achieved a **$1.99\times$ VM cycle reduction** and **$45.1\%$ code size reduction**.
>
> We formulated three Chapter 6 combinatorial optimization problems:
> 1. Pass Ordering via Branch-and-Bound over the permutation tree.
> 2. Loop Unrolling Factor via 0/1 Knapsack ILP.
> 3. Register Allocation via Kempe heuristic graph coloring, generating valid spill code when physical registers $K < \chi(G) = 4$.
>
> Differential testing across 500 random programs verified $100\%$ semantic equivalence between -O0 and -O3. In Track H, SQLite index selection was modeled as a 0/1 knapsack problem across 32 candidate indices, accelerating join queries by **$17.2\times$** while verifying result checksum invariance."

* **Code Reference**: Show `tracks/F_compiler/passes.py` and `tracks/H_dbms/index_selection_opt.py`.

---

### Minute 08:30 – 10:00: AI Failure Audit, Tool Comparison & Viva Defense
> **Speaker**:
> "To conclude, Section 8 mandates an honest account of AI collaboration. In `failures_log.md`, we documented 10 authentic AI coding bugs caught by our tests.
> Examples include:
> * **FAIL-01**: An operator precedence error in quadratic objectives that caused $f(x^*) = 0.0$ and produced impossible negative optimality gaps.
> * **FAIL-04**: An in-loop PRNG re-instantiation in the quantum simulator that silenced depolarizing noise.
> * **FAIL-08**: SQLite freelist reuse that falsely reported 0 KB footprint for candidate indices.
> * **FAIL-10**: A continuous floating-point upper bound that fell below discrete integer scores in `fuzzywuzzy`, violating admissibility.
>
> In our comparative evaluation against Claude Code (Sonnet 5.5) in `tool_comparison.md`, Antigravity produced the more complete, adaptive Levenberg-Marquardt optimizer with automated finite-difference tests, while Claude Code produced a compact fixed-damping routine.
>
> Thank you, and I look forward to your questions."

---

## Anticipated Viva Questions & Technical Answers

### Q1: Why did first-order GD fail to reach $10^{-6}$ on Rosenbrock in 5,000 steps while Gauss-Newton took only 3 steps?
> **Answer**: The Rosenbrock Hessian has condition number $\kappa \approx 2,500$ along the parabola $x_2 = x_1^2$. The maximum stable learning rate for gradient descent is bounded by $\eta < 2 / \lambda_{\max} \approx 2 / 2500 = 8 \times 10^{-4}$. With this small step size, progress along the eigenvector corresponding to $\lambda_{\min}$ is proportional to $\eta \lambda_{\min} \approx 10^{-3}$, requiring thousands of steps to traverse the curve. Gauss-Newton multiplies by $(J^T J + \lambda I)^{-1}$, effectively rotating and scaling the gradient into the natural curvature coordinates, achieving quadratic convergence near the minimum.

### Q2: What is the Nemhauser-Wolsey theorem and what assumptions are required for the $(1 - 1/e)$ bound?
> **Answer**: Nemhauser, Wolsey, and Fisher (1978) proved that for a normalized ($f(\emptyset) = 0$), monotone non-decreasing ($A \subseteq B \implies f(A) \le f(B)$) submodular set function, the greedy selection algorithm subject to a cardinality constraint $|S| \le k$ achieves $f(S_{\text{greedy}}) \ge (1 - 1/e) f(S^*) \approx 0.6321 \cdot f(S^*)$. In Track D, our facility location objective $f(S) = \sum_j \max_{i \in S} w_{ij}$ satisfies diminishing returns because $\max(a, c) - a = \max(0, c - a)$ is monotonically non-increasing in $a$.

### Q3: Why did MGDA produce a high Task 2 loss ($10.33$) compared to GradNorm ($0.013$)?
> **Answer**: MGDA seeks a Pareto-stationary point where $\min_{\alpha} \|\sum \alpha_i g_i\|^2 = 0$. If task gradients conflict and Task 1 reaches a point where its gradient $g_1 \approx 0$ or opposes $g_2$, the min-norm point sets $\alpha_1 \to 1.0$ and $\alpha_2 \to 0.0$. At this boundary, no common descent direction exists that decreases both tasks simultaneously, so MGDA terminates. GradNorm dynamically rescales task weights $w_i \propto (L_i(t) / L_i(0))^\alpha$, forcing the optimizer to allocate gradient magnitude to whichever task is lagging behind.

### Q4: Why is shot noise multinomial rather than Poisson?
> **Answer**: Projective quantum measurement collapses an $N$-qubit statevector $|\psi\rangle$ into one of $2^N$ discrete orthonormal computational basis states $|x\rangle$ with Born-rule probabilities $p(x) = |\langle x | \psi \rangle|^2$, where $\sum_x p(x) = 1$. Drawing $S$ measurement shots corresponds to $S$ independent categorical trials with fixed total count $S$, which is precisely a **Multinomial distribution** $\text{Multinomial}(S, \vec{p})$, with variance $\text{Var}(\hat{p}_x) = \frac{p_x(1 - p_x)}{S}$. Poisson noise describes unbounded counts from continuous rate processes, whereas quantum measurement outcomes are strictly constrained to sum to $S$.

### Q5: How was the SQLite freelist issue (FAIL-08) resolved?
> **Answer**: SQLite's storage engine does not return pages to the OS when an index is dropped; it pools them into an internal freelist (`PRAGMA freelist_count`). Measuring raw database file page counts before and after `CREATE INDEX` reported 0 net pages because the index reused existing freelist pages. We resolved this by computing the active page differential: $\Delta \text{pages} = (\text{page\_count} - \text{freelist\_count})_{\text{after}} - (\text{page\_count} - \text{freelist\_count})_{\text{before}}$, accurately measuring candidate index size regardless of freelist state.
