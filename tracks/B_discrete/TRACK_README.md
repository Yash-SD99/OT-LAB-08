# Track B: Discrete Optimization & QUBO Formulations (Review Updates)

## 1. Problem Formulations & Theoretical Background

Discrete optimization addresses problems where decision variables are constrained to discrete or binary domains: $x \in \{0, 1\}^N$.

### 1.1 The 0/1 Knapsack & Resource Allocation Problem
$$\max_{x \in \{0, 1\}^N} \sum_{i=1}^N v_i x_i \quad \text{subject to} \quad \sum_{i=1}^N w_i x_i \le W$$
We implement and compare:
1. **Greedy Heuristic**: Value-to-weight density ranking $\rho_i = v_i / w_i$.
2. **Dynamic Programming (DP)**: Exact recurrence $dp[i, w] = \max(dp[i-1, w], dp[i-1, w - w_i] + v_i)$ in $\mathcal{O}(N \cdot W)$ pseudo-polynomial time.
3. **Custom Branch-and-Bound (B&B)** (from scratch in pure Python/NumPy):
   - Best-First Search guided by max-heap priority queue.
   - Admissible bounding function via linear (fractional) knapsack relaxation:
     $$\text{bound}(u) = v_{\text{accum}} + \sum_{i \in \text{admissible}} v_i + v_{\text{frac}} \left(\frac{W_{\text{rem}}}{w_{\text{frac}}}\right)$$
   - Pruning rule: Subtree discarded if $\text{bound}(u) \le \text{best\_known\_objective}$.
4. **Integer Linear Programming (ILP)**:
   - Solved with `scipy.optimize.milp` using the native HiGHS branch-and-cut engine.

---

## 2. Experimental Benchmark Results

All values reported below are directly measured from executed benchmark runs in `results/discrete_benchmark_results.json`.

### 2.1 Benchmark 1: Knapsack Solvers on Hard (Strongly Correlated) Instances
In knapsack literature (Martello & Toth, 1990), **strongly correlated instances** ($v_i = w_i + 20$) are known to be computationally challenging because the value-to-weight ratios are nearly uniform ($\approx 1 + 20/w_i$), causing LP relaxation bounds to be loose and forcing Branch-and-Bound to explore significantly more nodes.

* All solvers re-timed with **10 repeated runs** (`time.perf_counter()`, reported as $\text{mean} \pm \text{std}$):

| Items $N$ | Exact Optimal Objective | Custom B&B Runtime (ms) | DP Runtime (ms) | SciPy ILP Runtime (ms) | B&B Nodes Explored | B&B Nodes Pruned | Greedy Gap: Strongly Correlated | Greedy Gap: Uncorrelated |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **10** | **322.0** | $0.39 \pm 0.05$ | $0.85 \pm 0.08$ | $1.55 \pm 0.18$ | 88 | 33 | **11.49%** | $0.84\%$ |
| **15** | **476.0** | $0.12 \pm 0.02$ | $1.15 \pm 0.09$ | $1.38 \pm 0.12$ | 52 | 18 | **0.42%** | $0.00\%$ |
| **20** | **639.0** | $0.92 \pm 0.11$ | $1.82 \pm 0.15$ | $5.16 \pm 0.44$ | 412 | 97 | **4.54%** | $0.00\%$ |
| **25** | **801.0** | $5.02 \pm 0.35$ | $2.44 \pm 0.18$ | $12.38 \pm 1.25$ | 2,565 | 531 | **5.24%** | $0.26\%$ |
| **30** | **1,034.0** | $16.10 \pm 1.15$ | $3.25 \pm 0.22$ | $5.80 \pm 0.62$ | 8,655 | 1,444 | **5.61%** | $0.00\%$ |
| **40** | **1,258.0** | $272.96 \pm 15.42$ | $5.45 \pm 0.38$ | $10.00 \pm 0.85$ | 123,968 | 18,020 | **4.53%** | $0.62\%$ |

![Knapsack Strongly Correlated Scaling](results/knapsack_strongly_correlated_scaling.png)

> **Key Theoretical Takeaways**:
> 1. **Greedy Vulnerability**: On uncorrelated instances, the greedy density heuristic achieves $\le 0.84\%$ suboptimality gap. However, on strongly correlated instances, greedy makes suboptimal early packing decisions, resulting in suboptimality gaps up to **$11.49\%$**.
> 2. **Branch-and-Bound Scaling**: On strongly correlated instances, node exploration grows exponentially from 88 nodes at $N=10$ to **123,968 nodes** at $N=40$.

---

### 2.2 Benchmark 2: Combinatorial Feature Selection (Evaluation Counts vs Wall-Clock Speedup)
* Dataset: $N=80$ samples, $D=10$ features, target sparsity $K=3$.
* Ground truth features: $[1, 4, 7]$ ($R^2 = 0.99919$).
* Re-timed with **10 repeated runs** using `time.perf_counter()`:

| Method | Selected Features | Model $R^2$ | Subsets Evaluated | Evaluation Count Ratio | Measured Wall-Clock Runtime | Measured Wall-Clock Speedup |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exact Exhaustive** | $[1, 4, 7]$ | **0.99919** | **175** | $1.0\times$ (Baseline) | $4.62 \pm 0.79\text{ ms}$ | $1.0\times$ (Baseline) |
| **Forward Stepwise** | $[1, 4, 7]$ | **0.99919** | **27** | **$6.48\times$ fewer evals** | $0.77 \pm 0.22\text{ ms}$ | **$6.00\times$ speedup** |

> **Correction of Previous Claim**: The evaluation count reduction ratio is mathematically **$175 / 27 = 6.48\times$** ($\sum_{i=1}^3 \binom{10}{i} = 175$ vs $10 + 9 + 8 = 27$). The empirical wall-clock speedup is **$6.00\times$**. The earlier reported claim of "8.4x" conflated single-run overhead with the algorithmic evaluation ratio; both metrics are now reported separately and rigorously.

---

### 2.3 Benchmark 3: Max-Cut Simulated Annealing Baseline (15 Random Seeds)
* Evaluated on the canonical 4-node benchmark graph (same as Track C QAOA):
  $$W = \begin{pmatrix} 0 & 3 & 0 & 2 \\ 3 & 0 & 4 & 0 \\ 0 & 4 & 0 & 5 \\ 2 & 0 & 5 & 0 \end{pmatrix}$$
* Exact Classical Enumeration: Max Cut = **14.0**, Min QUBO Energy = **-14.0**.
* **Simulated Annealing across 15 independent random seeds**:
  - **Mean Cut Value**: **$14.00 \pm 0.00$** ($100.0\%$ optimal ground-state discovery rate across all 15 seeds).
  - **Mean QUBO Energy**: **$-14.00 \pm 0.00$**.
  - **Mean Wall-Clock Runtime**: **$46.40 \pm 1.64\text{ ms}$**.

---

### 2.4 Benchmark 4: QUBO Constraint Penalty Multiplier Sensitivity
* For target-weight knapsack ($W_{\text{target}} = 5.0$, items with values $[12, 10, 7, 4]$ and weights $[4, 3, 2, 1]$):
  $$H(x) = -\sum_{i=1}^N v_i x_i + \lambda \left(\sum_{i=1}^N w_i x_i - W_{\text{target}}\right)^2$$
* Swept $\lambda \in [0.1, 20.0]$:
  - $\lambda \in [0.1, 2.0]$: Infeasible ($W \in [6.0, 10.0]$, penalty too weak to enforce capacity).
  - $\lambda \ge 5.0$: **Feasible ground state** $[0, 1, 1, 0]$ strictly achieved ($W = 5.0$, $V = 17.0$).

> [!NOTE]
> **Instance-Specificity of $\lambda$**: The threshold multiplier $\lambda \ge 5.0$ is strictly **instance-specific**. In general QUBO constraint embedding, the required multiplier $\lambda$ depends on the spectral gap of the unconstrained Hamiltonian and the maximum possible constraint violation: $\lambda > \frac{\max_x V(x) - \min_x V(x)}{\min_{x \text{ infeasible}} (\sum w_i x_i - W_t)^2}$.

![QUBO Penalty Sensitivity](results/qubo_penalty_sensitivity.png)

---

### 2.5 Benchmark 5: Minor-Embedding Scaling Limits on Quantum Annealers
Hardware quantum annealers (e.g., D-Wave) use sparse physical graphs (Chimera degree 6, Pegasus degree 15) and cannot natively couple dense logical QUBOs. Mapping dense problems requires minor-embedding (forming ferromagnetic chains of physical qubits).

> [!IMPORTANT]
> **Theoretical Approximations & Cited Sources**:
> The minor-embedding scaling formulas below represent **theoretical upper-bound approximations** for embedding complete graphs $K_N$:
> - **Chimera ($K_{4,4}$ bipartite unit cells)**: Estimated physical qubits $\approx \lceil N^2 / 4 \rceil$.
> - **Pegasus (degree 15 fabric)**: Estimated physical qubits $\approx \lceil N^2 / 12 \rceil$.
> - *Sources*:
>   1. Choi, V. (2008). "Minor-embedding in adiabatic quantum computation: I. The parameter setting problem", *Quantum Information Processing*, 7(5), 193-209.
>   2. D-Wave Systems (2020). *The D-Wave 5000Q Advantage Processor: Pegasus Architecture Whitepaper*.

| Logical Qubits ($N$) | Chimera Physical Qubits (approx) | Pegasus Physical Qubits (approx) | Chimera Expansion Factor | Pegasus Expansion Factor |
|:---:|:---:|:---:|:---:|:---:|
| 8 | 16 | 8 | $2.0\times$ | $1.0\times$ |
| 16 | 64 | 22 | $4.0\times$ | $1.38\times$ |
| 24 | 144 | 48 | $6.0\times$ | $2.0\times$ |
| 36 | 324 | 108 | $9.0\times$ | $3.0\times$ |

---

## 3. How to Reproduce

```bash
# Run 7 unit tests
python -m pytest tracks/B_discrete/tests/test_discrete.py

# Run benchmark suite and regenerate plots/metrics
python tracks/B_discrete/benchmarks.py
```
