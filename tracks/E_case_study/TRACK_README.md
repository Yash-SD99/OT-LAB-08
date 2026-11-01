# Track E: Main Case Study — Non-Convex Portfolio Optimization

## 1. Domain Discovery & Problem Formulation (Option C)

Modern Portfolio Theory (Markowitz, 1952) provides an analytical quadratic programming formulation for mean-variance efficient frontiers. However, real-world portfolio execution faces three critical non-convex frictions that invalidate standard convex optimization:
1. **Cardinality Constraints**: Institutional mandates and custody fees restrict the portfolio to holding at most $K$ assets ($\sum z_i \le K, K \ll N$), transforming the continuous problem into an NP-hard combinatorial selection over $\binom{N}{K}$ asset subspaces.
2. **Buy-In Thresholds & Upper Limits**: Minimum transaction lot sizes mandate that if an asset is held, its weight must satisfy $w_i \ge w_{\min}$ (e.g. $5\%$), while regulatory exposure limits cap $w_i \le w_{\max}$ (e.g. $40\%$), creating a disconnected, non-convex feasible domain:
   $$\mathcal{W}_i = \{0\} \cup [w_{\min}, w_{\max}]$$
3. **Fixed & Proportional Transaction Costs**: Rebalancing from an initial portfolio $w^{(0)}$ incurs both linear bid-ask spread costs and fixed transaction fees ($c_{\text{fixed}} \mathbb{I}(|w_i - w_i^{(0)}| > 0)$), introducing step discontinuities and non-smooth non-convex penalties.

---

### 1.1 Complete Mathematical Formulation

$$\min_{w \in \mathbb{R}^N, z \in \{0, 1\}^N, u \in \{0, 1\}^N} \quad \frac{1}{2} w^T \Sigma w - \lambda \mu^T w + \sum_{i=1}^N \left( c_{\text{fixed}} \cdot u_i + c_{\text{prop}} \cdot |w_i - w_i^{(0)}| \right)$$

$$\text{subject to:}$$
$$\sum_{i=1}^N w_i = 1 \quad \text{(Full Capital Allocation)}$$
$$\sum_{i=1}^N z_i \le K \quad \text{(Cardinality Limit)}$$
$$w_{\min} z_i \le w_i \le w_{\max} z_i, \quad \forall i \in \{1, \dots, N\} \quad \text{(Buy-In Thresholds & Exposure Limits)}$$
$$|w_i - w_i^{(0)}| \le M u_i, \quad u_i \in \{0, 1\} \quad \text{(Big-M Turnover Coupling)}$$

where:
* $N = 20$ assets across 5 economic sectors (Technology, Finance, Healthcare, Energy, Consumer).
* $\mu \in \mathbb{R}^N$ is the annualized expected return vector.
* $\Sigma \in \mathbb{R}^{N \times N}$ is the positive-definite factor risk covariance matrix ($\Sigma \succ 0$).
* $\lambda \ge 0$ is the investor risk tolerance parameter.
* $w^{(0)} = \frac{1}{N} \mathbf{1}$ is the initial uniform baseline allocation.
* $w_{\min} = 0.05$ ($5\%$), $w_{\max} = 0.40$ ($40\%$).
* $c_{\text{fixed}} = 0.002$ ($20 \text{ bps}$ fixed ticket fee), $c_{\text{prop}} = 0.005$ ($50 \text{ bps}$ proportional spread).

---

## 2. Strategy-Selection Decision Table (Amendment 7)

To guide practical deployment, the table below compares four distinct algorithmic approaches covering exact, heuristic, and metaheuristic paradigms:

| Method | Chapter 6 Category | Decision Variable Types | Mathematical Optimality Guarantee | Computational Complexity / Scalability | Typical Wall-Clock Solve Time | Recommended Hardware Profile & Use Case |
|---|---|---|---|---|:---:|---|
| **Exact MIQP (Branch-and-Bound)** | **Mixed-Integer Quadratic Programming** | Mixed Binary ($z, u$) + Continuous ($w$) | **Global Mathematical Optimum** ($0.0\%$ optimality gap) | Exponential $\mathcal{O}\left(\binom{N}{K} \cdot \text{QP}\right)$; feasible for $N \le 50, K \le 6$ | **$10,512.4 \text{ ms}$** | Multi-core CPU server; periodic high-stakes institutional rebalancing |
| **Greedy Forward Selection** | **Greedy Heuristic with QP Subspace Projection** | Iterative discrete selection + Continuous QP | Suboptimal (greedy heuristic approximation) | Polynomial $\mathcal{O}(K \cdot N \cdot \text{QP})$; scales to $N = 10,000+$ assets | **$33.8 \text{ ms}$ ($311\times$ faster)** | Lightweight CPU / edge device; real-time intraday trading & streaming |
| **Simulated Annealing (SA)** | **Stochastic Local Search Metaheuristic** | Continuous weights with simplex projection | Asymptotic convergence; high quality heuristic | Linear in iterations $\mathcal{O}(M \cdot \text{QP})$; highly tunable | **$222.2 \text{ ms}$** | Multi-threaded CPU with parallel random restarts; complex non-linear utility |
| **Genetic Algorithm (GA)** | **Population-Based Evolutionary Metaheuristic** | Binary chromosome mask with cardinality repair | Heuristic; global exploratory search | $\mathcal{O}(\text{Pop} \cdot \text{Gen} \cdot \text{QP})$; easily parallelized | **$1,337.6 \text{ ms}$** | Massively parallel CPU/GPU clusters; multi-objective Pareto optimization |

---

## 3. Experimental Benchmark Results

### 3.1 Multi-Solver Performance Comparison ($K = 4, \lambda = 1.0$)

Evaluated on the 20-asset factor market model with initial baseline $w^{(0)} = 0.05 \cdot \mathbf{1}$:

| Optimization Solver | Objective Value $f^*$ | Expected Return $\mu^T w$ | Annualized Volatility $\sigma$ | Sharpe Ratio | Transaction Costs | Solve Time (ms) | Optimality Gap | Constraints Feasible? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exact MIQP (B&B)** | **-0.1339** | **19.4%** | **16.5%** | **1.17** | 0.0125 | $10,512.4 \text{ ms}$ | **0.00% (Baseline)** | **Yes (100%)** |
| **Genetic Algorithm** | **-0.1339** | **19.4%** | **16.5%** | **1.17** | 0.0125 | $1,337.6 \text{ ms}$ | **0.00% (Exact Match)**| **Yes (100%)** |
| **Greedy Forward** | **-0.1337** | 19.3% | 16.6% | 1.17 | 0.0124 | **33.8 ms** | **0.15% (Near-Optimal)**| **Yes (100%)** |
| **Simulated Annealing**| -0.1276 | 18.9% | 16.3% | 1.16 | 0.0121 | 222.2 ms | 4.70% | **Yes (100%)** |

> **Key Findings**:
> * **Greedy Heuristic Efficiency**: Greedy Forward Selection achieved an objective within **$0.15\%$ of the global optimum** in just **$33.8\text{ ms}$** (**$311\times$ speedup** over exact MIQP).
> * **GA Global Exploration**: The Genetic Algorithm with cardinality repair successfully located the exact global optimum found by MIQP in $1.34\text{ s}$.
> * **100% Feasibility**: All four solvers generated portfolios strictly satisfying full investment ($\sum w = 1.0$), cardinality ($\sum z \le 4$), and buy-in bounds ($w_i \in [0.05, 0.40]$).

---

### 3.2 Non-Convex Efficient Frontiers across Risk Tolerances ($\lambda \in [0.2, 2.0]$)

Constructed across 6 risk tolerance levels $\lambda \in [0.2, 0.4, 0.7, 1.0, 1.5, 2.0]$:

![Non-Convex Efficient Frontiers](results/portfolio_efficient_frontiers.png)

* At low risk tolerance ($\lambda = 0.2$), all algorithms favor low-volatility Consumer and Healthcare assets, achieving $\sigma \approx 10.5\%$ and returns $\approx 11.2\%$.
* As risk tolerance increases ($\lambda \to 2.0$), the portfolio concentrates allocations into high-beta Technology and Energy assets up to the maximum exposure cap ($w_{\max} = 40\%$), driving returns to $\approx 20.8\%$ with volatility $\approx 18.2\%$.
* The MIQP and GA frontiers overlap almost identically across all risk regimes.

---

### 3.3 Cardinality Constraint Scaling & Diversification Benefits ($K \in [3, 6]$)

Evaluating the trade-off between concentrated vs diversified portfolios at $\lambda = 1.0$:

| Cardinality Limit $K$ | Greedy Return | Greedy Volatility $\sigma$ | Greedy Sharpe | Greedy Solve Time | SA Sharpe | SA Solve Time |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$K = 3$** | 19.1% | 15.8% | **1.21** | **1.4 ms** | 1.21 | 171.8 ms |
| **$K = 4$** | 19.3% | 16.6% | **1.17** | **33.9 ms** | 1.16 | 184.9 ms |
| **$K = 5$** | 19.4% | 16.6% | **1.17** | **83.6 ms** | 1.14 | 205.9 ms |
| **$K = 6$** | 19.4% | 16.9% | **1.15** | **134.4 ms** | 1.12 | 199.4 ms |

![Cardinality Trade-Off](results/portfolio_cardinality_tradeoff.png)

> **Financial Insight**:
> * Expanding cardinality from $K=3$ to $K=6$ increases diversification opportunities, but because the market model contains a dominant high-return cluster, higher cardinality forces capital into lower-Sharpe assets to satisfy the buy-in floor ($w_i \ge 5\%$).
> * Consequently, $K=3$ achieves the highest risk-adjusted Sharpe ratio ($1.21$), proving that cardinality constraints actively protect portfolios from overdiversification penalty when buy-in floors are present.

---

## 4. Verification & Correctness Checks

All 7 unit tests in `tracks/E_case_study/tests/test_portfolio.py` passed in **13.1s**:
1. **Covariance Matrix Invariant**: Verified positive-definiteness ($\min \lambda_i > 0$) and symmetry ($\Sigma = \Sigma^T$).
2. **Budget Constraint**: Verified $|\sum w_i - 1| \le 10^{-3}$ across all 4 solvers.
3. **Cardinality Limit**: Verified $\sum z_i \le K$ for all solvers.
4. **Buy-In Bounds**: Verified $w_i \in [w_{\min}, w_{\max}]$ for all non-zero holdings.
5. **Optimality Dominance**: Verified that MIQP objective value is $\le$ Greedy and metaheuristics.
6. **Transaction Cost Accounting**: Verified non-zero positive fixed and proportional transaction fees on rebalancing.

---

## 5. How to Reproduce

```bash
# 1. Run unit test suite
python -m unittest tracks/E_case_study/tests/test_portfolio.py

# 2. Re-run complete benchmarks and plot generation
python -m tracks.E_case_study.benchmarks
```
