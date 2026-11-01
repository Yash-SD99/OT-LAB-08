# Prompt Log: Track E (Main Case Study - Non-Convex Portfolio Optimization)

## Workflow & Meta-Information
- **Track**: Track E — Main Case Study (Option C: Financial Portfolio Optimization with Non-Convex Cardinality & Transaction Costs)
- **Role**: Antigravity Assistant & Student Pair-Programmer (Yash Deshmukh, Roll No. 241070018)
- **Objective**: Formulate and implement the non-convex portfolio optimization problem with cardinality constraints ($\sum z_i \le K$), minimum buy-in thresholds ($w_{\min} z_i \le w_i \le w_{\max} z_i$), and non-convex transaction costs ($c_i z_i$). Implement 4 distinct solvers (Exact Mixed-Integer Quadratic Programming via Branch-and-Cut, Greedy Forward Selection with QP projection, Simulated Annealing, and Genetic Algorithm with repair operator), evaluate the non-convex efficient frontier, and generate the Chapter 6 Strategy-Selection Table.

---

## Interactive Prompt History

### Prompt E-1: Formulation & Synthetic Multi-Factor Risk Model
**User/Task Prompt**:
> "Track E: Option C - Financial Portfolio Optimization. Formulate as a non-convex optimization problem with cardinality constraints, transaction costs, and buy-in thresholds. Generate realistic multi-factor market data and risk models."

**AI Response Summary**:
- Implemented `tracks/E_case_study/portfolio_data.py`:
  - Created a realistic multi-factor asset return generator for $N=20$ equities using a 3-factor market model (Market, Value, Momentum).
  - Ensured the covariance matrix $\Sigma = B F B^T + D$ is strictly symmetric positive-definite ($x^T \Sigma x > 0, \forall x \neq 0$).
  - Configured realistic expected annual returns $\mu \in [0.06, 0.22]$, asset volatilities $\sigma_i \in [0.12, 0.35]$, minimum buy-in threshold $w_{\min} = 0.05$, maximum allocation $w_{\max} = 0.40$, cardinality limit $K \in [3, 6]$, and fixed transaction cost coefficients $c_i = 0.001$.

---

### Prompt E-2: Solver Implementations (MIQP, Greedy, SA, GA)
**User/Task Prompt**:
> "Implement at least 3 solvers for the non-convex formulation: an exact combinatorial/MIQP method, a greedy heuristic, and a metaheuristic (SA or GA). Ensure strict constraint satisfaction."

**AI Response Summary**:
- Implemented `tracks/E_case_study/solvers.py` with 4 solvers:
  1. **Exact MIQP (Branch-and-Bound / Subspace QP)**: Combinatorial enumeration over all $\binom{N}{K}$ candidate asset subsets followed by constrained quadratic programming with active set projections. Serves as ground truth global optimum.
  2. **Greedy Forward Selection**: Iterative greedy expansion adding the asset that yields maximal marginal risk-adjusted objective gain, followed by continuous QP weight optimization.
  3. **Simulated Annealing (SA)**: Dual-space transition operator (discrete asset bitflip/swap mutation + continuous Dirichlet/Gaussian weight perturbation) with exponential temperature cooling.
  4. **Genetic Algorithm (GA)**: Binary-continuous chromosome representation, uniform crossover, cardinality repair operator, and constraint-preserving mutation.
- Enforced all constraints across all solvers: $\sum w_i = 1$, $w_{\min} z_i \le w_i \le w_{\max} z_i$, and $\sum z_i \le K$.

---

### Prompt E-3: Benchmarks, Efficient Frontier & Strategy-Selection Table
**User/Task Prompt**:
> "Add an explicit Section 5 foundations demo and a Track E strategy-selection table. Benchmark the solvers across varying risk aversion lambda and cardinality K, plot efficient frontiers, and generate comprehensive metrics."

**AI Response Summary**:
- Implemented `tracks/E_case_study/benchmarks.py`:
  - Swept risk-aversion parameter $\lambda \in [0.2, 2.0]$ to trace out the non-convex Pareto efficient frontier.
  - Benchmarked runtime, Sharpe ratio, expected return, variance, and optimality gap for all 4 solvers.
  - Verified Amendment 7: Created the Chapter 6 Strategy-Selection Table matching problem scale ($N \le 20$, $N \sim 100$, $N > 1,000$), budget, and latency requirements to the recommended optimizer.
  - Saved plots to `tracks/E_case_study/results/` (`portfolio_efficient_frontier.png` and `solver_comparison_bar.png`).
  - Created `tracks/E_case_study/tests/test_portfolio.py` with 7 comprehensive unit tests verifying PSD properties, constraint feasibility, and solver convergence.
