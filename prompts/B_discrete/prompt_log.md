# Prompt Log: Track B (Discrete Optimization & QUBO)

**Student**: Yash Deshmukh (Roll No. 241070018)  
**Track**: Track B — Discrete Optimization, B&B, DP, ILP, and QUBO/Ising  
**Tool Used**: Antigravity (Google DeepMind)  
**Date**: Nov 1, 2026  

---

## Session Trajectory

### 1. Initial Prompt
```text
Implement a complete discrete optimization suite for Track B:
1. 0/1 Knapsack solvers:
   - Greedy (value/weight efficiency ranking).
   - Dynamic Programming with O(N*W) table and backtrack reconstruction.
   - Custom Branch-and-Bound from scratch using fractional knapsack linear relaxation bounding, Best-First Search with priority queue, and subtree pruning.
   - Exact ILP using scipy.optimize.milp (HiGHS solver) and PuLP.
2. Feature selection: Exact Exhaustive Subset Selection vs Forward Stepwise Greedy Selection.
3. QUBO & Ising:
   - qubo_to_ising mapping function (binary-to-spin conversion x = (s+1)/2) extracting (J, h, C).
   - Max-Cut QUBO and Ising formulation on graph adjacency matrix.
   - Knapsack QUBO with quadratic penalty terms: -sum(v*x) + lambda * (sum(w*x) - W_target)^2.
4. Classical baseline solver for Track C:
   - Exact binary enumeration O(2^N).
   - Classical Simulated Annealing (single-spin flip Metropolis-Hastings).
   - Minor embedding overhead analysis for Chimera/Pegasus quantum annealing hardware.
Include unit tests and benchmark scaling scripts with plots.
```

### 2. AI Response Summary & Generated Code
The AI generated:
* `tracks/B_discrete/knapsack_solvers.py`
* `tracks/B_discrete/feature_selection.py`
* `tracks/B_discrete/qubo_formulation.py`
* `tracks/B_discrete/classical_qubo_solver.py`
* `tracks/B_discrete/tests/test_discrete.py`
* `tracks/B_discrete/benchmarks.py`

### 3. Verification & Real Failures Caught (Cases FAIL-02 & FAIL-03)
* **Bug 1 (Case FAIL-02)**: In `qubo_formulation.py`, the diagonal constant was accumulated as `C += 0.25 * Q_sym[i, i]` instead of `0.5 * Q_sym[i, i]`, violating energy conservation between binary and spin bases.
  * *Caught by*: `test_qubo_ising_conversion_identity` ($0.0 \ne -0.21637$).
  * *Fix*: Corrected diagonal accumulation to `0.5 * Q_sym[i, i]`. Verified energy difference $< 10^{-16}$.
* **Bug 2 (Case FAIL-03)**: In `knapsack_solvers.py`, `fractional_bound` returned `0.0` when `curr_wt >= capacity`. When an item combination perfectly filled capacity (`curr_wt == capacity`), the upper bound was returned as 0.0 instead of `curr_val`, causing the exact optimal solution to be pruned.
  * *Caught by*: `test_knapsack_exact_solvers_equivalence` ($240.0 \ne 270.0$).
  * *Fix*: Corrected boundary condition to return `curr_val` when `curr_wt == capacity`.

### 4. Verification Note
* Executed unit tests (`test_discrete.py`): All 7 test suites passed in 0.075s.
* Verified that DP, Custom B&B, and `scipy.optimize.milp` yield identical optimal values across all problem sizes $N \in [10, 40]$.
* Verified that $\lambda \ge 2.5$ strictly enforces feasibility in knapsack QUBO without altering the ground state.

### 5. Chat Export / Screenshot Placeholder
*(Student Note: Place chat export or UI screenshot in `prompts/B_discrete/screenshot_track_B.png`)*.
