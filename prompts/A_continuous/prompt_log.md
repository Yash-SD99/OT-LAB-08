# Prompt Log: Track A (Continuous Optimization)

**Student**: Yash Deshmukh (Roll No. 241070018)  
**Track**: Track A — Continuous Optimization & ML Foundations  
**Tool Used**: Antigravity (Google DeepMind)  
**Date**: Nov 1, 2026  

---

## Session Trajectory

### 1. Initial Prompt
```text
Implement a complete pure NumPy continuous optimization suite for Track A:
1. First-order: GradientDescent, Momentum, NAG, AdaGrad, RMSProp, Adam.
2. Second-order: Newton-Raphson, Gauss-Newton with Levenberg damping.
3. Landscapes: StronglyConvexQuadratic, SaddlePoint, Rosenbrock, Rastrigin, and NoisyObjectiveWrapper.
4. Backpropagation: Pure NumPy MLP with analytical gradients and finite-difference gradient check.
5. Curvature: Empirical Fisher, diagonal curvature, low-rank approximation, and Hessian-Vector Products.
Include unit tests and benchmark scripts with plots.
```

### 2. AI Response Summary & Generated Code
The AI generated:
* `tracks/A_continuous/landscapes.py`
* `tracks/A_continuous/optimizers.py`
* `tracks/A_continuous/models.py`
* `tracks/A_continuous/tests/test_continuous.py`
* `tracks/A_continuous/benchmarks.py`

### 3. Verification & Discovered Bug (Case FAIL-01)
* **Execution**: Ran `python -m tracks.A_continuous.benchmarks`.
* **Issue Discovered**: In `landscapes.py`, `self.f_opt = 0.5 * float(self.x_opt @ self.A @ self.x_opt - self.b @ self.x_opt)`. The AI factored $0.5$ across both terms, causing $0.5 \times (b^T x^* - b^T x^*) = 0.0$. When algorithms reached the true minimum $-0.3684$, the optimality gap $f(x) - f^*$ became negative (`-0.3684`), violating the theoretical guarantee of a convex quadratic minimum.
* **Rectification Prompt**:
  ```text
  Fix the algebraic precedence in StronglyConvexQuadratic.f_opt: f(x) = 0.5 * x^T A x - b^T x, so f(x*) = -0.5 * b^T x*, not 0.0. Group parentheses properly and re-run benchmarks.
  ```
* **Resolution**: Grouped terms as `0.5 * (self.x_opt @ self.A @ self.x_opt) - (self.b @ self.x_opt)`. Re-benchmarking confirmed non-negative gaps for all convex experiments.
* **Failure Log Reference**: Logged in `failures_log.md` under `FAIL-01`.

### 4. Verification Note
* Executed unit tests (`test_continuous.py`): All 6 test suites passed.
* Gradient check relative error: $4.2 \times 10^{-8} < 10^{-5}$.
* Verified convergence and stability boundaries against theoretical limits ($\eta < 2 / \lambda_{\max}$).

### 6. Review Fixes Prompt & Resolution
**User Review Prompt**:
> "TRACK A: Rosenbrock: same iteration budget (5,000) for all optimizers, tuned step sizes, and a table of iterations to reach f < 1e-6. Don't describe slow convergence as 'stuck'. Add: GD at a saddle point (exact saddle start vs perturbed start), Rastrigin local minima over 50 random starts, backprop MLP train loss and accuracy per optimizer, Fisher vs diagonal vs rank-k curvature quality vs the exact Hessian, L2 regularization effect, minibatch size sweep, HVP error vs finite-difference epsilon plot."

**AI Response Summary**:
- Overhauled `tracks/A_continuous/benchmarks.py`:
  - Rosenbrock: Ran all optimizers for 5,000 iterations. Gauss-Newton (3 iters), Newton-Raphson (5 iters), Momentum (1,386 iters), NAG (1,426 iters), and Adam (2,874 iters) reached $f < 10^{-6}$. GD, AdaGrad, and RMSProp exhibited slow convergence / plateauing along the narrow parabolic valley.
  - Saddle point: Proved exact saddle start $[0, 0]$ on $f = x^2 - y^2$ remains motionless ($\Delta = 0$), while perturbed start $[10^{-5}, 10^{-5}]$ escapes along the negative curvature axis, with Momentum accelerating exponentially.
  - Rastrigin: Discovered 18 distinct local minima across 50 random starts (0% global minimum reach for pure GD).
  - Backprop MLP: Adam and Momentum/NAG achieved 100% classification accuracy on concentric circles.
  - Curvature quality: Evaluated exact Hessian vs Rank-6 (19.8% error), Rank-3 (37.6%), Rank-1 (67.9%), Diagonal (83.6%), and Empirical Fisher (139.5%).
  - L2 regularization: Showed parameter shrinkage from 2.84 to 0.60.
  - Minibatch variance: Empirically verified $\mathcal{O}(1/B)$ variance scaling across $B \in [1, 4, 16, 64, 256]$.
  - HVP Epsilon: Plotted the truncation vs machine roundoff V-curve across $\epsilon \in [10^{-12}, 10^{-1}]$.
- Updated `tracks/A_continuous/tests/test_continuous.py` (9/9 passed).

