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

### 5. Chat Export / Screenshot Placeholder
*(Student Note: Place chat export or UI screenshot in `prompts/A_continuous/screenshot_track_A.png`)*.
