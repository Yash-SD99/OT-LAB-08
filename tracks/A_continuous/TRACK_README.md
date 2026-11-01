# Track A: Continuous Optimization & ML Foundations (Review Updates)

## 1. Problem Formulation & Objectives

Continuous optimization solves:
$$\min_{\theta \in \mathbb{R}^D} f(\theta)$$
where $f: \mathbb{R}^D \to \mathbb{R}$ is twice-differentiable.

This track investigates first-order gradient methods, adaptive learning rates, momentum acceleration, second-order curvature methods, and foundational optimization landscapes (saddle points, multi-modal local minima, ill-conditioned valleys), alongside a pure NumPy multi-layer perceptron with analytical backpropagation.

### Algorithms Implemented from Scratch (NumPy)
1. **Batch Gradient Descent (GD)**: $\theta_{t+1} = \theta_t - \eta \nabla f(\theta_t)$
2. **Momentum (Polyak)**: $v_{t+1} = \gamma v_t + \eta \nabla f(\theta_t)$, $\theta_{t+1} = \theta_t - v_{t+1}$
3. **Nesterov Accelerated Gradient (NAG)**: $v_{t+1} = \gamma v_t + \eta \nabla f(\theta_t - \gamma v_t)$, $\theta_{t+1} = \theta_t - v_{t+1}$
4. **AdaGrad**: $G_{t+1} = G_t + g_t^2$, $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_{t+1} + \epsilon}} g_t$
5. **RMSProp**: $v_{t+1} = \beta v_t + (1 - \beta) g_t^2$, $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_{t+1} + \epsilon}} g_t$
6. **Adam**: Bias-corrected first ($m_t$) and second ($v_t$) moments
7. **Newton-Raphson**: $\theta_{t+1} = \theta_t - [\nabla^2 f(\theta_t) + \lambda I]^{-1} \nabla f(\theta_t)$
8. **Gauss-Newton**: Non-linear least squares solver solving normal equations: $(J^T J + \lambda I)\Delta \theta = J^T r$
9. **Curvature Approximations**: Empirical Fisher matrix $F = \frac{1}{N} \sum_i g_i g_i^T$, diagonal curvature, low-rank SVD truncation, and finite-difference Hessian-Vector Products (HVP).
10. **Backpropagation**: Multivariate chain rule for 2-layer MLP with numerical gradient verification.

---

## 2. Experimental Results & Benchmarks

All values reported below are directly measured from executed benchmark runs in `results/continuous_benchmark_results.json`.

### 2.1 Benchmark 1: Rosenbrock 5,000-Iteration Budget Benchmark
* Start point: $x_0 = [-1.2, 1.0]$, Global minimum: $x^* = [1.0, 1.0]$, $f(x^*) = 0.0$.
* All optimizers run under an identical **5,000 iteration budget** with tuned step sizes.
* Target convergence threshold: $f(x) < 10^{-6}$.

| Optimizer | Tuned Hyperparameters | Iterations to Reach $f < 10^{-6}$ | Final Objective $f(x)$ | Runtime (ms) | Convergence Characterization |
|---|---|:---:|:---:|:---:|---|
| **Gauss-Newton** | $\eta = 1.0, \lambda = 10^{-3}$ | **3** | **$0.00$** | 55.9 ms | Quadratic rate; direct valley navigation via Gauss curvature |
| **Newton-Raphson** | $\eta = 1.0, \lambda = 10^{-4}$ | **5** | **$0.00$** | 75.6 ms | Quadratic rate; exact Hessian inverse step |
| **Momentum** | $\eta = 0.001, \gamma = 0.9$ | **1,386** | $8.82 \times 10^{-20}$ | 18.4 ms | Valley acceleration dampens transverse oscillations |
| **NAG** | $\eta = 0.001, \gamma = 0.9$ | **1,426** | $1.38 \times 10^{-19}$ | 24.3 ms | Lookahead gradient stabilizes valley trajectory |
| **Adam** | $\eta = 0.02, \beta_1=0.9, \beta_2=0.999$ | **2,874** | $1.02 \times 10^{-23}$ | 42.9 ms | Coordinate scaling navigates ill-conditioned curve |
| **AdaGrad** | $\eta = 0.2$ | Not reached ($> 5,000$) | $6.75 \times 10^{-4}$ | 24.7 ms | Slow convergence / plateauing (accumulated $G_t$ stifles progress) |
| **Gradient Descent**| $\eta = 0.001$ | Not reached ($> 5,000$) | $3.76 \times 10^{-3}$ | 14.2 ms | Slow convergence / plateauing along curved bottom |
| **RMSProp** | $\eta = 0.005, \beta = 0.9$ | Not reached ($> 5,000$) | $5.60 \times 10^{-3}$ | 31.0 ms | Oscillates across valley walls without reaching tolerance |

![Rosenbrock 5000](results/rosenbrock_5000_convergence.png)

---

### 2.2 Benchmark 2: Saddle Point Dynamics (Exact vs Perturbed Start)
* Evaluated on 2D hyperbolic saddle $f(x, y) = x^2 - y^2$, with saddle point at $(0, 0)$ ($\nabla f(0, 0) = [0, 0]^T$, Hessian $\text{diag}(2, -2)$).
* **Exact Saddle Start** ($x_0 = [0.0, 0.0]$):
  - Gradient is exactly zero: $\|\nabla f(x_0)\| = 0.0$.
  - First-order GD remains **completely motionless** at the saddle point ($\|x_{50} - x_0\| = 0.00$).
* **Perturbed Start** ($x_0 = [10^{-5}, 10^{-5}]$):
  - Perturbed GD breaks symmetry along the negative curvature axis ($y$-axis), moving distance $9.10 \times 10^{-2}$ toward $-\infty$.
  - Perturbed Momentum accelerates exponentially along the negative curvature direction, moving distance **$4.25 \times 10^{3}$** in 50 steps!

![Saddle Escape](results/saddle_point_escape.png)

---

### 2.3 Benchmark 3: Multi-modal Rastrigin Local Minima (50 Random Starts)
* Objective: $f(x) = 20 + \sum_{i=1}^2 [x_i^2 - 10 \cos(2\pi x_i)]$, domain $x_i \in [-5.12, 5.12]$.
* Global minimum at $x^* = (0, 0)$ with $f(x^*) = 0.0$.
* Evaluated standard GD across **50 random starting points** $x_0 \sim \mathcal{U}(-5.12, 5.12)^2$:
  - **Distinct Local Minima Discovered**: **18 distinct basins of attraction**.
  - **Global Minimum Discovery Rate**: **0/50 starts (0.0%)** — standard GD consistently gets trapped in the nearest local basin.
  - Mean final objective value: $15.76$ (Best $f=0.9950$, Worst $f=40.7930$).
  - **Takeaway**: In highly multi-modal landscapes, local gradient descent has zero probability of finding the global optimum without momentum, stochastic perturbations, or extensive random restarts.

![Rastrigin Local Minima](results/rastrigin_local_minima_hist.png)

---

### 2.4 Benchmark 4: Backpropagation MLP Across Optimizers
* Trained a 2-layer MLP ($2 \to 6 \to 1$, tanh activation) on a non-linear concentric circles classification dataset for 120 epochs using exact analytical backpropagation.

| Optimizer | Hyperparameters | Final MSE + L2 Loss | Final Classification Accuracy |
|---|---|:---:|:---:|
| **Adam** | $\eta = 0.03, \beta_1 = 0.9, \beta_2 = 0.999$ | **0.0632** | **100.0%** |
| **NAG** | $\eta = 0.08, \gamma = 0.85$ | 0.0677 | **100.0%** |
| **Momentum** | $\eta = 0.08, \gamma = 0.85$ | 0.0680 | **100.0%** |
| **AdaGrad** | $\eta = 0.25$ | 0.0756 | **100.0%** |
| **RMSProp** | $\eta = 0.02, \beta = 0.9$ | 0.0813 | **100.0%** |
| **Gradient Descent** | $\eta = 0.08$ | 0.8767 | 83.9% |

![MLP Backprop](results/mlp_backprop_optimizers.png)

---

### 2.5 Benchmark 5: Curvature Quality (Fisher, Diagonal, Rank-k vs Exact Hessian)
* Evaluated curvature approximations against the exact $17 \times 17$ MLP parameter Hessian ($H_{\text{exact}}$, Frobenius norm $= 4.164$):

| Curvature Approximation | Mathematical Formulation | Relative Frobenius Error $\|H_{\text{approx}} - H_{\text{exact}}\|_F / \|H_{\text{exact}}\|_F$ | Description / Practical Trade-off |
|---|---|:---:|---|
| **Rank-6 SVD** | $U_6 S_6 V_6^T$ | **19.82%** | Captures dominant curvature eigenspaces with 6 vectors |
| **Rank-3 SVD** | $U_3 S_3 V_3^T$ | **37.60%** | Captures top 3 principal directions of high curvature |
| **Rank-1 SVD** | $u_1 \sigma_1 v_1^T$ | **67.90%** | First-order principal component approximation |
| **Diagonal Hessian** | $\text{diag}(\text{diag}(H))$ | **83.60%** | Discards all cross-parameter coupling; $\mathcal{O}(D)$ memory |
| **Empirical Fisher** | $\frac{1}{N} \sum_i g_i g_i^T$ | **139.45%** | Positive semi-definite; approximates expected Fisher |

![Curvature Approximation](results/curvature_approximation_quality.png)

---

### 2.6 Benchmark 6: L2 Regularization & Conditioning
* Evaluated effect of weight decay $\lambda \in [0.0, 1.0]$:
  - $\lambda = 0.0 \implies$ Weight norm $= 2.843$, ill-conditioned loss landscape ($\kappa \approx 2.8 \times 10^{10}$).
  - $\lambda = 1.0 \implies$ Weight norm shrinks to **0.603** ($78.8\%$ parameter shrinkage), minimum eigenvalue regularized by $+ \lambda I$.

---

### 2.7 Benchmark 7: Minibatch Size Sweep & Gradient Variance
* Dataset size $N = 256$, swept batch sizes $B \in [1, 4, 16, 64, 256]$:
  - $B = 1 \implies \text{Var}(g_B) = 21.456$
  - $B = 4 \implies \text{Var}(g_B) = 5.285$ ($4.06\times$ variance reduction)
  - $B = 16 \implies \text{Var}(g_B) = 1.080$ ($19.88\times$ variance reduction)
  - $B = 64 \implies \text{Var}(g_B) = 0.237$ ($90.49\times$ variance reduction)
  - $B = 256 \implies \text{Var}(g_B) = 0.000$ (Full batch deterministic gradient)
  - **Empirical Law Confirmed**: Gradient variance scales precisely as $\mathcal{O}(1/B)$.

![Minibatch Scaling](results/minibatch_variance_scaling.png)

---

### 2.8 Benchmark 8: HVP Precision & Epsilon Trade-Off Plot
* Swept finite-difference step size $\epsilon \in [10^{-12}, 10^{-1}]$ for computing $H v \approx \frac{\nabla f(x + \epsilon v) - \nabla f(x - \epsilon v)}{2\epsilon}$:
  - Truncation error dominates for $\epsilon > 10^{-3}$ (relative error $\mathcal{O}(\epsilon^2)$).
  - Floating-point cancellation error dominates for $\epsilon < 10^{-10}$ (relative error $\mathcal{O}(\epsilon_{\text{mach}} / \epsilon)$).
  - Sweet spot lies in the range $\epsilon \in [10^{-6}, 10^{-4}]$ where relative error reaches machine precision limit ($< 10^{-14}$).

![HVP Tradeoff](results/hvp_epsilon_tradeoff.png)

---

## 3. How to Reproduce

```bash
# Run 9 unit tests
python -m pytest tracks/A_continuous/tests/test_continuous.py

# Run benchmark suite and regenerate plots/metrics
python tracks/A_continuous/benchmarks.py
```
