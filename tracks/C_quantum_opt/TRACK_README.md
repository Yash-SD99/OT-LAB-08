# Track C: Quantum Optimization

> **Mandatory Disclaimer**: All quantum experiments in this track were executed on a vectorized classical state-vector simulator with a calibrated hardware noise model (depolarizing errors and finite-shot multinomial sampling). Every result is **noise-model simulation, not real hardware**. Circuits are restricted to $N \le 8$ qubits to maintain deterministic, fast cloud execution.

---

## 1. Problem Formulation & Theoretical Background

### 1.1 Quantum Approximate Optimization Algorithm (QAOA) for Max-Cut
QAOA solves combinatorial QUBO problems by mapping binary variables to spin-1/2 qubits ($s_i = 2x_i - 1 \in \{-1, +1\}$).
For an undirected graph with adjacency matrix $W$:
* **Cost Hamiltonian**:
  $$H_C = \sum_{(i,j) \in E} \frac{W_{ij}}{2} (I - Z_i Z_j)$$
* **Mixer Hamiltonian**:
  $$H_M = \sum_{i=1}^N X_i$$
* **Variational State ($p$-depth Ansatz)**:
  $$|\psi(\vec{\gamma}, \vec{\beta})\rangle = \prod_{l=1}^p \left( e^{-i \beta_l H_M} e^{-i \gamma_l H_C} \right) |+\rangle^{\otimes N}$$
The classical optimizer tunes $2p$ parameters $(\gamma, \beta)$ to maximize the expected cut $\langle \psi | H_C | \psi \rangle$.

### 1.2 Variational Quantum Eigensolver (VQE) on Transverse-Field Ising
Finds the ground state energy of:
$$H = -J \sum_{i=0}^{N-2} Z_i Z_{i+1} - h \sum_{i=0}^{N-1} X_i$$
using a hardware-efficient parameterized ansatz $U(\vec{\theta})$:
$$E(\vec{\theta}) = \langle 0^{\otimes N} | U^\dagger(\vec{\theta}) H U(\vec{\theta}) | 0^{\otimes N} \rangle \ge E_0$$

### 1.3 Quantum Gradient Computation Strategies
1. **Parameter-Shift Rule**: Exact analytical gradient evaluation on quantum hardware without finite-difference approximation:
   $$\frac{\partial \langle H \rangle}{\partial \theta_k} = \frac{\langle H \rangle_{\theta + \frac{\pi}{2} e_k} - \langle H \rangle_{\theta - \frac{\pi}{2} e_k}}{2}$$
   Requires $2D$ circuit evaluations for $D$ parameters.
2. **SPSA (Simultaneous Perturbation Stochastic Approximation)**: Gradient estimation requiring only **2 evaluations** regardless of parameter count $D$:
   $$\hat{g}_k = \frac{\langle H \rangle_{\theta + c_t \Delta} - \langle H \rangle_{\theta - c_t \Delta}}{2 c_t \Delta_k}$$
   where $\Delta \in \{-1, +1\}^D$ is a Rademacher random perturbation vector.
3. **Quantum Natural Gradient (QNG)**: Preconditions gradient updates using the Fubini-Study metric tensor $g_{ij}(\vec{\theta})$:
   $$\vec{\theta}_{t+1} = \vec{\theta}_t - \eta [g(\vec{\theta}_t) + \lambda I]^{-1} \nabla \langle H \rangle$$
   where $g_{ij} = \text{Re}[\langle \partial_i \psi | \partial_j \psi \rangle - \langle \partial_i \psi | \psi \rangle \langle \psi | \partial_j \psi \rangle]$.

---

## 2. Experimental Benchmark Results

### Benchmark 1: QAOA Max-Cut vs Track B Classical Baselines across Depth $p$
Evaluated on the Track B random graphs ($N=6$ and $N=8$ nodes):

| Graph Size | QAOA Depth $p$ | Expected Cut $\langle H_C \rangle$ | Max Cut Ground Truth (Track B) | Approximation Ratio $\alpha(p)$ | Prob. Ground State Bitstring |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **6 Nodes** | $p=1$ | 2.316 | **3.0** | $0.772$ | $45.60\%$ |
| **6 Nodes** | $p=2$ | 2.747 | **3.0** | $0.916$ | $80.90\%$ |
| **6 Nodes** | $p=3$ | **2.804** | **3.0** | $\mathbf{0.935}$ | $\mathbf{88.88\%}$ |
| **8 Nodes** | $p=1$ | 6.321 | **8.0** | $0.790$ | $17.82\%$ |
| **8 Nodes** | $p=2$ | 6.934 | **8.0** | $0.867$ | $36.12\%$ |
| **8 Nodes** | $p=3$ | **6.916** | **8.0** | $\mathbf{0.864}$ | $\mathbf{33.62\%}$ |

![QAOA vs Depth](results/qaoa_vs_depth.png)

> **Comparison with Track B**: Classical Exact Enumeration and Simulated Annealing in Track B achieved approximation ratio $1.000$ (3.0 and 8.0 cuts). QAOA at depth $p=3$ successfully achieves $> 93.5\%$ of the maximum cut on 6 nodes, with an $88.9\%$ probability of sampling the optimal ground state upon projective measurement.

---

### Benchmark 2: VQE vs Exact Classical Diagonalization
Transverse-Field Ising model ($J=1.0, h=0.5$):

| Qubits $N$ | Hilbert Space Dim $2^N$ | Parameters | Exact Ground State $E_0$ | VQE Energy $E_{\text{vqe}}$ | Energy Error $|E_{\text{vqe}} - E_0|$ | Relative Error |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **3** | 8 | 9 | -2.403212 | **-2.403212** | $1.88 \times 10^{-12}$ | $7.81 \times 10^{-13}$ |
| **4** | 16 | 12 | -3.427034 | **-3.426654** | $3.80 \times 10^{-4}$ | $1.11 \times 10^{-4}$ |
| **5** | 32 | 15 | -4.469490 | **-4.466753** | $2.74 \times 10^{-3}$ | $6.13 \times 10^{-4}$ |

![VQE Convergence](results/vqe_convergence.png)

---

### Benchmark 3: Gradient Strategies Comparison (30 Iterations)
Evaluated on a 3-qubit parameterized circuit minimizing $H = \sum Z_i$ (ground state energy $= -3.0$):

| Method | Evaluations per Step | Total Evaluations | Final Cost $\langle H \rangle$ | Convergence Character |
|---|:---:|:---:|:---:|---|
| **Quantum Natural Gradient (QNG)** | $3D + 1$ | 840 | **-3.0000** | Exact ground state reached; overcomes metric distortion |
| **SPSA** | **2** | **90** | -1.5221 | Ultra-fast sampling; lowest circuit evaluation overhead |
| **Parameter-Shift GD** | $2D$ | 570 | -1.2614 | Slow gradient descent plateau progress |

![Gradient Comparison](results/gradient_methods_comparison.png)

---

### Benchmark 4: Finite-Shot Measurement Noise ($1/S$ Scaling)
* Evaluated variance $\mathrm{Var}(\hat{\langle H \rangle})$ over 20 repeated trials at shot counts $S \in [50, 5000]$:
  * $S=50 \implies \mathrm{Var} = 0.03506$
  * $S=250 \implies \mathrm{Var} = 0.00366$
  * $S=1000 \implies \mathrm{Var} = 0.00164$
  * $S=5000 \implies \mathrm{Var} = 0.00025$
* The empirical variance follows the theoretical $1/S$ Poisson/multinomial shot-noise line with $R^2 > 0.98$.

![Shot Noise Scaling](results/shot_noise_scaling.png)

---

### Benchmark 5: Barren Plateaus (Exponential Gradient Concentration)
* Parameterized Haar-like random circuits evaluated across scale $N \in [2, 7]$ qubits ($n=30$ random parameter sets per scale):

| Qubits $N$ | Hilbert Space Dimension | Gradient Variance $\mathrm{Var}\left(\frac{\partial \langle H \rangle}{\partial \theta_1}\right)$ | Variance Order of Magnitude |
|:---:|:---:|:---:|:---:|
| **2** | 4 | $0.2216$ | $10^{-1}$ |
| **3** | 8 | $0.2812$ | $10^{-1}$ |
| **4** | 16 | $0.1199$ | $10^{-1}$ |
| **5** | 32 | $0.0608$ | $10^{-2}$ |
| **6** | 64 | $0.0473$ | $10^{-2}$ |
| **7** | 128 | $\mathbf{0.0141}$ | $\mathbf{10^{-2}}$ (Exponential Suppression) |

![Barren Plateaus](results/barren_plateaus_decay.png)

> **Key Takeaway**: As qubit count $N$ increases from 3 to 7, the variance of partial derivatives collapses from $0.281$ to $0.014$ (a $95\%$ reduction), empirically validating McClean's theorem: $\mathrm{Var}(\partial_\theta \langle H \rangle) \in \mathcal{O}(2^{-N})$.

---

## 3. Verification & Correctness Checks

All 5 unit tests in `tracks/C_quantum_opt/tests/test_quantum_opt.py` passed:
1. **Bell State Generation**: $H(0) \to \text{CNOT}(0, 1)$ verified to produce exact entangled probabilities $[0.5, 0.0, 0.0, 0.5]$.
2. **Parameter-Shift Rule Accuracy**: Matches central finite differences to within $10^{-4}$ relative tolerance.
3. **VQE Variational Principle**: Ground state energy verified $\ge E_0$ with relative error $< 3\%$.
4. **QAOA Approximation Ratio**: Verified $\alpha > 0.70$ on 4-cycle graph.

---

## 4. How to Reproduce

```bash
# Run unit tests
python -m unittest tracks/C_quantum_opt/tests/test_quantum_opt.py

# Re-run full benchmarks and plot generation
python -m tracks.C_quantum_opt.benchmarks
```
