# Track I: Quantum Algorithms & QML Advantage

> [!IMPORTANT]
> **Mandatory Hardware Disclaimer**: All quantum computations in this track represent **noise-model statevector simulations executed on a classical host CPU**, not physical quantum hardware. Quantum query advantages reflect algorithmic oracle calls; wall-clock comparisons account for classical matrix-vector contraction overhead.

---

## 1. Executive Summary & Student Metadata
- **Student**: Yash Deshmukh
- **Roll Number**: 241070018
- **Institution**: Veermata Jijabai Technological Institute (VJTI), Mumbai
- **Track**: Track I — Quantum Algorithms & Quantum Machine Learning (QML) Advantage
- **Objective**: Implement from scratch in pure NumPy the foundational canon of quantum algorithms (Deutsch-Jozsa, Bernstein-Vazirani, Grover's Unstructured Search, Quantum Fourier Transform, Quantum Phase Estimation, and Shor's Order-Finding Algorithm for $N=15$), followed by a Variational Quantum Classifier (VQC) with an entangling ZZ-feature map trained via the analytical Parameter-Shift Rule against a parameter-matched Classical Multi-Layer Perceptron (MLP).

---

## 2. Canonical Quantum Algorithms: Theory & Results

### 2.1 Deutsch-Jozsa Algorithm (Exponential Query Advantage)
The Deutsch-Jozsa algorithm determines whether an unknown black-box Boolean function $f: \{0, 1\}^n \to \{0, 1\}$ is **constant** ($f(x) \equiv 0$ or $f(x) \equiv 1$) or **balanced** ($\sum_x f(x) = 2^{n-1}$) with zero probability of error in exactly **1 quantum query**.

#### Circuit Mechanics:
1. Initialize $(n + 1)$ qubits: $n$ input qubits in $|0\rangle^{\otimes n}$, 1 ancilla in $|1\rangle$.
2. Apply Hadamard gates:
   $$|\psi_1\rangle = H^{\otimes n} |0\rangle^{\otimes n} \otimes H |1\rangle = \frac{1}{\sqrt{2^n}} \sum_{x \in \{0, 1\}^n} |x\rangle \otimes |-\rangle$$
3. Query the oracle $U_f: |x\rangle|y\rangle \to |x\rangle|y \oplus f(x)\rangle$. Because $|-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$ is an eigenstate of the bit-flip operator $X$ with eigenvalue $-1$, phase kickback occurs:
   $$U_f |x\rangle|-\rangle = (-1)^{f(x)} |x\rangle|-\rangle$$
4. Apply Hadamard gates to the $n$ input qubits:
   $$|\psi_3\rangle = \frac{1}{2^n} \sum_{x, y \in \{0, 1\}^n} (-1)^{f(x) + x \cdot y} |y\rangle \otimes |-\rangle$$
5. Measure input qubits in the computational basis:
   - For constant $f(x) = c$: Amplitude of $|0\rangle^{\otimes n}$ is $\frac{(-1)^c}{2^n} \sum_{x} 1 = \pm 1 \implies P(|0\dots 0\rangle) = 1.0$.
   - For balanced $f(x)$: Amplitude of $|0\rangle^{\otimes n}$ is $\frac{1}{2^n} \sum_{x} (-1)^{f(x)} = 0 \implies P(|0\dots 0\rangle) = 0.0$.

#### Empirical Scaling Results ($n \in [2, 8]$):
| Input Qubits ($n$) | Evaluated Functions | Quantum Queries | Classical Deterministic Worst-Case | Empirical Query Speedup |
|:---:|:---:|:---:|:---:|:---:|
| 2 | Constant / Balanced | **1** | 3 | $3.0\times$ |
| 3 | Constant / Balanced | **1** | 5 | $5.0\times$ |
| 4 | Constant / Balanced | **1** | 9 | $9.0\times$ |
| 5 | Constant / Balanced | **1** | 17 | $17.0\times$ |
| 6 | Constant / Balanced | **1** | 33 | $33.0\times$ |
| 7 | Constant / Balanced | **1** | 65 | $65.0\times$ |
| 8 | Constant / Balanced | **1** | **129** | **$129.0\times$** |

---

### 2.2 Bernstein-Vazirani Algorithm (Linear Query Advantage)
Given an inner-product oracle $f(x) = s \cdot x \pmod 2$ parameterized by an unknown hidden bitstring $s \in \{0, 1\}^n$, the classical algorithm requires $n$ independent queries ($f(100\dots 0) = s_0, f(010\dots 0) = s_1$, etc.). The Bernstein-Vazirani algorithm reconstructs the entire $n$-bit string $s$ in **exactly 1 quantum query**.

#### Circuit Mechanics:
1. Ancilla initialized to $|1\rangle \to H \to |-\rangle$; inputs initialized to $|0\rangle^{\otimes n} \to H^{\otimes n} \to |+\rangle^{\otimes n}$.
2. Oracle applies CNOT from input qubit $i$ to ancilla if $s_i = 1$.
3. Phase kickback gives state:
   $$|\psi_2\rangle = \frac{1}{\sqrt{2^n}} \sum_{x} (-1)^{s \cdot x} |x\rangle \otimes |-\rangle$$
4. Applying $H^{\otimes n}$ performs the inverse Walsh-Hadamard transform, collapsing the superposition directly to $|s\rangle$:
   $$H^{\otimes n} \left( \frac{1}{\sqrt{2^n}} \sum_x (-1)^{s \cdot x} |x\rangle \right) = |s\rangle$$

#### Empirical Verification:
- Tested across random bitstrings $s$ of lengths $n = 2$ through $n = 8$ (e.g. `01`, `100`, `1010`, `01111`, `111010`, `1001110`, `11000011`).
- **Success Rate**: **100.0%** ($P(s) > 0.9999$ across all trials).
- **Quantum Queries**: **1** vs **$n$** classical queries.

---

### 2.3 Grover's Unstructured Search Algorithm (Quadratic Speedup)
For an unstructured database of $N = 2^n$ computational states with a single marked element $w$, classical linear search requires on average $N/2$ queries. Grover's algorithm finds $w$ in $\mathcal{O}(\sqrt{N})$ queries.

#### Geometric Invariant:
State space is spanned by the orthonormal basis $\{|w\rangle, |w^\perp\rangle\}$, where $|w^\perp\rangle = \frac{1}{\sqrt{N-1}}\sum_{x \neq w}|x\rangle$. The uniform superposition $|s\rangle = \sin(\theta)|w\rangle + \cos(\theta)|w^\perp\rangle$ with initial angle $\sin(\theta) = 1/\sqrt{N}$.
Each Grover iteration applies:
$$G = - D \cdot O_w = (2|s\rangle\langle s| - I)(I - 2|w\rangle\langle w|)$$
Each application rotates the state vector toward $|w\rangle$ by angle $2\theta$. The exact optimal iteration count is:
$$R = \text{round}\left( \frac{\pi / 2 - \theta}{2\theta} \right)$$

#### Empirical Scaling Table ($N = 4$ to $1024$):
| Qubits ($n$) | Database Size ($N$) | Grover Iterations ($R$) | Success Prob $P(w)$ | Classical Avg Queries ($N/2$) | Query Speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 4 | **1** | **100.0%** | 2.0 | $2.0\times$ |
| 3 | 8 | **2** | **94.53%** | 4.0 | $2.0\times$ |
| 4 | 16 | **3** | **96.13%** | 8.0 | $2.67\times$ |
| 5 | 32 | **4** | **99.92%** | 16.0 | $4.0\times$ |
| 6 | 64 | **6** | **99.66%** | 32.0 | $5.33\times$ |
| 7 | 128 | **8** | **99.56%** | 64.0 | $8.0\times$ |
| 8 | 256 | **12** | **99.99%** | 128.0 | $10.67\times$ |
| 9 | 512 | **17** | **99.94%** | 256.0 | $15.06\times$ |
| 10 | 1024 | **25** | **99.95%** | 512.0 | **$20.48\times$** |

> [!NOTE]
> **Simulation Reality & Crossover Analysis**:
> On physical quantum hardware, each Grover query takes $\mathcal{O}(1)$ physical gate time, yielding a genuine $\mathcal{O}(\sqrt{N})$ wall-clock speedup. On a **classical CPU statevector simulator**, maintaining and contracting the $2^n$-dimensional state vector incurs an exponential cost $\mathcal{O}(2^n)$. As observed in `grover_scaling_crossover.png`, classical in-memory linear scan ($0.05\text{ ms}$ at $N=1024$) outperforms the classical Python statevector simulator ($42\text{ ms}$). This demonstrates why quantum advantage requires physical quantum processors rather than classical emulation.

---

### 2.4 Quantum Fourier Transform (QFT) & Quantum Phase Estimation (QPE)
- **QFT Matrix Equivalence**: Verified that our native gate decomposition (Hadamards, controlled-phase rotations $R_k = \text{diag}(1, e^{2\pi i / 2^k})$, and SWAP reversals) matches the exact theoretical Discrete Fourier Transform matrix with Frobenius error $< 10^{-12}$:
  $$\| U_{QFT} - \text{DFT} \|_F = 5.80 \times 10^{-15}$$
- **QPE Phase Accuracy**: Tested on phase rotation gate $P(2\pi \theta)$ with eigenstate $|1\rangle$.
  - For dyadic rational phases ($\theta = 0.0, 0.25, 0.5, 0.75, 0.375$) with $m=4$ counting qubits:
    $$\text{Phase Error} = |\hat{\theta} - \theta| = 0.0000 \quad (P(\hat{\theta}) = 1.0000)$$

---

### 2.5 Shor's Period-Finding Algorithm (Factoring $N=15$)
Shor's algorithm reduces the problem of factoring composite integer $N$ to order-finding for $f(x) = a^x \pmod N$, where $\gcd(a, N) = 1$.

#### Execution on $N = 15$ with Coprime Base $a = 7$:
1. **Classical Modular Sequence**:
   $$7^0 \equiv 1, \quad 7^1 \equiv 7, \quad 7^2 \equiv 4, \quad 7^3 \equiv 13, \quad 7^4 \equiv 1 \pmod{15} \implies r = 4$$
2. **Quantum Phase Estimation Circuit**:
   - Counting register of $m=4$ qubits (basis states $0$ to $15$).
   - Unitary $U_a |y\rangle = |a \cdot y \pmod N\rangle$ has eigenvalues $e^{2\pi i (s / r)}$ for $s \in \{0, 1, 2, 3\}$.
   - In big-endian place-value notation ($2^{m-1-j}$), QPE produces measurement peaks at:
     $$k = \frac{s \cdot 2^m}{r} \in \{0, 4, 8, 12\}$$
     corresponding to measured phases $\theta \in \{0.0, 0.25, 0.5, 0.75\}$.
3. **Factor Extraction via Continued Fractions**:
   - For measured peak $\theta = 0.25 = 1/4$ or $0.75 = 3/4$, the denominator yields period $r = 4$.
   - Since $r$ is even and $a^{r/2} = 7^2 = 49 \not\equiv -1 \pmod{15}$:
     $$p_1 = \gcd(7^2 - 1, 15) = \gcd(48, 15) = 3$$
     $$p_2 = \gcd(7^2 + 1, 15) = \gcd(50, 15) = 5$$
   - **Result**: Successfully factored $15$ into non-trivial prime factors $\{3, 5\}$!

---

## 3. Quantum Machine Learning: VQC vs Classical MLP

### 3.1 Non-Linear Dataset & Model Architecture
We evaluated classification on a 2D non-linear concentric circles dataset ($160$ samples, $120$ train / $40$ test, scaled into $[0.1\pi, 0.9\pi]$).

```
[2D Inputs (x0, x1)] 
         │
         ▼
[Quantum ZZ-Feature Map] ──> H x H ──> Rz(2*x0) (x) Rz(2*x1) ──> Rzz(2(pi - x0)(pi - x1))
         │
         ▼
[Parameterized Ansatz (2 Layers)] ──> Ry(theta0, theta1) ──> CNOT ──> Ry(theta2, theta3) [x2]
         │
         ▼
[Pauli-Z Observable <Z0>] ──> Continuous Prediction y_hat in [-1, +1]
```

- **Variational Quantum Classifier (VQC)**:
  - 2 qubits, 2 variational layers.
  - **8 trainable parameters** ($\theta \in \mathbb{R}^8$).
  - Gradients computed via the exact analytical **Parameter-Shift Rule**:
    $$\frac{\partial \langle Z_0 \rangle}{\partial \theta_k} = \frac{1}{2}\left[ \langle Z_0 \rangle_{\theta_k + \pi/2} - \langle Z_0 \rangle_{\theta_k - \pi/2} \right]$$
  - Verified against numerical central differences: $\| \nabla_{ps} - \nabla_{num} \| < 10^{-4}$.
- **Classical Multi-Layer Perceptron (MLP)**:
  - Architecture: $2 \to 3 \to 1$ ($2$ inputs, $3$ hidden neurons with `tanh`, $1$ output with `tanh`).
  - **13 trainable parameters** ($W_1: 6, b_1: 3, W_2: 3, b_2: 1$).
  - Gradients computed via analytical backpropagation.

### 3.2 Empirical Comparison
| Model | Trainable Parameters | Train Accuracy | Test Accuracy | Final MSE Loss | Training Time (40 epochs) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Variational Quantum Classifier (VQC)** | **8** | **91.7%** | **95.0%** | **0.4359** | 17.58 s |
| **Classical MLP** | 13 | 86.7% | 77.5% | 0.4652 | **0.01 s** |

#### Key Insights:
1. **Parameter Efficiency & Expressibility**: With only 8 variational angles, the VQC achieved **95.0% test accuracy** on the non-linear concentric circles dataset, outperforming the 13-parameter classical MLP (77.5% test accuracy). The quantum ZZ-feature map maps non-linear circular geometries into orthogonal hyperplanes in the 4-dimensional Hilbert space.
2. **Computational Footprint**: On classical simulation hardware, parameter-shift training requires $2 \times P = 16$ circuit evaluations per gradient step, causing VQC training to take 17.58 seconds compared to 10 milliseconds for vectorized backpropagation.

---

## 4. Verification & Unit Tests
The test suite in [`tests/test_quantum_algos.py`](file:///c:/Users/Yash%20Deshmukh/Documents/Coding%20Notes/Projects/OTLAB08/tracks/I_quantum_algos_qml/tests/test_quantum_algos.py) passes 8/8 tests in 5.68 seconds:
- `test_deutsch_jozsa_classification`: PASSED
- `test_bernstein_vazirani_recovery`: PASSED
- `test_grovers_search_success_rate`: PASSED
- `test_qft_and_inverse_qft_unitarity`: PASSED
- `test_qpe_exact_phases`: PASSED
- `test_shors_factoring_15`: PASSED
- `test_vqc_parameter_shift_gradient`: PASSED
- `test_vqc_non_linear_classification`: PASSED

---

## 5. Artifact Manifest
- [`quantum_circuits.py`](file:///c:/Users/Yash%20Deshmukh/Documents/Coding%20Notes/Projects/OTLAB08/tracks/I_quantum_algos_qml/quantum_circuits.py): Vectorized quantum circuit simulator with elementary, controlled, QFT, and Grover operators.
- [`fundamental_algos.py`](file:///c:/Users/Yash%20Deshmukh/Documents/Coding%20Notes/Projects/OTLAB08/tracks/I_quantum_algos_qml/fundamental_algos.py): Implementations of DJ, BV, Grover, QPE, and Shor's order finding.
- [`vqc_qml.py`](file:///c:/Users/Yash%20Deshmukh/Documents/Coding%20Notes/Projects/OTLAB08/tracks/I_quantum_algos_qml/vqc_qml.py): Variational Quantum Classifier with ZZ-feature map and Parameter-Shift gradients vs Classical MLP.
- [`benchmarks.py`](file:///c:/Users/Yash%20Deshmukh/Documents/Coding%20Notes/Projects/OTLAB08/tracks/I_quantum_algos_qml/benchmarks.py): Scalability benchmarks and plot generator.
- [`results/grover_scaling_crossover.png`](file:///c:/Users/Yash%20Deshmukh/Documents/Coding%20Notes/Projects/OTLAB08/tracks/I_quantum_algos_qml/results/grover_scaling_crossover.png): Grover query scaling $\mathcal{O}(\sqrt{N})$ vs classical $\mathcal{O}(N)$ and simulator wall-clock reality.
- [`results/vqc_vs_mlp_decision_boundary.png`](file:///c:/Users/Yash%20Deshmukh/Documents/Coding%20Notes/Projects/OTLAB08/tracks/I_quantum_algos_qml/results/vqc_vs_mlp_decision_boundary.png): Decision boundaries and MSE training loss trajectories.
- [`results/qft_qpe_shor_diagnostics.png`](file:///c:/Users/Yash%20Deshmukh/Documents/Coding%20Notes/Projects/OTLAB08/tracks/I_quantum_algos_qml/results/qft_qpe_shor_diagnostics.png): Measurement probability spectrum and period extraction for $N=15$.
- [`results/quantum_algos_benchmark_results.json`](file:///c:/Users/Yash%20Deshmukh/Documents/Coding%20Notes/Projects/OTLAB08/tracks/I_quantum_algos_qml/results/quantum_algos_benchmark_results.json): Raw measured metrics.
