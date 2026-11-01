# Prompt Log: Track C (Quantum Optimization)

**Student**: Yash Deshmukh (Roll No. 241070018)  
**Track**: Track C — Quantum Optimization (QAOA, VQE, PQC, Parameter-Shift, SPSA, QNG)  
**Tool Used**: Antigravity (Google DeepMind)  
**Date**: Nov 1, 2026  

---

## Session Trajectory

### 1. Initial Prompt
```text
Implement a complete quantum optimization suite for Track C:
1. Quantum state-vector simulator from scratch in NumPy:
   - N-qubit state vector |psi> in C^{2^N}
   - Gates: H, X, Y, Z, Rx, Ry, Rz, CNOT, CZ, Rzz
   - Expectation values, Born-rule multinomial shot sampling, and depolarizing hardware noise channel.
2. QAOA Max-Cut solver:
   - Solves the Track B Max-Cut QUBO across depths p in [1, 2, 3].
   - Compares approximation ratio and ground-state sampling probability against Track B classical baselines.
3. VQE:
   - Hardware-efficient ansatz for Transverse-Field 1D Ising model.
   - Compares ground state energy against exact classical diagonalization.
4. Quantum Gradient Methods:
   - Parameter-shift rule exact analytical derivatives.
   - SPSA (2 evaluations per step).
   - Quantum Natural Gradient (Fubini-Study metric tensor).
5. Empirical Noise & Scaling Analysis:
   - Finite shot noise variance (1/S).
   - Depolarizing gate error degradation.
   - Barren Plateaus gradient variance collapse across N in [2..7] qubits.
Mark all outputs as "noise-model simulation, not real hardware".
```

### 2. AI Response Summary & Generated Code
The AI generated:
* `tracks/C_quantum_opt/quantum_simulator.py`
* `tracks/C_quantum_opt/qaoa_maxcut.py`
* `tracks/C_quantum_opt/vqe_ising.py`
* `tracks/C_quantum_opt/pqc_qnn.py`
* `tracks/C_quantum_opt/noise_and_barren.py`
* `tracks/C_quantum_opt/tests/test_quantum_opt.py`
* `tracks/C_quantum_opt/benchmarks.py`

### 3. Verification & Bug Rectification
* **Execution**: Ran `test_quantum_opt.py` and `benchmarks.py`.
* **Issue Discovered**: In `noise_and_barren.py`, `Optional` was missing from `from typing import Dict, List, Tuple`. In `quantum_simulator.py`, `apply_depolarizing_noise` re-instantiated a local RNG with `seed=seed` inside each gate call, causing the first pseudo-random float `0.7739` to repeat, preventing noise from being applied for $p < 0.77$.
* **Rectification**: Added `Optional` to typing imports. Added persistent `self.rng = np.random.default_rng(seed)` to `QuantumCircuitSimulator` to allow continuous generator state progression across gates.
* **Verification Note**: Re-running tests confirmed all 5 unit tests passed. Re-running benchmarks confirmed depolarizing noise degradation and Barren Plateau variance decay ($0.281 \to 0.014$).

### 4. Chat Export / Screenshot Placeholder
*(Student Note: Place chat export or UI screenshot in `prompts/C_quantum_opt/screenshot_track_C.png`)*.
