# AI Coding Failure & Rectification Log

As mandated by Section 8 of the assignment, this log records authentic cases where AI-generated code was incorrect, unsafe, mathematically flawed, inefficient, or introduced regressions during the development of Tracks A through J. 

Every case documented below was caught by automated unit tests or mathematical invariant assertions, investigated, corrected, and linked to a key lesson learned.

---

## Summary Table

| ID | Track | Failure Classification | Root Cause | Detection Mechanism | Resolution |
|---|:---:|---|---|---|---|
| **FAIL-01** | Track A (Continuous) | Mathematical / Operator Precedence Error | Flawed parentheses in `f_opt = 0.5 * float(x_opt @ A @ x_opt - b @ x_opt)` scaled the linear term by 0.5, causing theoretical minimum to evaluate to 0.0 instead of $-0.5 b^T A^{-1} b$, yielding negative optimality gaps. | Automated verification: $f(x) - f^* < 0$ caught in sensitivity benchmark output. | Corrected grouping to `0.5 * (x_opt @ A @ x_opt) - (b @ x_opt)` ensuring mathematical equivalence to $-0.5 b^T x^*$. |
| **FAIL-02** | Track B (Discrete / QUBO) | Algebraic Mapping Error | In `qubo_to_ising`, the constant diagonal offset contribution was set to `C += 0.25 * Q[i, i]` instead of `0.5 * Q[i, i]`, violating energy conservation between binary and spin bases. | Unit test assertion failure: `test_qubo_ising_conversion_identity` reported discrepancy between $x^T Q x$ and $H_{\text{Ising}}(s)$. | Corrected diagonal constant accumulation to `C += 0.5 * Q_sym[i, i]`, achieving machine-precision equivalence ($< 10^{-16}$). |
| **FAIL-03** | Track B (Discrete / B&B) | Algorithmic Bounding Logic Error | In custom Branch-and-Bound, `fractional_bound` returned `0.0` when `curr_wt >= capacity`. When an item combination exactly filled capacity (`curr_wt == capacity`), the upper bound was returned as 0.0 instead of `curr_val`, causing the optimal incumbent to be pruned prematurely. | Known-instance test failure: B&B returned 240.0 while DP and `scipy.optimize.milp` found 270.0. | Corrected boundary logic to return `curr_val` when `curr_wt == capacity` and prune only when strictly exceeding capacity. |
| **FAIL-04** | Track C (Quantum Opt) | Random Number Generator State Reset | In `apply_depolarizing_noise`, a local `rng = np.random.default_rng(seed)` was instantiated on every gate call. Since the seed was identical each gate, the random draw was always the same value (0.7739), meaning noise was never applied when error rate $p < 0.77$. | Test failure in `test_depolarizing_noise_fidelity_decay`: fidelity under noise $p=0.1$ remained identical to $1.0$. | Moved `rng` to persistent state on `QuantumCircuitSimulator` instance (`self.rng`), advancing the PRNG sequence across gates. |
| **FAIL-05** | Track F (Compiler Opt) | Over-conservative Liveness in Dead Code Elimination | In `DeadCodeEliminationPass`, instruction removal was restricted via `if instr.dest in used_vars or not instr.dest.startswith("t_"):`, which prevented non-temporary user variables from ever being pruned even when unread. | Unit test failure: Dead assignment elimination failed to eliminate unused store variables, leaving instruction count unoptimized. | Redefined liveness strictly through backward def-use graph analysis over active variable references. |
| **FAIL-06** | Track F (Compiler Opt) | Loop Back-Edge Liveness Blindness in DCE | Single backward liveness pass failed to propagate variable usage across loop back-edges (`goto start_lbl`), causing induction variable updates (`iter = iter - 1`) to be mistakenly pruned as dead code. | Differential testing failure: 100% divergence in while-loop execution between -O0 and -O3. | Implemented multi-pass backward fixed-point iteration until the live set reaches convergence. |
| **FAIL-07** | Track F (Compiler Opt) | Premature Invalidation of Available Expressions in CSE | In `CommonSubexpressionEliminationPass`, invalidation filtered expressions where `v != d_str` after recording, deleting the newly recorded result temporary immediately upon definition. | Unit test failure: `test_common_subexpression_elimination` showed `ADD` remained instead of reusing previous temporary via `ASSIGN`. | Restructured invalidation to run prior to adding the new expression to the available cache. |
| **FAIL-08** | Track H (DBMS Optimization) | SQLite Freelist Blindness in Storage Footprint Measurement | Measuring candidate index storage size via raw `PRAGMA page_count` before and after `CREATE INDEX` failed because dropped indices left unallocated pages in SQLite's freelist, which subsequent index builds reused without expanding total pages, yielding 0 KB measured storage for 7 out of 10 candidate indices. | Benchmark execution showed 0.0 KB footprint for multiple candidate indices, distorting the 0/1 Knapsack constraints. | Accounted for internal freelist pages by tracking active pages: $\Delta \text{pages} = (\text{page\_count} - \text{freelist\_count})_{\text{after}} - (\text{page\_count} - \text{freelist\_count})_{\text{before}}$. |

---

## Detailed Incident Reports

### Case FAIL-01: Operator Precedence in Quadratic Objective Minimum (Track A)
* **Track Affected**: Track A (Continuous Optimization)
* **Initial Prompt**: "Generate the StronglyConvexQuadratic class with analytical optimum f_opt."
* **Flawed Code Excerpt**:
  ```python
  # Flawed line generated by AI:
  self.f_opt = 0.5 * float(self.x_opt @ self.A @ self.x_opt - self.b @ self.x_opt)
  ```
* **Failure Description & Invariant Violated**:
  For $f(x) = \frac{1}{2} x^T A x - b^T x$, since $A x^* = b$, we have $x^{*T} A x^* = b^T x^*$. 
  The AI factored $0.5$ across the entire difference: $0.5 \times (b^T x^* - b^T x^*) = 0.0$.
  Consequently, $f^*$ was computed as $0.0$ regardless of $A$ and $b$, violating the theoretical property that $f(x^*) = -\frac{1}{2} b^T A^{-1} b$. When algorithms found the true negative minimum, the computed optimality gap $f(x) - f^*$ became negative, which is mathematically impossible for a global minimum of a strongly convex function.
* **How It Was Detected**:
  Inspecting `continuous_benchmark_results.json` revealed negative gap values (`-0.3684`) during the step-size sweep.
* **Corrective Fix**:
  Changed expression to:
  ```python
  self.f_opt = float(0.5 * (self.x_opt @ self.A @ self.x_opt) - (self.b @ self.x_opt))
  ```
* **Lesson Learned & Takeaway for Viva**:
  Never assume scalar objective simplifications in AI-generated code are algebraically correct. Always verify invariants like $f(x) - f^* \ge 0$ for convex functions against known analytical closed-form solutions before trusting benchmark metrics.

---

### Case FAIL-02: Constant Offset Error in QUBO-to-Ising Spin Mapping (Track B)
* **Track Affected**: Track B (Discrete Optimization / QUBO)
* **Initial Prompt**: "Write a function qubo_to_ising(Q) to convert QUBO matrix to Ising Hamiltonian (J, h, C)."
* **Flawed Code Excerpt**:
  ```python
  for i in range(N):
      h[i] += 0.5 * Q_sym[i, i]
      C += 0.25 * Q_sym[i, i]  # FLAWED: Divided by 4 instead of 2
  ```
* **Failure Description & Invariant Violated**:
  When substituting $x_i = \frac{s_i + 1}{2}$ into the diagonal term $Q_{ii} x_i^2 = Q_{ii} x_i$:
  $$Q_{ii} \frac{s_i + 1}{2} = \frac{1}{2} Q_{ii} s_i + \frac{1}{2} Q_{ii}$$
  The constant contribution is $\frac{1}{2} Q_{ii}$, not $\frac{1}{4} Q_{ii}$. The AI confused the linear substitution with the quadratic cross-term substitution $\frac{s_i+1}{2} \frac{s_j+1}{2} = \frac{1}{4} (s_i s_j + s_i + s_j + 1)$. Consequently, the baseline Ising energy shifted away from the true QUBO energy by $\frac{1}{4} \sum_i Q_{ii}$.
* **How It Was Detected**:
  Unit test `test_qubo_ising_conversion_identity` asserted $x^T Q x == H_{\text{Ising}}(s)$ for random states. The test failed with `AssertionError: 0.0 != -0.21637`.
* **Corrective Fix**:
  Updated the accumulation to `C += 0.5 * Q_sym[i, i]`. Re-running the test verified exact zero difference ($0.00 \times 10^{-16}$).
* **Lesson Learned & Takeaway for Viva**:
  Quantum mapping transformations must preserve the exact Hamiltonian spectrum. A constant energy shift in the classical mapping propagates directly into biased phase rotations in QAOA circuits in Track C.

---

### Case FAIL-03: Boundary Misjudgment in Branch-and-Bound Linear Relaxation (Track B)
* **Track Affected**: Track B (Discrete Optimization / Custom B&B)
* **Initial Prompt**: "Implement a custom Branch-and-Bound solver for 0/1 Knapsack with fractional linear relaxation bounding."
* **Flawed Code Excerpt**:
  ```python
  def fractional_bound(level, curr_val, curr_wt):
      if curr_wt >= instance.capacity:
          return 0.0  # FLAWED: When curr_wt == capacity, upper bound is curr_val, not 0.0!
  ```
* **Failure Description & Invariant Violated**:
  In Branch-and-Bound, an upper bound must be an *admissible* (upper) bound on the achievable total objective from that subtree. If `curr_wt == capacity`, the sack is exactly full, so zero *additional* weight can be packed, meaning the maximum possible value achievable from this node is exactly `curr_val`. The AI conflated "additional achievable value is 0" with "total upper bound is 0". When a path perfectly packed the knapsack to capacity (50.0), its upper bound was set to 0.0, causing the node to be immediately pruned because $0.0 \le \text{best\_value}$.
* **How It Was Detected**:
  Unit test `test_knapsack_exact_solvers_equivalence` compared DP, Custom B&B, and `scipy.optimize.milp` on an instance with values $[60, 100, 120, 80, 30]$ and weights $[10, 20, 30, 15, 5]$ with capacity $50$. DP and ILP found optimal value $270.0$ (packing items $0, 1, 3, 4$ with total weight $50.0$), but B&B returned $240.0$.
* **Corrective Fix**:
  Corrected the boundary condition:
  ```python
  if curr_wt > instance.capacity:
      return 0.0
  if curr_wt == instance.capacity:
      return float(curr_val)
  ```
* **Lesson Learned & Takeaway for Viva**:
  Branch-and-Bound correctness hinges entirely on the admissibility of the bounding function ($\text{bound}(u) \ge f(x)$ for all leaves $x$ in subtree $u$). Underestimating the bound at boundary conditions breaks admissibility and silently prunes the global optimum.

---

### Case FAIL-04: PRNG Re-initialization Defeating Quantum Noise Simulation (Track C)
* **Track Affected**: Track C (Quantum Optimization / Noise)
* **Initial Prompt**: "Implement a local depolarizing error channel in QuantumCircuitSimulator."
* **Flawed Code Excerpt**:
  ```python
  def apply_depolarizing_noise(self, qubit: int, p: float):
      rng = np.random.default_rng(self.seed) # FLAWED: fresh RNG reset on every gate
      if rng.random() < p:
          # apply random Pauli X, Y, or Z
  ```
* **Failure Description & Invariant Violated**:
  Because `rng = np.random.default_rng(self.seed)` was called inside `apply_depolarizing_noise` for every single gate, the first call to `rng.random()` always yielded the exact same pseudo-random float: $0.773956$. For any realistic gate error rate $p < 0.77$ (such as $p = 0.05$ or $p = 0.10$), the conditional `rng.random() < p` was always False. The noise channel was thus silently skipped on 100% of gates.
* **How It Was Detected**:
  In `test_depolarizing_noise_fidelity_decay`, the output state fidelity under noise rate $p = 0.1$ was asserted to be $< 0.99$. The test failed because the fidelity was exactly $1.00000000$.
* **Corrective Fix**:
  Created a persistent `self.rng = np.random.default_rng(seed)` attribute on `QuantumCircuitSimulator` so that PRNG state advances deterministically and continuously across gate operations.
* **Lesson Learned & Takeaway for Viva**:
  Stateless PRNG calls in looped or event-driven simulations will repeatedly sample the same prefix of pseudo-random numbers, turning stochastic models completely deterministic. Always manage PRNG states explicitly as instance members.

---

### Case FAIL-05: Over-conservative Liveness in Dead Code Elimination (Track F)
* **Track Affected**: Track F (Compiler Optimization / DCE Pass)
* **Initial Prompt**: "Implement DeadCodeEliminationPass in IR optimization pipeline."
* **Flawed Code Excerpt**:
  ```python
  if instr.dest in used_vars or not instr.dest.startswith("t_"):
      new_instructions.append(instr) # FLAWED: assumed non-t_ vars are external outputs
  ```
* **Failure Description & Invariant Violated**:
  The AI attempted to prevent pruning user variables by checking `not instr.dest.startswith("t_")`. In a compiler intermediate representation, any dead variable assignment (e.g. `x = 5` followed by no reads before function return) is candidate dead code unless it affects exported return registers or observable side-effects. The heuristic prevented dead stores to user-named variables from ever being eliminated, reducing optimization efficiency.
* **How It Was Detected**:
  In `test_output_equivalence_across_optimizations`, a dead assignment benchmark had 8 instructions. After DCE, it still had 8 instructions instead of shrinking to 4 instructions, failing the assertion `len(opt_ir) < len(ir)`.
* **Corrective Fix**:
  Formalized backward liveness analysis: only the designated return variable and variables referenced in downstream expressions remain in the live set.
* **Lesson Learned & Takeaway for Viva**:
  Heuristic shortcuts (like checking string prefixes) fail compiler invariants. Compilers require formal dataflow analysis (like backward reaching definitions and live-variable analysis) rather than naming conventions.

---

### Case FAIL-06: Loop Back-Edge Blindness in Dead Code Elimination (Track F)
* **Track Affected**: Track F (Compiler Optimization / Liveness Analysis)
* **Initial Prompt**: "Implement a backward liveness analysis pass to remove dead code in DeadCodeEliminationPass."
* **Flawed Code Excerpt**:
  ```python
  used_vars = set()
  # Single backward pass:
  for instr in reversed(instructions):
      if instr.op in ("PRINT", "RETURN", "JUMP_IF_ZERO"):
          used_vars.add(instr.src1)
      elif instr.dest in used_vars:
          used_vars.add(instr.src1)
  ```
* **Failure Description & Invariant Violated**:
  In a program with a `while` loop, the induction variable increment (`iter = iter - 1`) appears near the bottom of the loop body, while its usage (`t_cond = iter > 0`) appears earlier at the top of the loop. In a single backward sweep starting from `return res`, `iter` is not yet live when the sweep passes `iter = iter - 1`. Because the backward jump `goto while_start` was not treated as a control-flow cycle propagating liveness backwards, `iter = iter - 1` was classified as dead code and completely stripped, freezing loop counters in infinite or divergent states.
* **How It Was Detected**:
  Differential testing against 50 randomized programs resulted in immediate divergence between `-O0` and `-O3` outputs (`O0=[92], O3=[82]`), with inspection revealing `iter = t_14` missing from the optimized TAC.
* **Corrective Fix**:
  Implemented fixed-point liveness iteration: the backward sweep repeats over the instruction sequence until `len(used_vars)` converges to a fixed point, correctly capturing dependencies across loop back-edges.
* **Lesson Learned & Takeaway for Viva**:
  Control-flow loops introduce cyclic dataflow dependencies. A single linear pass across basic blocks can never correctly determine liveness; compiler dataflow analyses must iterate to a fixed point across all back-edges.

---

### Case FAIL-07: Premature Available Expression Invalidation in CSE (Track F)
* **Track Affected**: Track F (Compiler Optimization / CSE Pass)
* **Initial Prompt**: "Implement CommonSubexpressionEliminationPass with basic block cache invalidation."
* **Flawed Code Excerpt**:
  ```python
  # Add new expression:
  available_expressions[expr_key] = str(instr.dest)
  optimized.append(instr)

  # FLAWED: Invalidated immediately after adding!
  if instr.dest:
      d_str = str(instr.dest)
      available_expressions = {
          k: v for k, v in available_expressions.items()
          if k[1] != d_str and k[2] != d_str and v != d_str # 'v != d_str' killed newly inserted entry!
      }
  ```
* **Failure Description & Invariant Violated**:
  When an instruction like `t1 = x + y` was evaluated, `expr_key = ('ADD', 'x', 'y')` was stored in `available_expressions` with value `v = "t1"`. Immediately afterward, the destination invalidator ran with `d_str = "t1"`. The predicate `v != d_str` evaluated to `False` for the freshly added entry, instantly purging it from the dictionary. Consequently, subsequent identical expressions like `t2 = x + y` could never find the cached value.
* **How It Was Detected**:
  Unit test `test_common_subexpression_elimination` failed with `AssertionError: 'ADD' != 'ASSIGN'`.
* **Corrective Fix**:
  Moved the operand invalidation step to run *prior* to inserting the new expression, ensuring previous obsolete values are pruned while newly computed expressions remain cached.
* **Lesson Learned & Takeaway for Viva**:
  Ordering of cache invalidation vs insertion in stateful optimization passes is critical. Invalidating after insertion risks evicting the newly computed state.

---

### Case FAIL-08: SQLite Freelist Blindness in Index Storage Profiling (Track H)
* **Track Affected**: Track H (Relational DBMS Optimization / Index Selection)
* **Initial Prompt**: "Measure index storage size using PRAGMA page_count before and after CREATE INDEX."
* **Flawed Code Excerpt**:
  ```python
  p_before = cur.execute("PRAGMA page_count;").fetchone()[0]
  cur.execute(info["sql"]) # CREATE INDEX
  conn.commit()
  p_after = cur.execute("PRAGMA page_count;").fetchone()[0]
  cur.execute(f"DROP INDEX {idx_name};")
  size_bytes = (p_after - p_before) * page_size
  ```
* **Failure Description & Invariant Violated**:
  When SQLite drops an index (`DROP INDEX`), it does not immediately truncate or shrink the database file. Instead, it marks the allocated pages as free, moving them to an internal *freelist*. When the loop iterated to evaluate the *next* candidate index, SQLite fulfilled the new B-tree page allocations directly from the existing freelist rather than allocating new pages from the operating system. Consequently, `p_after` equaled `p_before`, and `(p_after - p_before)` evaluated to `0`, falsely reporting that 7 out of 10 candidate indices occupied 0 bytes of disk storage. This severely corrupted the 0/1 Knapsack ILP capacity constraints.
* **How It Was Detected**:
  Inspecting benchmark logs revealed 0.0 KB footprints for major indexes like `idx_orders_cust` and `idx_order_items_oid` despite them indexing 50,000 to 100,000 rows.
* **Corrective Fix**:
  Queried `PRAGMA freelist_count;` and computed the net *active* pages:
  $$\text{active\_pages} = \text{page\_count} - \text{freelist\_count}$$
  $$\Delta \text{pages} = \text{active\_after} - \text{active\_before}$$
  This accurately isolates the exact page footprint consumed by the index regardless of whether pages were drawn from the freelist or the OS.
* **Lesson Learned & Takeaway for Viva**:
  Storage engines decouple physical OS file size from logical database allocations via page freelists. Profiling storage requirements in embedded DBMSs requires querying internal free-space tracking (`PRAGMA freelist_count` or `VACUUM`) rather than raw file length.

---

### Case FAIL-09: QPE Bit-Endianness Inversion in Shor's Order Finding (Track I)
* **Track Affected**: Track I (Quantum Algorithms & QML / QPE & Shor's Algorithm)
* **Initial Prompt**: "Implement Quantum Phase Estimation and apply it to modular exponentiation in Shor's algorithm for factoring N = 15."
* **Flawed Code Excerpt**:
  ```python
  # In QuantumPhaseEstimation:
  for j in range(self.m):
      counting_qubit = j
      angle = 2.0 * np.pi * self.theta * (2**j)  # FLAW: assumed qubit j is LSB (2^j)
      qc.cphase(counting_qubit, self.target_qubit, angle)
  ```
* **Failure Description & Invariant Violated**:
  In our vectorized statevector simulator, qubit indexing is big-endian where qubit 0 is the most significant bit (MSB, binary place value $2^{m-1}$) and qubit $m-1$ is the least significant bit (LSB, place value $2^0$). The initial QPE loop applied controlled phase rotations $U^{2^j}$ to qubit $j$, mistakenly associating qubit $j$ with weight $2^j$. When the subsequent inverse QFT ($QFT^\dagger$) operated on the register, the phases across the computational basis states were bit-reversed and reflected ($1 - \theta$). Consequently, for a true phase of $\theta = 0.25$ (integer peak 4 out of 16), QPE produced peak 14 ($\theta \approx 0.875$). This phase distortion caused continued fractions to fail to recover the denominator $r = 4$, resulting in Shor's algorithm completely failing to find the non-trivial factors of 15.
* **How It Was Detected**:
  Running `ShorsAlgorithmDemo(15, 7, m_counting=4).run()` returned `Factors: [] Found r: None` with peak indices `[0, 15, 1, 14]` instead of `[0, 4, 8, 12]`.
* **Corrective Fix**:
  Corrected the controlled phase exponent to match each qubit's binary place value in big-endian representation:
  ```python
  weight = 2**(self.m - 1 - j)
  angle = 2.0 * np.pi * self.theta * weight
  qc.cphase(counting_qubit, self.target_qubit, angle)
  ```
  Immediately afterward, QPE estimated phases $\theta \in \{0, 0.25, 0.5, 0.75\}$ with probability 1.0, accurately yielding peaks at $\{0, 4, 8, 12\}$ and recovering non-trivial factors $[3, 5]$ of 15.
* **Lesson Learned & Takeaway for Viva**:
  Quantum algorithms (especially Fourier-based methods like QPE and Shor's algorithm) require strict consistency between qubit endianness, binary place values, and inverse Fourier basis conventions. Inverting bit-endianness reflects phase values across the unit circle.

---

*(Additional failure incidents will be appended dynamically as each track's test suite executes)*.
