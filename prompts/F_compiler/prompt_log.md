# Prompt Log: Track F (Compiler Optimization & Problem Formulations)

## Workflow & Meta-Information
- **Track**: Track F — Compiler Optimization
- **Role**: Antigravity Assistant & Student Pair-Programmer (Yash Deshmukh, Roll No. 241070018)
- **Objective**: Design, implement, optimize, and benchmark a complete Three-Address Code (TAC) Intermediate Representation (IR) compiler and virtual machine with a Toy Source Language frontend, hardware-calibrated cycle cost model, 6 classical/advanced passes, 3 Chapter 6 optimization formulations, and a 100-test differential testing suite.

---

## Interactive Prompt History

### Prompt F-1: Toy Source Language Frontend (Lexer, Parser, AST, Lowering)
**User/Task Prompt**:
> "Add a toy source language with lexer, parser (AST), lowering to TAC and a code generator, and write the benchmark programs in the toy language. Support variable declarations, while loops, if-else branches, binary arithmetic, and returns."

**AI Response Summary**:
- Implemented `tracks/F_compiler/toy_language.py` with:
  - Lexer tokenizing numbers, identifiers, operators, keywords (`var`, `while`, `if`, `else`, `return`), and comments (`//`).
  - Recursive descent `Parser` generating AST nodes (`VarDecl`, `Assignment`, `WhileStatement`, `IfStatement`, `BinaryExpr`).
  - `TACLowerer` lowering AST into Three-Address Code instructions (`TACInstr`).

**Verification / Human Audit**:
- Tested frontend with while loops and conditional blocks. Fixed syntax error handling for comments.

---

### Prompt F-2: Hardware Opcode Execution Cost Model & VM Integer Semantics
**User/Task Prompt**:
> "Document the cycle cost model per opcode, use it consistently, and also report wall-clock VM time and code size. Check strength reduction and constant folding match the VM's integer semantics (negative division, shifts, overflow); only apply transformations where valid."

**AI Response Summary**:
- In `tracks/F_compiler/ir_compiler.py`:
  - Documented and calibrated `OPCODE_CYCLES`: ALU/Bitwise (1 cycle), Multiply (3 cycles), Divide/Mod (10 cycles), Branches (2 cycles), Labels (0 cycles).
  - Enforced C-style integer truncation towards zero: `int(v1 / v2)`.
  - Preserved runtime division-by-zero exceptions (constant folding prohibited from folding $x / 0$).
  - Prohibited signed $x / 2^k \to x \gg k$ transformations due to truncation vs floor right shift divergence on negative integers.
  - Added wall-clock timing using `time.perf_counter()`.

---

### Prompt F-3: Optimization Passes (LICM & Guarded Loop Unrolling)
**User/Task Prompt**:
> "Implement a real loop unrolling pass using the optimal factor from the formulation, and add LICM if feasible."

**AI Response Summary**:
- In `tracks/F_compiler/passes.py`:
  - Added `LoopInvariantCodeMotionPass`: Identifies instructions inside while loops whose operands are invariant, hoisting them into pre-headers.
  - Added `LoopUnrollPass`: Replicates loop body by factor $u$ with intermediate condition checks (`JUMP_IF_ZERO`), ensuring non-multiple trip counts do not overshoot.
  - Added multi-level optimization driver: `-O0`, `-O1`, `-O2`, `-O3`.

**Verification / Human Audit**:
- Caught bug **FAIL-06** in DCE: single backward pass pruned loop updates (`iter = iter - 1`). Fixed by adding fixed-point liveness iteration.
- Caught bug **FAIL-07** in CSE: invalidation ran after insertion, evicting the newly computed entry. Fixed by reordering invalidation before insertion.

---

### Prompt F-4: Differential Testing Suite
**User/Task Prompt**:
> "Add differential testing: a random program generator comparing -O0 vs -O3 outputs over many random inputs, including branches and loops (CSE/DCE across blocks)."

**AI Response Summary**:
- Implemented `tracks/F_compiler/differential_testing.py`:
  - Generates randomized programs combining arithmetic, CSE, strength reduction, conditional branches, dead code, and while loops.
  - Executed 50 to 100 randomized test cases comparing `-O0` and `-O3`.
  - Verified 100% output identity (0.00% divergence rate).

---

### Prompt F-5: Formulations & Comparison against Baselines
**User/Task Prompt**:
> "Name the Chapter 6 method for each formulation (phase ordering, unroll factor, register allocation) and compare against exhaustive search and a greedy baseline; compare Kempe coloring to exact coloring and show spills when K < chromatic number."

**AI Response Summary**:
- In `tracks/F_compiler/formulation_opt.py`:
  - **Phase Ordering**: Formulated as Dynamic Programming & Branch-and-Bound (Chapter 6). Compared against Exhaustive Permutation Search and Greedy Selection.
  - **Loop Unroll Factor**: Formulated as 0/1 Knapsack / Integer Linear Programming (ILP) (Chapter 6). Compared against Exhaustive Search and Greedy Maximum-Fit.
  - **Register Allocation**: Formulated as Chaitin-Kempe Degree-Ordered Graph Coloring (Chapter 6). Compared against Exact Backtracking Coloring and Naive First-Fit.
  - **Spill Proof**: Constructed $K_4$ complete interference graph ($\chi(K_4)=4$). Demonstrated that for $K \in \{2, 3\}$, both Kempe and Exact require 2 and 1 spills, while for $K \ge 4$, 0 spills occur.

---

### Prompt F-6: Larger Realistic Benchmarks & Accurate Compile Time
**User/Task Prompt**:
> "Add larger benchmarks (nested loops, 100+ instructions) and report speedups honestly; do not headline trivial constant chains. Measure compile time with perf_counter and repeated runs on the larger programs."

**AI Response Summary**:
- Implemented 4 realistic kernels in `benchmarks.py`:
  1. Matrix-Vector Dot Product (Nested while loops, LICM, CSE, strength reduction)
  2. Polynomial Evaluation (4th-order polynomial with power-of-2 multiplications)
  3. Box Filter Stencil (1D moving window with division and clipping)
  4. Prime Factor Search (Nested search loops with modulo arithmetic)
- Measured compile time via 200 repeated runs of `time.perf_counter()`.
- Generated `compiler_benchmark_results.json`, `larger_benchmarks_speedup.png`, and `register_spill_chromatic_bound.png`.
