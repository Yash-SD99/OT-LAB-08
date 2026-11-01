# Prompt Log: Track F (Compiler Optimization)

## Workflow & Meta-Information
- **Track**: Track F — Compiler Optimization
- **Role**: Antigravity Assistant & Student Pair-Programmer (Yash Deshmukh, Roll No. 241070018)
- **Objective**: Design, implement, optimize, and benchmark a complete Three-Address Code (TAC) Intermediate Representation (IR) compiler and virtual machine with 4 classical passes, plus formulation of Phase Ordering (DP/Greedy), Loop Unrolling (ILP/Knapsack), and Register Allocation (Kempe Graph Coloring).

---

## Interactive Prompt History

### Prompt F-1: IR Architecture & Execution Engine
**User/Task Prompt**:
> "Design a Three-Address Code (TAC) Intermediate Representation and VM execution engine in Python with zero external runtime dependencies. Support arithmetic (`+`, `-`, `*`, `/`, `%`, `**`), bitwise (`<<`, `>>`, `&`, `|`, `^`), assignments, conditional branches (`if_false`), unconditional jumps (`goto`), labels, and return instructions. Track instruction execution counts and memory state."

**AI Response Summary**:
- Created `Instruction` dataclass with `op`, `dest`, `arg1`, `arg2`.
- Created `ThreeAddressCode` container and `TACVirtualMachine`.
- Implemented step-by-step interpreter enforcing correct operator semantics, label resolution table, and cycle counters.

**Verification / Human Audit**:
- Tested basic blocks with arithmetic expressions.
- Verified that division by zero raises appropriate runtime errors.

---

### Prompt F-2: Optimization Passes Pipeline
**User/Task Prompt**:
> "Implement four classical compiler optimization passes operating directly on TAC:
> 1. Constant Folding (`ConstantFoldingPass`): evaluate constant sub-expressions at compile time.
> 2. Common Subexpression Elimination (`CommonSubexpressionEliminationPass`): identify and eliminate redundant identical operations across basic blocks.
> 3. Dead Code Elimination (`DeadCodeEliminationPass`): eliminate assignments to variables that are never subsequently read.
> 4. Strength Reduction (`StrengthReductionPass`): convert expensive operations into cheap ones (e.g. $x \times 2^k \to x \ll k$, $x \times 0 \to 0$, $x + 0 \to x$)."

**AI Response Summary**:
- Implemented each pass with a clean `.run(tac)` interface returning a modified `ThreeAddressCode` instance.
- Wrapped into an optimization pipeline orchestrator supporting levels `O0` (none), `O1` (ConstFold + DCE), `O2` (O1 + CSE), and `O3` (O2 + Strength Reduction iterated until fixed point).

**Verification / Human Audit**:
- Caught bug **FAIL-05**: DCE initially had a crude heuristic `not instr.dest.startswith("t_")` which failed to prune user dead variables. Replaced with backward liveness analysis.

---

### Prompt F-3: Optimization Formulations (Phase Ordering, Unrolling, Register Allocation)
**User/Task Prompt**:
> "As required by Track F and Amendment 6:
> 1. Formulate Phase Ordering as a search problem over pass permutations; implement a DP/Greedy pass search that maximizes instruction reduction under compile-time constraints.
> 2. Formulate Loop Unroll Factor as a 0/1 Knapsack / ILP problem balancing speedup against instruction cache budget.
> 3. Formulate Register Allocation as Kempe Graph Coloring over interference graphs of variable live ranges.
> 4. Measure compile time vs runtime speedup trade-offs."

**AI Response Summary**:
- In `formulation_opt.py`:
  - Implemented `search_optimal_pass_ordering` using depth-bounded branch search with compile-time measurement.
  - Implemented `optimize_loop_unroll_factor` modeling instruction bloat vs loop branch reduction using integer linear bounds.
  - Implemented `allocate_registers_graph_coloring` modeling interference graphs and Chaitin-Kempe heuristic coloring with spilling detection.

**Verification / Human Audit**:
- Ran tests in `tests/test_compiler.py`: verified strict output equivalence between unoptimized and optimized code, and confirmed that graph coloring allocates valid, non-interfering register maps.

---

### Prompt F-4: Benchmarking Suite & Artifact Generation
**User/Task Prompt**:
> "Run the full benchmark suite across 5 realistic benchmark kernels (Constant Folding chain, CSE arithmetic diamond, Dead Code elimination loop, Strength Reduction kernel, and Matrix Multiplication loop). Output metrics to JSON and generate comparative bar plots."

**AI Response Summary**:
- Created `benchmarks.py` measuring original instructions, optimized instructions, reduction %, compile time ($\mu\text{s}$), and VM cycles.
- Generated `compiler_benchmark_results.json` and `compiler_pass_impact.png`.
- Documented full results in `tracks/F_compiler/TRACK_README.md`.
