# Prompt Log: Track G (xv6 Operating System Scheduler Optimization)

## Workflow & Meta-Information
- **Track**: Track G — xv6 Operating System Optimization
- **Role**: Antigravity Assistant & Student Pair-Programmer (Yash Deshmukh, Roll No. 241070018)
- **Objective**: Design, formulate, patch, and benchmark a Proportional-Share / Lottery Scheduler and `sys_schedstat` observability system call in MIT xv6-riscv, with automated GitHub Codespaces tooling and 3 user-level workload benchmarks.

---

## Interactive Prompt History

### Prompt G-1: Kernel Architecture & Proportional-Share Formulation
**User/Task Prompt**:
> "xv6: I will use GitHub Codespaces (Linux), not Windows. Give me the exact apt install commands for qemu-system-misc and the RISC-V toolchain, the build steps, the patch/diff, user-level benchmark programs and the test plan. Mark all xv6 results 'NOT YET MEASURED' until I run them. Never fabricate xv6 numbers. Also add a second kernel feature: a syscall for scheduling stats, 3 workloads, and usertests passing."

**AI Response Summary**:
- Formulated CPU scheduling as a proportional-share resource allocation optimization problem:
  $$\max \sum w_i \log(T_i) \implies P(i) = \frac{w_i}{\sum w_j}$$
- Designed Feature 1: Lottery / Priority Scheduler in `kernel/proc.c` replacing default Round-Robin.
- Designed Feature 2: `sys_schedstat` system call copying `struct schedstat` (`pid`, `tickets`, `ticks_run`, `ticks_wait`, `sched_count`) to user space.
- Created `setup_codespaces.sh` with exact apt commands, repository clone, patch application, and build flow.

---

### Prompt G-2: Unified Kernel Patch Creation
**User/Task Prompt**:
> "Generate the complete unified git diff patch for xv6-riscv implementing the lottery scheduler and sys_schedstat, modifying proc.h, proc.c, syscall.h, syscall.c, sysproc.c, trap.c, user.h, usys.pl, and Makefile."

**AI Response Summary**:
- Generated `patches/0001_lottery_scheduler_and_schedstat.patch`:
  - Added lottery fields and counters to `struct proc`.
  - Implemented pseudo-random LCG lottery draw over cumulative runnable tickets in `scheduler()`.
  - Added `sys_settickets` and `sys_schedstat` with safe `copyout`.
  - Added timer accounting in `kernel/trap.c`.
  - Added Makefile entries for all 4 benchmark programs.

---

### Prompt G-3: User-Level Workloads & Fairness Benchmark Harness
**User/Task Prompt**:
> "Write the 3 user-level workloads (CPU-bound, I/O-bound, mixed) and a fairness benchmark harness."

**AI Response Summary**:
- Implemented:
  1. `user/workload_cpu.c`: $32 \times 32$ integer matrix multiplication over 500 iterations.
  2. `user/workload_io.c`: 50 rounds of interactive pipe I/O and sleep yields.
  3. `user/workload_mixed.c`: 30 alternating phases of CPU bursts (70%) and I/O sleeps (30%).
  4. `user/sched_bench.c`: Spawns 3 concurrent processes with tickets $10, 20, 30$ ($1:2:3$), measuring empirical CPU ticks.

---

### Prompt G-4: Test Plan & Honesty Invariant
**User/Task Prompt**:
> "Document the complete test plan and ensure all benchmark numbers are marked NOT YET MEASURED."

**AI Response Summary**:
- Created `test_plan.md` with instructions for running `usertests` in QEMU and executing workloads.
- Created `results/sched_benchmark_results.json` and `TRACK_README.md` with all quantitative metrics explicitly marked `"NOT YET MEASURED on Host (To be benchmarked in GitHub Codespaces)"`.
