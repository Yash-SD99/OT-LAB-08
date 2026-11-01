# Track G: xv6 Operating System Optimization

> **Mandatory Rule Compliance**: Because xv6 requires a 64-bit RISC-V QEMU emulator environment and cannot run natively on the development laptop, all quantitative xv6 benchmark metrics below are marked **"NOT YET MEASURED on Host (To be benchmarked in GitHub Codespaces)"**. No benchmark numbers are fabricated.

---

## 1. Problem Formulation & Operating System Architecture

The operating system scheduler governs the multiplexing of hardware CPU cores among competing user processes. In the baseline MIT xv6 operating system:
- **Baseline Scheduler**: A naive Round-Robin (RR) scheduler that iterates sequentially through the fixed process table (`proc[NPROC]`).
- **Deficiencies of Baseline**:
  1. *No Priority or Resource Differentiation*: Every runnable process receives an identical time slice regardless of importance, latency requirements, or user priority.
  2. *Lack of Observability*: xv6 maintains no record of cumulative process CPU execution time, waiting latency, or context-switch counts.

### 1.1 Mathematical Formulation of Proportional-Share Lottery Scheduling
We formulate CPU resource allocation as a weighted proportional-share optimization problem (Waldspurger & Weihl, 1994):
$$\max_{\{T_i\}} \sum_{i \in \mathcal{R}} w_i \log(T_i) \quad \text{subject to} \quad \sum_{i \in \mathcal{R}} T_i \le T_{\text{total}}$$
where:
- $\mathcal{R}$ is the set of active `RUNNABLE` processes.
- $w_i > 0$ represents the lottery tickets (or priority weight) allocated to process $i$.
- Total active tickets:
  $$W = \sum_{j \in \mathcal{R}} w_j$$
- The probability that process $i$ is chosen to execute during any scheduling quantum is:
  $$P(i) = \frac{w_i}{W}$$
- By the Law of Large Numbers, the expected CPU time share allocated to process $i$ over $N$ scheduling rounds asymptotically converges to:
  $$\lim_{N \to \infty} \frac{T_i}{\sum_j T_j} = \frac{w_i}{W}$$

---

## 2. Implemented Kernel Features (xv6-riscv)

The implementation is packaged as a clean, unified patch against the MIT xv6-riscv master branch (`patches/0001_lottery_scheduler_and_schedstat.patch`):

### Feature 1: Proportional-Share Lottery Scheduler (`kernel/proc.c`)
- **Per-Process State**: Added `tickets`, `ticks_run`, `ticks_wait`, and `sched_count` to `struct proc` in `kernel/proc.h`.
- **Initialization**: Newly allocated processes in `allocproc()` receive default tickets ($10$) and zeroed tick counters. On `fork()`, child processes inherit the parent's ticket allocation.
- **Lottery Algorithm**:
  1. Computes total tickets $W = \sum_{p \in \mathcal{R}} p\text{->tickets}$.
  2. Generates a pseudo-random winning ticket $r \in [0, W - 1]$ using a fast linear congruential generator (LCG).
  3. Scans runnable processes, accumulating tickets until the counter exceeds $r$, context-switching to the winner and incrementing `sched_count`.
- **Clock Accounting (`kernel/trap.c`)**: On each timer interrupt (`which_dev == 2`), the executing process's `ticks_run` counter is atomically incremented before yielding.

### Feature 2: Scheduling Statistics System Call (`sys_schedstat`)
As required by Amendment 5, a new system call enables user-space observability:
```c
struct schedstat {
  int pid;         // Process ID
  int tickets;     // Lottery tickets held
  int ticks_run;   // Cumulative clock ticks spent executing on CPU
  int ticks_wait;  // Cumulative clock ticks spent waiting in RUNNABLE state
  int sched_count; // Number of times context-switched onto CPU
};

int schedstat(struct schedstat *st);
int settickets(int tickets);
```
- Safe copyout via `copyout(p->pagetable, addr, (char*)&st, sizeof(st))`.

---

## 3. Benchmark Workload Programs (`tracks/G_xv6/user/`)

Three specialized workload programs evaluate scheduler behavior across distinct operating regimes:

1. **`workload_cpu.c` (CPU-Bound)**:
   - Computes 500 iterations of $32 \times 32$ integer matrix multiplication.
   - Evaluates pure compute throughput and execution ticks under assigned ticket weights.
2. **`workload_io.c` (I/O-Bound)**:
   - Executes 50 interactive rounds of pipe write, sleep yields (`sleep(1)`), and pipe reads.
   - Evaluates turnaround latency and waiting time under voluntary CPU yields.
3. **`workload_mixed.c` (Mixed CPU & I/O)**:
   - Executes 30 alternating phases (70% CPU arithmetic bursts, 30% I/O sleep).
   - Models realistic multitasking workloads.
4. **`sched_bench.c` (Fairness Harness)**:
   - Spawns 3 concurrent workers with ticket ratios:
     - Worker 1: **10 tickets** ($16.7\%$ expected share)
     - Worker 2: **20 tickets** ($33.3\%$ expected share)
     - Worker 3: **30 tickets** ($50.0\%$ expected share)
   - Queries `schedstat` upon worker termination to measure empirical share vs theoretical ratio ($1 : 2 : 3$).

---

## 4. Benchmark Measurement Plan & Status

| Workload Benchmark | Metric Target | Theoretical Expectation | Measured Result | Status |
|---|---|:---:|:---:|:---:|
| **Kernel Integrity (`usertests`)** | System call & memory regression | `ALL TESTS PASSED` | **NOT YET MEASURED** | To be run in Codespaces |
| **CPU Workload (`workload_cpu 10`)** | Execution ticks & context switches | Proportional to compute loop | **NOT YET MEASURED** | To be run in Codespaces |
| **CPU Workload (`workload_cpu 30`)** | Execution ticks & context switches | Priority acceleration | **NOT YET MEASURED** | To be run in Codespaces |
| **I/O Workload (`workload_io 30`)** | Sleep vs wait tick ratio | Immediate interactive dispatch | **NOT YET MEASURED** | To be run in Codespaces |
| **Mixed Workload (`workload_mixed 20`)** | Total elapsed ticks | Fair time-sharing | **NOT YET MEASURED** | To be run in Codespaces |
| **Proportional Share Worker 1** | CPU Share ($10 / 60$ tickets) | $\mathbf{16.7\%}$ | **NOT YET MEASURED** | To be run in Codespaces |
| **Proportional Share Worker 2** | CPU Share ($20 / 60$ tickets) | $\mathbf{33.3\%}$ | **NOT YET MEASURED** | To be run in Codespaces |
| **Proportional Share Worker 3** | CPU Share ($30 / 60$ tickets) | $\mathbf{50.0\%}$ | **NOT YET MEASURED** | To be run in Codespaces |

---

## 5. Execution Guide (GitHub Codespaces)

```bash
# 1. Run automated setup in Codespaces terminal
chmod +x tracks/G_xv6/setup_codespaces.sh
./tracks/G_xv6/setup_codespaces.sh

# 2. Launch xv6 in QEMU
cd $HOME/xv6-riscv
make qemu

# 3. Inside xv6 shell:
$ usertests
$ sched_bench
$ workload_cpu 20
$ workload_io 30
$ workload_mixed 20

# 4. Exit QEMU: Press 'Ctrl-a' then 'x'
```
