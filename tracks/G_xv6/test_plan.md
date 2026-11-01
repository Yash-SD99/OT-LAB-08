# Track G: xv6 Operating System Optimization — Test Plan & Execution Guide

## 1. Environment & Target Machine
- **Host**: GitHub Codespaces (Ubuntu 22.04 / 24.04 LTS, 2 vCPUs, 8 GB RAM).
- **Target Architecture**: RISC-V 64-bit (`riscv64`).
- **Emulator**: QEMU 7.x+ (`qemu-system-riscv64`).
- **Toolchain**: `gcc-riscv64-unknown-elf`, `binutils-riscv64-unknown-elf`.

> **Mandatory Rule Compliance**: Because execution requires a RISC-V environment with QEMU, all xv6 performance benchmark values in this repository are marked **"NOT YET MEASURED on Host (To be benchmarked in GitHub Codespaces)"**. Under no circumstances are kernel benchmark numbers fabricated.

---

## 2. Automated Setup Instructions (GitHub Codespaces)

Open a terminal inside your GitHub Codespace and execute:

```bash
# 1. Navigate to the project root
cd /workspaces/OTLAB08

# 2. Make the setup script executable and run it
chmod +x tracks/G_xv6/setup_codespaces.sh
./tracks/G_xv6/setup_codespaces.sh
```

### What `setup_codespaces.sh` Performs:
1. Installs `build-essential`, `gcc-riscv64-unknown-elf`, and `qemu-system-misc`.
2. Clones the official MIT xv6 repository (`https://github.com/mit-pdos/xv6-riscv.git`) into `$HOME/xv6-riscv`.
3. Verifies and applies `patches/0001_lottery_scheduler_and_schedstat.patch`.
4. Copies benchmark user programs (`workload_cpu.c`, `workload_io.c`, `workload_mixed.c`, `sched_bench.c`) into xv6 `user/`.
5. Compiles the kernel and user disk image (`make -j$(nproc)`).

---

## 3. Step-by-Step Test & Verification Plan

### Step 3.1: Kernel Correctness & Regression Testing (`usertests`)
To ensure that replacing the Round-Robin scheduler with the Proportional-Share Lottery Scheduler and adding `sys_schedstat` introduced zero kernel regressions:

1. Launch xv6 in QEMU:
   ```bash
   cd $HOME/xv6-riscv
   make qemu
   ```
2. In the xv6 shell (`$`), run the full official test suite:
   ```bash
   $ usertests
   ```
3. **Success Criteria**:
   - `usertests` completes with `ALL TESTS PASSED`.
   - No kernel panic, lock deadlocks, or page table faults.

---

### Step 3.2: Benchmark Workload Execution

#### Workload A: CPU-Bound Matrix Multiplication Benchmark
Tests pure computational execution under high ticket allocations:
```bash
$ workload_cpu 10
$ workload_cpu 30
```
- **Recorded Metrics**: Elapsed ticks, `ticks_run`, `sched_count`, checksum.

#### Workload B: I/O-Bound Interactive Pipe/Sleep Benchmark
Tests latency and responsiveness of interactive processes alternating between pipe reads and sleep yields:
```bash
$ workload_io 30
```
- **Recorded Metrics**: Total elapsed ticks, sleep ticks, `ticks_run`, `ticks_wait`.

#### Workload C: Mixed CPU & I/O Benchmark
Simulates realistic multitasking (70% CPU bursts, 30% I/O wait):
```bash
$ workload_mixed 20
```
- **Recorded Metrics**: Total elapsed ticks, context switches, `ticks_run`.

#### Workload D: Multi-Process Lottery Fairness Harness (`sched_bench`)
Spawns 3 concurrent worker processes with ticket allocations of $10$, $20$, and $30$ tickets (ratio $1 : 2 : 3$):
```bash
$ sched_bench
```
- **Theoretical Expectation**:
  $$\text{CPU Share}_i = \frac{\text{tickets}_i}{\sum \text{tickets}} = \left[\frac{10}{60}, \frac{20}{60}, \frac{30}{60}\right] = [16.7\%, 33.3\%, 50.0\%]$$
- **Success Criteria**:
  - Measured `ticks_run` ratio closely matches the theoretical ticket ratio ($1 : 2 : 3$).
  - System call `sys_schedstat` correctly retrieves statistics for each worker without memory corruption.

---

### Step 3.3: Exiting QEMU and Recording Measured Numbers
1. To exit QEMU: Press `Ctrl-a`, release, then press `x`.
2. Copy the printed output from the QEMU terminal into `tracks/G_xv6/results/sched_benchmark_results.json`.
3. Commit and push the updated results from GitHub Codespaces:
   ```bash
   git add tracks/G_xv6/results/
   git commit -m "[AI:antigravity] Record actual Codespaces QEMU benchmark numbers for Track G"
   git push origin main
   ```
