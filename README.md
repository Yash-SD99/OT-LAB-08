# Optimization Techniques: Lab Assignment 8 (14-Day Vibe-Coding Challenge)
### **Optimize Anything: Comprehensive 10-Track Optimization Portfolio**

* **Student Name**: Yash Deshmukh
* **Roll Number**: 241070018
* **Institution**: Veermata Jijabai Technological Institute (VJTI), Mumbai
* **Course**: Optimization Techniques (Lab Assignment 8)
* **GitHub Repository**: [Yash-SD99/OTLAB08](https://github.com/Yash-SD99/OTLAB08)
* **Development Methodology**: AI-Assisted Vibe Coding (Strict Disclosure & Verification Workflow)

---

## 1. Project Overview

This repository contains the complete, reproducible work for the 14-day Optimization Techniques challenge. Across ten compulsory tracks (**Tracks A through J**), we formulate, implement from scratch, benchmark, and critically evaluate the fundamental optimization algorithms taught in Chapter 6, alongside specialized applications in operating systems (xv6), compiler engineering, database systems, quantum computing, and software engineering.

Every implementation adheres to strict scientific and engineering rigor:
1. **NumPy-Native From Scratch**: Core optimization algorithms, neural backpropagation, second-order solvers, and branch-and-bound logic are written natively in pure NumPy to guarantee complete mechanistic understanding for viva examination.
2. **Zero Fabricated Results**: Every figure and metric is generated directly from executed benchmarks. Where cloud limitations or simulations apply (e.g. quantum noise models or xv6 on Codespaces), they are explicitly documented as such.
3. **Statistical Integrity**: All stochastic algorithms are evaluated across $\ge 10$ randomized runs with fixed seeds, reporting `mean ± std`.
4. **Verifiable Equivalence**: All discrete, compiler, and DBMS optimizations verify that system outputs and query results remain strictly invariant under optimization.

---

## 2. Benchmark Environment & Hardware Specifications

To guarantee universal accessibility and zero-barrier reproducibility on free cloud infrastructure, workloads are designated for specific free environments:

| Track Group | Benchmark Platform | Execution Hardware Profile | OS / Runtime |
|---|---|---|---|
| **Tracks A, B, C, D, E, H, I, J** | Google Colab / Kaggle | Intel Xeon @ 2.20GHz (2 vCPUs), 13 GB RAM, No GPU (CPU-only benchmarked) | Linux (Ubuntu 22.04 LTS), Python 3.10+ |
| **Track F (Compiler)** | GitHub Codespaces / Colab | 2-Core AMD/Intel Virtual Processor, 8 GB RAM | Linux (Ubuntu 22.04 LTS), Python 3.10+ |
| **Track G (xv6 OS Kernel)** | GitHub Codespaces | 2-Core Linux VM with QEMU RISC-V (`qemu-system-misc`, `gcc-riscv64-unknown-elf`) | Ubuntu 22.04, RISC-V QEMU 7.0+ |

> **Note on Quantum Tracks (C & I)**: All quantum experiments run on simulated backends with calibrated hardware noise models (depolarizing and thermal relaxation noise), explicitly reported as *"noise-model simulation, not physical QPU hardware"*. Maximum circuit width is kept to $\le 8$ qubits to maintain deterministic, fast cloud execution.

---

## 3. Strict AI Tools Register (Section 8 Compliance)

| Tool Name & Model Version | Tracks Utilized | Dates Used | Kinds of Work Conducted | Notes & Attribution |
|---|:---:|:---:|---|---|
| **Antigravity (Google DeepMind)** | Tracks A–J | Nov 2026 | Architecture planning, pure NumPy algorithmic implementation, test suite generation, benchmarking scripts, LaTeX documentation | Primary pair-programming agent |
| **Claude Code (claude-sonnet-5-5, Anthropic)** | Track A | Nov 2026 | Comparative evaluation on Gauss-Newton with Levenberg-Marquardt damping | Benchmark comparison in [`tool_comparison.md`](tool_comparison.md) |

*(Detailed prompt trajectories, AI responses, revisions, and commit hashes are logged under the [`prompts/`](prompts/) directory)*.

---

## 4. Repository Structure

```text
OTLAB08/
├── README.md                          # Master project documentation & AI tools register
├── requirements.txt                   # Universal dependencies
├── setup_colab.ipynb                  # One-click runner notebook for Google Colab
├── CHAPTER6_CHECKLIST.md              # Complete Chapter 6 syllabus coverage matrix
├── failures_log.md                    # Audited log of real AI code failures caught by tests
├── tool_comparison.md                 # Comparative assessment between AI assistants
│
├── tracks/
│   ├── A_continuous/                  # 1st & 2nd order methods, Backprop, HVP, Fisher, Landscapes
│   ├── B_discrete/                    # Knapsack, Branch-and-Bound, DP, QUBO/Ising & Penalty terms
│   ├── C_quantum_opt/                 # QAOA Max-Cut, VQE, PQC, Parameter-Shift, SPSA, Barren Plateaus
│   ├── D_submodular_parallel_dist/    # Submodular Greedy, Multiprocessing, FedAvg, MGDA/GradNorm
│   ├── E_case_study/                  # Option C: Non-Convex Portfolio Optimization (Cardinality + TC)
│   ├── F_compiler/                    # TAC Mini-Compiler, 4 Optimization Passes, Pass Scheduling
│   ├── G_xv6/                         # xv6 Priority/Lottery Scheduler Patch, QEMU Benchmarks
│   ├── H_dbms/                        # SQLite Index Selection (Knapsack/ILP) & Query Plan Optimization
│   ├── I_quantum_algos_qml/           # Hand-coded DJ/Grover/QFT, QML Classifier vs Classical MLP
│   └── J_software_eng/                # Target >1,000 LOC Service Profiling & Algorithmic Optimization
│
├── prompts/                           # Track-by-track prompt logs and conversation traces
├── report/                            # Final comprehensive Technical Report
└── demo/                              # Video demonstration script and slides
```

---

## 5. Quick-Start Guide

### Running on Google Colab / Kaggle
1. Open the repository in Google Colab or clone it:
   ```bash
   git clone https://github.com/Yash-SD99/OTLAB08.git
   cd OTLAB08
   pip install -r requirements.txt
   ```
2. Run any track's benchmark script, e.g. Track A:
   ```bash
   python -m tracks.A_continuous.benchmarks
   ```
3. Run the automated test suite (71 passing tests across all tracks):
   ```bash
   pytest --ignore=tracks/J_software_eng/fuzzywuzzy_upstream
   ```

### Running Track G (xv6 Kernel) on GitHub Codespaces
1. Open the repository in GitHub Codespaces.
2. Execute the environment setup:
   ```bash
   bash tracks/G_xv6/setup_codespaces.sh
   ```
3. Follow the instructions in `tracks/G_xv6/TRACK_README.md` to compile and launch xv6 under QEMU.
