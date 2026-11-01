# Prompt Log: Track H (DBMS Optimization & Index Selection)

## Workflow & Meta-Information
- **Track**: Track H — Relational DBMS Optimization
- **Role**: Antigravity Assistant & Student Pair-Programmer (Yash Deshmukh, Roll No. 241070018)
- **Objective**: Design, populate, optimize, and benchmark an embedded SQLite relational database with 162,000 synthetic records, formulate the Index Selection Problem (ISP) as a Chapter 6 0/1 Knapsack ILP problem, verify EXPLAIN QUERY PLAN transitions, and enforce 100% output result checksum invariance.

---

## Interactive Prompt History

### Prompt H-1: Schema Design & Fast Database Generator
**User/Task Prompt**:
> "DBMS: SQLite (zero setup) with a documented synthetic dataset of stated size, plus EXPLAIN QUERY PLAN verification. Add TPC-H SF 0.1 only if it loads easily."

**AI Response Summary**:
- Created `tracks/H_dbms/db_generator.py` with 4 relational tables:
  - `customers` (10,000 rows)
  - `products` (2,000 rows)
  - `orders` (50,000 rows)
  - `order_items` (100,000 rows)
  - Total: 162,000 records.
- Optimized generation via transactional batching (`executemany` with memory journal mode), generating all 162,000 records in 0.82 seconds.

---

### Prompt H-2: Workload Suite, Candidate Indices & Formulations
**User/Task Prompt**:
> "Define representative queries (point lookups, range filters, aggregations, joins), candidate index pool, and formulate index selection as a Chapter 6 optimization problem under storage constraints."

**AI Response Summary**:
- Implemented `tracks/H_dbms/index_selection_opt.py`:
  - 6 workload queries ($Q_1$ to $Q_6$) covering point lookups, range filters, 4-table joins, aggregations, and subqueries.
  - 10 candidate single- and multi-column indices on `orders`, `order_items`, `customers`, and `products`.
  - Formulated Index Selection as 0/1 Knapsack / ILP:
    $$\max \sum \Delta L_i x_i \quad \text{s.t.} \quad \sum S(i) x_i \le B$$
  - Implemented 3 solvers:
    1. Chapter 6 0/1 Knapsack ILP via `scipy.optimize.milp`
    2. Greedy marginal-benefit-to-storage ratio selection
    3. Exhaustive subspace enumeration ($2^{10} = 1024$ configurations)

---

### Prompt H-3: Benchmarking Suite & Storage Profiling
**User/Task Prompt**:
> "Measure query latencies, index footprints, throughput, verify EXPLAIN QUERY PLAN index transitions, and verify query checksums before and after optimization."

**AI Response Summary**:
- Created `tracks/H_dbms/benchmarks.py`.
- Caught and resolved bug **FAIL-08**: Dropping SQLite indices caused unallocated pages to pool in SQLite's freelist, so subsequent index creations reported 0 bytes. Corrected by tracking active pages:
  $$\text{active\_pages} = \text{page\_count} - \text{freelist\_count}$$
- Evaluated workload under Baseline, Greedy, ILP, and Full Index configurations.
- Verified 100% SHA-256 result checksum identity across all queries.
- Generated `dbms_query_latency_comparison.png` and `dbms_storage_vs_speedup_frontier.png`.
