# Track H: Relational DBMS Optimization (SQLite)

## 1. Problem Formulation & Theoretical Background

Relational database systems (RDBMS) process query workloads over multi-table relational schemas where query execution latency is heavily constrained by disk I/O and memory cache scanning.
* **Without Secondary Indices**: Queries filtering or joining on non-primary columns must perform sequential **full table scans** ($\mathcal{O}(N)$), examining every record page in the table.
* **With Secondary Indices**: The database engine constructs self-balancing $\text{B}^+$-Trees ($\mathcal{O}(\log N)$), enabling logarithmic search, range scans, and index-nested-loop joins.
* **The Index Selection Trade-Off**: Secondary indices are not free; every additional index consumes disk space and memory buffer pool cache, and slows down data modification (`INSERT`/`UPDATE`/`DELETE`).

---

### 1.1 Mathematical Formulation of the Index Selection Problem (ISP)

We formulate index selection as a **0/1 Knapsack & Integer Linear Programming (ILP)** optimization problem (Chapter 6):

$$\max_{\{x_i \in \{0, 1\}\}} \sum_{i \in \mathcal{C}} \Delta L(i) \cdot x_i$$
$$\text{subject to} \sum_{i \in \mathcal{C}} S(i) \cdot x_i \le B_{\text{storage}}$$

where:
* $\mathcal{C} = \{I_1, I_2, \dots, I_m\}$ is the set of $m = 10$ candidate single- and multi-column indices.
* $x_i \in \{0, 1\}$ is the binary decision variable indicating whether candidate index $i$ is created.
* $S(i)$ is the measured physical disk storage cost of index $i$ in bytes (accounting for SQLite freelist pages).
* $B_{\text{storage}}$ is the user's allocated storage budget (e.g. $2.5 \text{ MB}$).
* $\Delta L(i)$ is the weighted workload latency savings achieved by index $i$:
  $$\Delta L(i) = \sum_{q \in Q} w_q \cdot \max\left(0, L(q, \emptyset) - L(q, \{I_i\})\right)$$
  for workload queries $q \in Q$ with operational priority weights $w_q$.

---

## 2. Experimental Setup & Database Schema

Evaluated on an embedded SQLite database populated with **162,000 synthetic records** across 4 normalized relational tables ([`db_generator.py`](file:///c:/Users/Yash%20Deshmukh/Documents/Coding%20Notes/Projects/OTLAB08/tracks/H_dbms/db_generator.py)):

| Table | Record Count | Schema Attributes | Key Relationships |
|---|:---:|---|---|
| **`customers`** | 10,000 | `cust_id` (PK), `name`, `segment`, `country`, `created_year` | Referenced by `orders.cust_id` |
| **`products`** | 2,000 | `product_id` (PK), `name`, `category`, `price`, `stock_quantity` | Referenced by `order_items.product_id` |
| **`orders`** | 50,000 | `order_id` (PK), `cust_id` (FK), `order_date`, `order_status`, `total_amount` | Central transaction fact table |
| **`order_items`**| 100,000 | `item_id` (PK), `order_id` (FK), `product_id` (FK), `quantity`, `unit_price`, `discount` | Heavy line-item aggregation table |

### Workload Query Suite ($Q_1$ to $Q_6$)
1. **$Q_1$ (Point Lookup)**: Point search on `orders` by `cust_id` and `order_status` ($w=2.0$).
2. **$Q_2$ (Range Filter & Order)**: Date and amount range scan on `orders` with `ORDER BY total_amount DESC, order_id ASC LIMIT 50` ($w=1.5$).
3. **$Q_3$ (Multi-Table 4-Way Join)**: 4-table join (`customers` $\bowtie$ `orders` $\bowtie$ `order_items` $\bowtie$ `products`) with 2D aggregation ($w=3.0$).
4. **$Q_4$ (Line-Item Aggregation)**: Heavy group-by on `order_items` filtered by discount ($w=1.5$).
5. **$Q_5$ (Correlated Subquery)**: Customer segment filter with nested existence check on orders ($w=2.0$).
6. **$Q_6$ (Product Range Search)**: Multi-attribute range scan on `products` catalog ($w=1.0$).

---

## 3. Experimental Benchmark Results

### 3.1 Candidate Index Storage Footprints & Marginal Benefits
Accurately isolated using active page tracking ($\Delta \text{pages} = \text{page\_count} - \text{freelist\_count}$):

| Candidate Index | Target Table | Indexed Columns | Storage Footprint | Marginal Latency Benefit | Benefit / Size Ratio |
|---|---|---|:---:|:---:|:---:|
| `idx_orders_cust` | `orders` | `(cust_id)` | **512.0 KB** | **6.566 ms** | $1.28 \times 10^{-2}$ |
| `idx_orders_status_cust` | `orders` | `(order_status, cust_id)` | **956.0 KB** | **6.573 ms** | $0.69 \times 10^{-2}$ |
| `idx_orders_date_amt` | `orders` | `(order_date, total_amount)` | **1348.0 KB** | **3.869 ms** | $0.29 \times 10^{-2}$ |
| `idx_orders_amt` | `orders` | `(total_amount)` | **800.0 KB** | **5.612 ms** | $0.70 \times 10^{-2}$ |
| `idx_order_items_oid` | `order_items` | `(order_id)` | **1084.0 KB** | 0.000 ms | — |
| `idx_order_items_pid` | `order_items` | `(product_id)` | **1044.0 KB** | 0.000 ms | — |
| `idx_order_items_disc_pid` | `order_items`| `(discount, product_id)` | **1804.0 KB** | 0.000 ms | — |
| `idx_customers_segment` | `customers` | `(segment)` | **184.0 KB** | 0.000 ms | — |
| `idx_customers_country` | `customers` | `(country)` | **104.0 KB** | 0.000 ms | — |
| `idx_products_cat_price_stock` | `products` | `(category, price, stock)` | **56.0 KB** | **0.225 ms** | $0.40 \times 10^{-2}$ |

---

### 3.2 Optimization Formulation Comparison (Budget = 2.5 MB)

| Solver Method | Chapter 6 Category | Configurations Evaluated | Total Storage Consumed | Budget Utilized | Objective Benefit | Solve Time | Optimal Selected Indices |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **0/1 Knapsack ILP** | **Chapter 6: ILP (`scipy.optimize.milp`)** | 1 (Exact Branch-and-Cut) | **2,324.0 KB** | **90.78%** | **18.98 ms** | **7,742.0 $\mu$s** | `['idx_orders_cust', 'idx_orders_status_cust', 'idx_orders_amt', 'idx_products_cat_price_stock']` |
| **Greedy Selection** | Greedy Heuristic (Marginal Ratio) | 10 steps | 2,324.0 KB | 90.78% | 18.98 ms | **27.1 $\mu$s** | `['idx_orders_cust', 'idx_orders_amt', 'idx_orders_status_cust', 'idx_products_cat_price_stock']` |
| **Exhaustive Search** | Brute-Force Permutations | 1,024 configurations | 2,324.0 KB | 90.78% | 18.98 ms | 1,200.0 $\mu$s | `['idx_orders_cust', 'idx_orders_status_cust', 'idx_orders_amt', 'idx_products_cat_price_stock']` |

> **Formulation Findings**:
> * Under the 2.5 MB constraint, Chapter 6 0/1 Knapsack ILP selects the exact 4-index subset that maximizes cumulative latency reduction while fitting strictly within $2,324 \text{ KB} \le 2,560 \text{ KB}$.
> * Both the Greedy heuristic and the Exhaustive 1,024-configuration search confirm that this subset is provably the global optimum ($18.98\text{ ms}$ benefit).

---

### 3.3 Query Execution Plan Transitions & Checksum Invariance

| Query | Baseline Plan (Unindexed) | Optimized Plan (Chapter 6 ILP) | Index Utilized | Result Checksum Match? |
|---|---|---|:---:|:---:|
| **$Q_1$ (Point Lookup)** | `SCAN orders` (50,000 rows) | `SEARCH orders USING INDEX idx_orders_status_cust` | **`idx_orders_status_cust`** | **100% Exact Match** |
| **$Q_2$ (Range Sort)** | `SCAN orders` | `SEARCH orders USING INDEX idx_orders_amt (total_amount>?)` | **`idx_orders_amt`** | **100% Exact Match** |
| **$Q_3$ (Multi-Table Join)**| `SCAN oi` | `SCAN oi` (uses PK lookups on joined tables) | PK | **100% Exact Match** |
| **$Q_4$ (Line Item Agg)** | `SCAN order_items` | `SCAN order_items` | None | **100% Exact Match** |
| **$Q_5$ (Subquery)** | `SCAN customers` | `SEARCH customers USING INTEGER PRIMARY KEY (rowid=?)` | PK | **100% Exact Match** |
| **$Q_6$ (Product Range)** | `SCAN products` (2,000 rows) | `SEARCH products USING INDEX idx_products_cat_price_stock` | **`idx_products_cat_price_stock`**| **100% Exact Match** |

> **Zero Query Divergence**: Across all queries, SHA-256 result checksums are verified **100% IDENTICAL** between unindexed baseline and optimized database execution.

---

## 4. Visualizations & Artifacts

- **Query Latency Bar Chart**: [`results/dbms_query_latency_comparison.png`](results/dbms_query_latency_comparison.png) (Per-query latency across Baseline, Greedy, and Chapter 6 ILP).
- **Storage-Speedup Pareto Frontier**: [`results/dbms_storage_vs_speedup_frontier.png`](results/dbms_storage_vs_speedup_frontier.png) (Demonstrating the trade-off frontier between storage footprint in KB and workload speedup).
- **Machine-Readable Metrics**: [`results/dbms_benchmark_results.json`](results/dbms_benchmark_results.json).

---

## 5. Verification & Correctness Checks

All 6 automated unit tests in `tracks/H_dbms/tests/test_dbms.py` passed in **1.49s**:
1. **Schema & Records**: Verified 162,000 rows across all 4 tables.
2. **Freelist Accounting**: Verified positive, non-zero storage measurements across candidate indices.
3. **Query Plan Transitions**: Verified transition from `SCAN orders` to `SEARCH orders USING INDEX idx_orders_status_cust`.
4. **Result Checksum Invariance**: Verified zero data divergence across all workload queries.
5. **ILP Budget Feasibility**: Verified that ILP solutions never violate the storage budget.
6. **Optimality Equivalence**: Verified that Chapter 6 ILP matches Exhaustive search and beats/matches Greedy.

---

## 6. How to Reproduce

```bash
# 1. Run unit test suite
python -m unittest tracks/H_dbms/tests/test_dbms.py

# 2. Re-run database generator (if needed)
python -m tracks.H_dbms.db_generator

# 3. Re-run benchmarks and plot generation
python -m tracks.H_dbms.benchmarks
```
