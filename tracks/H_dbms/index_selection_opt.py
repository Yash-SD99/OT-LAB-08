"""Index Selection Optimization for Relational Databases (Track H).

Formulates the Index Selection Problem (ISP) as a Chapter 6 optimization problem:
1. 0/1 Knapsack & Integer Linear Programming (ILP) using scipy.optimize.milp & DP
2. Greedy Marginal-Benefit Selection
3. Exhaustive Subspace Enumeration Baseline
4. EXPLAIN QUERY PLAN verification (SCAN vs SEARCH USING INDEX)
5. Strict Result Checksum Invariance Verification
"""

from __future__ import annotations
import copy
import hashlib
import itertools
import os
import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


# -----------------------------------------------------------------------------
# 1. Workload Query Suite
# -----------------------------------------------------------------------------
WORKLOAD_QUERIES: Dict[str, Dict[str, Any]] = {
    "Q1_PointLookup": {
        "sql": "SELECT * FROM orders WHERE cust_id = 4520 AND order_status = 'COMPLETED';",
        "description": "Point lookup on foreign key (cust_id) with status filter",
        "weight": 2.0
    },
    "Q2_RangeSort": {
        "sql": "SELECT * FROM orders WHERE total_amount >= 2800.0 AND order_date >= '2023-01-01' ORDER BY total_amount DESC, order_id ASC LIMIT 50;",
        "description": "Range filter on date and amount with ordered limit and deterministic tie-breaker",
        "weight": 1.5
    },
    "Q3_MultiTableJoin": {
        "sql": """
        SELECT c.country, p.category, SUM(oi.quantity * oi.unit_price * (1.0 - oi.discount)) AS total_rev
        FROM customers c
        JOIN orders o ON c.cust_id = o.cust_id
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        WHERE o.order_date >= '2023-01-01'
        GROUP BY c.country, p.category
        ORDER BY c.country, p.category;
        """,
        "description": "4-table join with date filter and multi-dimensional aggregation",
        "weight": 3.0
    },
    "Q4_LineItemAggregation": {
        "sql": """
        SELECT product_id, SUM(quantity) as total_qty
        FROM order_items
        WHERE discount >= 0.15
        GROUP BY product_id
        HAVING total_qty > 100
        ORDER BY product_id;
        """,
        "description": "Heavy aggregation over order_items table with discount filter",
        "weight": 1.5
    },
    "Q5_CorrelatedSubquery": {
        "sql": """
        SELECT cust_id, name, segment
        FROM customers
        WHERE segment = 'ENTERPRISE'
          AND cust_id IN (SELECT DISTINCT cust_id FROM orders WHERE total_amount > 3200.0)
        ORDER BY cust_id;
        """,
        "description": "Customer segment lookup with subquery existence check on orders",
        "weight": 2.0
    },
    "Q6_ProductRange": {
        "sql": """
        SELECT product_id, name, price, stock_quantity
        FROM products
        WHERE category = 'ELECTRONICS'
          AND price BETWEEN 250.0 AND 750.0
          AND stock_quantity > 50
        ORDER BY product_id;
        """,
        "description": "Multi-column range search on product catalog",
        "weight": 1.0
    }
}


# -----------------------------------------------------------------------------
# 2. Candidate Index Pool
# -----------------------------------------------------------------------------
CANDIDATE_INDEX_POOL: Dict[str, Dict[str, str]] = {
    "idx_orders_cust": {
        "table": "orders",
        "sql": "CREATE INDEX idx_orders_cust ON orders(cust_id);"
    },
    "idx_orders_status_cust": {
        "table": "orders",
        "sql": "CREATE INDEX idx_orders_status_cust ON orders(order_status, cust_id);"
    },
    "idx_orders_date_amt": {
        "table": "orders",
        "sql": "CREATE INDEX idx_orders_date_amt ON orders(order_date, total_amount);"
    },
    "idx_orders_amt": {
        "table": "orders",
        "sql": "CREATE INDEX idx_orders_amt ON orders(total_amount);"
    },
    "idx_order_items_oid": {
        "table": "order_items",
        "sql": "CREATE INDEX idx_order_items_oid ON order_items(order_id);"
    },
    "idx_order_items_pid": {
        "table": "order_items",
        "sql": "CREATE INDEX idx_order_items_pid ON order_items(product_id);"
    },
    "idx_order_items_disc_pid": {
        "table": "order_items",
        "sql": "CREATE INDEX idx_order_items_disc_pid ON order_items(discount, product_id);"
    },
    "idx_customers_segment": {
        "table": "customers",
        "sql": "CREATE INDEX idx_customers_segment ON customers(segment);"
    },
    "idx_customers_country": {
        "table": "customers",
        "sql": "CREATE INDEX idx_customers_country ON customers(country);"
    },
    "idx_products_cat_price_stock": {
        "table": "products",
        "sql": "CREATE INDEX idx_products_cat_price_stock ON products(category, price, stock_quantity);"
    }
}


class DatabaseBenchmarkEngine:
    """Manages SQLite benchmark database, index creation, latency profiling, and checksum verification."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._page_size = self._query_page_size()

    def _query_page_size(self) -> int:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA page_size;")
        ps = cur.fetchone()[0]
        conn.close()
        return ps

    def drop_all_secondary_indices(self):
        """Drops all custom secondary indices to restore raw baseline state."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        for idx_name in CANDIDATE_INDEX_POOL.keys():
            cur.execute(f"DROP INDEX IF EXISTS {idx_name};")
        conn.commit()
        conn.close()

    def apply_index_set(self, index_names: List[str]):
        """Creates a specified set of candidate indices."""
        self.drop_all_secondary_indices()
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        for idx_name in index_names:
            if idx_name in CANDIDATE_INDEX_POOL:
                cur.execute(CANDIDATE_INDEX_POOL[idx_name]["sql"])
        conn.commit()
        conn.close()

    def measure_index_sizes(self) -> Dict[str, int]:
        """Measures exact storage cost in bytes for each candidate index.
        Correctly accounts for SQLite's internal freelist pages:
        active_pages = page_count - freelist_count
        """
        sizes = {}
        for idx_name, info in CANDIDATE_INDEX_POOL.items():
            self.drop_all_secondary_indices()
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("PRAGMA page_count;")
            pc_before = cur.fetchone()[0]
            cur.execute("PRAGMA freelist_count;")
            fl_before = cur.fetchone()[0]
            active_before = pc_before - fl_before

            cur.execute(info["sql"])
            conn.commit()

            cur.execute("PRAGMA page_count;")
            pc_after = cur.fetchone()[0]
            cur.execute("PRAGMA freelist_count;")
            fl_after = cur.fetchone()[0]
            active_after = pc_after - fl_after

            cur.execute(f"DROP INDEX {idx_name};")
            conn.commit()
            conn.close()

            delta_pages = max(1, active_after - active_before)
            sizes[idx_name] = delta_pages * self._page_size
        self.drop_all_secondary_indices()
        return sizes

    def execute_query_with_metrics(
        self,
        query_sql: str,
        runs: int = 5
    ) -> Dict[str, Any]:
        """Executes query, measures mean latency, parses query plan, and computes SHA-256 result checksum."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # 1. EXPLAIN QUERY PLAN
        cur.execute(f"EXPLAIN QUERY PLAN {query_sql}")
        plan_rows = cur.fetchall()
        plan_details = [row[3] for row in plan_rows]
        uses_index = any("USING INDEX" in text or "USING COVERING INDEX" in text for text in plan_details)

        # 2. Warmup & Canonical Result Fetch for Checksum
        cur.execute(query_sql)
        raw_results = cur.fetchall()
        # Sort canonically before hashing to respect relational set identity
        hasher = hashlib.sha256()
        for row in sorted(raw_results):
            hasher.update(str(row).encode("utf-8"))
        checksum = hasher.hexdigest()

        # 3. Repeated Latency Measurement
        latencies_ms = []
        for _ in range(runs):
            t0 = time.perf_counter()
            cur.execute(query_sql)
            cur.fetchall()
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat_ms)

        conn.close()
        return {
            "mean_latency_ms": float(np.mean(latencies_ms)),
            "std_latency_ms": float(np.std(latencies_ms)),
            "row_count": len(raw_results),
            "checksum": checksum,
            "plan_details": plan_details,
            "uses_index": uses_index
        }

    def evaluate_workload(
        self,
        index_set: List[str],
        runs_per_query: int = 5
    ) -> Dict[str, Any]:
        """Evaluates entire workload suite under specified index configuration."""
        self.apply_index_set(index_set)

        query_metrics = {}
        total_weighted_latency = 0.0
        workload_t0 = time.perf_counter()

        for q_name, q_info in WORKLOAD_QUERIES.items():
            qm = self.execute_query_with_metrics(q_info["sql"], runs=runs_per_query)
            qm["weight"] = q_info["weight"]
            qm["weighted_latency_ms"] = qm["mean_latency_ms"] * q_info["weight"]
            total_weighted_latency += qm["weighted_latency_ms"]
            query_metrics[q_name] = qm

        workload_wall_ms = (time.perf_counter() - workload_t0) * 1000.0

        return {
            "index_set": list(index_set),
            "num_indices": len(index_set),
            "query_metrics": query_metrics,
            "total_weighted_latency_ms": float(total_weighted_latency),
            "workload_wall_ms": float(workload_wall_ms),
            "qps": float(len(WORKLOAD_QUERIES) / max(workload_wall_ms / 1000.0, 1e-6))
        }


# -----------------------------------------------------------------------------
# 3. Optimization Formulations (Knapsack ILP, Greedy, Exhaustive)
# -----------------------------------------------------------------------------

def solve_index_selection_ilp(
    index_candidates: List[str],
    index_sizes: Dict[str, int],
    index_benefits: Dict[str, float],
    storage_budget_bytes: int
) -> Dict[str, Any]:
    """Chapter 6 Method: 0/1 Knapsack / Integer Linear Programming Formulation.
    
    Decision variables: x_i in {0, 1} for each candidate index i.
    Maximize: sum_i Benefit(i) * x_i
    Subject to: sum_i Size(i) * x_i <= StorageBudget
    """
    t0 = time.perf_counter()
    n = len(index_candidates)
    c = np.array([-index_benefits[name] for name in index_candidates], dtype=float) # Minimize negative benefit
    A = np.array([[index_sizes[name] for name in index_candidates]], dtype=float)
    b_u = np.array([storage_budget_bytes], dtype=float)
    b_l = np.array([0.0], dtype=float)

    constraints = LinearConstraint(A, b_l, b_u)
    integrality = np.ones(n) # Binary 0/1
    bounds = Bounds(0, 1)

    res = milp(c=c, integrality=integrality, constraints=constraints, bounds=bounds)
    solve_time_us = (time.perf_counter() - t0) * 1e6

    if res.success:
        chosen_indices = [index_candidates[i] for i in range(n) if res.x[i] > 0.5]
        total_size = sum(index_sizes[i] for i in chosen_indices)
        total_benefit = sum(index_benefits[i] for i in chosen_indices)
    else:
        chosen_indices = []
        total_size = 0
        total_benefit = 0.0

    return {
        "method": "Chapter 6: 0/1 Knapsack ILP (scipy.optimize.milp)",
        "chosen_indices": chosen_indices,
        "total_benefit_ms": float(total_benefit),
        "total_storage_bytes": int(total_size),
        "storage_budget_bytes": int(storage_budget_bytes),
        "budget_utilized_pct": float(total_size / max(storage_budget_bytes, 1) * 100.0),
        "solve_time_us": float(solve_time_us)
    }


def solve_index_selection_greedy(
    index_candidates: List[str],
    index_sizes: Dict[str, int],
    index_benefits: Dict[str, float],
    storage_budget_bytes: int
) -> Dict[str, Any]:
    """Baseline 1: Greedy Marginal-Benefit-to-Storage Ratio Selection.
    At each step, greedily selects the candidate index with the highest Benefit(i) / Size(i)
    that still fits within the remaining storage budget.
    """
    t0 = time.perf_counter()
    # Compute benefit-to-cost ratio
    ratios = [
        (name, index_benefits[name] / max(index_sizes[name], 1))
        for name in index_candidates
    ]
    ratios.sort(key=lambda x: x[1], reverse=True)

    chosen: List[str] = []
    current_size = 0
    total_benefit = 0.0

    for name, ratio in ratios:
        size = index_sizes[name]
        if current_size + size <= storage_budget_bytes and index_benefits[name] > 0:
            chosen.append(name)
            current_size += size
            total_benefit += index_benefits[name]

    solve_time_us = (time.perf_counter() - t0) * 1e6
    return {
        "method": "Greedy Marginal-Benefit Selection",
        "chosen_indices": chosen,
        "total_benefit_ms": float(total_benefit),
        "total_storage_bytes": int(current_size),
        "storage_budget_bytes": int(storage_budget_bytes),
        "budget_utilized_pct": float(current_size / max(storage_budget_bytes, 1) * 100.0),
        "solve_time_us": float(solve_time_us)
    }


def solve_index_selection_exhaustive(
    index_candidates: List[str],
    index_sizes: Dict[str, int],
    index_benefits: Dict[str, float],
    storage_budget_bytes: int
) -> Dict[str, Any]:
    """Baseline 2: Exhaustive Search over all 2^N candidate configurations."""
    t0 = time.perf_counter()
    best_combo: Tuple[str, ...] = ()
    best_benefit = -1.0
    best_size = 0
    evaluated = 0

    n = len(index_candidates)
    for r in range(n + 1):
        for combo in itertools.combinations(index_candidates, r):
            evaluated += 1
            combo_size = sum(index_sizes[i] for i in combo)
            if combo_size <= storage_budget_bytes:
                combo_benefit = sum(index_benefits[i] for i in combo)
                if combo_benefit > best_benefit:
                    best_benefit = combo_benefit
                    best_combo = combo
                    best_size = combo_size

    solve_time_us = (time.perf_counter() - t0) * 1e6
    return {
        "method": "Exhaustive Configuration Search",
        "chosen_indices": list(best_combo),
        "total_benefit_ms": float(best_benefit),
        "total_storage_bytes": int(best_size),
        "storage_budget_bytes": int(storage_budget_bytes),
        "budget_utilized_pct": float(best_size / max(storage_budget_bytes, 1) * 100.0),
        "evaluated_configurations": evaluated,
        "solve_time_us": float(solve_time_us)
    }
