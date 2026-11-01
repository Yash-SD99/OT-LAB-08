"""Unit and Verification Tests for Track H: Relational DBMS Optimization (SQLite).

Verifies:
1. Database Schema Integrity & Record Population
2. Accurate Storage Measurement under SQLite Freelist Accounting
3. EXPLAIN QUERY PLAN Index Transitions (SCAN -> SEARCH USING INDEX)
4. Query Result Checksum Invariance (Zero Divergence)
5. 0/1 Knapsack ILP Feasibility under Storage Budget Constraints
6. Optimality Alignment: Chapter 6 ILP vs Greedy vs Exhaustive
"""

from __future__ import annotations
import os
import sqlite3
import unittest
import numpy as np

from tracks.H_dbms.db_generator import DB_PATH, populate_database
from tracks.H_dbms.index_selection_opt import (
    WORKLOAD_QUERIES,
    CANDIDATE_INDEX_POOL,
    DatabaseBenchmarkEngine,
    solve_index_selection_ilp,
    solve_index_selection_greedy,
    solve_index_selection_exhaustive
)


class TestDBMSOptimization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure database exists
        if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) < 1_000_000:
            populate_database(DB_PATH, num_customers=1000, num_products=500, num_orders=2000, num_order_items=4000)
        cls.engine = DatabaseBenchmarkEngine(DB_PATH)
        cls.engine.drop_all_secondary_indices()

    @classmethod
    def tearDownClass(cls):
        cls.engine.drop_all_secondary_indices()

    def test_database_schema_and_tables(self):
        """Verifies that all 4 tables exist with records."""
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        for tbl in ["customers", "products", "orders", "order_items"]:
            cur.execute(f"SELECT COUNT(*) FROM {tbl};")
            count = cur.fetchone()[0]
            self.assertGreater(count, 0, f"Table {tbl} is empty!")
        conn.close()

    def test_index_storage_measurement_freelist(self):
        """Verifies that index storage measurement produces positive byte footprints."""
        sizes = self.engine.measure_index_sizes()
        for name, sz in sizes.items():
            self.assertGreater(sz, 0, f"Index {name} has invalid size 0!")
            self.assertLessEqual(sz, 10 * 1024 * 1024, f"Index {name} has unrealistically large size!")

    def test_explain_query_plan_transition(self):
        """Verifies that creating an index converts full table scan to B-tree search."""
        q_sql = WORKLOAD_QUERIES["Q1_PointLookup"]["sql"]

        # Baseline: No index -> SCAN
        self.engine.drop_all_secondary_indices()
        m_base = self.engine.execute_query_with_metrics(q_sql, runs=1)
        self.assertFalse(m_base["uses_index"])
        self.assertTrue(any("SCAN orders" in p for p in m_base["plan_details"]))

        # Optimized: With index -> SEARCH USING INDEX
        self.engine.apply_index_set(["idx_orders_status_cust"])
        m_opt = self.engine.execute_query_with_metrics(q_sql, runs=1)
        self.assertTrue(m_opt["uses_index"])
        self.assertTrue(any("USING INDEX idx_orders_status_cust" in p for p in m_opt["plan_details"]))

    def test_query_result_checksum_invariance(self):
        """Verifies that query outputs are 100% identical between unindexed and indexed configurations."""
        self.engine.drop_all_secondary_indices()
        baseline_res = self.engine.evaluate_workload([], runs_per_query=1)

        # Apply optimal index set
        self.engine.apply_index_set(["idx_orders_status_cust", "idx_orders_amt", "idx_products_cat_price_stock"])
        opt_res = self.engine.evaluate_workload(["idx_orders_status_cust", "idx_orders_amt", "idx_products_cat_price_stock"], runs_per_query=1)

        for q_name in WORKLOAD_QUERIES.keys():
            b_hash = baseline_res["query_metrics"][q_name]["checksum"]
            o_hash = opt_res["query_metrics"][q_name]["checksum"]
            self.assertEqual(
                b_hash, o_hash,
                f"Query {q_name} output differed between baseline and indexed database!"
            )

    def test_ilp_knapsack_budget_feasibility(self):
        """Verifies that Chapter 6 0/1 Knapsack ILP never exceeds storage budget."""
        candidates = ["idx1", "idx2", "idx3", "idx4"]
        sizes = {"idx1": 500, "idx2": 800, "idx3": 1200, "idx4": 1500}
        benefits = {"idx1": 10.0, "idx2": 25.0, "idx3": 30.0, "idx4": 35.0}
        budget = 2000

        res = solve_index_selection_ilp(candidates, sizes, benefits, budget)
        self.assertLessEqual(res["total_storage_bytes"], budget)
        self.assertGreater(res["total_benefit_ms"], 0.0)

    def test_ilp_vs_greedy_vs_exhaustive_optimality(self):
        """Verifies that Chapter 6 ILP finds the globally optimal knapsack solution matching exhaustive search."""
        candidates = ["idxA", "idxB", "idxC", "idxD", "idxE"]
        sizes = {"idxA": 300, "idxB": 400, "idxC": 500, "idxD": 600, "idxE": 700}
        benefits = {"idxA": 12.0, "idxB": 18.0, "idxC": 24.0, "idxD": 28.0, "idxE": 35.0}
        budget = 1100

        ilp = solve_index_selection_ilp(candidates, sizes, benefits, budget)
        exh = solve_index_selection_exhaustive(candidates, sizes, benefits, budget)
        grd = solve_index_selection_greedy(candidates, sizes, benefits, budget)

        # ILP must equal exhaustive optimum
        self.assertAlmostEqual(ilp["total_benefit_ms"], exh["total_benefit_ms"], places=4)
        # ILP must be greater than or equal to Greedy
        self.assertGreaterEqual(ilp["total_benefit_ms"], grd["total_benefit_ms"] - 1e-6)


if __name__ == "__main__":
    unittest.main()
