"""Benchmarking Suite for Relational DBMS Optimization (Track H).

Evaluates SQLite under:
1. Baseline (Unindexed / Primary Keys only)
2. Greedy Marginal-Benefit Index Selection
3. Chapter 6 0/1 Knapsack ILP Index Selection (scipy.optimize.milp)
4. Full Exhaustive Candidate Indexing

Measures:
- Latency per query (ms) across repeated runs
- Workload throughput (QPS)
- Index storage overhead (KB/MB)
- Output result SHA-256 checksums (verifying zero query divergence)
- EXPLAIN QUERY PLAN transitions from full table scans to B-tree index lookups
"""

from __future__ import annotations
import json
import os
import sqlite3
import time
from typing import Any, Dict, List
import matplotlib.pyplot as plt
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


def run_dbms_benchmarks() -> Dict[str, Any]:
    print("===================================================================")
    print(" Running Track H Relational DBMS Optimization Benchmarks (SQLite)  ")
    print("===================================================================")

    # 1. Ensure database is populated
    if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) < 1_000_000:
        print(f"Generating synthetic database at {DB_PATH}...")
        populate_database(DB_PATH)
    else:
        print(f"Using existing synthetic database: {os.path.getsize(DB_PATH) / 1024 / 1024:.2f} MB")

    engine = DatabaseBenchmarkEngine(DB_PATH)
    engine.drop_all_secondary_indices()

    # 2. Profile Candidate Index Storage Costs
    print("\nMeasuring candidate index storage footprints...")
    index_sizes = engine.measure_index_sizes()
    for name, size_bytes in index_sizes.items():
        print(f"  - {name:<30}: {size_bytes / 1024:>8.1f} KB")

    # 3. Profile Baseline Workload (Unindexed)
    print("\nEvaluating Baseline Workload (0 secondary indices, full scans)...")
    baseline_eval = engine.evaluate_workload(index_set=[], runs_per_query=5)
    print(f"  Baseline Total Weighted Latency: {baseline_eval['total_weighted_latency_ms']:.2f} ms")
    print(f"  Baseline Workload QPS:            {baseline_eval['qps']:.1f} queries/s")

    # 4. Profile Marginal Benefits of Individual Indices per Query
    print("\nProfiling marginal latency benefits per candidate index...")
    index_benefits = {}
    candidates = list(CANDIDATE_INDEX_POOL.keys())

    for idx_name in candidates:
        engine.apply_index_set([idx_name])
        b_total = 0.0
        for q_name, q_info in WORKLOAD_QUERIES.items():
            qm = engine.execute_query_with_metrics(q_info["sql"], runs=5)
            if qm["uses_index"]:
                base_lat = baseline_eval["query_metrics"][q_name]["mean_latency_ms"]
                delta_ms = max(0.0, base_lat - qm["mean_latency_ms"])
                b_total += delta_ms * q_info["weight"]
        index_benefits[idx_name] = round(b_total, 4)
        print(f"  - {idx_name:<30}: Benefit = {index_benefits[idx_name]:>6.3f} ms saved | Size = {index_sizes[idx_name]/1024:>6.1f} KB")
    engine.drop_all_secondary_indices()

    # 5. Formulate Optimization under Storage Budget (Budget = 2.5 MB)
    storage_budget_bytes = int(2.5 * 1024 * 1024)
    print(f"\nSolving Index Selection Problem (Storage Budget: {storage_budget_bytes / 1024 / 1024:.1f} MB)...")

    ilp_res = solve_index_selection_ilp(candidates, index_sizes, index_benefits, storage_budget_bytes)
    greedy_res = solve_index_selection_greedy(candidates, index_sizes, index_benefits, storage_budget_bytes)
    exh_res = solve_index_selection_exhaustive(candidates, index_sizes, index_benefits, storage_budget_bytes)

    print(f"  [Chapter 6 ILP] Chosen {len(ilp_res['chosen_indices'])} indices: {ilp_res['chosen_indices']}")
    print(f"                  Benefit = {ilp_res['total_benefit_ms']:.2f} ms | Storage = {ilp_res['total_storage_bytes']/1024:.1f} KB | Solve Time = {ilp_res['solve_time_us']:.1f} us")
    print(f"  [Greedy]        Chosen {len(greedy_res['chosen_indices'])} indices: {greedy_res['chosen_indices']}")
    print(f"                  Benefit = {greedy_res['total_benefit_ms']:.2f} ms | Storage = {greedy_res['total_storage_bytes']/1024:.1f} KB | Solve Time = {greedy_res['solve_time_us']:.1f} us")
    print(f"  [Exhaustive]    Evaluated {exh_res['evaluated_configurations']} configs | Optimal Benefit = {exh_res['total_benefit_ms']:.2f} ms")

    # 6. Execute and Verify Workload under Selected Index Configurations
    print("\nExecuting workload across indexing strategies...")
    configs = {
        "Baseline_Unindexed": [],
        "Greedy_Selection": greedy_res["chosen_indices"],
        "Chapter6_ILP": ilp_res["chosen_indices"],
        "All_Candidates": candidates
    }

    eval_results = {}
    for cfg_name, idx_list in configs.items():
        res = engine.evaluate_workload(index_set=idx_list, runs_per_query=5)
        res["storage_bytes"] = sum(index_sizes[i] for i in idx_list)
        res["speedup"] = baseline_eval["total_weighted_latency_ms"] / max(res["total_weighted_latency_ms"], 1e-6)
        eval_results[cfg_name] = res
        print(f"  Configuration: {cfg_name:<20} | Latency: {res['total_weighted_latency_ms']:>6.2f} ms | Speedup: {res['speedup']:>5.2f}x | Storage: {res['storage_bytes']/1024:>6.1f} KB")

    # 7. Checksum Invariance Verification
    print("\nVerifying Query Result Checksum Invariance (Zero Divergence)...")
    checksum_verification = {}
    all_checksums_match = True

    baseline_metrics = eval_results["Baseline_Unindexed"]["query_metrics"]
    ilp_metrics = eval_results["Chapter6_ILP"]["query_metrics"]

    for q_name in WORKLOAD_QUERIES.keys():
        b_hash = baseline_metrics[q_name]["checksum"]
        opt_hash = ilp_metrics[q_name]["checksum"]
        matches = (b_hash == opt_hash)
        checksum_verification[q_name] = {
            "baseline_hash": b_hash,
            "optimized_hash": opt_hash,
            "matches": matches,
            "baseline_plan": baseline_metrics[q_name]["plan_details"],
            "optimized_plan": ilp_metrics[q_name]["plan_details"],
            "uses_index": ilp_metrics[q_name]["uses_index"]
        }
        if not matches:
            all_checksums_match = False
        print(f"  - {q_name:<25}: Hash Match = {matches} | Plan = {ilp_metrics[q_name]['plan_details'][0]}")

    assert all_checksums_match, "FATAL: Query result checksum divergence detected between baseline and indexed database!"
    print("All query result checksums verified 100% IDENTICAL.")

    master_results = {
        "candidate_index_sizes": index_sizes,
        "candidate_index_benefits": index_benefits,
        "formulation_solvers": {
            "Chapter6_ILP": ilp_res,
            "Greedy": greedy_res,
            "Exhaustive": exh_res
        },
        "configurations_evaluation": eval_results,
        "checksum_verification": checksum_verification
    }

    # 8. Save Results & Plots
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "dbms_benchmark_results.json"), "w") as f:
        json.dump(master_results, f, indent=2)

    generate_plots(master_results, out_dir)
    print("\nDBMS benchmarks complete. Results saved to tracks/H_dbms/results/")
    return master_results


def generate_plots(data: Dict[str, Any], out_dir: str):
    """Generates per-query latency comparison and storage-vs-latency trade-off plots."""
    configs = data["configurations_evaluation"]
    query_names = list(WORKLOAD_QUERIES.keys())

    # Plot 1: Per-Query Latency Comparison (Log scale)
    fig, ax1 = plt.subplots(figsize=(11, 5))
    x = np.arange(len(query_names))
    width = 0.25

    baseline_lats = [configs["Baseline_Unindexed"]["query_metrics"][q]["mean_latency_ms"] for q in query_names]
    greedy_lats = [configs["Greedy_Selection"]["query_metrics"][q]["mean_latency_ms"] for q in query_names]
    ilp_lats = [configs["Chapter6_ILP"]["query_metrics"][q]["mean_latency_ms"] for q in query_names]

    r1 = ax1.bar(x - width, baseline_lats, width, label='Baseline (Unindexed)', color='#d62728', alpha=0.9)
    r2 = ax1.bar(x, greedy_lats, width, label='Greedy Index Selection', color='#ff7f0e', alpha=0.9)
    r3 = ax1.bar(x + width, ilp_lats, width, label='Chapter 6 0/1 Knapsack ILP', color='#2ca02c', alpha=0.9)

    ax1.set_ylabel('Mean Execution Latency (ms)', fontsize=11, fontweight='bold')
    ax1.set_title('Track H: Query Latency Reduction under Index Selection (SQLite)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(query_names, rotation=15, ha='right', fontsize=9)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dbms_query_latency_comparison.png"), dpi=300)
    plt.close()

    # Plot 2: Storage Overhead vs Workload Speedup
    fig, ax2 = plt.subplots(figsize=(8, 4.5))
    for cfg_name, res in configs.items():
        storage_kb = res["storage_bytes"] / 1024.0
        speedup = res["speedup"]
        ax2.scatter(storage_kb, speedup, s=120, zorder=5, label=cfg_name)
        ax2.annotate(f"{cfg_name}\n({speedup:.2f}x)", (storage_kb, speedup),
                     textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)

    ax2.set_xlabel('Index Storage Overhead (KB)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Workload Speedup Factor (x)', fontsize=11, fontweight='bold')
    ax2.set_title('Track H: Index Storage vs Workload Speedup Pareto Frontier', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dbms_storage_vs_speedup_frontier.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    run_dbms_benchmarks()
