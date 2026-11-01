"""Comprehensive Benchmarking Suite for Track D: Submodular, Parallel, and Distributed Optimization.

Generates reproducible CSV/JSON benchmark data and visualization figures:
1. Submodular Greedy vs Exact Optimal and Nemhauser-Wolsey Bound (1 - 1/e)
2. Process-Based Parallel Strong Scaling, Efficiency, and Amdahl's Law Fitting (P in [1, 2, 4] across 15 seeds)
3. Static vs Work-Stealing on Identical Workloads & Seeds (mean +/- std)
4. Pipeline Parallelism (1F1B bubble analysis), IPC Communication Overhead, and Gustafson Weak Scaling
5. Federated Learning (FedAvg with E in {1, 5, 10} vs Centralized SGD across 15 seeds)
6. Multi-Task Optimization (Equal Weighting vs MGDA vs GradNorm across 15 seeds with Pareto plot)
"""

import json
import os
import platform
import time
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tracks.D_submodular_parallel_distributed.submodular_opt import (
    FacilityLocationSubmodular,
    verify_diminishing_returns,
    greedy_submodular_selection,
    exact_optimal_selection
)
from tracks.D_submodular_parallel_distributed.parallel_scheduling import (
    StaticProcessScheduler,
    WorkStealingSimulator,
    fit_amdahl_gustafson,
    evaluate_pipeline_parallelism,
    measure_ipc_overhead,
    evaluate_gustafson_weak_scaling
)
from tracks.D_submodular_parallel_distributed.federated_fedavg import (
    DistributedDataset,
    run_centralized_training,
    run_fedavg,
    run_federated_multi_seed_benchmarks
)
from tracks.D_submodular_parallel_distributed.multitask_mgda_gradnorm import (
    TwoTaskModel,
    train_equal_weighting,
    train_mgda,
    train_gradnorm,
    run_multitask_pareto_benchmarks
)


def get_hardware_specifications() -> Dict:
    """Collects and returns exact host hardware specifications."""
    try:
        import psutil
        total_ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        phys_cores = psutil.cpu_count(logical=False)
    except ImportError:
        total_ram_gb = "unknown"
        phys_cores = "unknown"
        
    specs = {
        "logical_cpu_count": os.cpu_count(),
        "physical_cores": phys_cores,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "system_ram_gb": total_ram_gb,
        "parallel_execution_engine": "multiprocessing.ProcessPoolExecutor (independent OS processes)"
    }
    return specs


# ==============================================================================
# 1. Submodular Greedy vs Exact Optimal & Guarantee Verification
# ==============================================================================

def run_submodular_benchmarks(results_dir: str) -> Dict:
    """Benchmark 1: Submodular Greedy vs Exact Optimal across budget k."""
    print("Running Benchmark 1: Submodular Greedy vs Exact Optimal (Facility Location)...")
    np.random.seed(42)
    N_cands = 14
    N_targets = 25
    candidates = np.random.uniform(-3, 3, size=(N_cands, 2))
    targets = np.random.uniform(-3, 3, size=(N_targets, 2))
    obj = FacilityLocationSubmodular(candidates, targets, sigma=1.5)
    
    audit = verify_diminishing_returns(obj, n_pairs_to_test=100, seed=42)
    
    budgets = [1, 2, 3, 4, 5]
    greedy_vals = []
    exact_vals = []
    ratios = []
    
    for k in budgets:
        res_g = greedy_submodular_selection(obj, k)
        res_e = exact_optimal_selection(obj, k)
        greedy_vals.append(res_g["final_value"])
        exact_vals.append(res_e["final_value"])
        ratios.append(float(res_g["final_value"] / res_e["final_value"]))
        
    plt.figure(figsize=(8, 5))
    plt.plot(budgets, exact_vals, 's-', label="Exact Optimal f(S*)", lw=2, color="#1f77b4")
    plt.plot(budgets, greedy_vals, 'o-', label="Greedy Selection f(S_greedy)", lw=2, color="#2ca02c")
    theoretical_bound = [v * (1.0 - 1.0 / np.e) for v in exact_vals]
    plt.plot(budgets, theoretical_bound, 'r--', label="Nemhauser Bound (1 - 1/e) * f(S*)", lw=2)
    plt.xlabel("Budget / Selected Facilities (k)")
    plt.ylabel("Submodular Objective Value f(S)")
    plt.title("Submodular Maximization: Greedy vs Exact vs (1 - 1/e) Bound")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "submodular_guarantee.png"), dpi=200)
    plt.close()
    
    return {
        "diminishing_returns_audit": audit,
        "budgets": budgets,
        "greedy_values": greedy_vals,
        "exact_values": exact_vals,
        "approximation_ratios": ratios,
        "nemhauser_theoretical_lower_bound": float(1.0 - 1.0 / np.e),
        "submodular_proof_note": "f(S) is submodular because max_{i in S} w_{ij} satisfies diminishing returns: max(a, c) - a >= max(b, c) - b for a <= b."
    }


# ==============================================================================
# 2 & 3. Process Parallel Scaling & Work Stealing (Identical Workload, 15 Seeds)
# ==============================================================================

def run_parallel_scaling_benchmarks(results_dir: str) -> Dict:
    """Benchmark 2 & 3: Process Parallel Strong Scaling & Work Stealing across 15 seeds."""
    print("Running Benchmark 2 & 3: Process Parallel Scaling & Work Stealing (15 seeds)...")
    p_workers = [1, 2, 4]
    n_seeds = 15
    
    static_results = {p: [] for p in p_workers}
    ws_results = {p: [] for p in p_workers}
    static_imbalances = {p: [] for p in p_workers}
    ws_stolen_counts = {p: [] for p in p_workers}
    
    for s in range(n_seeds):
        # Generate identical heavy-tailed Pareto workload for seed s
        rng = np.random.default_rng(500 + s * 19)
        intensities = list(rng.pareto(a=1.5, size=20) + 0.15)
        
        for p in p_workers:
            # 1. Static Multiprocessing Process Execution
            sched_static = StaticProcessScheduler(n_workers=p)
            res_static = sched_static.run(intensities)
            static_results[p].append(res_static["makespan_ms"])
            static_imbalances[p].append(res_static["load_imbalance_std_ms"])
            
            # 2. Simulated Work-Stealing on the EXACT SAME task list
            sched_ws = WorkStealingSimulator(n_workers=p)
            res_ws = sched_ws.run(intensities, seed=s)
            ws_results[p].append(res_ws["makespan_ms"])
            ws_stolen_counts[p].append(res_ws["stolen_tasks_count"])
            
    summary_static = {}
    summary_ws = {}
    mean_static_runtimes = []
    
    for p in p_workers:
        m_stat = float(np.mean(static_results[p]))
        s_stat = float(np.std(static_results[p]))
        mean_static_runtimes.append(m_stat)
        
        summary_static[p] = {
            "mean_makespan_ms": m_stat,
            "std_makespan_ms": s_stat,
            "mean_imbalance_std_ms": float(np.mean(static_imbalances[p])),
            "speedup": float(mean_static_runtimes[0] / m_stat),
            "efficiency": float((mean_static_runtimes[0] / m_stat) / p)
        }
        
        m_ws = float(np.mean(ws_results[p]))
        s_ws = float(np.std(ws_results[p]))
        summary_ws[p] = {
            "mean_makespan_ms": m_ws,
            "std_makespan_ms": s_ws,
            "mean_stolen_tasks": float(np.mean(ws_stolen_counts[p])),
            "speedup": float(mean_static_runtimes[0] / m_ws)
        }
        print(f"  P={p}: Static Makespan = {m_stat:.1f} +/- {s_stat:.1f} ms (Speedup: {summary_static[p]['speedup']:.2f}x) | "
              f"Work-Stealing (sim) = {m_ws:.1f} +/- {s_ws:.1f} ms")
        
    amdahl_fit = fit_amdahl_gustafson(p_workers, mean_static_runtimes)
    
    # Plot Process Speedup and Amdahl Law Fit
    plt.figure(figsize=(8, 5))
    plt.plot(p_workers, amdahl_fit["measured_speedups"], 'bo-', label="Measured Multiprocessing Speedup S(P)", lw=2)
    plt.plot(p_workers, amdahl_fit["amdahl_speedup_fit"], 'r--', label=f"Amdahl's Fit (serial fraction={amdahl_fit['fitted_serial_fraction']:.2f})", lw=2)
    plt.plot(p_workers, p_workers, 'k:', label="Ideal Linear Speedup S(P) = P", alpha=0.6)
    plt.xlabel("Process Worker Count (P)")
    plt.ylabel("Speedup S(P)")
    plt.xticks(p_workers)
    plt.title("Process-Based Parallel Strong Scaling & Amdahl's Law Fit")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "parallel_speedup_amdahl.png"), dpi=200)
    plt.close()
    
    return {
        "p_workers": p_workers,
        "n_seeds": n_seeds,
        "static_multiprocessing": summary_static,
        "work_stealing_simulation": summary_ws,
        "amdahl_analysis": amdahl_fit
    }


# ==============================================================================
# 4. Pipeline Parallelism, IPC Overhead, and Gustafson Weak Scaling
# ==============================================================================

def run_pipeline_ipc_gustafson_benchmarks(results_dir: str) -> Dict:
    """Benchmark 4: Pipeline bubble analysis, IPC overhead, and Gustafson weak scaling."""
    print("Running Benchmark 4: Pipeline Parallelism, IPC Overhead & Gustafson Weak Scaling...")
    # 1. Pipeline Parallelism Bubble Overhead
    pipeline_res = {}
    for m in [4, 8, 16, 32]:
        pipeline_res[f"M_{m}"] = evaluate_pipeline_parallelism(n_stages=4, n_microbatches=m, stage_time_ms=5.0)
        
    # 2. IPC Communication Overhead
    ipc_res = measure_ipc_overhead(message_sizes_kb=[1, 10, 100, 1000])
    
    # 3. Gustafson Weak Scaling
    weak_res = evaluate_gustafson_weak_scaling(p_workers=[1, 2, 4], base_workload_size=8, n_seeds=5)
    
    return {
        "pipeline_parallelism": pipeline_res,
        "ipc_overhead": ipc_res,
        "gustafson_weak_scaling": weak_res
    }


# ==============================================================================
# 5. Federated Learning (FedAvg across 15 Seeds)
# ==============================================================================

def run_federated_benchmarks(results_dir: str) -> Dict:
    """Benchmark 5: Federated Learning (FedAvg with E in {1, 5, 10} vs Centralized) across 15 seeds."""
    print("Running Benchmark 5: Federated Learning (FedAvg vs Centralized SGD, 15 seeds)...")
    fed_res = run_federated_multi_seed_benchmarks(n_seeds=15)
    
    # Plot sample convergence trajectory from seed 42
    dataset_demo = DistributedDataset(n_clients=10, n_features=8, samples_per_client=50, seed=42)
    res_central = run_centralized_training(dataset_demo, max_epochs=35, lr=0.05)
    res_fed_1 = run_fedavg(dataset_demo, n_rounds=25, local_epochs=1, lr=0.05)
    res_fed_5 = run_fedavg(dataset_demo, n_rounds=25, local_epochs=5, lr=0.05)
    res_fed_10 = run_fedavg(dataset_demo, n_rounds=25, local_epochs=10, lr=0.05)
    
    plt.figure(figsize=(8, 5))
    plt.plot(res_fed_1["loss_history"], label="FedAvg (E=1 local step)", lw=2)
    plt.plot(res_fed_5["loss_history"], label="FedAvg (E=5 local steps)", lw=2)
    plt.plot(res_fed_10["loss_history"], label="FedAvg (E=10 local steps)", lw=2)
    plt.axhline(res_central["final_loss"], color='k', ls='--', label=f"Centralized SGD (loss={res_central['final_loss']:.3f})")
    plt.xlabel("Communication Round")
    plt.ylabel("Global Validation Loss (MSE)")
    plt.title("Federated Learning: Convergence vs Local Epochs E (Non-IID)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fedavg_convergence.png"), dpi=200)
    plt.close()
    
    return fed_res


# ==============================================================================
# 6. Multi-Task Optimization (Equal Weighting vs MGDA vs GradNorm across 15 Seeds)
# ==============================================================================

def run_multitask_benchmarks(results_dir: str) -> Dict:
    """Benchmark 6: Multi-Task Optimization (Equal Weighting vs MGDA vs GradNorm) across 15 seeds."""
    print("Running Benchmark 6: Multi-Task Optimization (15 seeds with Pareto plot)...")
    mt_res = run_multitask_pareto_benchmarks(n_seeds=15)
    
    # Pareto Plot: Task 1 Loss vs Task 2 Loss
    plt.figure(figsize=(8, 5))
    plt.scatter(mt_res["equal_weighting"]["all_l1"], mt_res["equal_weighting"]["all_l2"],
                color="#1f77b4", marker='o', s=60, alpha=0.8,
                label=f"Equal Weighting (Total: {mt_res['equal_weighting']['mean_total_loss']:.3f} +/- {mt_res['equal_weighting']['std_total_loss']:.3f})")
    plt.scatter(mt_res["mgda"]["all_l1"], mt_res["mgda"]["all_l2"],
                color="#ff7f0e", marker='s', s=60, alpha=0.8,
                label=f"MGDA Pareto (Total: {mt_res['mgda']['mean_total_loss']:.3f} +/- {mt_res['mgda']['std_total_loss']:.3f})")
    plt.scatter(mt_res["gradnorm"]["all_l1"], mt_res["gradnorm"]["all_l2"],
                color="#2ca02c", marker='^', s=70, alpha=0.8,
                label=f"GradNorm Dynamic (Total: {mt_res['gradnorm']['mean_total_loss']:.3f} +/- {mt_res['gradnorm']['std_total_loss']:.3f})")
                
    plt.xlabel("Task 1 Loss (L1)")
    plt.ylabel("Task 2 Loss (L2)")
    plt.title("Multi-Task Optimization: Pareto Trade-off Distribution across 15 Seeds")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "multitask_comparison.png"), dpi=200)
    plt.close()
    
    return mt_res


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    hw_specs = get_hardware_specifications()
    print("=== Track D: Distributed & Parallel Optimization Benchmarks ===")
    print(f"Host Specs: {hw_specs['platform']} | Logical CPUs: {hw_specs['logical_cpu_count']} | "
          f"Physical Cores: {hw_specs['physical_cores']} | RAM: {hw_specs['system_ram_gb']} GB")
    print(f"Parallel Execution: {hw_specs['parallel_execution_engine']}\n")
    
    all_results = {}
    all_results["hardware_specifications"] = hw_specs
    all_results["benchmark_1_submodular"] = run_submodular_benchmarks(results_dir)
    all_results["benchmark_2_3_parallel"] = run_parallel_scaling_benchmarks(results_dir)
    all_results["benchmark_4_pipeline_ipc_gustafson"] = run_pipeline_ipc_gustafson_benchmarks(results_dir)
    all_results["benchmark_5_federated"] = run_federated_benchmarks(results_dir)
    all_results["benchmark_6_multitask"] = run_multitask_benchmarks(results_dir)
    
    output_path = os.path.join(results_dir, "parallel_dist_benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\nAll Track D benchmarks completed successfully!")
    print(f"Results and plots saved to: {results_dir}")


if __name__ == "__main__":
    main()
