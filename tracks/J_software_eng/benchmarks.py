"""Comprehensive Profiling and Benchmarking Suite for Track J.

Profiles and benchmarks SeatGeek fuzzywuzzy (1,675 LOC open-source library):
1. cProfile trace of baseline vs Chapter 6 Branch-and-Bound optimizer.
2. Wall-clock runtime scaling (mean +/- std over 5 trials) across corpus sizes M in [500, 1000, 2000, 3000].
3. tracemalloc peak memory profiling.
4. Strict numerical equivalence verification (100% identical top match and score).
5. Generation of publication-grade diagnostic plots in results/.
6. JSON output of all measured metrics.
"""

from collections import Counter
import cProfile
import io
import json
import os
import pstats
import sys
import time
import tracemalloc
from typing import Dict, List, Tuple

# Ensure project and upstream directories are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_DIR = os.path.join(CURRENT_DIR, "fuzzywuzzy_upstream")
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))

for p in [PROJECT_ROOT, CURRENT_DIR, UPSTREAM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib.pyplot as plt
import numpy as np

from fuzzywuzzy import fuzz, process
from tracks.J_software_eng.formulated_optimizer import (
    FormulatedFuzzyOptimizer,
    PreindexedCandidateCorpus,
    BKTree,
)


RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==============================================================================
# 1. Realistic Dataset Generator
# ==============================================================================

def generate_enterprise_corpus(n_entries: int = 3000, seed: int = 42) -> List[str]:
    """Generates a realistic corpus of corporate entities and organization names."""
    rng = np.random.default_rng(seed)
    
    prefixes = [
        "Advanced", "Global", "Apex", "Vertex", "Quantum", "Cyber", "Starlight", "Beacon",
        "Pacific", "Atlantic", "Summit", "Vanguard", "Pinnacle", "Dynamic", "Omni",
        "Strategic", "Unified", "Horizon", "National", "Imperial", "First", "Universal"
    ]
    domains = [
        "Technologies", "Solutions", "Pharmaceuticals", "Financial", "Capital", "Logistics",
        "Consulting", "Aerospace", "Biotech", "Semiconductor", "Robotics", "Analytics",
        "Networks", "Holdings", "Energy", "Systems", "Enterprises", "Industrial", "Communications"
    ]
    suffixes = ["Inc", "Corp", "LLC", "Group", "Ltd", "Partners", "Holdings", "International"]
    locations = [
        "North America", "Europe", "Asia", "London", "New York", "Tokyo", "Singapore",
        "Zurich", "Boston", "Silicon Valley", "Toronto", "Sydney", "Mumbai", "Frankfurt"
    ]
    
    corpus_set = set()
    # Add real landmark brands
    base_brands = [
        "Apple Inc", "Microsoft Corporation", "Amazon Web Services", "Alphabet Google",
        "Meta Platforms", "Nvidia Corporation", "Tesla Motors", "JPMorgan Chase Bank",
        "Berkshire Hathaway", "Johnson & Johnson", "Visa International", "Mastercard Inc",
        "Pfizer Pharmaceuticals", "Novartis International", "Cisco Systems", "Oracle Corporation"
    ]
    for b in base_brands:
        corpus_set.add(b)
        
    while len(corpus_set) < n_entries:
        p = rng.choice(prefixes)
        d = rng.choice(domains)
        s = rng.choice(suffixes)
        loc = rng.choice(locations)
        pattern = rng.integers(0, 4)
        if pattern == 0:
            name = f"{p} {d} {s}"
        elif pattern == 1:
            name = f"{p} {d} of {loc}"
        elif pattern == 2:
            name = f"{p} {d} {s} ({loc})"
        else:
            name = f"{p} & {rng.choice(prefixes)} {d}"
        corpus_set.add(name)
        
    return sorted(list(corpus_set))


def generate_noisy_queries() -> List[str]:
    """Generates representative real-world noisy search queries."""
    return [
        "microsft corperation", "amazn web servcs", "alphabt googl", "jpmorgan chas bank",
        "berkshir hathaway", "tesl motrs", "novartis pharmceutical", "cisco systms",
        "oracl corporatn", "appl inc", "meta platfrms", "johnson & jhnson",
        "globel technolgy", "apex solutns llc", "vanguard captal group", "summit logistcs",
        "pinnacle semconductor", "quantum robocs", "unifed networks inc", "beacon aerospac",
        "pacfic biotch", "cyber consultng group", "horizon enrgy", "atlantc systms",
        "starligt analytcs", "omni communcatns", "first industral llc", "natnal partners",
        "imperal holdings", "universl solutns ltd"
    ]


# ==============================================================================
# 2. cProfile Profiling Harness
# ==============================================================================

def run_cprofile_audit(queries: List[str], corpus: List[str]) -> Tuple[Dict, Dict]:
    """Runs cProfile on baseline vs Chapter 6 Branch-and-Bound optimizer."""
    print("\n--- Running cProfile Profiling Audit ---")
    
    # 1. Baseline cProfile
    pr_base = cProfile.Profile()
    pr_base.enable()
    for q in queries[:10]:
        process.extractOne(q, corpus, scorer=fuzz.ratio)
    pr_base.disable()
    
    s_base = io.StringIO()
    ps_base = pstats.Stats(pr_base, stream=s_base).sort_stats("cumulative")
    ps_base.print_stats(10)
    base_profile_str = s_base.getvalue()
    
    # 2. Optimized cProfile
    opt = FormulatedFuzzyOptimizer(scorer=fuzz.ratio)
    preindexed = PreindexedCandidateCorpus(corpus)
    
    pr_opt = cProfile.Profile()
    pr_opt.enable()
    for q in queries[:10]:
        opt.extractOne(q, preindexed)
    pr_opt.disable()
    
    s_opt = io.StringIO()
    ps_opt = pstats.Stats(pr_opt, stream=s_opt).sort_stats("cumulative")
    ps_opt.print_stats(10)
    opt_profile_str = s_opt.getvalue()
    
    # Extract total calls and time for SequenceMatcher.ratio
    base_stats = {
        "profile_dump": base_profile_str,
        "total_calls": ps_base.total_calls,
        "total_time": ps_base.total_tt
    }
    opt_stats = {
        "profile_dump": opt_profile_str,
        "total_calls": ps_opt.total_calls,
        "total_time": ps_opt.total_tt,
        "optimizer_pruning": dict(opt.stats)
    }
    
    print("\n[Baseline Top Calls]:\n", "\n".join(base_profile_str.splitlines()[:12]))
    print("\n[Optimized Top Calls]:\n", "\n".join(opt_profile_str.splitlines()[:12]))
    
    return base_stats, opt_stats


# ==============================================================================
# 3. Wall-Clock Scaling & Equivalence Benchmarks
# ==============================================================================

def run_scaling_benchmarks(queries: List[str], full_corpus: List[str]) -> List[Dict]:
    """Evaluates runtime across scaling corpus sizes M in [500, 1000, 2000, 3000]."""
    print("\n--- Running Wall-Clock Runtime Scaling Benchmarks ---")
    sizes = [500, 1000, 2000, 3000]
    n_trials = 5
    scaling_results = []
    
    for M in sizes:
        sub_corpus = full_corpus[:M]
        preindexed = PreindexedCandidateCorpus(sub_corpus)
        opt = FormulatedFuzzyOptimizer(scorer=fuzz.ratio)
        
        # Verify exact equivalence first on all queries
        for q in queries:
            base_res = process.extractOne(q, sub_corpus, scorer=fuzz.ratio)
            opt_res = opt.extractOne(q, preindexed)
            assert base_res == opt_res, f"Equivalence mismatch: Base={base_res} vs Opt={opt_res} for query '{q}'"
            
        # 1. Benchmark Baseline
        base_times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            for q in queries:
                process.extractOne(q, sub_corpus, scorer=fuzz.ratio)
            base_times.append(time.perf_counter() - t0)
            
        # 2. Benchmark Optimized BnB
        opt_times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            for q in queries:
                opt.extractOne(q, preindexed)
            opt_times.append(time.perf_counter() - t0)
            
        mean_base = float(np.mean(base_times))
        std_base = float(np.std(base_times))
        mean_opt = float(np.mean(opt_times))
        std_opt = float(np.std(opt_times))
        speedup = mean_base / mean_opt
        
        entry = {
            "corpus_size": M,
            "n_queries": len(queries),
            "baseline_time_mean": mean_base,
            "baseline_time_std": std_base,
            "optimized_time_mean": mean_opt,
            "optimized_time_std": std_opt,
            "speedup_factor": speedup,
            "latency_per_query_base_ms": (mean_base / len(queries)) * 1000.0,
            "latency_per_query_opt_ms": (mean_opt / len(queries)) * 1000.0
        }
        scaling_results.append(entry)
        print(f"M={M:4d}: Baseline={mean_base:.3f}s (+/-{std_base:.3f}) | Optimized={mean_opt:.3f}s (+/-{std_opt:.3f}) | Speedup={speedup:.2f}x ({entry['latency_per_query_opt_ms']:.2f} ms/query)")
        
    return scaling_results


# ==============================================================================
# 4. Memory Profiling (tracemalloc)
# ==============================================================================

def run_memory_profiling(queries: List[str], corpus: List[str]) -> Dict:
    """Profiles memory peak and allocation delta using tracemalloc."""
    print("\n--- Running Memory Allocation Profiling ---")
    
    # Baseline Memory
    tracemalloc.start()
    for q in queries:
        process.extractOne(q, corpus, scorer=fuzz.ratio)
    curr_base, peak_base = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Optimized Memory (including preindexed corpus)
    tracemalloc.start()
    preindexed = PreindexedCandidateCorpus(corpus)
    opt = FormulatedFuzzyOptimizer(scorer=fuzz.ratio)
    for q in queries:
        opt.extractOne(q, preindexed)
    curr_opt, peak_opt = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    mem_data = {
        "baseline_peak_kb": peak_base / 1024.0,
        "optimized_peak_kb": peak_opt / 1024.0,
        "delta_kb": (peak_opt - peak_base) / 1024.0
    }
    print(f"Peak Memory: Baseline={mem_data['baseline_peak_kb']:.1f} KB | Optimized={mem_data['optimized_peak_kb']:.1f} KB")
    return mem_data


# ==============================================================================
# 5. Diagnostic Plot Generator
# ==============================================================================

def generate_plots(scaling_results: List[Dict], base_profile: Dict, opt_profile: Dict, mem_data: Dict) -> None:
    """Generates all publication-grade diagnostic plots in results/."""
    # Plot 1: Runtime Speedup Scaling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    
    M_vals = [d["corpus_size"] for d in scaling_results]
    t_base = [d["baseline_time_mean"] for d in scaling_results]
    t_base_err = [d["baseline_time_std"] for d in scaling_results]
    t_opt = [d["optimized_time_mean"] for d in scaling_results]
    t_opt_err = [d["optimized_time_std"] for d in scaling_results]
    speedups = [d["speedup_factor"] for d in scaling_results]
    
    ax1.errorbar(M_vals, t_base, yerr=t_base_err, fmt="r--o", label="Baseline (fuzzywuzzy.process)", capsize=4, linewidth=2)
    ax1.errorbar(M_vals, t_opt, yerr=t_opt_err, fmt="g-s", label="Optimized (Chapter 6 BnB)", capsize=4, linewidth=2)
    ax1.set_xlabel("Corpus Size $M$ (Candidate Strings)", fontsize=11)
    ax1.set_ylabel("Total Runtime (seconds, 30 queries)", fontsize=11)
    ax1.set_title("Runtime Scaling vs Corpus Size", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)
    
    ax2.plot(M_vals, speedups, "b-^", linewidth=2.5, markersize=8)
    for m, s in zip(M_vals, speedups):
        ax2.annotate(f"{s:.1f}x", xy=(m, s), xytext=(m, s + 0.5), ha="center", fontweight="bold", fontsize=10)
    ax2.set_xlabel("Corpus Size $M$ (Candidate Strings)", fontsize=11)
    ax2.set_ylabel("Speedup Factor ($T_{base} / T_{opt}$)", fontsize=11)
    ax2.set_title("Empirical Speedup Scaling", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.suptitle("Track J: SeatGeek fuzzywuzzy Optimization (1,675 LOC)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p1 = os.path.join(RESULTS_DIR, "runtime_speedup_scaling.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    
    # Plot 2: Call Count & Execution Breakdown
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    labels = ["Baseline Linear Scan", "Optimized BnB"]
    total_calls = [base_profile["total_calls"], opt_profile["total_calls"]]
    cpu_times = [base_profile["total_time"], opt_profile["total_time"]]
    
    ax1.bar(labels, total_calls, color=["salmon", "mediumseagreen"], edgecolor="black", width=0.5)
    for i, v in enumerate(total_calls):
        ax1.text(i, v * 1.02, f"{v:,} calls", ha="center", fontweight="bold")
    ax1.set_ylabel("Total Python Function Calls", fontsize=11)
    ax1.set_title("cProfile Function Call Reduction", fontsize=12, fontweight="bold")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.6)
    
    ax2.bar(labels, cpu_times, color=["firebrick", "seagreen"], edgecolor="black", width=0.5)
    for i, v in enumerate(cpu_times):
        ax2.text(i, v * 1.02, f"{v:.3f} s", ha="center", fontweight="bold")
    ax2.set_ylabel("cProfile Cumulative CPU Time (s)", fontsize=11)
    ax2.set_title("cProfile Execution Time (10 queries)", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    p2 = os.path.join(RESULTS_DIR, "cprofile_call_breakdown.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    
    # Plot 3: Latency Per Query & Memory Footprint
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    lat_base = [d["latency_per_query_base_ms"] for d in scaling_results]
    lat_opt = [d["latency_per_query_opt_ms"] for d in scaling_results]
    
    ax1.plot(M_vals, lat_base, "r--o", label="Baseline ms/query", linewidth=2)
    ax1.plot(M_vals, lat_opt, "g-s", label="Optimized ms/query", linewidth=2)
    ax1.set_xlabel("Corpus Size $M$", fontsize=11)
    ax1.set_ylabel("Latency per Query (ms)", fontsize=11)
    ax1.set_title("Per-Query Latency Scaling", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)
    
    mem_labels = ["Baseline Peak", "Optimized Peak\n(incl. Index Cache)"]
    mem_vals = [mem_data["baseline_peak_kb"], mem_data["optimized_peak_kb"]]
    ax2.bar(mem_labels, mem_vals, color=["cornflowerblue", "royalblue"], edgecolor="black", width=0.5)
    for i, v in enumerate(mem_vals):
        ax2.text(i, v * 1.02, f"{v:.1f} KB", ha="center", fontweight="bold")
    ax2.set_ylabel("Peak Memory (KB)", fontsize=11)
    ax2.set_title("tracemalloc Memory Footprint", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    p3 = os.path.join(RESULTS_DIR, "memory_and_latency_distribution.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    
    print(f"\nAll plots saved to {RESULTS_DIR}")


# ==============================================================================
# 6. Main Runner
# ==============================================================================

def main():
    print("=" * 70)
    print("Executing Track J: Software Engineering Optimization")
    print("Target: SeatGeek fuzzywuzzy (1,675 LOC Open-Source Library, MIT)")
    print("=" * 70)
    
    full_corpus = generate_enterprise_corpus(n_entries=3000, seed=42)
    queries = generate_noisy_queries()
    
    base_profile, opt_profile = run_cprofile_audit(queries, full_corpus[:1000])
    scaling_results = run_scaling_benchmarks(queries, full_corpus)
    mem_data = run_memory_profiling(queries, full_corpus[:1000])
    
    generate_plots(scaling_results, base_profile, opt_profile, mem_data)
    
    payload = {
        "metadata": {
            "track": "Track J: Software Engineering Optimization",
            "project_name": "SeatGeek/fuzzywuzzy",
            "license": "MIT",
            "project_lines_of_code": 1675,
            "student": "Yash Deshmukh",
            "roll_no": "241070018",
            "institution": "VJTI Mumbai",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        },
        "scaling_benchmarks": scaling_results,
        "cprofile_summary": {
            "baseline_total_calls": base_profile["total_calls"],
            "baseline_total_time_sec": base_profile["total_time"],
            "optimized_total_calls": opt_profile["total_calls"],
            "optimized_total_time_sec": opt_profile["total_time"],
            "call_reduction_ratio": base_profile["total_calls"] / opt_profile["total_calls"]
        },
        "memory_profile": mem_data
    }
    
    json_path = os.path.join(RESULTS_DIR, "software_eng_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Benchmark results JSON written to {json_path}")


if __name__ == "__main__":
    main()
