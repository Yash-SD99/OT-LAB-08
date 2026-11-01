"""Comprehensive Benchmark Suite for Quantum Algorithms & QML Advantage (Track I).

Generates:
1. Deutsch-Jozsa query scaling (1 vs 2^{n-1} + 1).
2. Bernstein-Vazirani query scaling (1 vs n).
3. Grover's search scaling & simulation crossover analysis.
4. QPE precision and Shor's algorithm order finding for N=15.
5. VQC vs Classical MLP classification on non-linear dataset.
6. Publication-quality diagnostic plots in results/.
7. Detailed JSON results summary.

DISCLAIMER: All quantum simulations represent noise-model simulation on classical CPU, not real hardware.
"""

import json
import os
import sys
import time
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib.pyplot as plt
import numpy as np

from tracks.I_quantum_algos_qml.fundamental_algos import (
    DeutschJozsaAlgorithm,
    make_dj_oracle,
    BernsteinVaziraniAlgorithm,
    GroverSearchAlgorithm,
    QuantumPhaseEstimation,
    ShorsAlgorithmDemo,
)
from tracks.I_quantum_algos_qml.vqc_qml import (
    generate_concentric_circles,
    VariationalQuantumClassifier,
    ClassicalMLP,
)


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def benchmark_deutsch_jozsa() -> List[Dict]:
    """Evaluates Deutsch-Jozsa across input sizes n in [2, 8]."""
    results = []
    print("\n--- Running Deutsch-Jozsa Benchmarks ---")
    for n in range(2, 9):
        dj = DeutschJozsaAlgorithm(n)
        
        # Test Constant
        t0 = time.perf_counter()
        res_const = dj.run(make_dj_oracle(n, "constant_one"))
        t_const = time.perf_counter() - t0
        
        # Test Balanced
        t0 = time.perf_counter()
        res_bal = dj.run(make_dj_oracle(n, "balanced", seed=42))
        t_bal = time.perf_counter() - t0
        
        entry = {
            "n_inputs": n,
            "quantum_queries": 1,
            "classical_deterministic_worst": (2**(n - 1)) + 1,
            "speedup_queries": ((2**(n - 1)) + 1) / 1.0,
            "const_classified": res_const["classification"],
            "bal_classified": res_bal["classification"],
            "wall_clock_time_sec": t_const + t_bal
        }
        results.append(entry)
        print(f"n={n}: Quantum=1 query | Classical={entry['classical_deterministic_worst']} queries | Ratio={entry['speedup_queries']:.1f}x")
    return results


def benchmark_bernstein_vazirani() -> List[Dict]:
    """Evaluates Bernstein-Vazirani across bitstring lengths n in [2, 8]."""
    results = []
    print("\n--- Running Bernstein-Vazirani Benchmarks ---")
    rng = np.random.default_rng(42)
    for n in range(2, 9):
        secret = "".join(str(b) for b in rng.choice([0, 1], size=n))
        bv = BernsteinVaziraniAlgorithm(secret)
        
        t0 = time.perf_counter()
        res = bv.run()
        dt = time.perf_counter() - t0
        
        entry = {
            "n_bits": n,
            "secret": secret,
            "recovered": res["recovered_secret"],
            "success": res["success"],
            "quantum_queries": 1,
            "classical_queries": n,
            "wall_clock_time_sec": dt
        }
        results.append(entry)
        print(f"n={n}: Secret={secret} -> Recovered={res['recovered_secret']} | Q=1 vs C={n}")
    return results


def benchmark_grover_scaling() -> List[Dict]:
    """Evaluates Grover search scaling and simulation crossover for n in [2, 10]."""
    results = []
    print("\n--- Running Grover Search Scaling & Crossover ---")
    rng = np.random.default_rng(123)
    
    for n in range(2, 11):
        N = 2**n
        target = int(rng.integers(0, N))
        grover = GroverSearchAlgorithm(n_qubits=n, target_state=target)
        
        # Measure quantum simulation wall-clock
        t0 = time.perf_counter()
        res = grover.run()
        t_quantum_sim = time.perf_counter() - t0
        
        # Measure classical linear search wall-clock
        t0 = time.perf_counter()
        # Classical linear scan
        db = np.arange(N)
        found_idx = -1
        classical_queries = 0
        for i in range(N):
            classical_queries += 1
            if db[i] == target:
                found_idx = i
                break
        t_classical = time.perf_counter() - t0
        
        entry = {
            "n_qubits": n,
            "database_size": N,
            "target": target,
            "optimal_iterations": res["optimal_iterations"],
            "success_probability": res["success_probability"],
            "theoretical_grover_queries": res["optimal_iterations"],
            "theoretical_classical_avg_queries": N / 2.0,
            "query_speedup": (N / 2.0) / res["optimal_iterations"],
            "quantum_sim_time_sec": t_quantum_sim,
            "classical_search_time_sec": t_classical
        }
        results.append(entry)
        print(f"n={n} (N={N:4d}): Grover Queries={entry['theoretical_grover_queries']:3d} | Classical Avg={entry['theoretical_classical_avg_queries']:5.1f} | P(success)={res['success_probability']:.4f}")
    return results


def benchmark_qpe_and_shor() -> Dict:
    """Evaluates QPE phase accuracy and Shor's order finding on N=15."""
    print("\n--- Running QPE & Shor's Algorithm (N=15) ---")
    qpe_trials = []
    test_thetas = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75]
    for th in test_thetas:
        for m in [3, 4, 5]:
            qpe = QuantumPhaseEstimation(m_counting=m, target_theta=th)
            res = qpe.run()
            qpe_trials.append({
                "m_counting": m,
                "true_theta": th,
                "estimated_theta": res["estimated_theta"],
                "phase_error": res["phase_error"],
                "probability": res["max_probability"]
            })

    # Run Shor's algorithm factoring N=15 with a=7
    shor = ShorsAlgorithmDemo(N=15, a=7, m_counting=4)
    shor_res = shor.run()
    print(f"Shor N=15, a=7: Found order r={shor_res['found_order_r']}, Factors={shor_res['factors']}, Success={shor_res['success']}")

    return {
        "qpe_trials": qpe_trials,
        "shor_factoring": {
            "N": shor_res["N"],
            "a": shor_res["a"],
            "true_order_r": shor_res["true_order_r"],
            "found_order_r": shor_res["found_order_r"],
            "factors": shor_res["factors"],
            "success": shor_res["success"],
            "peak_indices": shor_res["peak_indices"],
            "simulated_counts": shor_res["simulated_counts"].tolist()
        }
    }


def benchmark_vqc_vs_mlp() -> Dict:
    """Trains and compares VQC vs Classical MLP on non-linear concentric circles."""
    print("\n--- Running VQC vs Classical MLP Benchmarks ---")
    X, y = generate_concentric_circles(n_samples=160, noise=0.08, factor=0.45, seed=42)
    split = 120
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # 1. Variational Quantum Classifier (8 parameters)
    vqc = VariationalQuantumClassifier(n_layers=2, seed=42)
    t0 = time.perf_counter()
    vqc_losses = vqc.fit(X_train, y_train, epochs=40, lr=0.1, batch_size=24, seed=42)
    t_vqc = time.perf_counter() - t0
    vqc_train_acc = float(np.mean(vqc.predict_classes(X_train) == y_train))
    vqc_test_acc = float(np.mean(vqc.predict_classes(X_test) == y_test))

    # 2. Classical MLP (13 parameters)
    mlp = ClassicalMLP(hidden_dim=3, seed=42)
    t0 = time.perf_counter()
    mlp_losses = mlp.fit(X_train, y_train, epochs=40, lr=0.1, batch_size=24, seed=42)
    t_mlp = time.perf_counter() - t0
    mlp_train_acc = float(np.mean(mlp.predict_classes(X_train) == y_train))
    mlp_test_acc = float(np.mean(mlp.predict_classes(X_test) == y_test))

    print(f"VQC (8 params):  Train Acc={vqc_train_acc*100:.1f}% | Test Acc={vqc_test_acc*100:.1f}% | Final Loss={vqc_losses[-1]:.4f} | Time={t_vqc:.2f}s")
    print(f"MLP (13 params): Train Acc={mlp_train_acc*100:.1f}% | Test Acc={mlp_test_acc*100:.1f}% | Final Loss={mlp_losses[-1]:.4f} | Time={t_mlp:.2f}s")

    # Generate decision boundary grids for visualization
    x0_min, x0_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    x1_min, x1_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x0_min, x0_max, 35), np.linspace(x1_min, x1_max, 35))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    vqc_grid_preds = vqc.predict(grid_points).reshape(xx.shape)
    mlp_grid_preds = mlp.predict(grid_points).reshape(xx.shape)

    return {
        "vqc": {
            "n_params": vqc.n_params,
            "train_acc": vqc_train_acc,
            "test_acc": vqc_test_acc,
            "final_loss": float(vqc_losses[-1]),
            "loss_history": [float(l) for l in vqc_losses],
            "train_time_sec": t_vqc,
            "grid_preds": vqc_grid_preds.tolist()
        },
        "mlp": {
            "n_params": mlp.n_params,
            "train_acc": mlp_train_acc,
            "test_acc": mlp_test_acc,
            "final_loss": float(mlp_losses[-1]),
            "loss_history": [float(l) for l in mlp_losses],
            "train_time_sec": t_mlp,
            "grid_preds": mlp_grid_preds.tolist()
        },
        "grid_meta": {
            "xx": xx.tolist(),
            "yy": yy.tolist(),
            "X_train": X_train.tolist(),
            "y_train": y_train.tolist(),
            "X_test": X_test.tolist(),
            "y_test": y_test.tolist()
        }
    }


def generate_plots(grover_data: List[Dict], vqc_mlp_data: Dict, shor_data: Dict) -> None:
    """Generates all publication-grade plots in results/."""
    # Plot 1: Grover Scaling & Crossover
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    N_vals = [d["database_size"] for d in grover_data]
    q_grover = [d["theoretical_grover_queries"] for d in grover_data]
    q_classical = [d["theoretical_classical_avg_queries"] for d in grover_data]
    t_q_sim = [d["quantum_sim_time_sec"] for d in grover_data]
    t_c_scan = [d["classical_search_time_sec"] for d in grover_data]

    # Subplot 1: Query Complexity
    ax1.plot(N_vals, q_classical, "r--o", label="Classical Linear Search O(N/2)", linewidth=2)
    ax1.plot(N_vals, q_grover, "b-s", label=r"Grover Search $\mathcal{O}(\sqrt{N})$", linewidth=2)
    ax1.set_xlabel("Database Size $N = 2^n$", fontsize=11)
    ax1.set_ylabel("Query Count to Oracle", fontsize=11)
    ax1.set_title("Grover Query Advantage: $O(\\sqrt{N})$ vs $O(N)$", fontsize=12, fontweight="bold")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    # Subplot 2: Wall-Clock Simulator Reality (Honesty Rule)
    ax2.plot(N_vals, t_c_scan, "r--o", label="Classical Memory Lookup", linewidth=2)
    ax2.plot(N_vals, t_q_sim, "b-s", label="Statevector Simulator Overhead (Classical CPU)", linewidth=2)
    ax2.set_xlabel("Database Size $N = 2^n$", fontsize=11)
    ax2.set_ylabel("Wall-Clock Time (seconds)", fontsize=11)
    ax2.set_title("Simulator Reality: $\\mathcal{O}(2^n)$ Matrix Contraction", fontsize=12, fontweight="bold")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)
    
    plt.suptitle("Grover Scaling & Crossover (noise-model simulation, not real hardware)", fontsize=13, fontstyle="italic")
    plt.tight_layout()
    plot1_path = os.path.join(RESULTS_DIR, "grover_scaling_crossover.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()

    # Plot 2: VQC vs MLP Decision Boundaries & Training Loss
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    meta = vqc_mlp_data["grid_meta"]
    xx = np.array(meta["xx"])
    yy = np.array(meta["yy"])
    X_train = np.array(meta["X_train"])
    y_train = np.array(meta["y_train"])
    
    vqc_grid = np.array(vqc_mlp_data["vqc"]["grid_preds"])
    mlp_grid = np.array(vqc_mlp_data["mlp"]["grid_preds"])

    # VQC Decision Boundary
    ax = axes[0]
    ax.contourf(xx, yy, vqc_grid, levels=20, cmap="coolwarm", alpha=0.7)
    ax.contour(xx, yy, vqc_grid, levels=[0.0], colors="k", linewidths=2)
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c="blue", edgecolors="k", label="Class +1")
    ax.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], c="red", edgecolors="k", label="Class -1")
    ax.set_title(f"VQC (8 params): Test Acc {vqc_mlp_data['vqc']['test_acc']*100:.1f}%\n(noise-model simulation, not real hardware)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Feature $x_0$ (radians)")
    ax.set_ylabel("Feature $x_1$ (radians)")
    ax.legend(loc="lower right")

    # Classical MLP Decision Boundary
    ax = axes[1]
    ax.contourf(xx, yy, mlp_grid, levels=20, cmap="coolwarm", alpha=0.7)
    ax.contour(xx, yy, mlp_grid, levels=[0.0], colors="k", linewidths=2)
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c="blue", edgecolors="k", label="Class +1")
    ax.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], c="red", edgecolors="k", label="Class -1")
    ax.set_title(f"Classical MLP (13 params): Test Acc {vqc_mlp_data['mlp']['test_acc']*100:.1f}%\n(Classical CPU Baseline)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Feature $x_0$ (radians)")
    ax.legend(loc="lower right")

    # Loss Curves
    ax = axes[2]
    epochs = np.arange(1, len(vqc_mlp_data["vqc"]["loss_history"]) + 1)
    ax.plot(epochs, vqc_mlp_data["vqc"]["loss_history"], "b-", label=f"VQC MSE Loss (Final={vqc_mlp_data['vqc']['final_loss']:.3f})", linewidth=2)
    ax.plot(epochs, vqc_mlp_data["mlp"]["loss_history"], "r--", label=f"MLP MSE Loss (Final={vqc_mlp_data['mlp']['final_loss']:.3f})", linewidth=2)
    ax.set_xlabel("Epochs", fontsize=11)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=11)
    ax.set_title("Training Loss Convergence", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plot2_path = os.path.join(RESULTS_DIR, "vqc_vs_mlp_decision_boundary.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()

    # Plot 3: QPE & Shor's Factorization Spectrum
    fig, ax = plt.subplots(figsize=(9, 4.8))
    counts = np.array(shor_data["shor_factoring"]["simulated_counts"])
    states = np.arange(len(counts))
    bars = ax.bar(states, counts, color="indigo", alpha=0.8, edgecolor="black")
    
    # Highlight peaks
    for idx in shor_data["shor_factoring"]["peak_indices"]:
        bars[idx].set_color("crimson")
        phase = idx / len(counts)
        ax.annotate(f"State {idx}\n$\\theta={phase:.2f}$", xy=(idx, counts[idx]),
                    xytext=(idx, counts[idx] + 0.03), ha="center", fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1))

    ax.set_xlabel("Counting Register State $k \\in [0, 15]$ (Binary Fraction $k/16$)", fontsize=11)
    ax.set_ylabel("Measurement Probability", fontsize=11)
    ax.set_title(f"Shor's Factoring Spectrum: $N=15, a=7$ $\\rightarrow$ Period $r=4$ $\\rightarrow$ Factors {shor_data['shor_factoring']['factors']}\n(noise-model simulation, not real hardware)", fontsize=12, fontweight="bold")
    ax.set_xticks(states)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot3_path = os.path.join(RESULTS_DIR, "qft_qpe_shor_diagnostics.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()

    print(f"\nAll plots saved successfully to {RESULTS_DIR}")


def main():
    print("=" * 70)
    print("Executing Track I: Quantum Algorithms & QML Advantage Benchmarks")
    print("NOTE: All results represent noise-model simulation on classical CPU.")
    print("=" * 70)

    dj_data = benchmark_deutsch_jozsa()
    bv_data = benchmark_bernstein_vazirani()
    grover_data = benchmark_grover_scaling()
    shor_data = benchmark_qpe_and_shor()
    vqc_mlp_data = benchmark_vqc_vs_mlp()

    # Save plots
    generate_plots(grover_data, vqc_mlp_data, shor_data)

    # Save comprehensive results JSON
    benchmark_payload = {
        "metadata": {
            "track": "Track I: Quantum Algorithms & QML Advantage",
            "student": "Yash Deshmukh",
            "roll_no": "241070018",
            "institution": "VJTI Mumbai",
            "disclaimer": "noise-model simulation, not real hardware",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        },
        "deutsch_jozsa": dj_data,
        "bernstein_vazirani": bv_data,
        "grover_scaling": grover_data,
        "qpe_and_shor": shor_data,
        "vqc_vs_mlp": {
            "vqc": {
                "n_params": vqc_mlp_data["vqc"]["n_params"],
                "train_acc": vqc_mlp_data["vqc"]["train_acc"],
                "test_acc": vqc_mlp_data["vqc"]["test_acc"],
                "final_loss": vqc_mlp_data["vqc"]["final_loss"],
                "train_time_sec": vqc_mlp_data["vqc"]["train_time_sec"]
            },
            "mlp": {
                "n_params": vqc_mlp_data["mlp"]["n_params"],
                "train_acc": vqc_mlp_data["mlp"]["train_acc"],
                "test_acc": vqc_mlp_data["mlp"]["test_acc"],
                "final_loss": vqc_mlp_data["mlp"]["final_loss"],
                "train_time_sec": vqc_mlp_data["mlp"]["train_time_sec"]
            }
        }
    }

    json_path = os.path.join(RESULTS_DIR, "quantum_algos_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(benchmark_payload, f, indent=2)
    print(f"Benchmark results JSON written to {json_path}")


if __name__ == "__main__":
    main()
