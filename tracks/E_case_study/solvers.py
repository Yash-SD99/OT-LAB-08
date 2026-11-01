"""Portfolio Optimization Solvers under Non-Convex Cardinality & Transaction Costs (Track E).

Implements four distinct algorithmic strategies:
1. Exact / Branch-and-Bound MIQP Solver (Global Optimum via Subspace Branching)
2. Greedy Forward Selection Solver (Greedy Heuristic with QP Projection)
3. Simulated Annealing Solver (Stochastic Local Search with Simplex Projection)
4. Genetic Algorithm Solver (Population-Based Evolutionary Metaheuristic)
"""

from __future__ import annotations
import copy
import itertools
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.optimize import minimize

from tracks.E_case_study.portfolio_data import PortfolioDataset


@dataclass
class PortfolioSolution:
    weights: np.ndarray                 # w in R^N
    active_mask: np.ndarray             # z in {0, 1}^N
    cardinality: int                    # sum(z)
    objective_value: float              # f(w, z)
    expected_return: float              # mu^T w
    portfolio_variance: float           # w^T Sigma w
    transaction_cost: float             # Fixed + Proportional costs
    sharpe_ratio: float                 # (mu^T w) / sqrt(w^T Sigma w)
    solve_time_ms: float                # Wall-clock execution time
    is_feasible: bool                   # All constraints strictly satisfied
    method_name: str


def compute_portfolio_metrics(
    w: np.ndarray,
    data: PortfolioDataset,
    lambda_risk: float = 1.0,
    tol: float = 1e-4
) -> Tuple[float, float, float, float, bool]:
    """Computes objective, return, variance, transaction cost, and feasibility."""
    n = len(w)
    z = (w > tol).astype(int)

    # 1. Expected Return & Risk
    port_ret = float(data.expected_returns @ w)
    port_var = float(w @ data.covariance_matrix @ w)

    # 2. Transaction Costs (Non-smooth fixed charge + proportional cost)
    turnover = np.abs(w - data.initial_weights)
    trades = (turnover > tol).astype(float)
    fixed_cost = data.fixed_trans_cost * np.sum(trades)
    prop_cost = data.prop_trans_cost * np.sum(turnover)
    trans_cost = fixed_cost + prop_cost

    # 3. Overall Non-Convex Objective: min 0.5 * Var - lambda * Return + TransCost
    obj = 0.5 * port_var - lambda_risk * port_ret + trans_cost

    # 4. Feasibility Check
    budget_ok = abs(np.sum(w) - 1.0) <= 1e-3
    nonneg_ok = np.all(w >= -1e-5)
    bounds_ok = True
    for i in range(n):
        if z[i] == 1:
            if w[i] < data.min_weight - 1e-4 or w[i] > data.max_weight + 1e-4:
                bounds_ok = False
                break
        else:
            if abs(w[i]) > tol:
                bounds_ok = False
                break

    is_feasible = budget_ok and nonneg_ok and bounds_ok
    return obj, port_ret, port_var, trans_cost, is_feasible


def solve_continuous_qp_subproblem(
    active_indices: List[int],
    data: PortfolioDataset,
    lambda_risk: float = 1.0
) -> Optional[np.ndarray]:
    """Solves the continuous convex QP over a fixed subset of active assets:
    min_{w_S} 0.5 * w_S^T Sigma_{SS} w_S - lambda * mu_S^T w_S + sum c_prop * |w_i - w_0|
    s.t. sum w_S = 1, w_min <= w_i <= w_max
    """
    k = len(active_indices)
    if k == 0 or k * data.min_weight > 1.0001 or k * data.max_weight < 0.9999:
        return None  # Infeasible subset

    mu_s = data.expected_returns[active_indices]
    sigma_s = data.covariance_matrix[np.ix_(active_indices, active_indices)]
    w0_s = data.initial_weights[active_indices]

    def sub_obj(w_sub):
        var = w_sub @ sigma_s @ w_sub
        ret = mu_s @ w_sub
        prop = data.prop_trans_cost * np.sum(np.abs(w_sub - w0_s))
        return 0.5 * var - lambda_risk * ret + prop

    # Constraints: sum(w_sub) = 1
    constraints = [{'type': 'eq', 'fun': lambda w_sub: np.sum(w_sub) - 1.0}]
    bounds = [(data.min_weight, data.max_weight) for _ in range(k)]

    # Initial guess: equal weighting
    w_init = np.ones(k) / k

    res = minimize(sub_obj, w_init, method='SLSQP', bounds=bounds, constraints=constraints,
                   options={'ftol': 1e-7, 'maxiter': 200})

    if res.success and abs(np.sum(res.x) - 1.0) <= 1e-3:
        w_full = np.zeros(len(data.expected_returns))
        w_full[active_indices] = np.clip(res.x, data.min_weight, data.max_weight)
        # Renormalize to exact 1.0
        w_full[active_indices] /= np.sum(w_full[active_indices])
        return w_full
    return None


# =============================================================================
# Solver 1: Exact / Branch-and-Bound MIQP Solver
# =============================================================================
class MIQPSolver:
    """Chapter 6 Method: Branch-and-Bound / Subspace Search for Exact Global Optimum."""

    def __init__(self, data: PortfolioDataset, max_cardinality: int = 4, lambda_risk: float = 1.0):
        self.data = data
        self.max_cardinality = max_cardinality
        self.lambda_risk = lambda_risk

    def solve(self) -> PortfolioSolution:
        t0 = time.perf_counter()
        n = len(self.data.expected_returns)
        best_w = np.zeros(n)
        best_obj = float('inf')

        # Continuous relaxation lower bound (unconstrained Markowitz)
        # Branch over all subsets of size k in [ceil(1/w_max), max_cardinality]
        min_k = int(np.ceil(1.0 / self.data.max_weight))
        max_k = self.max_cardinality

        for k in range(min_k, max_k + 1):
            for subset in itertools.combinations(range(n), k):
                w_cand = solve_continuous_qp_subproblem(list(subset), self.data, self.lambda_risk)
                if w_cand is not None:
                    obj, _, _, _, feasible = compute_portfolio_metrics(w_cand, self.data, self.lambda_risk)
                    if feasible and obj < best_obj:
                        best_obj = obj
                        best_w = w_cand

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        obj, port_ret, port_var, trans_cost, feasible = compute_portfolio_metrics(
            best_w, self.data, self.lambda_risk
        )
        sharpe = port_ret / max(np.sqrt(port_var), 1e-6)

        return PortfolioSolution(
            weights=best_w,
            active_mask=(best_w > 1e-4).astype(int),
            cardinality=int(np.sum(best_w > 1e-4)),
            objective_value=obj,
            expected_return=port_ret,
            portfolio_variance=port_var,
            transaction_cost=trans_cost,
            sharpe_ratio=sharpe,
            solve_time_ms=elapsed_ms,
            is_feasible=feasible,
            method_name="Exact MIQP (Branch-and-Bound)"
        )


# =============================================================================
# Solver 2: Greedy Forward Selection Solver
# =============================================================================
class GreedyPortfolioSolver:
    """Chapter 6 Method: Greedy Forward Selection based on Sharpe Ratios and Marginal Risk."""

    def __init__(self, data: PortfolioDataset, max_cardinality: int = 4, lambda_risk: float = 1.0):
        self.data = data
        self.max_cardinality = max_cardinality
        self.lambda_risk = lambda_risk

    def solve(self) -> PortfolioSolution:
        t0 = time.perf_counter()
        n = len(self.data.expected_returns)

        # Standalone Sharpe ratios
        volatilities = np.sqrt(np.diag(self.data.covariance_matrix))
        sharpe_standalone = self.data.expected_returns / np.maximum(volatilities, 1e-6)
        ranked_assets = np.argsort(-sharpe_standalone)

        min_k = int(np.ceil(1.0 / self.data.max_weight))
        current_subset: List[int] = list(ranked_assets[:min_k])
        best_w = solve_continuous_qp_subproblem(current_subset, self.data, self.lambda_risk)
        best_obj = float('inf')
        if best_w is not None:
            best_obj, _, _, _, _ = compute_portfolio_metrics(best_w, self.data, self.lambda_risk)

        # Greedily add remaining assets up to max_cardinality
        for _ in range(len(current_subset), self.max_cardinality):
            best_candidate = None
            best_cand_w = None
            best_cand_obj = best_obj

            for candidate in range(n):
                if candidate not in current_subset:
                    trial_subset = current_subset + [candidate]
                    w_trial = solve_continuous_qp_subproblem(trial_subset, self.data, self.lambda_risk)
                    if w_trial is not None:
                        obj_trial, _, _, _, feas = compute_portfolio_metrics(w_trial, self.data, self.lambda_risk)
                        if feas and obj_trial < best_cand_obj:
                            best_cand_obj = obj_trial
                            best_candidate = candidate
                            best_cand_w = w_trial

            if best_candidate is not None:
                current_subset.append(best_candidate)
                best_obj = best_cand_obj
                best_w = best_cand_w
            else:
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        obj, port_ret, port_var, trans_cost, feasible = compute_portfolio_metrics(
            best_w, self.data, self.lambda_risk
        )
        sharpe = port_ret / max(np.sqrt(port_var), 1e-6)

        return PortfolioSolution(
            weights=best_w,
            active_mask=(best_w > 1e-4).astype(int),
            cardinality=int(np.sum(best_w > 1e-4)),
            objective_value=obj,
            expected_return=port_ret,
            portfolio_variance=port_var,
            transaction_cost=trans_cost,
            sharpe_ratio=sharpe,
            solve_time_ms=elapsed_ms,
            is_feasible=feasible,
            method_name="Greedy Forward Selection"
        )


# =============================================================================
# Solver 3: Simulated Annealing Solver
# =============================================================================
class SimulatedAnnealingPortfolioSolver:
    """Chapter 6 Method: Simulated Annealing Metaheuristic with Discrete-Continuous Perturbations."""

    def __init__(
        self,
        data: PortfolioDataset,
        max_cardinality: int = 4,
        lambda_risk: float = 1.0,
        initial_temp: float = 1.0,
        cooling_rate: float = 0.95,
        max_iterations: int = 150,
        seed: int = 42
    ):
        self.data = data
        self.max_cardinality = max_cardinality
        self.lambda_risk = lambda_risk
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.seed = seed

    def solve(self) -> PortfolioSolution:
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        n = len(self.data.expected_returns)

        # Initial feasible state
        min_k = int(np.ceil(1.0 / self.data.max_weight))
        current_subset = list(rng.choice(n, size=min_k, replace=False))
        current_w = solve_continuous_qp_subproblem(current_subset, self.data, self.lambda_risk)
        while current_w is None:
            current_subset = list(rng.choice(n, size=min_k, replace=False))
            current_w = solve_continuous_qp_subproblem(current_subset, self.data, self.lambda_risk)

        current_obj, _, _, _, _ = compute_portfolio_metrics(current_w, self.data, self.lambda_risk)
        best_w = np.copy(current_w)
        best_obj = current_obj
        temp = self.initial_temp

        for it in range(self.max_iterations):
            neighbor_subset = list(current_subset)
            # Mutation 1: Swap one active asset with an inactive asset
            if rng.random() < 0.6:
                idx_remove = rng.integers(0, len(neighbor_subset))
                candidate_adds = [i for i in range(n) if i not in neighbor_subset]
                if candidate_adds:
                    neighbor_subset[idx_remove] = rng.choice(candidate_adds)
            # Mutation 2: Add an asset if below max_cardinality
            elif len(neighbor_subset) < self.max_cardinality and rng.random() < 0.5:
                candidate_adds = [i for i in range(n) if i not in neighbor_subset]
                if candidate_adds:
                    neighbor_subset.append(rng.choice(candidate_adds))
            # Mutation 3: Remove an asset if above min_k
            elif len(neighbor_subset) > min_k:
                idx_remove = rng.integers(0, len(neighbor_subset))
                neighbor_subset.pop(idx_remove)

            cand_w = solve_continuous_qp_subproblem(neighbor_subset, self.data, self.lambda_risk)
            if cand_w is not None:
                cand_obj, _, _, _, feas = compute_portfolio_metrics(cand_w, self.data, self.lambda_risk)
                if feas:
                    delta = cand_obj - current_obj
                    if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-6)):
                        current_subset = neighbor_subset
                        current_w = cand_w
                        current_obj = cand_obj

                        if current_obj < best_obj:
                            best_obj = current_obj
                            best_w = np.copy(current_w)

            temp *= self.cooling_rate

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        obj, port_ret, port_var, trans_cost, feasible = compute_portfolio_metrics(
            best_w, self.data, self.lambda_risk
        )
        sharpe = port_ret / max(np.sqrt(port_var), 1e-6)

        return PortfolioSolution(
            weights=best_w,
            active_mask=(best_w > 1e-4).astype(int),
            cardinality=int(np.sum(best_w > 1e-4)),
            objective_value=obj,
            expected_return=port_ret,
            portfolio_variance=port_var,
            transaction_cost=trans_cost,
            sharpe_ratio=sharpe,
            solve_time_ms=elapsed_ms,
            is_feasible=feasible,
            method_name="Simulated Annealing"
        )


# =============================================================================
# Solver 4: Genetic Algorithm Solver
# =============================================================================
class GeneticAlgorithmPortfolioSolver:
    """Chapter 6 Method: Genetic Algorithm with Cardinality Repair Operator."""

    def __init__(
        self,
        data: PortfolioDataset,
        max_cardinality: int = 4,
        lambda_risk: float = 1.0,
        population_size: int = 30,
        generations: int = 25,
        mutation_rate: float = 0.20,
        seed: int = 42
    ):
        self.data = data
        self.max_cardinality = max_cardinality
        self.lambda_risk = lambda_risk
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.seed = seed

    def solve(self) -> PortfolioSolution:
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        n = len(self.data.expected_returns)
        min_k = int(np.ceil(1.0 / self.data.max_weight))

        def repair(mask: np.ndarray) -> np.ndarray:
            active = np.where(mask == 1)[0]
            if len(active) > self.max_cardinality:
                # Drop excess assets
                drop_count = len(active) - self.max_cardinality
                drop_indices = rng.choice(active, size=drop_count, replace=False)
                mask[drop_indices] = 0
            elif len(active) < min_k:
                # Add random missing assets
                inactive = np.where(mask == 0)[0]
                add_count = min_k - len(active)
                add_indices = rng.choice(inactive, size=add_count, replace=False)
                mask[add_indices] = 1
            return mask

        # Initial Population
        population = []
        for _ in range(self.population_size):
            mask = np.zeros(n, dtype=int)
            k_init = rng.integers(min_k, self.max_cardinality + 1)
            init_idx = rng.choice(n, size=k_init, replace=False)
            mask[init_idx] = 1
            population.append(mask)

        best_w = np.zeros(n)
        best_obj = float('inf')

        for gen in range(self.generations):
            scores = []
            solutions = []
            for mask in population:
                active_idx = list(np.where(mask == 1)[0])
                w_cand = solve_continuous_qp_subproblem(active_idx, self.data, self.lambda_risk)
                if w_cand is not None:
                    obj, _, _, _, feas = compute_portfolio_metrics(w_cand, self.data, self.lambda_risk)
                    if feas:
                        fitness = -obj
                        scores.append(fitness)
                        solutions.append(w_cand)
                        if obj < best_obj:
                            best_obj = obj
                            best_w = w_cand
                        continue
                scores.append(-9999.0)
                solutions.append(np.zeros(n))

            # Tournament Selection & Crossover
            next_pop = []
            for _ in range(self.population_size // 2):
                # Tournament 1
                i1, i2 = rng.choice(self.population_size, size=2, replace=False)
                parent1 = population[i1] if scores[i1] > scores[i2] else population[i2]
                # Tournament 2
                j1, j2 = rng.choice(self.population_size, size=2, replace=False)
                parent2 = population[j1] if scores[j1] > scores[j2] else population[j2]

                # Two-point crossover
                pt = rng.integers(1, n)
                child1 = np.concatenate([parent1[:pt], parent2[pt:]])
                child2 = np.concatenate([parent2[:pt], parent1[pt:]])

                # Mutation
                if rng.random() < self.mutation_rate:
                    bit = rng.integers(0, n)
                    child1[bit] = 1 - child1[bit]
                if rng.random() < self.mutation_rate:
                    bit = rng.integers(0, n)
                    child2[bit] = 1 - child2[bit]

                next_pop.append(repair(child1))
                next_pop.append(repair(child2))

            population = next_pop

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        obj, port_ret, port_var, trans_cost, feasible = compute_portfolio_metrics(
            best_w, self.data, self.lambda_risk
        )
        sharpe = port_ret / max(np.sqrt(port_var), 1e-6)

        return PortfolioSolution(
            weights=best_w,
            active_mask=(best_w > 1e-4).astype(int),
            cardinality=int(np.sum(best_w > 1e-4)),
            objective_value=obj,
            expected_return=port_ret,
            portfolio_variance=port_var,
            transaction_cost=trans_cost,
            sharpe_ratio=sharpe,
            solve_time_ms=elapsed_ms,
            is_feasible=feasible,
            method_name="Genetic Algorithm"
        )
