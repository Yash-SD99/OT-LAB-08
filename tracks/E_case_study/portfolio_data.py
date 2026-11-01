"""Realistic Financial Market Dataset Generator (Track E).

Generates realistic expected returns and positive-definite covariance matrices
for N=20 assets across multiple economic sectors using a multi-factor risk model:
    Sigma = B @ B.T + D
where:
- B is the factor loadings matrix (market, size, value, momentum factors)
- D is the diagonal idiosyncratic risk matrix
guaranteeing mathematical positive-definiteness (Sigma > 0).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


SECTORS = {
    "TECH": ["AAPL_synth", "MSFT_synth", "GOOG_synth", "NVDA_synth"],
    "FINANCE": ["JPM_synth", "BAC_synth", "GS_synth", "V_synth"],
    "HEALTHCARE": ["JNJ_synth", "PFE_synth", "UNH_synth", "ABBV_synth"],
    "ENERGY": ["XOM_synth", "CVX_synth", "COP_synth", "SLB_synth"],
    "CONSUMER": ["PG_synth", "KO_synth", "WMT_synth", "MCD_synth"],
}


@dataclass
class PortfolioDataset:
    asset_names: List[str]
    expected_returns: np.ndarray        # mu in R^N
    covariance_matrix: np.ndarray       # Sigma in R^{N x N}
    initial_weights: np.ndarray         # w_0 in R^N
    min_weight: float                   # w_min (e.g. 0.05)
    max_weight: float                   # w_max (e.g. 0.40)
    fixed_trans_cost: float             # c_fixed (e.g. 0.002)
    prop_trans_cost: float              # c_prop (e.g. 0.005)


def generate_portfolio_market_data(
    seed: int = 42,
    num_factors: int = 3,
    min_weight: float = 0.05,
    max_weight: float = 0.40,
    fixed_trans_cost: float = 0.002,
    prop_trans_cost: float = 0.005
) -> PortfolioDataset:
    """Generates synthetic multi-asset market parameters with guaranteed positive-definite Sigma."""
    rng = np.random.default_rng(seed)

    asset_names = []
    for sector, tickers in SECTORS.items():
        asset_names.extend(tickers)
    n = len(asset_names)

    # 1. Expected annual returns (5% to 22%)
    base_returns = {
        "TECH": 0.18,
        "FINANCE": 0.12,
        "HEALTHCARE": 0.10,
        "ENERGY": 0.14,
        "CONSUMER": 0.08
    }
    mu = np.zeros(n)
    idx = 0
    for sector, tickers in SECTORS.items():
        for _ in tickers:
            mu[idx] = base_returns[sector] + rng.normal(0, 0.02)
            idx += 1

    # 2. Factor Risk Model Covariance Matrix: Sigma = B @ B.T + diag(s_eps^2)
    # B: factor exposures (e.g. Market Beta, Size, Value)
    B = rng.normal(0.08, 0.03, size=(n, num_factors))
    # Idiosyncratic variance
    s_eps = rng.uniform(0.04, 0.08, size=n)
    D = np.diag(s_eps ** 2)

    Sigma = B @ B.T + D
    # Symmetrize and ensure positive definiteness
    Sigma = 0.5 * (Sigma + Sigma.T)
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-5:
        Sigma += (1e-5 - min_eig + 1e-4) * np.eye(n)

    # 3. Initial Baseline Portfolio: Uniform allocation w_0 = 1/N
    w_0 = np.ones(n) / n

    return PortfolioDataset(
        asset_names=asset_names,
        expected_returns=mu,
        covariance_matrix=Sigma,
        initial_weights=w_0,
        min_weight=min_weight,
        max_weight=max_weight,
        fixed_trans_cost=fixed_trans_cost,
        prop_trans_cost=prop_trans_cost
    )
