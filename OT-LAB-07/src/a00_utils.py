# %% [markdown]
# # Lab Assignment 7 — Multiobjective Optimization: 37 Algorithms, Trade-offs and Applications
# **Optimization Techniques Lab · Yash Deshmukh · Roll 241070018 (R = 18)**
#
# Everything below is written from scratch in **Python + NumPy + Matplotlib only** (Section 2A). Run top to bottom.
# Single notebook layout:
# * **A1** `moo_utils` (dominance, Deb rules, sorting, crowding, HV, IGD, Spacing) + problems P1–P14 + unit tests
# * **A2/A3** all 37 algorithms: docstring, unit test, fixed-seed runs, plots, HV / IGD / Spacing, implementation log
# * **Problem studies** (the "Investigate" questions of P1–P14)
# * **B** comparison table · **C** deep dive (algorithms **18 MOEA/D, 30 MOMA, 5 Value Function**) · **D** head-to-head · **E** selection · **F** viva
#
# Conventions: every objective is **minimised** (Max f ≡ Min −f). Budget **10,000 evaluations**, seeds fixed (`seed=0` for single runs, seeds 0–9 for comparisons).
# Set `QUICK = True` below to smoke-test the notebook in ~2 minutes with smaller budgets.

# %%
import numpy as np, time, math, itertools, warnings, sys
import matplotlib.pyplot as plt
try:
    from IPython.display import display, Markdown
except Exception:
    display = print
    Markdown = lambda s: s
warnings.filterwarnings("ignore")
plt.rcParams.update({'figure.dpi': 80, 'axes.grid': True, 'grid.alpha': .3, 'figure.max_open_warning': 0})

QUICK = False            # True = tiny budgets for a fast smoke test
BUDGET = 3000 if QUICK else 10000
NSEEDS = 3 if QUICK else 10
SEEDS = list(range(NSEEDS))

def md(s): display(Markdown(s))
def md_table(headers, rows, fmt="{:.4g}"):
    f = lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
    s = "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n"
    for r in rows: s += "| " + " | ".join(f(v) for v in r) + " |\n"
    md(s)

# %% [markdown]
# ## A1 · Shared toolkit `moo_utils`
# Dominance test, Deb's constrained dominance (3 rules), fast non-dominated sorting, crowding distance, 2-D hypervolume (exact) / Monte-Carlo hypervolume (M ≥ 3), IGD, Spacing, normalisation, Das–Dennis reference points.

# %%
# ------------------------------ moo_utils ------------------------------
def dominates(a, b):
    """a dominates b (minimisation): a<=b everywhere and a<b somewhere."""
    return bool(np.all(a <= b) and np.any(a < b))

def cdominates(fa, cva, fb, cvb):
    """Deb's constrained domination. Rule 1: feasible beats infeasible. Rule 2: both infeasible -> smaller violation wins.
    Rule 3: both feasible -> ordinary dominance."""
    if cva <= 0 and cvb > 0: return True
    if cva > 0 and cvb <= 0: return False
    if cva > 0 and cvb > 0: return cva < cvb
    return dominates(fa, fb)

def dom_matrix(F, CV=None):
    """D[i,j] = True if i dominates j (Deb-constrained if CV is given)."""
    D = (F[:, None, :] <= F[None, :, :]).all(2) & (F[:, None, :] < F[None, :, :]).any(2)
    if CV is not None:
        fe = CV <= 0
        D = np.where(fe[:, None] & fe[None, :], D, False)
        D |= fe[:, None] & ~fe[None, :]
        D |= (~fe[:, None] & ~fe[None, :]) & (CV[:, None] < CV[None, :])
    return D

def nds(F, CV=None):
    """Fast non-dominated sorting (Deb): domination counts, peel fronts. Returns list of index arrays."""
    D = dom_matrix(F, CV); cnt = D.sum(0).astype(int); alive = np.ones(len(F), bool); fronts = []
    while alive.any():
        cur = np.where(alive & (cnt == 0))[0]
        fronts.append(cur); alive[cur] = False; cnt = cnt - D[cur].sum(0)
    return fronts

def crowding(F):
    """Crowding distance of one front (boundary points = inf)."""
    n, m = F.shape; d = np.zeros(n)
    if n <= 2: return np.full(n, np.inf)
    for j in range(m):
        o = np.argsort(F[:, j], kind='stable'); fj = F[o, j]; rg = fj[-1] - fj[0]
        d[o[0]] = d[o[-1]] = np.inf
        if rg > 0: d[o[1:-1]] += (fj[2:] - fj[:-2]) / rg
    return d

def pareto2d(F):
    """Non-dominated subset of a 2-objective set (O(n log n) sweep)."""
    o = np.lexsort((F[:, 1], F[:, 0])); Fs = F[o]
    prev = np.minimum.accumulate(np.r_[np.inf, Fs[:-1, 1]])
    return Fs[Fs[:, 1] < prev]

def hv2d(F, ref):
    """Exact 2-D hypervolume w.r.t. reference point ref (minimisation)."""
    P = F[(F < ref).all(1)]
    if len(P) == 0: return 0.0
    P = P[np.argsort(P[:, 0])]; hv = 0.0; best = ref[1]
    for f1, f2 in P:
        if f2 < best: hv += (ref[0] - f1) * (best - f2); best = f2
    return hv

_MC = {}
def hv_mc(F, ref, K=40000, seed=123):
    """Monte-Carlo hypervolume in the box [0,ref] (objectives already normalised to ~[0,1]); common random numbers."""
    M = F.shape[1]
    key = (M, K, seed)
    if key not in _MC: _MC[key] = np.random.default_rng(seed).random((K, M))
    S = _MC[key] * ref
    P = F[(F < ref).all(1)]
    if len(P) == 0: return 0.0
    hit = np.zeros(K, bool)
    for a in range(0, K, 10000): hit[a:a + 10000] = (P[None, :, :] <= S[a:a + 10000, None, :]).all(2).any(1)
    return hit.mean() * np.prod(ref)

def hypervolume(F, ref):
    F = np.asarray(F, float); ref = np.asarray(ref, float)
    return hv2d(F, ref) if F.shape[1] == 2 else hv_mc(F, ref)

def igd(Z, F):
    """Mean distance from each true-front point in Z to the nearest obtained point in F (smaller = better)."""
    if len(F) == 0: return np.inf
    return np.sqrt(((Z[:, None, :] - F[None, :, :]) ** 2).sum(2)).min(1).mean()

def spacing(F):
    """Standard deviation of nearest-neighbour distances along the obtained front (smaller = more even)."""
    if len(F) < 3: return np.nan
    D = np.sqrt(((F[:, None, :] - F[None, :, :]) ** 2).sum(2)); np.fill_diagonal(D, np.inf)
    return D.min(1).std()

def normalize(F, ideal, nadir): return (F - ideal) / np.maximum(nadir - ideal, 1e-12)

def das_dennis(M, p):
    """Das–Dennis simplex lattice: all weight vectors with components in {0,1/p,..,1} summing to 1."""
    def rec(M, p):
        if M == 1: return [[p]]
        return [[i] + r for i in range(p + 1) for r in rec(M - 1, p - i)]
    return np.array(rec(M, p), float) / p

def thin(Z, n=400):
    if len(Z) <= n: return Z
    o = np.lexsort(Z.T[::-1]); return Z[o[np.linspace(0, len(Z) - 1, n).astype(int)]]

# %% [markdown]
# ### Unit tests of `moo_utils` on hand-checkable cases
# *Car example*: six cars (price in $k, fuel use in L/100 km, both minimised). *Deb's table*: four candidate designs (two feasible, two infeasible).

# %%
cars = np.array([[20, 8], [25, 6], [30, 5], [22, 9], [28, 7], [35, 4.5]], float)   # A B C D E F
names = list("ABCDEF")
fr = nds(cars)
print("Car example, front 1:", [names[i] for i in fr[0]], "| front 2:", [names[i] for i in fr[1]])
assert sorted(fr[0]) == [0, 1, 2, 5] and sorted(fr[1]) == [3, 4]
assert abs(hv2d(cars, np.array([40., 10.])) - 82.5) < 1e-9                      # 40+30+10+2.5 by hand
assert np.isinf(crowding(cars[fr[0]])[[0, 3]]).all()
cd = crowding(cars[fr[0]][np.argsort(cars[fr[0], 0])])
# interior point B: (30-20)/15 + (8-5)/3.5 ; point C: (35-25)/15 + (6-4.5)/3.5
assert abs(cd[1] - ((30 - 20) / 15 + (8 - 5) / 3.5)) < 1e-9
# Deb's constrained-dominance table
deb = [("p1", (2, 3), 0.0), ("p2", (3, 4), 0.0), ("p3", (1, 1), 0.5), ("p4", (0.5, 0.5), 1.2)]
rows = []
for (na, fa, ca), (nb, fb, cb) in itertools.combinations(deb, 2):
    w = na if cdominates(np.array(fa), ca, np.array(fb), cb) else (nb if cdominates(np.array(fb), cb, np.array(fa), ca) else "none")
    rule = "1 (feasible > infeasible)" if (ca <= 0) != (cb <= 0) else ("2 (smaller violation)" if ca > 0 else "3 (dominance)")
    rows.append([na, nb, w, rule])
md_table(["a", "b", "winner", "Deb rule"], rows)
assert rows[0][2] == "p1" and rows[2][2] == "p1" and rows[5][2] == "p3"   # p3 (cv .5) beats p4 though p4 has better objectives
assert abs(igd(np.array([[0., 0.], [1., 1.]]), np.array([[0., 0.], [1., 0.]])) - 0.5) < 1e-12
assert abs(spacing(np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]]))) < 1e-12
print("moo_utils unit tests: PASS")
