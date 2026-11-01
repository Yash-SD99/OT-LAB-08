# %% [markdown]
# ## A2 · Evolutionary framework (shared by algorithms 10–35)
# `Variation` (SBX + polynomial mutation for real, order crossover + swap/inversion for permutations, uniform crossover + bit flip for binary), base class `MOO` with
# `initialize / evaluate / iterate / run`, an external-archive helper `Arc`, selection helpers and the algorithm registry.
# Algorithms **without** feasibility rules handle constraints with a static penalty (`prob.pen`); only Constrained NSGA-II (21) uses Deb's rules.

# %%
class Variation:
    def __init__(s, prob, rng, pc=.9, eta_c=15., eta_m=20., pm=None):
        s.p, s.rng, s.pc, s.eta_c, s.eta_m = prob, rng, pc, eta_c, eta_m; s.pm = 1.0 / prob.n if pm is None else pm; s.perm_mut = 'both'; s.perm_cross = 'ox'   # perm_mut: both|swap|inversion|none
    def sbx(s, A, B):
        u = s.rng.random(A.shape); e = 1 / (s.eta_c + 1)
        beta = np.where(u <= .5, (2 * u) ** e, (1 / (2 * (1 - u) + 1e-12)) ** e)
        c1 = .5 * ((1 + beta) * A + (1 - beta) * B); c2 = .5 * ((1 - beta) * A + (1 + beta) * B)
        do = (s.rng.random((len(A), 1)) < s.pc) & (s.rng.random(A.shape) < .5)
        return np.clip(np.where(do, c1, A), s.p.lb, s.p.ub), np.clip(np.where(do, c2, B), s.p.lb, s.p.ub)
    def pmut(s, C):
        u = s.rng.random(C.shape); mk = s.rng.random(C.shape) < s.pm; e = 1 / (s.eta_m + 1)
        d = np.where(u < .5, (2 * u) ** e - 1, 1 - (2 * (1 - u)) ** e); return np.clip(C + mk * d * (s.p.ub - s.p.lb), s.p.lb, s.p.ub)
    def ox(s, a, b):
        n = len(a); i, j = sorted(s.rng.choice(n, 2, replace=False)); c = -np.ones(n, int); c[i:j + 1] = a[i:j + 1]; seg = set(a[i:j + 1].tolist())
        rest = [g for g in np.roll(b, -(j + 1)) if g not in seg]; c[[(j + 1 + k) % n for k in range(len(rest))]] = rest; return c
    def cross(s, A, B):
        k = s.p.kind
        if k == 'real': return s.sbx(A, B)
        if k == 'perm':
            do = (s.rng.random(len(A)) < s.pc) & (s.perm_cross == 'ox')
            return (np.array([s.ox(a, b) if d else a.copy() for a, b, d in zip(A, B, do)]), np.array([s.ox(b, a) if d else b.copy() for a, b, d in zip(A, B, do)]))
        m = (s.rng.random(A.shape) < .5) & (s.rng.random((len(A), 1)) < s.pc); return np.where(m, B, A), np.where(m, A, B)
    def mutate(s, C):
        k = s.p.kind
        if k == 'real': return s.pmut(C)
        C = C.copy()
        if k == 'perm':
            for r in range(len(C)):
                if s.rng.random() < .3 and s.perm_mut != 'none':
                    i, j = sorted(s.rng.choice(C.shape[1], 2, replace=False))
                    if s.perm_mut == 'swap' or (s.perm_mut == 'both' and s.rng.random() < .5): C[r, [i, j]] = C[r, [j, i]]                 # swap mutation
                    else: C[r, i:j + 1] = C[r, i:j + 1][::-1]                            # inversion mutation
            return C
        return np.where(s.rng.random(C.shape) < s.pm, 1 - C, C)
    def offspring(s, P):
        K = len(P)
        if K % 2: P = np.vstack([P, P[-1:]])
        c1, c2 = s.cross(P[0::2], P[1::2]); C = np.empty_like(P); C[0::2], C[1::2] = c1, c2; return s.mutate(C)[:K]
    def one(s, a, b): return s.mutate(s.cross(a[None], b[None])[0])[0]
    def neighbor(s, X, scale=.05):
        """Small move used by local search / SA: Gaussian (real), swap (perm), single bit flip (binary)."""
        k = s.p.kind; X = X.copy()
        if k == 'real': return np.clip(X + s.rng.normal(size=X.shape) * scale * (s.p.ub - s.p.lb), s.p.lb, s.p.ub)
        for r in range(len(X)):
            if k == 'perm': i, j = s.rng.choice(X.shape[1], 2, replace=False); X[r, [i, j]] = X[r, [j, i]]
            else: i = s.rng.integers(X.shape[1]); X[r, i] = 1 - X[r, i]
        return X

def tournament(rank, crowd, K, rng):
    """Binary tournament with the crowded-comparison rule: lower rank wins, ties -> larger crowding."""
    a = rng.integers(0, len(rank), K); b = rng.integers(0, len(rank), K)
    return np.where((rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] >= crowd[b])), a, b)

def grid_index(F, div=10, lo=None, hi=None):
    lo = F.min(0) if lo is None else lo; hi = F.max(0) if hi is None else hi
    g = np.clip(((F - lo) / np.maximum(hi - lo, 1e-12) * div).astype(int), 0, div - 1); return (g * (div ** np.arange(F.shape[1]))).sum(1)

class Arc:
    """External archive of non-dominated solutions with capacity cap; truncation by crowding distance or by grid (remove from the most crowded cell)."""
    def __init__(s, cap=100, deb=False, trunc='crowd', div=10, rng=None):
        s.cap, s.deb, s.trunc, s.div, s.rng = cap, deb, trunc, div, rng or np.random.default_rng(0); s.X = None
    def __len__(s): return 0 if s.X is None else len(s.X)
    def add(s, X, F, CV, Fp):
        if s.X is not None: X, F, CV, Fp = np.vstack([s.X, X]), np.vstack([s.F, F]), np.r_[s.CV, CV], np.vstack([s.Fp, Fp])
        Fs, CVs = (F, CV) if s.deb else (Fp, None); nd = nds(Fs, CVs)[0]; X, F, CV, Fp = X[nd], F[nd], CV[nd], Fp[nd]
        _, u = np.unique(np.round(Fp, 10), axis=0, return_index=True); u = np.sort(u); X, F, CV, Fp = X[u], F[u], CV[u], Fp[u]
        while len(X) > s.cap:
            if s.trunc == 'crowd': j = np.argmin(crowding(Fp))
            else:
                gi = grid_index(Fp, s.div); ids, cnt = np.unique(gi, return_counts=True); big = ids[cnt == cnt.max()]
                j = s.rng.choice(np.where(np.isin(gi, big))[0])
            keep = np.arange(len(X)) != j; X, F, CV, Fp = X[keep], F[keep], CV[keep], Fp[keep]
        s.X, s.F, s.CV, s.Fp = X, F, CV, Fp

class MOO:
    """Base class. initialize() -> evaluate() -> iterate() until the evaluation budget is spent -> result() (non-dominated, feasible)."""
    name = 'MOO'
    def __init__(s, prob, N=100, seed=0, deb=False, **kw):
        s.prob, s.N, s.seed, s.deb, s.t, s.nfe = prob, N, seed, deb, 0, 0
        s.rng = np.random.default_rng(seed); s.var = Variation(prob, s.rng, pc=kw.get('pc', .9), eta_c=kw.get('eta_c', 15.), eta_m=kw.get('eta_m', 20.)); s.__dict__.update(kw)
    def evaluate(s, X):
        F, CV = s.prob.evaluate(X); s.nfe += len(F); return F, CV, s.prob.pen(F, CV)
    def sk(s, F, CV, Fp): return (F, CV) if s.deb else (Fp, None)       # what the sorting/dominance sees
    def initialize(s): s.X = s.prob.init(s.N, s.rng); s.F, s.CV, s.Fp = s.evaluate(s.X)
    def iterate(s): raise NotImplementedError
    def run(s, max_nfe=10000):
        s.max_nfe = max_nfe; s.initialize()
        while s.nfe < max_nfe: s.iterate(); s.t += 1
        return s.result()
    def out(s): return s.X, s.F, s.CV
    def result(s):
        X, F, CV = s.out(); ok = CV <= 0; X, F = X[ok], F[ok]
        if len(F) == 0: return X, F
        nd = nds(F)[0]; X, F = X[nd], F[nd]; _, u = np.unique(np.round(F, 9), axis=0, return_index=True); return X[u], F[u]

def run_alg(cls, prob, seed=0, nfe=None, **kw):
    nfe = BUDGET if nfe is None else nfe; t = time.time(); a = cls(prob, seed=seed, **kw); X, F = a.run(nfe); return X, F, time.time() - t, a

# %% [markdown]
# ## Family: non-elitist evolutionary methods (algorithms 10–13)

# %%
class NonElitist(MOO):
    """Generational scheme without elitism: the offspring replace the population entirely. Subclasses define mate_idx()."""
    def mate_idx(s): raise NotImplementedError
    def iterate(s):
        Q = s.var.offspring(s.X[s.mate_idx()]); s.X = Q; s.F, s.CV, s.Fp = s.evaluate(Q)

def share_matrix(Fn, sigma): D = np.sqrt(((Fn[:, None] - Fn[None]) ** 2).sum(2)); return np.maximum(0, 1 - D / sigma)
def minmax(F): return (F - F.min(0)) / np.maximum(F.max(0) - F.min(0), 1e-12)

def vega_select(Fp, m, k, rng):
    """Alg 10 core: sub-population m is filled by binary tournaments on objective m ONLY."""
    a = rng.integers(0, len(Fp), k); b = rng.integers(0, len(Fp), k); return np.where(Fp[a, m] <= Fp[b, m], a, b)
class VEGA(NonElitist):
    """Alg 10 VEGA (Schaffer 1985). Steps: (1) random population; (2) for each objective m select N/M parents by tournament on f_m; (3) shuffle the M sub-populations together;
    (4) crossover + mutation -> new population (no elitism). Default: N=100, SBX pc=.9, PM 1/n."""
    name = 'VEGA'
    def mate_idx(s):
        M = s.prob.M; pool = np.concatenate([vega_select(s.Fp, m, s.N // M + 1, s.rng) for m in range(M)])[:s.N]; return s.rng.permutation(pool)

def moga_rank(F): return 1 + dom_matrix(F).sum(0)                            # 1 + number of dominators
def moga_fitness(F, sigma=.1):
    N = len(F); rank = moga_rank(F); order = np.argsort(rank, kind='stable'); raw = np.empty(N); raw[order] = np.linspace(N, 1, N)
    fit = raw.copy()
    for r in np.unique(rank): fit[rank == r] = raw[rank == r].mean()          # same rank -> same raw fitness (average)
    m = share_matrix(minmax(F), sigma).sum(1); sh = fit / m
    for r in np.unique(rank): sh[rank == r] *= fit[rank == r].mean() / sh[rank == r].mean()    # rescale so the class average is preserved
    return rank, fit, sh
class MOGA(NonElitist):
    """Alg 11 MOGA (Fonseca & Fleming 1993). Steps: (1) rank_i = 1 + #individuals dominating i; (2) raw fitness interpolated N..1 by rank, averaged inside a rank;
    (3) fitness sharing with niche radius sigma (normalised objective space); (4) fitness-proportionate selection; (5) variation. sigma=.1."""
    name = 'MOGA'
    def mate_idx(s):
        _, _, sh = moga_fitness(s.Fp, getattr(s, 'sigma', .1)); return s.rng.choice(s.N, s.N, p=sh / sh.sum())

def nsga_fitness(F, sigma=.1):
    N = len(F); fit = np.zeros(N); dummy = float(N); fronts = nds(F); Sh = share_matrix(minmax(F), sigma)
    for fr in fronts:
        m = Sh[np.ix_(fr, fr)].sum(1); fit[fr] = dummy / m; dummy = fit[fr].min() * .9            # sharing only INSIDE a front
    return fit, fronts
class NSGA(NonElitist):
    """Alg 12 NSGA (Srinivas & Deb 1994). Steps: (1) classify into non-dominated fronts; (2) front 1 gets dummy fitness N, shared inside the front (niche count);
    (3) next front's dummy fitness = 0.9*min shared fitness of the previous front; (4) fitness-proportionate selection; (5) variation. sigma=.1 (objective space)."""
    name = 'NSGA'
    def mate_idx(s): fit, _ = nsga_fitness(s.Fp, getattr(s, 'sigma', .1)); return s.rng.choice(s.N, s.N, p=fit / fit.sum())

def npga_winner(F, a, b, S, niche):
    """Alg 13 core: compare candidates a,b against comparison set S. Dominated-by-set loses; if both or none are dominated, the smaller niche count wins."""
    da = bool(np.any(np.all(F[S] <= F[a], 1) & np.any(F[S] < F[a], 1))); db = bool(np.any(np.all(F[S] <= F[b], 1) & np.any(F[S] < F[b], 1)))
    if da != db: return b if da else a
    return a if niche[a] <= niche[b] else b
class NPGA(NonElitist):
    """Alg 13 NPGA (Horn, Nafpliotis & Goldberg 1994). Steps: (1) pick two candidates; (2) draw a comparison set of t_dom=10% of N; (3) a candidate dominated by someone in the set loses;
    (4) tie (both/none dominated) -> smaller niche count (sharing, sigma=.1) wins; (5) variation. No elitism."""
    name = 'NPGA'
    def mate_idx(s):
        N = s.N; niche = share_matrix(minmax(s.Fp), getattr(s, 'sigma', .1)).sum(1); td = max(2, int(.1 * N)); w = []
        for _ in range(N):
            a, b = s.rng.integers(0, N, 2); w.append(npga_winner(s.Fp, a, b, s.rng.integers(0, N, td), niche))
        return np.array(w)

# %% [markdown]
# ## Family: elitist methods — NSGA-II (14), Constrained NSGA-II (21), MOMA (30), R-NSGA-II (35)

# %%
class NSGA2(MOO):
    """Alg 14 NSGA-II (Deb et al. 2002). Steps: (1) random P0; (2) rank (fast non-dominated sort) + crowding distance; (3) binary tournament with crowded comparison;
    (4) SBX + PM -> Q_t; (5) R = P_t + Q_t; (6) sort R into fronts, fill N slots front by front, truncate the last front by crowding distance (elitism). N=100, pc=.9, eta_c=15, eta_m=20."""
    name = 'NSGA-II'
    def initialize(s): super().initialize(); s.rank_crowd()
    def rank_crowd(s):
        Fs, CVs = s.sk(s.F, s.CV, s.Fp); fr = nds(Fs, CVs); s.rank = np.zeros(len(Fs), int); s.crowd = np.zeros(len(Fs))
        for r, f in enumerate(fr): s.rank[f] = r; s.crowd[f] = s.crowd_measure(Fs, f)
    def crowd_measure(s, Fs, f): return crowding(Fs[f])
    def offspring(s): return s.var.offspring(s.X[tournament(s.rank, s.crowd, s.N, s.rng)])
    def survive(s, F, CV, Fp, N):
        Fs, CVs = s.sk(F, CV, Fp); sel = []
        for fr in nds(Fs, CVs):
            if len(sel) + len(fr) <= N: sel += fr.tolist()
            else:
                o = np.argsort(-s.crowd_measure(Fs, fr), kind='stable'); sel += fr[o[:N - len(sel)]].tolist(); break
        return np.array(sel)
    def merge(s, Q, QF, QCV, QFp):
        X, F, CV, Fp = np.vstack([s.X, Q]), np.vstack([s.F, QF]), np.r_[s.CV, QCV], np.vstack([s.Fp, QFp]); k = s.survive(F, CV, Fp, s.N)
        s.X, s.F, s.CV, s.Fp = X[k], F[k], CV[k], Fp[k]; s.rank_crowd()
    def iterate(s): Q = s.offspring(); s.merge(Q, *s.evaluate(Q))

class CNSGA2(NSGA2):
    """Alg 21 Constrained NSGA-II (Deb 2002, Part 4). Same as NSGA-II but 'dominates' is replaced by Deb's constrained domination:
    (R1) feasible beats infeasible; (R2) both infeasible -> smaller violation; (R3) both feasible -> Pareto dominance. No penalty parameter."""
    name = 'C-NSGA-II'
    def __init__(s, prob, **kw): super().__init__(prob, deb=True, **kw)

class MOMA(NSGA2):
    """Alg 30 MOMA (memetic: NSGA-II global search + hill-climbing local search, Lamarckian). Steps: (1)-(4) as NSGA-II to create Q_t; (5) with prob p_ls pick offspring and improve each by
    L hill-climbing steps on a random-weight Tchebycheff scalarisation (accept a neighbour only if the scalar value improves); (6) write improved solutions back; (7) NSGA-II survival.
    Defaults: p_ls=.2, L=5 (neighbour = Gaussian step 5% of range / swap / bit flip)."""
    name = 'MOMA'
    def tcheb(s, Fp, w): return np.max(w * s.prob.norm(Fp), 1)
    def local_search(s, X, F, CV, Fp, w, L):
        for _ in range(L):
            Y = s.var.neighbor(X, getattr(s, 'ls_scale', .05)); YF, YCV, YFp = s.evaluate(Y)
            better = s.tcheb(YFp, w) < s.tcheb(Fp, w)
            X, F, CV, Fp = np.where(better[:, None], Y, X), np.where(better[:, None], YF, F), np.where(better, YCV, CV), np.where(better[:, None], YFp, Fp)
        return X, F, CV, Fp
    def iterate(s):
        Q = s.offspring(); QF, QCV, QFp = s.evaluate(Q); p = getattr(s, 'p_ls', .2); sel = np.where(s.rng.random(len(Q)) < p)[0]
        if len(sel):
            w = s.rng.dirichlet(np.ones(s.prob.M), len(sel)) + 1e-6
            Q[sel], QF[sel], QCV[sel], QFp[sel] = s.local_search(Q[sel], QF[sel], QCV[sel], QFp[sel], w, getattr(s, 'L', 5))
        s.merge(Q, QF, QCV, QFp)

def pref_order(Fn, ref_n, eps):
    """Alg 35 core: order solutions of one front by distance to the DM reference point(s), with ε-clearing (a solution within eps of an already chosen one is postponed)."""
    d = np.sqrt(((Fn[:, None, :] - ref_n[None]) ** 2).sum(2)).min(1); left = list(np.argsort(d)); order = []
    while left:
        chosen = []; rest = []
        for i in left:
            if all(np.linalg.norm(Fn[i] - Fn[j]) > eps for j in chosen): chosen.append(i)
            else: rest.append(i)
        order += chosen; left = rest
    return np.array(order)
class RNSGA2(NSGA2):
    """Alg 35 R-NSGA-II (Deb & Sundar 2006). Steps: as NSGA-II, but inside each front crowding distance is replaced by the preference rank = order of the (normalised) Euclidean distance to the DM
    reference point, with ε-clearing to keep diversity (eps=.002). `ref` (natural units, minimisation form) must be supplied."""
    name = 'R-NSGA-II'
    def crowd_measure(s, Fs, f):
        Fn = s.prob.norm(Fs[f]); rn = s.prob.norm(np.atleast_2d(s.ref)); o = pref_order(Fn, rn, getattr(s, 'eps', .002)); c = np.empty(len(f)); c[o] = -np.arange(len(f)); return c
