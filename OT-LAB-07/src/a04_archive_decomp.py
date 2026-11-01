# %% [markdown]
# ## Family: reference-point, archive-based and decomposition methods (algorithms 15–20)

# %%
def nsga3_normalize(F):
    """NSGA-III adaptive normalisation: translate by the ideal point, find extreme points with an achievement scalarising function, intercepts of the hyperplane."""
    z = F.min(0); Ft = F - z; M = F.shape[1]; ext = []
    for j in range(M):
        w = np.full(M, 1e-6); w[j] = 1; ext.append(np.argmin(np.max(Ft / w, 1)))
    try:
        a = np.linalg.solve(Ft[ext], np.ones(M)); ic = 1 / a
        if np.any(a <= 1e-12) or not np.all(np.isfinite(ic)): raise np.linalg.LinAlgError
    except np.linalg.LinAlgError: ic = Ft.max(0)
    return Ft / np.maximum(ic, 1e-9)
def associate(Fn, W):
    """Assign each (normalised) point to the reference line with the smallest PERPENDICULAR distance."""
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True); proj = Fn @ Wn.T; d = np.sqrt(np.maximum((Fn ** 2).sum(1)[:, None] - proj ** 2, 0))
    r = d.argmin(1); return r, d[np.arange(len(Fn)), r]
class NSGA3(MOO):
    """Alg 15 NSGA-III (Deb & Jain 2014). Steps: (1) Das–Dennis reference points; (2) random mating, SBX+PM; (3) R=P+Q, non-dominated sort, take whole fronts until >=N (last front F_l);
    (4) adaptive normalisation (ideal, extreme points, hyperplane intercepts); (5) associate every member to its nearest reference line (perpendicular distance); (6) niche counts rho_j from fronts before F_l;
    (7) niche-preservation: repeatedly take the reference point with the smallest rho_j and add one F_l member associated with it (closest if rho_j=0, random otherwise). Divisions p=12 (M=3), 4 (M=5)."""
    name = 'NSGA-III'
    def __init__(s, prob, p=None, **kw):
        M = prob.M; p = p or {2: 99, 3: 12, 5: 4}.get(M, 3); W = das_dennis(M, p); N = kw.pop('N', int(math.ceil(len(W) / 4) * 4)); super().__init__(prob, N=N, **kw); s.W = W
    def niching(s, K, rho, ref, dist):
        rho = rho.copy(); avail = np.ones(len(ref), bool); active = np.ones(len(rho), bool); chosen = []
        while len(chosen) < K:
            cand = np.where(active)[0]; js = cand[rho[cand] == rho[cand].min()]; j = s.rng.choice(js); mem = np.where(avail & (ref == j))[0]
            if len(mem) == 0: active[j] = False; continue
            pick = mem[np.argmin(dist[mem])] if rho[j] == 0 else s.rng.choice(mem)
            chosen.append(pick); avail[pick] = False; rho[j] += 1
        return np.array(chosen)
    def survive(s, F, CV, Fp, N):
        fronts = nds(Fp); sel = []; Fl = None
        for fr in fronts:
            if len(sel) + len(fr) <= N: sel += fr.tolist()
            else: Fl = fr; break
            if len(sel) == N: break
        if len(sel) == N or Fl is None: return np.array(sel)
        S = np.array(sel + Fl.tolist()); Fn = nsga3_normalize(Fp[S]); ref, dist = associate(Fn, s.W); nsel = len(sel)
        rho = np.bincount(ref[:nsel], minlength=len(s.W)); pick = s.niching(N - nsel, rho, ref[nsel:], dist[nsel:])
        return np.array(sel + Fl[pick].tolist())
    def iterate(s):
        Q = s.var.offspring(s.X[s.rng.permutation(s.N)]); QF, QCV, QFp = s.evaluate(Q)
        X, F, CV, Fp = np.vstack([s.X, Q]), np.vstack([s.F, QF]), np.r_[s.CV, QCV], np.vstack([s.Fp, QFp]); k = s.survive(F, CV, Fp, s.N)
        s.X, s.F, s.CV, s.Fp = X[k], F[k], CV[k], Fp[k]

def spea2_fitness(F):
    """Alg 16 core. S(i)=#dominated by i; raw R(i)=sum of S(j) over dominators j; density D(i)=1/(sigma_k+2), k=sqrt(n); fitness=R+D (<1 <=> non-dominated)."""
    n = len(F); D = dom_matrix(F); S = D.sum(1); R = D.T.astype(float) @ S
    dist = np.sqrt(((F[:, None] - F[None]) ** 2).sum(2)); np.fill_diagonal(dist, np.inf); k = max(1, min(n - 1, int(np.sqrt(n)))); sk = np.sort(dist, 1)[:, k - 1]
    return R + 1 / (sk + 2), R
def spea2_truncate(F, K):
    """Archive truncation: repeatedly delete the point whose sorted distance list to the others is lexicographically smallest."""
    idx = list(range(len(F))); Dm = np.sqrt(((F[:, None] - F[None]) ** 2).sum(2))
    while len(idx) > K:
        sub = Dm[np.ix_(idx, idx)].copy(); np.fill_diagonal(sub, np.inf); S = np.sort(sub, 1)[:, :8]; idx.pop(np.lexsort(S.T[::-1])[0])
    return np.array(idx)
class SPEA2(MOO):
    """Alg 16 SPEA2 (Zitzler, Laumanns & Thiele 2001). Steps: (1) population P (N) + fixed-size archive A (Nbar=N); (2) fitness = raw (strength of dominators) + k-th nearest-neighbour density;
    (3) A_next = all non-dominated (fitness<1); fill with best dominated if too small, truncate by nearest-neighbour rule if too large; (4) binary tournament on A_next by fitness; (5) SBX+PM -> new P."""
    name = 'SPEA2'
    def initialize(s): super().initialize(); s.AX = np.empty((0,) + s.X.shape[1:], s.X.dtype); s.AF = np.empty((0, s.prob.M)); s.ACV = np.empty(0); s.AFp = np.empty((0, s.prob.M)); s.env()
    def env(s):
        X, F, CV, Fp = np.vstack([s.AX, s.X]), np.vstack([s.AF, s.F]), np.r_[s.ACV, s.CV], np.vstack([s.AFp, s.Fp]); fit, _ = spea2_fitness(Fp); Nb = s.N
        nd = np.where(fit < 1)[0]
        if len(nd) < Nb: k = np.argsort(fit)[:Nb]
        elif len(nd) > Nb: k = nd[spea2_truncate(Fp[nd], Nb)]
        else: k = nd
        s.AX, s.AF, s.ACV, s.AFp = X[k], F[k], CV[k], Fp[k]; s.afit = spea2_fitness(s.AFp)[0]
    def iterate(s):
        a = s.rng.integers(0, len(s.AX), s.N); b = s.rng.integers(0, len(s.AX), s.N); w = np.where(s.afit[a] <= s.afit[b], a, b)
        s.X = s.var.offspring(s.AX[w]); s.F, s.CV, s.Fp = s.evaluate(s.X); s.env()
    def out(s): return s.AX, s.AF, s.ACV

class PESA2(MOO):
    """Alg 17 PESA-II (Corne et al. 2001). Steps: (1) small internal population (N=20 children per generation) and an external archive (cap 100) divided into hyper-boxes (grid, 10 divisions per objective);
    (2) region-based selection: pick a hyper-box by binary tournament (the less crowded box wins), then a random archive member inside it; (3) SBX+PM; (4) evaluate; (5) insert non-dominated children,
    delete dominated members; if the archive overflows delete from the most crowded box."""
    name = 'PESA-II'
    def __init__(s, prob, N=20, cap=100, div=10, **kw): super().__init__(prob, N=N, **kw); s.arc = Arc(cap, trunc='grid', div=div, rng=s.rng); s.div = div
    def initialize(s): super().initialize(); s.arc.add(s.X, s.F, s.CV, s.Fp)
    def pick(s):
        gi = grid_index(s.arc.Fp, s.div); ids, cnt = np.unique(gi, return_counts=True); a, b = s.rng.integers(0, len(ids), 2)
        box = ids[a] if cnt[a] <= cnt[b] else ids[b]; return s.rng.choice(np.where(gi == box)[0])      # less crowded box wins the tournament
    def iterate(s):
        Q = np.array([s.var.one(s.arc.X[s.pick()], s.arc.X[s.pick()]) for _ in range(s.N)]); QF, QCV, QFp = s.evaluate(Q); s.arc.add(Q, QF, QCV, QFp)
    def out(s): return s.arc.X, s.arc.F, s.arc.CV

def tcheb(Fv, w, z, zn): return np.max(w * np.abs(Fv - z) / np.maximum(zn - z, 1e-12), axis=-1)
def neighbours(W, T): D = np.sqrt(((W[:, None] - W[None]) ** 2).sum(2)); return np.argsort(D, 1, kind='stable')[:, :T]
class MOEAD(MOO):
    """Alg 18 MOEA/D (Zhang & Li 2007) with Tchebycheff decomposition. Steps: (1) N weight vectors lambda_i, neighbourhood B_i = the T closest weight vectors; (2) one solution per subproblem, ideal point z*;
    (3) for each i: pick two parents from B_i (prob delta) or the whole population, SBX+PM -> y; (4) update z*; (5) for neighbours j (at most nr replacements): if g(y|lambda_j,z*) <= g(x_j|lambda_j,z*) then x_j=y.
    g(x|lambda,z*)=max_k lambda_k |f_k-z*_k| (objectives normalised by the current ideal/nadir estimate). Defaults: N=100 (M=2), T=20, nr=2, delta=0.9."""
    name = 'MOEA/D'
    def __init__(s, prob, T=20, nr=2, delta=.9, p=None, **kw):
        M = prob.M
        if M == 2: W = np.c_[np.linspace(0, 1, kw.get('N', 100)), 1 - np.linspace(0, 1, kw.get('N', 100))]
        else: W = das_dennis(M, p or {3: 13, 5: 5}.get(M, 4))
        kw['N'] = len(W); super().__init__(prob, **kw); s.W = np.maximum(W, 1e-6); s.T, s.nr, s.delta = T, nr, delta; s.B = neighbours(s.W, min(T, len(W)))
    def initialize(s): super().initialize(); s.z = s.Fp.min(0); s.zn = s.Fp.max(0)
    def g(s, Fv, j): return tcheb(Fv, s.W[j], s.z, s.zn)
    def child(s, i, P): a, b = s.rng.choice(P, 2, replace=False) if len(P) > 1 else (P[0], P[0]); return s.var.one(s.X[a], s.X[b])
    def update(s, i, y, P):
        yF, yCV, yFp = s.evaluate(y[None]); s.z = np.minimum(s.z, yFp[0]); cnt = 0
        for j in s.rng.permutation(P):
            if s.g(yFp[0], j) <= s.g(s.Fp[j], j): s.X[j], s.F[j], s.CV[j], s.Fp[j] = y, yF[0], yCV[0], yFp[0]; cnt += 1
            if cnt >= s.nr: break
    def iterate(s):
        for i in s.rng.permutation(s.N):
            P = s.B[i] if s.rng.random() < s.delta else np.arange(s.N); s.update(i, s.child(i, P), P)
        s.zn = s.Fp.max(0)

class MOEADDE(MOEAD):
    """Alg 19 MOEA/D-DE (Li & Zhang 2009). MOEA/D whose variation is differential evolution: y = x_r1 + F (x_r2 - x_r3) (r's from B_i), binomial crossover with x_i (CR=1), polynomial mutation.
    Also adds the nr limit that stops one good child taking over a whole neighbourhood. F=0.5, CR=1.0, delta=0.9, nr=2, T=20."""
    name = 'MOEA/D-DE'
    def child(s, i, P):
        r = s.rng.choice(P, 3, replace=len(P) < 3); y = s.X[r[0]] + getattr(s, 'F_de', .5) * (s.X[r[1]] - s.X[r[2]]); m = s.rng.random(s.prob.n) < getattr(s, 'CR', 1.0); m[s.rng.integers(s.prob.n)] = True
        y = np.where(m, y, s.X[i]); return s.var.pmut(np.clip(y, s.prob.lb, s.prob.ub)[None])[0]

def aco_tour(W, rng):
    """Probabilistic route construction: next city j from city c with prob proportional to W[c,j] (=tau^alpha*eta^beta) over unvisited cities."""
    n = len(W); cur = int(rng.integers(n)); tour = [cur]; vis = np.zeros(n, bool); vis[cur] = True
    for _ in range(n - 1):
        p = W[cur] * ~vis; c = np.cumsum(p); nxt = min(int(np.searchsorted(c, rng.random() * c[-1])), n - 1)
        while vis[nxt]: nxt = int(rng.integers(n))
        tour.append(nxt); vis[nxt] = True; cur = nxt
    return np.array(tour)
def edge_matrix(tour): n = len(tour); E = np.zeros((n, n)); a, b = tour, np.roll(tour, -1); E[a, b] = 1; E[b, a] = 1; return E
class MOEADACO(MOEAD):
    """Alg 20 MOEA/D-ACO (Ke, Zhang & Battiti 2013). Steps: (1) N Tchebycheff subproblems with weights lambda_i; each has its own pheromone matrix tau_i and heuristic eta_i = 1/(lambda1*d_ij + lambda2*t_ij) (normalised);
    (2) one ant per subproblem builds a tour with P ~ tau^alpha * eta^beta; (3) evaluate, update ideal point, neighbours' solutions replaced if the child is better for their Tchebycheff function (nr=2);
    (4) pheromone of subproblem i is evaporated (rho) and reinforced by the current solutions of its neighbourhood B_i, weighted by their quality for lambda_i. alpha=1, beta=2, rho=.1, N=20, T=5."""
    name = 'MOEA/D-ACO'
    def __init__(s, prob, N=20, T=5, alpha=1., beta=2., rho=.1, **kw): super().__init__(prob, N=N, T=T, **kw); s.alpha, s.beta, s.rho = alpha, beta, rho
    def initialize(s):
        super().initialize(); n = s.prob.n; s.tau = np.ones((s.N, n, n)); Dn = s.prob.D / s.prob.D[s.prob.D > 0].mean(); Tn = s.prob.T / s.prob.T[s.prob.T > 0].mean()
        s.eta = np.array([1 / (np.maximum(w[0] * Dn + w[1] * Tn, 1e-9)) for w in s.W]); s.eta[:, np.arange(n), np.arange(n)] = 0
    def iterate(s):
        for i in s.rng.permutation(s.N):
            Wm = (s.tau[i] ** s.alpha) * (s.eta[i] ** s.beta); y = aco_tour(Wm, s.rng); s.update(i, y, s.B[i])
        s.zn = s.Fp.max(0); n = s.prob.n
        for i in range(s.N):
            Bi = s.B[i]; g = s.g(s.Fp[Bi], i); delta = (g.min() + 1e-9) / (g + 1e-9); dep = sum(d * edge_matrix(s.X[j]) for d, j in zip(delta, Bi)) / len(Bi)
            s.tau[i] = np.clip((1 - s.rho) * s.tau[i] + s.rho * dep, .05, 5.)
