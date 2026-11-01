# %% [markdown]
# ## Family: swarm, physics- and human-inspired metaheuristics (algorithms 22–29)

# %%
def cell_leader(arc, n, rng, div=10):
    """Leader selection from an archive: choose a hyper-box with probability ~ 1/(#members) (less crowded boxes favoured), then a random member inside it."""
    gi = grid_index(arc.Fp, div); ids, cnt = np.unique(gi, return_counts=True); p = (1 / cnt) / (1 / cnt).sum(); cells = rng.choice(len(ids), n, p=p)
    return np.array([rng.choice(np.where(gi == ids[c])[0]) for c in cells])

def pso_velocity(w, c1, c2, v, x, pb, gb, r1, r2): return w * v + c1 * r1 * (pb - x) + c2 * r2 * (gb - x)
class MOPSO(MOO):
    """Alg 22 MOPSO (Coello, Pulido & Lechuga 2004). Steps: (1) swarm, velocities 0, personal bests, external archive with adaptive grid; (2) global best of each particle = archive member drawn from a
    less-crowded hyper-box; (3) v = w v + c1 r1 (pbest-x) + c2 r2 (gbest-x), x += v (reflect at bounds); (4) turbulence mutation with a rate that decays; (5) evaluate; pbest replaced if the new position dominates
    (random if non-dominated); (6) update archive (non-dominated, grid truncation). w=.4, c1=c2=1.5, swarm 100, archive 100."""
    name = 'MOPSO'
    def __init__(s, prob, N=100, w=.4, c1=1.5, c2=1.5, cap=100, div=10, **kw): super().__init__(prob, N=N, **kw); s.w, s.c1, s.c2, s.div = w, c1, c2, div; s.arc = Arc(cap, trunc='grid', div=div, rng=s.rng)
    def initialize(s): super().initialize(); s.V = np.zeros_like(s.X, float); s.PB, s.PBF = s.X.copy(), s.Fp.copy(); s.arc.add(s.X, s.F, s.CV, s.Fp)
    def iterate(s):
        N, n = s.X.shape; gb = s.arc.X[cell_leader(s.arc, N, s.rng, s.div)]; V = pso_velocity(s.w, s.c1, s.c2, s.V, s.X, s.PB, gb, s.rng.random((N, n)), s.rng.random((N, n)))
        X = s.X + V; lo, hi = X < s.prob.lb, X > s.prob.ub; X = np.clip(X, s.prob.lb, s.prob.ub); V = np.where(lo | hi, -V, V)
        rate = .5 * (1 - s.nfe / s.max_nfe) ** 3; m = s.rng.random(N) < rate; j = s.rng.integers(0, n, N); step = (s.prob.ub - s.prob.lb)[j] * rate * s.rng.uniform(-1, 1, N)
        X[m, j[m]] = np.clip(X[m, j[m]] + step[m], s.prob.lb[j[m]], s.prob.ub[j[m]])
        F, CV, Fp = s.evaluate(X); dn = np.all(Fp <= s.PBF, 1) & np.any(Fp < s.PBF, 1); od = np.all(s.PBF <= Fp, 1) & np.any(s.PBF < Fp, 1)
        rep = dn | (~dn & ~od & (s.rng.random(N) < .5)); s.PB[rep], s.PBF[rep] = X[rep], Fp[rep]; s.X, s.V, s.F, s.CV, s.Fp = X, V, F, CV, Fp; s.arc.add(X, F, CV, Fp)
    def out(s): return s.arc.X, s.arc.F, s.arc.CV

def gwo_pos(X, Xa, Xb, Xd, a, rng):
    """GWO encircling: X_k = L_k - A_k |C_k L_k - X| for leaders alpha, beta, delta; new position = mean of the three. A=2a r1 - a, C=2 r2."""
    out = 0
    for L in (Xa, Xb, Xd): A = 2 * a * rng.random(X.shape) - a; C = 2 * rng.random(X.shape); out = out + L - A * np.abs(C * L - X)
    return out / 3
class MOGWO(MOO):
    """Alg 23 MOGWO (Mirjalili et al. 2016). Steps: (1) wolf pack + archive (grid); (2) alpha, beta, delta = three leaders drawn from the least crowded archive boxes; (3) every wolf moves to the mean of the
    three encircling positions, with a decreasing from 2 to 0; (4) evaluate, update archive. Binary problems: continuous position in [0,1] decoded by a sigmoid transfer. pack 50, archive 100."""
    name = 'MOGWO'
    def __init__(s, prob, N=50, cap=100, div=10, **kw): super().__init__(prob, N=N, **kw); s.div = div; s.arc = Arc(cap, trunc='grid', div=div, rng=s.rng)
    def initialize(s):
        p = s.prob; s.Xc = p.lb + s.rng.random((s.N, p.n)) * (p.ub - p.lb); s.X = p.decode(s.Xc, s.rng); s.F, s.CV, s.Fp = s.evaluate(s.X); s.arc.add(s.X, s.F, s.CV, s.Fp)
    def leaders(s):
        if len(s.arc) < 3: idx = s.rng.integers(0, len(s.arc), 3)
        else: idx = cell_leader(s.arc, 3, s.rng, s.div)
        return idx
    def iterate(s):
        p = s.prob; ia, ib, idl = s.leaders(); C = s.arc.X.astype(float) if p.kind != 'bin' else s.arc.X.astype(float) * .8 + .1                  # bits -> continuous guide
        a = 2 * (1 - s.nfe / s.max_nfe); s.Xc = np.clip(gwo_pos(s.Xc, C[ia], C[ib], C[idl], a, s.rng), p.lb, p.ub)
        s.X = p.decode(s.Xc, s.rng); s.F, s.CV, s.Fp = s.evaluate(s.X); s.arc.add(s.X, s.F, s.CV, s.Fp)
    def out(s): return s.arc.X, s.arc.F, s.arc.CV

def dom_pair(a, b): return bool(np.all(a <= b) and np.any(a < b))
class MOABC(MOO):
    """Alg 24 MOABC (artificial bee colony, Pareto version). Steps: (1) SN=50 food sources, trial counters, archive; (2) employed bees: v_ij = x_ij + phi (x_ij - x_kj) on one random dimension, greedy DOMINANCE selection
    (replace if v dominates x; keep if x dominates v; non-dominated pair -> replace with prob .5); (3) onlooker bees choose sources with probability ~ 1/(1+rank) (non-dominated rank) and repeat (2);
    (4) scouts replace sources whose trial counter exceeds limit=20 by random ones; (5) archive of non-dominated solutions (crowding truncation)."""
    name = 'MOABC'
    def __init__(s, prob, N=50, limit=20, cap=100, **kw): super().__init__(prob, N=N, **kw); s.limit = limit; s.arc = Arc(cap, trunc='crowd', rng=s.rng)
    def initialize(s): super().initialize(); s.trial = np.zeros(s.N, int); s.arc.add(s.X, s.F, s.CV, s.Fp)
    def candidate(s, idx):
        N, n = s.X.shape; V = s.X[idx].copy(); k = (idx + s.rng.integers(1, N, len(idx))) % N; j = s.rng.integers(0, n, len(idx)); phi = s.rng.uniform(-1, 1, len(idx)); r = np.arange(len(idx))
        V[r, j] = np.clip(V[r, j] + phi * (V[r, j] - s.X[k, j]), s.prob.lb[j], s.prob.ub[j]); return V
    def greedy(s, idx, V):
        VF, VCV, VFp = s.evaluate(V)
        for r, i in enumerate(idx):
            if dom_pair(VFp[r], s.Fp[i]): rep = True
            elif dom_pair(s.Fp[i], VFp[r]): rep = False
            else: rep = s.rng.random() < .5
            if rep: s.X[i], s.F[i], s.CV[i], s.Fp[i], s.trial[i] = V[r], VF[r], VCV[r], VFp[r], 0
            else: s.trial[i] += 1
        s.arc.add(V, VF, VCV, VFp)
    def iterate(s):
        idx = np.arange(s.N); s.greedy(idx, s.candidate(idx))                                                           # employed bees
        rank = np.zeros(s.N);
        for r, f in enumerate(nds(s.Fp)): rank[f] = r
        p = 1 / (1 + rank); idx = s.rng.choice(s.N, s.N, p=p / p.sum()); s.greedy(idx, s.candidate(idx))                  # onlookers
        sc = np.where(s.trial > s.limit)[0]
        if len(sc):                                                                                                     # scouts
            Xn = s.prob.init(len(sc), s.rng); F, CV, Fp = s.evaluate(Xn); s.X[sc], s.F[sc], s.CV[sc], s.Fp[sc], s.trial[sc] = Xn, F, CV, Fp, 0; s.arc.add(Xn, F, CV, Fp)
    def out(s): return s.arc.X, s.arc.F, s.arc.CV

class MOACO(MOO):
    """Alg 25 MOACO (two pheromone matrices, bi-objective TSP). Steps: (1) tau_dist, tau_time = 1; heuristics eta_dist=1/d, eta_time=1/t (normalised); (2) ant k gets weight lambda_k=k/(m-1) and builds a tour with
    P ~ [tau1^l tau2^(1-l)]^alpha [eta1^l eta2^(1-l)]^beta; (3) evaluate, update non-dominated archive; (4) evaporate (rho=.1) and deposit rho/(f_k+.1) for 10 random archive tours on each matrix; tau clipped to [.05,10].
    m=20 ants, alpha=1, beta=2."""
    name = 'MOACO'
    def __init__(s, prob, N=20, alpha=1., beta=2., rho=.1, cap=100, **kw): super().__init__(prob, N=N, **kw); s.alpha, s.beta, s.rho = alpha, beta, rho; s.arc = Arc(cap, rng=s.rng)
    def initialize(s):
        n = s.prob.n; s.tau = [np.ones((n, n)), np.ones((n, n))]; D, T = s.prob.D, s.prob.T; s.eta = [D.mean() / np.maximum(D, 1e-9), T.mean() / np.maximum(T, 1e-9)]
        for e in s.eta: np.fill_diagonal(e, 0)
        s.X = s.prob.init(s.N, s.rng); s.F, s.CV, s.Fp = s.evaluate(s.X); s.arc.add(s.X, s.F, s.CV, s.Fp)
    def weights(s, l): return ((s.tau[0] ** l) * (s.tau[1] ** (1 - l))) ** s.alpha * ((s.eta[0] ** l) * (s.eta[1] ** (1 - l))) ** s.beta
    def iterate(s):
        lam = np.linspace(0, 1, s.N); X = np.array([aco_tour(s.weights(l), s.rng) for l in lam]); F, CV, Fp = s.evaluate(X); s.arc.add(X, F, CV, Fp)
        pick = s.rng.choice(len(s.arc), min(10, len(s.arc)), replace=False); Fn = s.prob.norm(s.arc.F[pick])
        for k in range(2):
            s.tau[k] *= (1 - s.rho)
            for t, f in zip(s.arc.X[pick], Fn[:, k]): s.tau[k] += s.rho / (max(f, 0) + .1) * edge_matrix(t)
            s.tau[k] = np.clip(s.tau[k], .05, 10.)
    def out(s): return s.arc.X, s.arc.F, s.arc.CV

def sa_accept_prob(dE, T):
    """Boltzmann acceptance: improvements (dE<=0) always accepted, deteriorations with prob exp(-dE/T)."""
    return 1.0 if dE <= 0 else float(np.exp(-dE / T))
class MOSA(MOO):
    """Alg 26 MOSA (Suppapitnarm et al. 2000, archive-based, return-to-base). Steps: (1) one current solution, archive of non-dominated solutions; (2) neighbour move (Gaussian / swap / bit flip) whose size shrinks with T;
    (3) energy change dE = sum_k (f_k(new)-f_k(cur))/range_k; accept with Boltzmann probability, always accept if the new point enters the archive; (4) geometric cooling T0=.5 -> 1e-3 over the budget;
    (5) every 200 iterations restart from a random archive member."""
    name = 'MOSA'
    def __init__(s, prob, N=1, T0=.5, Tend=1e-3, cap=100, base=200, **kw): super().__init__(prob, N=N, **kw); s.T0, s.Tend, s.base = T0, Tend, base; s.arc = Arc(cap, rng=s.rng)
    def initialize(s): super().initialize(); s.T = s.T0; s.arc.add(s.X, s.F, s.CV, s.Fp)
    def iterate(s):
        T = s.T0 * (s.Tend / s.T0) ** (s.nfe / s.max_nfe); y = s.var.neighbor(s.X, .02 + .3 * T / s.T0); yF, yCV, yFp = s.evaluate(y)
        dE = ((yFp[0] - s.Fp[0]) / s.prob.rng_).sum(); entered = not np.any(np.all(s.arc.Fp <= yFp[0], 1))
        if entered: s.arc.add(y, yF, yCV, yFp)
        if entered or s.rng.random() < sa_accept_prob(dE, T): s.X, s.F, s.CV, s.Fp = y, yF, yCV, yFp
        if s.t % s.base == s.base - 1:
            j = s.rng.integers(len(s.arc)); s.X, s.F, s.CV, s.Fp = s.arc.X[j:j + 1], s.arc.F[j:j + 1], s.arc.CV[j:j + 1], s.arc.Fp[j:j + 1]
    def out(s): return s.arc.X, s.arc.F, s.arc.CV

def rand_idx(rng, N, k=3):
    P = np.argsort(rng.random((N, N)), 1); return np.array([P[i][P[i] != i][:k] for i in range(N)])
def de_mutant(xr1, xr2, xr3, F): return xr1 + F * (xr2 - xr3)
class MODE(MOO):
    """Alg 27 MODE (Pareto differential evolution). Steps: (1) population N=100; (2) DE/rand/1 mutant v = x_r1 + F (x_r2 - x_r3), F=.5; (3) binomial crossover CR=.9 -> trial u;
    (4) selection: the trial replaces its TARGET only if it dominates it; (5) every evaluated trial is offered to an external non-dominated archive (cap 100, crowding) which is the reported front."""
    name = 'MODE'
    def __init__(s, prob, N=100, F_de=.5, CR=.9, cap=100, **kw): super().__init__(prob, N=N, **kw); s.F_de, s.CR = F_de, CR; s.arc = Arc(cap, rng=s.rng)
    def initialize(s): super().initialize(); s.arc.add(s.X, s.F, s.CV, s.Fp)
    def iterate(s):
        N, n = s.X.shape; r = rand_idx(s.rng, N); V = de_mutant(s.X[r[:, 0]], s.X[r[:, 1]], s.X[r[:, 2]], s.F_de); m = s.rng.random((N, n)) < s.CR; m[np.arange(N), s.rng.integers(0, n, N)] = True
        U = np.clip(np.where(m, V, s.X), s.prob.lb, s.prob.ub); F, CV, Fp = s.evaluate(U); rep = np.all(Fp <= s.Fp, 1) & np.any(Fp < s.Fp, 1)
        s.X[rep], s.F[rep], s.CV[rep], s.Fp[rep] = U[rep], F[rep], CV[rep], Fp[rep]; s.arc.add(U, F, CV, Fp)
    def out(s): return s.arc.X, s.arc.F, s.arc.CV

def teacher_phase(X, T, TF, rng): return X + rng.random(X.shape) * (T - TF * X.mean(0))
class MOTLBO(NSGA2):
    """Alg 28 MO-TLBO (teaching-learning-based optimisation, multiobjective). Steps: (1) class of N learners, rank + crowding as in NSGA-II; (2) TEACHER phase: teacher = random member of the first front,
    x' = x + r (T - TF*mean), TF shrinks 2 -> 1; (3) LEARNER phase: pick a random partner j, move towards it if j is better (rank, crowding) else away: x' = x + r (x_j - x) or x + r (x - x_j);
    (4) parents + both offspring sets are merged and cut back to N by non-dominated rank + crowding (elitist). Real variables only."""
    name = 'MO-TLBO'
    def iterate(s):
        N = s.N; TF = 2 - s.nfe / s.max_nfe; front = np.where(s.rank == 0)[0]; T = s.X[s.rng.choice(front, N)]
        Xt = np.clip(teacher_phase(s.X, T, TF, s.rng), s.prob.lb, s.prob.ub); j = (np.arange(N) + s.rng.integers(1, N, N)) % N
        jb = (s.rank[j] < s.rank) | ((s.rank[j] == s.rank) & (s.crowd[j] > s.crowd)); r = s.rng.random(s.X.shape)
        Xl = np.clip(np.where(jb[:, None], s.X + r * (s.X[j] - s.X), s.X + r * (s.X - s.X[j])), s.prob.lb, s.prob.ub); Q = np.vstack([Xt, Xl]); s.merge(Q, *s.evaluate(Q))

def bat_freq(fmin, fmax, beta): return fmin + (fmax - fmin) * beta
class MOBA(MOO):
    """Alg 29 MOBA (multiobjective bat algorithm). Steps: (1) N=50 bats with velocity 0, loudness A=1, pulse rate r=0, external archive; (2) f = fmin + (fmax-fmin)*beta; v += (x - x_leader) f; x += v, where the
    leader is a random archive member; (3) with prob (1-r) replace the move by a local walk around the leader, x = leader + eps*mean(A)*0.1*range; (4) evaluate; accept if rand<A and the new bat dominates (or is non-dominated
    with prob .5); on acceptance A *= alpha, r = r0 (1 - exp(-gamma t)); (5) archive. fmin=0, fmax=2, alpha=gamma=.9, r0=.5."""
    name = 'MOBA'
    def __init__(s, prob, N=50, fmin=0., fmax=2., alpha=.9, gamma=.9, r0=.5, cap=100, **kw): super().__init__(prob, N=N, **kw); s.fmin, s.fmax, s.alpha, s.gamma, s.r0 = fmin, fmax, alpha, gamma, r0; s.arc = Arc(cap, rng=s.rng)
    def initialize(s): super().initialize(); s.V = np.zeros_like(s.X, float); s.A = np.ones(s.N); s.r = np.zeros(s.N); s.arc.add(s.X, s.F, s.CV, s.Fp)
    def iterate(s):
        N, n = s.X.shape; p = s.prob; L = s.arc.X[s.rng.integers(0, len(s.arc), N)]; f = bat_freq(s.fmin, s.fmax, s.rng.random((N, 1)))
        s.V = np.clip(s.V + (s.X - L) * f, -(p.ub - p.lb), p.ub - p.lb); Xn = np.clip(s.X + s.V, p.lb, p.ub); lw = s.rng.random(N) > s.r
        walk = np.clip(L + s.rng.uniform(-1, 1, (N, n)) * s.A.mean() * .1 * (p.ub - p.lb), p.lb, p.ub); Xn = np.where(lw[:, None], walk, Xn)
        F, CV, Fp = s.evaluate(Xn); dn = np.all(Fp <= s.Fp, 1) & np.any(Fp < s.Fp, 1); od = np.all(s.Fp <= Fp, 1) & np.any(s.Fp < Fp, 1)
        acc = (s.rng.random(N) < s.A) & (dn | (~dn & ~od & (s.rng.random(N) < .5)))
        s.X[acc], s.F[acc], s.CV[acc], s.Fp[acc] = Xn[acc], F[acc], CV[acc], Fp[acc]; s.A[acc] *= s.alpha; s.r[acc] = s.r0 * (1 - np.exp(-s.gamma * (s.t + 1))); s.arc.add(Xn, F, CV, Fp)
    def out(s): return s.arc.X, s.arc.F, s.arc.CV
