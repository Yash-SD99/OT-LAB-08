# %% [markdown]
# ## Family: many-objective and indicator-based methods (algorithms 31–34)

# %%
def apd_values(F, V, gamma, zmin, frac, M, alpha=2.):
    """Alg 31 core. Angle-penalised distance: APD = (1 + P) ||f - zmin||, P = M (t/tmax)^alpha theta / gamma_v, theta = angle between f - zmin and its reference vector."""
    Ft = F - zmin; nrm = np.linalg.norm(Ft, axis=1) + 1e-12; cos = (Ft / nrm[:, None]) @ V.T; a = cos.argmax(1); th = np.arccos(np.clip(cos[np.arange(len(F)), a], -1, 1))
    return (1 + M * frac ** alpha * th / gamma[a]) * nrm, a
class RVEA(MOO):
    """Alg 31 RVEA (Cheng et al. 2016). Steps: (1) unit reference vectors V (Das–Dennis); (2) random mating -> offspring; (3) P+Q: associate each solution to the reference vector with the smallest angle;
    (4) in every sub-population keep the solution with the smallest angle-penalised distance (penalty grows with generation count); (5) every fr*tmax generations rescale V by the current objective range
    (reference vector adaptation). alpha=2, fr=0.1. M=3: 12 divisions; M=5: 4."""
    name = 'RVEA'
    def __init__(s, prob, p=None, alpha=2., fr=.1, **kw):
        M = prob.M; W = das_dennis(M, p or {3: 12, 5: 4}.get(M, 3)); s_N = kw.pop('N', len(W)); super().__init__(prob, N=s_N, **kw)
        s.V0 = W / np.linalg.norm(W, axis=1, keepdims=True); s.alpha, s.fr = alpha, fr; s.set_V(s.V0)
    def set_V(s, V): s.V = V / np.linalg.norm(V, axis=1, keepdims=True); c = s.V @ s.V.T; np.fill_diagonal(c, -1); s.gamma = np.maximum(np.arccos(np.clip(c.max(1), -1, 1)), 1e-6)
    def select(s, F, frac):
        apd, a = apd_values(F, s.V, s.gamma, F.min(0), frac, s.prob.M, s.alpha); keep = []
        for v in np.unique(a): m = np.where(a == v)[0]; keep.append(m[np.argmin(apd[m])])
        return np.array(keep)
    def iterate(s):
        tmax = max(1, s.max_nfe // s.N); Q = s.var.offspring(s.X[s.rng.integers(0, len(s.X), s.N)]); QF, QCV, QFp = s.evaluate(Q)
        X, F, CV, Fp = np.vstack([s.X, Q]), np.vstack([s.F, QF]), np.r_[s.CV, QCV], np.vstack([s.Fp, QFp]); k = s.select(Fp, min(1., s.t / tmax))
        s.X, s.F, s.CV, s.Fp = X[k], F[k], CV[k], Fp[k]
        if s.t % max(1, int(s.fr * tmax)) == 0 and s.t > 0: rg = np.maximum(s.Fp.max(0) - s.Fp.min(0), 1e-9); s.set_V(s.V0 * rg)

def eps_indicator(Fn): """I_eps+(x_i, x_j) = max_k (f_k(x_i) - f_k(x_j)): the smallest shift that makes x_i weakly dominate x_j.""" ; return Fn[:, None, :].__sub__(Fn[None, :, :]).max(2)
def ibea_fitness(I, kappa=.05):
    """Fitness(x_j) = - sum_{i != j} exp(-I(x_i, x_j) / (c kappa)),  c = max |I| (larger is better)."""
    c = np.abs(I).max() + 1e-12; E = np.exp(-I / (c * kappa)); np.fill_diagonal(E, 0); return -E.sum(0), E
class IBEA(MOO):
    """Alg 32 IBEA (Zitzler & Künzli 2004), additive ε-indicator. Steps: (1) population N; (2) binary tournament on indicator fitness; (3) SBX+PM; (4) union of 2N, normalise objectives;
    (5) fitness from pairwise I_eps+ values; (6) repeatedly delete the worst-fitness individual and update the others' fitness until N remain. No explicit diversity mechanism. kappa=.05."""
    name = 'IBEA'
    def fit(s, Fp): return ibea_fitness(eps_indicator(minmax(Fp)), getattr(s, 'kappa', .05))
    def initialize(s): super().initialize(); s.pf, _ = s.fit(s.Fp)
    def iterate(s):
        a = s.rng.integers(0, s.N, s.N); b = s.rng.integers(0, s.N, s.N); w = np.where(s.pf[a] >= s.pf[b], a, b); Q = s.var.offspring(s.X[w]); QF, QCV, QFp = s.evaluate(Q)
        X, F, CV, Fp = np.vstack([s.X, Q]), np.vstack([s.F, QF]), np.r_[s.CV, QCV], np.vstack([s.Fp, QFp]); fit, E = ibea_fitness(eps_indicator(minmax(Fp)), getattr(s, 'kappa', .05)); alive = np.ones(len(X), bool)
        while alive.sum() > s.N: w = np.argmin(np.where(alive, fit, np.inf)); alive[w] = False; fit = fit + E[w]
        k = np.where(alive)[0]; s.X, s.F, s.CV, s.Fp = X[k], F[k], CV[k], Fp[k]; s.pf, _ = s.fit(s.Fp)

def hv_contributions(Fn, ref):
    """Alg 33 core. Exclusive hypervolume contribution of every point of a (normalised) set: exact in 2-D, Monte-Carlo for M>=3."""
    n = len(Fn)
    if Fn.shape[1] == 2:
        o = np.argsort(Fn[:, 0]); P = Fn[o]; c = np.zeros(n); right = np.r_[P[1:, 0], ref[0]]; up = np.r_[ref[1], P[:-1, 1]]
        c[o] = np.maximum(right - P[:, 0], 0) * np.maximum(up - P[:, 1], 0) * (P[:, 0] < ref[0]) * (P[:, 1] < ref[1]); return c
    tot = hypervolume(Fn, ref); return np.array([tot - hypervolume(np.delete(Fn, i, 0), ref) for i in range(n)])
class SMSEMOA(MOO):
    """Alg 33 SMS-EMOA (Beume, Naujoks & Emmerich 2007). Steady-state (mu+1). Steps: (1) population N=50; (2) two random parents -> one child (SBX+PM); (3) non-dominated sort of the N+1 individuals;
    (4) in the worst front delete the individual with the smallest exclusive hypervolume contribution (reference point = 1.1 in normalised space)."""
    name = 'SMS-EMOA'
    def __init__(s, prob, N=50, **kw): super().__init__(prob, N=N, **kw)
    def iterate(s):
        a, b = s.rng.choice(s.N, 2, replace=False); y = s.var.one(s.X[a], s.X[b])[None]; yF, yCV, yFp = s.evaluate(y)
        X, F, CV, Fp = np.vstack([s.X, y]), np.vstack([s.F, yF]), np.r_[s.CV, yCV], np.vstack([s.Fp, yFp]); last = nds(Fp)[-1]
        w = last[0] if len(last) == 1 else last[np.argmin(hv_contributions(s.prob.norm(Fp[last]), s.prob.ref))]
        k = np.delete(np.arange(len(X)), w); s.X, s.F, s.CV, s.Fp = X[k], F[k], CV[k], Fp[k]

def hype_alpha(n, k):
    """HypE weights: alpha_j = (1/j) prod_{i=1}^{j-1} (k-i)/(n-i): share of a sampled point dominated by j individuals when k individuals are to be removed."""
    a = np.zeros(n + 1)
    for j in range(1, n + 1): a[j] = np.prod([(k - i) / (n - i) for i in range(1, j)]) / j if j <= k else 0.
    return a
def hype_fitness(Fn, k, S, vol):
    """Monte-Carlo hypervolume-based fitness: S = uniform samples in the box [0,ref]; fitness_i = vol/|S| * sum over samples dominated by i of alpha_{#dominators}."""
    dom = (Fn[None, :, :] <= S[:, None, :]).all(2); cnt = dom.sum(1); w = hype_alpha(len(Fn), k)[cnt]; return (dom * w[:, None]).sum(0) * vol / len(S)
class HypE(MOO):
    """Alg 34 HypE (Bader & Zitzler 2011). Steps: (1) population N; (2) mating by tournament on HypE fitness; (3) SBX+PM; (4) union: take whole non-dominated fronts while they fit;
    (5) in the last front remove, one at a time, the individual with the smallest MC-estimated hypervolume fitness (k = number still to remove); samples: 1000 per selection. Reference point 1.1 (normalised)."""
    name = 'HypE'
    def __init__(s, prob, N=100, nsamp=1000, **kw): super().__init__(prob, N=N, **kw); s.nsamp = nsamp
    def samples(s): return s.rng.random((s.nsamp, s.prob.M)) * s.prob.ref, float(np.prod(s.prob.ref))
    def survive(s, Fp, N):
        fronts = nds(Fp); sel = []
        for fr in fronts:
            if len(sel) + len(fr) <= N: sel += fr.tolist()
            else:
                S, vol = s.samples(); cand = sel + fr.tolist(); Fn = s.prob.norm(Fp[cand]); dom = (Fn[None] <= S[:, None, :]).all(2); alive = np.ones(len(cand), bool); k = len(cand) - N
                while alive.sum() > N:
                    d = dom[:, alive]; cnt = d.sum(1); w = hype_alpha(alive.sum(), alive.sum() - N)[cnt]; fit = (d * w[:, None]).sum(0); fit_full = np.full(len(cand), np.inf)
                    fit_full[np.where(alive)[0]] = fit; fit_full[:len(sel)] = np.inf; alive[np.argmin(fit_full)] = False
                sel = [c for c, a in zip(cand, alive) if a]; break
        return np.array(sel)
    def iterate(s):
        S, vol = s.samples(); fit = hype_fitness(s.prob.norm(s.Fp), s.N // 2, S, vol); a = s.rng.integers(0, s.N, s.N); b = s.rng.integers(0, s.N, s.N)
        Q = s.var.offspring(s.X[np.where(fit[a] >= fit[b], a, b)]); QF, QCV, QFp = s.evaluate(Q)
        X, F, CV, Fp = np.vstack([s.X, Q]), np.vstack([s.F, QF]), np.r_[s.CV, QCV], np.vstack([s.Fp, QFp]); k = s.survive(Fp, s.N); s.X, s.F, s.CV, s.Fp = X[k], F[k], CV[k], Fp[k]
