# %% [markdown]
# ## A1 · Problems P1–P14
# Each problem returns objective values **F** (minimisation form) and a constraint violation **CV** (0 = feasible).
# `prob.norm(F)` maps to ≈[0,1] with the problem's ideal/nadir; the HV reference point is **1.1** in normalised space for every problem.
# Reference (true) fronts `prob.Z`: analytic for P1, P2, P3, P7, P10, P14; dense sampling + Pareto filter for P4, P5, P6 (labelled *reference*); P8/P9 have none (best-known union is used when needed).

# %%
class Problem:
    def __init__(s, name, M, n, lb, ub, fn, kind='real', ideal=None, nadir=None, Z=None, label=''):
        s.name, s.M, s.n, s.kind, s.fn, s.label = name, M, n, kind, fn, label
        s.lb, s.ub = np.broadcast_to(np.asarray(lb, float), (n,)).copy(), np.broadcast_to(np.asarray(ub, float), (n,)).copy()
        s.Z = Z
        if Z is not None and ideal is None: ideal, nadir = Z.min(0), Z.max(0)
        s.ideal, s.nadir = (None if ideal is None else np.asarray(ideal, float)), (None if nadir is None else np.asarray(nadir, float))
        s.ref = np.full(M, 1.1)
    def evaluate(s, X):
        X = np.atleast_2d(X); F, CV = s.fn(X); return np.asarray(F, float), np.asarray(CV, float)
    @property
    def rng_(s): return np.maximum(s.nadir - s.ideal, 1e-12)
    def norm(s, F): return (F - s.ideal) / s.rng_
    def pen(s, F, CV):
        """Static penalty used by algorithms WITHOUT feasibility rules: F + 10*CV*range."""
        return F + 10.0 * CV[:, None] * s.rng_[None, :]
    @property
    def Zn(s): return None if s.Z is None else s.norm(s.Z)
    def init(s, N, rng):
        if s.kind == 'real': return s.lb + rng.random((N, s.n)) * (s.ub - s.lb)
        if s.kind == 'perm': return np.array([rng.permutation(s.n) for _ in range(N)])
        return (rng.random((N, s.n)) < 0.5).astype(int)
    def decode(s, Xc, rng=None):
        """Continuous-position swarms: bits for binary problems, identity for real ones."""
        if s.kind == 'bin': return (rng.random(Xc.shape) < 1 / (1 + np.exp(-10 * (Xc - 0.5)))).astype(int)
        return Xc
    def __repr__(s): return s.name

def P1():
    f = lambda X: (np.c_[(X[:, 0] - 3) ** 2, (X[:, 0] - 7) ** 2], np.zeros(len(X)))
    x = np.linspace(3, 7, 400); Z = np.c_[(x - 3) ** 2, (x - 7) ** 2]
    return Problem('P1 factory speed', 2, 1, 1, 10, f, ideal=[0, 0], nadir=[16, 16], Z=Z)

def P2(integer=False):
    def f(X):
        x = np.round(X[:, 0]) if integer else X[:, 0]
        return np.c_[100 / x, x ** 2], np.zeros(len(X))
    x = np.arange(1, 11.) if integer else np.linspace(1, 10, 400)
    return Problem('P2 fleet' + (' (int)' if integer else ' (real)'), 2, 1, 1, 10, f, ideal=[10, 1], nadir=[100, 100], Z=np.c_[100 / x, x ** 2])

def P3(n=10):
    def f(X):
        g = 1 + X[:, 1:].sum(1); return np.c_[X[:, 0], g * (1 - (X[:, 0] / g) ** 2)], np.zeros(len(X))
    x = np.linspace(0, 1, 500)
    return Problem('P3 ZDT2', 2, n, 0, 1, f, ideal=[0, 0], nadir=[1, 1], Z=np.c_[x, 1 - x ** 2])

MU = np.array([.08, .12, .15, .10])
SIG = np.array([[.010, .002, .001, .003], [.002, .040, .006, .004], [.001, .006, .090, .005], [.003, .004, .005, .020]])
def P4(mode='repair'):
    def weights(X):
        if mode == 'repair':
            s = X.sum(1, keepdims=True); return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 0.25), np.zeros(len(X))
        return X, np.maximum(0, np.abs(X.sum(1) - 1) - 1e-3)
    def f(X):
        W, cv = weights(X); return np.c_[np.einsum('ij,jk,ik->i', W, SIG, W), -W @ MU], cv
    r = np.random.default_rng(0); W = np.vstack([r.dirichlet(np.full(4, a), 250000) for a in (.15, .5, 1.)] + [np.eye(4)])
    F = np.c_[np.einsum('ij,jk,ik->i', W, SIG, W), -W @ MU]; Z = pareto2d(F)
    p = Problem('P4 portfolio (%s)' % mode, 2, 4, 0, 1, f, Z=thin(Z, 600)); p.weights = lambda X: weights(X)[0]; return p

def P5():
    L, Pl, E, rho = 1., 1000., 200e9, 7850.
    def f(X):
        b, h = X[:, 0], X[:, 1]
        return np.c_[rho * L * b * h, 4 * Pl * L ** 3 / (E * b * h ** 3)], np.maximum(0, 2.4e-5 - b * h ** 2) / 2.4e-5
    bb, hh = np.meshgrid(np.linspace(.01, .1, 1200), np.linspace(.02, .2, 1200)); X = np.c_[bb.ravel(), hh.ravel()]
    F, cv = f(X); Z = pareto2d(F[cv <= 0])
    p = Problem('P5 beam', 2, 2, [.01, .02], [.1, .2], f, Z=thin(Z, 600))
    # ideal/nadir of the *reference front*; deflection spans decades -> use log10 scaling for normalisation
    p.norm = lambda F, p=p: (np.log10(F) - np.log10(p.Z).min(0)) / np.maximum(np.log10(p.Z).max(0) - np.log10(p.Z).min(0), 1e-12)
    p.ideal, p.nadir = p.Z.min(0), p.Z.max(0)
    p.pen = lambda F, CV, p=p: F + 10.0 * CV[:, None] * p.rng_[None, :]
    return p

def P6():
    def f(X):
        x1, x2 = X[:, 0], X[:, 1]
        g1 = (x1 - 5) ** 2 + x2 ** 2 - 25; g2 = 7.7 - ((x1 - 8) ** 2 + (x2 + 3) ** 2)
        return np.c_[4 * x1 ** 2 + 4 * x2 ** 2, (x1 - 5) ** 2 + (x2 - 5) ** 2], np.maximum(0, g1) + np.maximum(0, g2)
    a, b = np.meshgrid(np.linspace(0, 5, 1500), np.linspace(0, 3, 900)); X = np.c_[a.ravel(), b.ravel()]
    F, cv = f(X); Z = pareto2d(F[cv <= 0])
    return Problem('P6 Binh-Korn', 2, 2, [0, 0], [5, 3], f, Z=thin(Z, 500))

def P7():
    def f(X): return np.c_[-0.8 * X[:, 0], -1.2 * X[:, 1]], np.maximum(0, X[:, 0] + X[:, 1] - 90)
    x1 = np.linspace(10, 70, 400); Z = np.c_[-0.8 * x1, -1.2 * (90 - x1)]
    return Problem('P7 water', 2, 2, [10, 20], [70, 80], f, ideal=[-56, -96], nadir=[-8, -24], Z=Z)

def P8():
    r = np.random.RandomState(7); C = r.uniform(0, 100, (15, 2))      # == numpy.random.seed(7); uniform(0,100,(15,2))
    Dm = np.sqrt(((C[:, None] - C[None]) ** 2).sum(2)); U = np.triu(r.uniform(.5, 2., (15, 15)), 1); U = U + U.T; Tm = Dm * U
    def f(X):
        nx = np.roll(X, -1, 1); return np.c_[Dm[X, nx].sum(1), Tm[X, nx].sum(1)], np.zeros(len(X))
    p = Problem('P8 TSP-15', 2, 15, 0, 14, f, kind='perm'); p.C, p.D, p.T = C, Dm, Tm
    # ideal / nadir from 2-opt tours of each single objective (fixed, used for normalisation only)
    def two_opt(Cm, seed):
        rg = np.random.default_rng(seed); t = rg.permutation(15); best = Cm[t, np.roll(t, -1)].sum(); imp = True
        while imp:
            imp = False
            for i in range(14):
                for j in range(i + 2, 15):
                    u = np.r_[t[:i + 1], t[i + 1:j + 1][::-1], t[j + 1:]]; c = Cm[u, np.roll(u, -1)].sum()
                    if c < best - 1e-12: t, best, imp = u, c, True
        return t, best
    td = min((two_opt(Dm, s) for s in range(30)), key=lambda z: z[1])[0]; tt = min((two_opt(Tm, s) for s in range(30)), key=lambda z: z[1])[0]
    Fd, Ft = f(td[None])[0][0], f(tt[None])[0][0]; p.tour_d, p.tour_t = td, tt
    p.ideal, p.nadir = np.array([Fd[0], Ft[1]]), np.array([Ft[0], Fd[1]]); return p

def P9():
    from sklearn.datasets import load_breast_cancer                    # raw data loading only
    d = load_breast_cancer(); X = (d.data - d.data.mean(0)) / d.data.std(0); y = d.target.astype(int); n = len(y)
    fold = np.random.default_rng(0).permutation(n) % 5
    D2 = ((X.T[:, :, None] - X.T[:, None, :]) ** 2).astype(np.float32).reshape(30, -1)       # per-feature squared differences
    same = (fold[:, None] == fold[None, :]) | np.eye(n, dtype=bool)
    def acc(mask):
        """5-fold CV accuracy of a hand-coded 5-NN classifier on the selected features."""
        D = (mask.astype(np.float32) @ D2).reshape(n, n); D[same] = np.inf
        nn = np.argpartition(D, 5, axis=1)[:, :5]; pred = (y[nn].sum(1) >= 3).astype(int); return (pred == y).mean()
    def f(M):
        F = np.zeros((len(M), 2)); cv = np.zeros(len(M))
        for i, m in enumerate(M):
            k = int(m.sum())
            if k == 0: F[i] = [.5, 0]; cv[i] = 1
            else: F[i] = [1 - acc(m), k]
        return F, cv
    p = Problem('P9 feature selection', 2, 30, 0, 1, f, kind='bin', ideal=[0, 1], nadir=[.4, 30]); p.acc = acc; p.feat = d.feature_names; return p

def P10(M=3):
    n = M + 9
    def f(X):
        g = ((X[:, M - 1:] - .5) ** 2).sum(1); c = np.cos(X[:, :M - 1] * np.pi / 2); s = np.sin(X[:, :M - 1] * np.pi / 2); F = np.zeros((len(X), M))
        for m in range(M):                     # f_{m+1}: cos product of first M-1-m vars times sin(x_{M-1-m})
            F[:, m] = (1 + g) * np.prod(c[:, :M - 1 - m], axis=1) * (s[:, M - 1 - m] if m > 0 else 1)
        return F, np.zeros(len(X))
    W = das_dennis(M, 12 if M == 3 else 6); Z = W / np.linalg.norm(W, axis=1, keepdims=True)
    return Problem('P10 DTLZ2 M=%d' % M, M, n, 0, 1, f, ideal=np.zeros(M), nadir=np.ones(M), Z=Z)

def P11():
    def f(X):
        a, b = X[:, 0], X[:, 1]
        return np.c_[-(4 * a + 6 * b), -(a + b)], np.maximum(0, 2 * a + 3 * b - 1200) / 1200 + np.maximum(0, 4 * a + 2 * b - 1600) / 1600
    return Problem('P11 production', 2, 2, 0, 400, f, ideal=[-2400, -500], nadir=[0, 0])

SUP = np.array([[120, 8, 10, 92], [100, 7, 14, 88], [140, 9, 7, 95], [90, 6, 18, 80], [130, 8, 9, 90]], float)

def P14():
    def f(X):
        x1, x2 = X[:, 0], X[:, 1]; return np.c_[.4 + .05 * x1 + .012 * x2, -0.2 * x2 / (.5 + .03 * x1)], np.zeros(len(X))
    x2 = np.linspace(40, 90, 400); Z = np.c_[.4 + .05 * 13 + .012 * x2, -0.2 * x2 / (.5 + .03 * 13)]
    return Problem('P14 laptop', 2, 2, [13, 40], [17, 90], f, Z=Z)

PROB = {k: v for k, v in dict(P1=P1(), P2=P2(), P3=P3(), P4=P4(), P5=P5(), P6=P6(), P7=P7(), P8=P8(), P9=P9(), P10_3=P10(3), P10_5=P10(5), P11=P11(), P14=P14()).items()}
PROB['P2i'] = P2(True)
print("Problems ready:", ", ".join(PROB))

# %% [markdown]
# ### Sanity checks from the problem bank (all verified by hand in the statement)

# %%
chk = lambda p, x: p.evaluate(np.array([x], float))
assert np.allclose(chk(PROB['P1'], [5])[0], [4, 4]) and np.allclose(chk(PROB['P1'], [3])[0], [0, 16])
assert np.allclose(chk(PROB['P2'], [2])[0], [50, 4]) and np.allclose(chk(PROB['P2'], [10])[0], [10, 100])
assert np.allclose(PROB['P3'].Z[:, 1], 1 - PROB['P3'].Z[:, 0] ** 2)
F, cv = chk(PROB['P4'], [0, 0, 1, 0]); assert np.allclose(F[0], [0.09, -0.15])
F, cv = chk(PROB['P5'], [.01, .05]); assert np.allclose(F[0], [3.925, 0.016]) and cv[0] == 0
F, cv = chk(PROB['P6'], [0, 0]); assert np.allclose(F[0], [0, 50]) and cv[0] == 0
F, cv = chk(PROB['P6'], [5, 3]); assert np.allclose(F[0], [136, 4]) and cv[0] == 0
assert np.allclose(chk(PROB['P7'], [10, 80])[0], [-8, -96]) and np.allclose(chk(PROB['P7'], [70, 20])[0], [-56, -24])
x = np.random.default_rng(1).random((5, 12)); F = PROB['P10_3'].evaluate(x)[0]; g = ((x[:, 2:] - .5) ** 2).sum(1)
x0 = x.copy(); x0[:, 2:] = .5; F0 = PROB['P10_3'].evaluate(x0)[0]; assert np.allclose((F0 ** 2).sum(1), 1)
assert np.allclose(chk(PROB['P14'], [13, 40])[0], [1.53, -8.989], atol=2e-3) and np.allclose(chk(PROB['P14'], [13, 90])[0], [2.13, -20.2247], atol=2e-3)
F, cv = chk(PROB['P11'], [300, 200]); assert np.allclose(F[0], [-2400, -500]) and cv[0] == 0
print("P1–P14 sanity checks: PASS | P8 coordinates (seed 7):\n", np.round(PROB['P8'].C, 2).tolist())
print("P8 shortest-distance tour objs:", np.round(PROB['P8'].evaluate(PROB['P8'].tour_d[None])[0][0], 1), "| shortest-time tour objs:", np.round(PROB['P8'].evaluate(PROB['P8'].tour_t[None])[0][0], 1))
