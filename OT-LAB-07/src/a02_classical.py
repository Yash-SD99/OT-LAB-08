# %% [markdown]
# ## A2 · Classical methods (algorithms 1–9)
# **Sub-problem solver.** Section 2A allows `scipy.optimize` for the single-objective sub-problem; to stay dependency-free this notebook uses its own constrained differential evolution (Deb feasibility rule) + pattern-search polish, `solve_so`.
# The scalarisation, sweeps over weights / ε / targets, deviation variables and priority logic are all coded below.

# %%
class Counter: n = 0

def solve_so(fvec, lb, ub, seed=0, pop=40, gens=200, x0=None):
    """Minimise a scalar function (vectorised over rows) with constraint violation. fvec(X)->(values, cv).
    DE/rand/1/bin with feasibility rule, followed by shrinking pattern search."""
    rng = np.random.default_rng(seed); lb = np.asarray(lb, float); ub = np.asarray(ub, float); n = len(lb)
    better = lambda fu, cu, fv, cv: np.where((cu <= 1e-9) & (cv <= 1e-9), fu < fv, cu < cv)
    X = lb + rng.random((pop, n)) * (ub - lb)
    if x0 is not None: X[0] = x0
    fv, cv = fvec(X)
    for g in range(gens):
        r = rng.integers(0, pop, (pop, 3)); Fm = rng.uniform(.4, .9, (pop, 1)); V = X[r[:, 0]] + Fm * (X[r[:, 1]] - X[r[:, 2]])
        m = rng.random((pop, n)) < .9; m[np.arange(pop), rng.integers(0, n, pop)] = True
        U = np.clip(np.where(m, V, X), lb, ub); fu, cu = fvec(U); b = better(fu, cu, fv, cv)
        X[b], fv[b], cv[b] = U[b], fu[b], cu[b]
    i = 0
    for k in range(1, pop):
        if better(fv[k:k + 1], cv[k:k + 1], fv[i:i + 1], cv[i:i + 1])[0]: i = k
    x, f, c = X[i].copy(), fv[i], cv[i]; step = 0.05 * (ub - lb)
    dirs = np.vstack([np.eye(n), -np.eye(n), rng.normal(size=(2 * n, n))])
    for it in range(300):
        C = np.clip(x + step[None, :] * dirs, lb, ub); fc, cc = fvec(C)
        j = 0
        for k in range(1, len(C)):
            if better(fc[k:k + 1], cc[k:k + 1], fc[j:j + 1], cc[j:j + 1])[0]: j = k
        if better(fc[j:j + 1], cc[j:j + 1], np.array([f]), np.array([c]))[0]: x, f, c = C[j].copy(), fc[j], cc[j]
        else:
            step = step * 0.5
            if step.max() < 1e-12: break
    return x, f, c

def vecF(prob):
    """Normalised objective matrix + cv for a batch of decision vectors (counts evaluations)."""
    def g(X):
        F, cv = prob.evaluate(X); Counter.n += len(X); return prob.norm(F), cv
    return g

def sweep_solve(prob, scal, extra_cv=None, seeds=0, x0=None, **kw):
    g = vecF(prob)
    def fvec(X):
        Fn, cv = g(X); v = scal(Fn)
        if extra_cv is not None: cv = cv + extra_cv(Fn)
        return v, cv
    return solve_so(fvec, prob.lb, prob.ub, seed=seeds, x0=x0, **kw)

def front_from(prob, xs):
    X = np.array(xs); F, cv = prob.evaluate(X); ok = cv <= 1e-9; X, F = X[ok], F[ok]
    nd = nds(F)[0]; X, F = X[nd], F[nd]; _, u = np.unique(np.round(F, 6), axis=0, return_index=True); return X[u], F[u]   # distinct points only

# ---- 1 Weighted Sum
def weighted_sum(prob, n_w=21, seed=0):
    """Alg 1 Weighted Sum. Steps: (1) normalise objectives; (2) for each weight w in [0,1] minimise w*f1+(1-w)*f2; (3) collect solutions; (4) keep non-dominated."""
    xs = [sweep_solve(prob, lambda Fn, w=w: w * Fn[:, 0] + (1 - w) * Fn[:, 1], seeds=seed)[0] for w in np.linspace(0, 1, n_w)]
    return front_from(prob, xs)
# ---- 2 epsilon-constraint
def eps_constraint(prob, n_e=21, seed=0, free=0):
    """Alg 2 ε-constraint. Steps: (1) keep objective `free` as the objective; (2) bound the other by ε (swept between ideal and nadir); (3) solve; (4) keep non-dominated."""
    other = 1 - free; xs = []
    for e in np.linspace(0, 1, n_e):
        xs.append(sweep_solve(prob, lambda Fn: Fn[:, free], extra_cv=lambda Fn, e=e: np.maximum(0, Fn[:, other] - e), seeds=seed)[0])
    return front_from(prob, xs)
# ---- 3 Benson
def benson(prob, n_s=12, seed=0, w=(.5, .5)):
    """Alg 3 Benson. Steps: (1) pick a feasible start x0 (reference point F(x0)); (2) maximise sum_i w_i*l_i with l_i = F_i(x0)-F_i(x) >= 0; (3) result is a Pareto point that dominates x0; (4) repeat for several x0."""
    rng = np.random.default_rng(seed); xs = []; w = np.asarray(w)
    for k in range(n_s):
        for _ in range(200):
            x0 = prob.lb + rng.random(prob.n) * (prob.ub - prob.lb)
            if prob.evaluate(x0[None])[1][0] <= 0: break
        f0 = prob.norm(prob.evaluate(x0[None])[0])[0]
        xs.append(sweep_solve(prob, lambda Fn: -(w * (f0 - Fn)).sum(1), extra_cv=lambda Fn: np.maximum(0, Fn - f0).sum(1), seeds=seed + k, x0=x0)[0])
    return front_from(prob, xs)
# ---- 4 weight metric
def weight_metric(prob, p=2, n_w=21, seed=0):
    """Alg 4 Weight metric. Steps: (1) ideal point z*=0 (normalised); (2) minimise (sum (w_i f_i)^p)^(1/p) (p=1,2) or max_i w_i f_i (p=inf, Tchebycheff); (3) sweep w; (4) keep non-dominated."""
    def sc(Fn, w, p=p): return np.max(w * Fn, 1) if np.isinf(p) else ((w * Fn) ** p).sum(1) ** (1 / p)
    xs = [sweep_solve(prob, lambda Fn, w=np.array([a, 1 - a]): sc(Fn, w), seeds=seed)[0] for a in np.linspace(0.02, .98, n_w)]
    return front_from(prob, xs)
# ---- 5 value function
def value_function(prob, U, seed=0, **kw):
    """Alg 5 Value function. Steps: (1) DM supplies a utility U(f_normalised) to maximise; (2) minimise -U subject to constraints; (3) report the single preferred Pareto solution."""
    x, f, c = sweep_solve(prob, lambda Fn: -U(Fn), seeds=seed, **kw); F = prob.evaluate(x[None])[0][0]; return x, F, -f
def cobb_douglas(a): return lambda Fn: np.clip(1 - Fn[:, 0], 1e-9, None) ** a * np.clip(1 - Fn[:, 1], 1e-9, None) ** (1 - a)
# ---- 6 goal programming
def goal_programming(prob, goals, wts, types, sign=-1, seed=0):
    """Alg 6 Goal programming. Steps: (1) goals g_i in natural units; (2) deviation variables d+_i=max(0,f_i-g_i), d-_i=max(0,g_i-f_i) (so f_i + d-_i - d+_i = g_i);
    (3) minimise sum w_i*(d- for '>=' goals, d+ for '<=' goals) subject to the original constraints."""
    goals = np.asarray(goals, float); wts = np.asarray(wts, float); g = vecF(prob)
    def dev(F): nat = sign * F; return np.maximum(0, nat - goals), np.maximum(0, goals - nat)
    def fvec(X):
        F, cv = prob.evaluate(X); Counter.n += len(X); dp, dm = dev(F)
        return (wts * np.where(np.array(types) == '>=', dm, dp)).sum(1), cv
    x, f, c = solve_so(fvec, prob.lb, prob.ub, seed=seed); F = prob.evaluate(x[None])[0]; dp, dm = dev(F)
    return x, sign * F[0], dp[0], dm[0], f
# ---- 7 lexicographic
def lexicographic(prob, order, tol=5e-3, seed=0):
    """Alg 7 Lexicographic. Steps: for k in priority order: (1) minimise f_k subject to f_j <= f_j* + tol for every higher-priority j; (2) record f_k*; (3) continue with next priority."""
    best = {}
    for k in order:
        def fvec(X, k=k):
            F, cv = prob.evaluate(X); Counter.n += len(X)
            c = cv + sum(np.maximum(0, F[:, j] - best[j] - tol * max(1, abs(best[j]))) for j in best)
            return F[:, k], c
        x, f, c = solve_so(fvec, prob.lb, prob.ub, seed=seed); best[k] = f
    return x, prob.evaluate(x[None])[0][0], best
# ---- 8 TOPSIS, 9 VIKOR
def topsis(D, w, benefit):
    """Alg 8 TOPSIS. Steps: vector-normalise, weight, ideal/anti-ideal, Euclidean distances, closeness C = d-/(d+ + d-)."""
    D = np.asarray(D, float); V = D / np.sqrt((D ** 2).sum(0)) * w; benefit = np.asarray(benefit, bool)
    ideal, anti = np.where(benefit, V.max(0), V.min(0)), np.where(benefit, V.min(0), V.max(0))
    dp, dm = np.sqrt(((V - ideal) ** 2).sum(1)), np.sqrt(((V - anti) ** 2).sum(1)); C = dm / (dp + dm); return C, np.argsort(-C)
def vikor(D, w, benefit, v=.5):
    """Alg 9 VIKOR. Steps: f*,f- per criterion; S_i=sum w_j (f*_j-f_ij)/(f*_j-f-_j); R_i=max_j(...); Q_i=v(S-S*)/(S--S*)+(1-v)(R-R*)/(R--R*); rank by Q (small=best); check C1 (advantage) and C2 (stability)."""
    D = np.asarray(D, float); benefit = np.asarray(benefit, bool)
    fb, fw = np.where(benefit, D.max(0), D.min(0)), np.where(benefit, D.min(0), D.max(0))
    T = w * (fb - D) / np.where(fb == fw, 1, fb - fw); S, R = T.sum(1), T.max(1)
    Q = v * (S - S.min()) / max(S.max() - S.min(), 1e-12) + (1 - v) * (R - R.min()) / max(R.max() - R.min(), 1e-12)
    o = np.argsort(Q); J = len(Q); c1 = (Q[o[1]] - Q[o[0]]) >= 1 / (J - 1); c2 = (o[0] == np.argmin(S)) or (o[0] == np.argmin(R))
    return dict(S=S, R=R, Q=Q, order=o, C1=bool(c1), C2=bool(c2))

# %% [markdown]
# ### Unit tests (algorithms 1–9): each checks the key rule of the method

# %%
p1 = PROB['P1']; p11 = PROB['P11']; p7 = PROB['P7']
T = {}
x, F, _ = value_function(p1, cobb_douglas(.5)); T[5] = abs(x[0] - 5) < 1e-2                                   # symmetric utility -> mid point (4,4)
X1, F1 = weighted_sum(p1, 5); T[1] = abs(F1[np.argmin(np.abs(F1[:, 0] - F1[:, 1]))].mean() - 4) < .5 and F1[:, 0].min() < 1e-6   # w=.5 -> x=5, w=1 -> x=3
X2, F2 = eps_constraint(p1, 5); T[2] = np.isclose(F2[np.argmin(np.abs(F2[:, 1] - 4))][0], 4, atol=.05) and F2[:, 1].max() <= 16 + 1e-6   # f2<=4 (eps=.25 normalised) -> min f1 = 4
f0 = p1.norm(p1.evaluate(np.array([[2.]]))[0])[0]; Fb = p1.evaluate(sweep_solve(p1, lambda Fn: -(.5 * (f0 - Fn)).sum(1), extra_cv=lambda Fn: np.maximum(0, Fn - f0).sum(1), x0=np.array([2.]))[0][None])[0][0]
F0 = p1.evaluate(np.array([[2.]]))[0][0]; T[3] = bool(np.all(Fb <= F0 + 1e-6) and np.any(Fb < F0 - 1e-3)) and 3 - 1e-3 <= np.sqrt(Fb[0]) + 3 <= 7 + 1e-3        # Benson result dominates its start and is Pareto (x in [3,7])
Fw = [p1.evaluate(weight_metric(p1, pp, 3)[0])[0] for pp in (1, 2, np.inf)]; T[4] = all(len(f) >= 1 for f in Fw)
Tt = topsis(np.array([[1, 1], [2, 2], [3, 3.]]), np.array([.5, .5]), [True, True]); T[8] = list(Tt[1]) == [2, 1, 0]
Vv = vikor(np.array([[1, 1], [2, 2], [3, 3.]]), np.array([.5, .5]), [True, True]); T[9] = list(Vv['order']) == [2, 1, 0] and Vv['Q'][2] == 0
xg, nat, dp, dm, _ = goal_programming(p11, [2400, 600], [1, 1], ['>=', '>=']); T[6] = abs(dm[1] - 100) < .5 and abs(dm[0]) < .5
xl, Fl, bl = lexicographic(p11, [0, 1]); T[7] = np.allclose(xl, [300, 200], atol=1.0)
print({k: ('PASS' if v else 'FAIL') for k, v in sorted(T.items())}); assert all(T.values())
UNIT = {k: v for k, v in T.items()}

# %% [markdown]
# ### Alg 1 — Weighted Sum · Alg 2 — ε-constraint (runs on their test problems, seed 0)

# %%
def run_classical(no, name, fn, keys, label=None):
    ents = []
    for k in keys:
        Counter.n = 0; t = time.time(); X, F = fn(PROB[k]); dt = time.time() - t
        s = log(no, k, PROB[k], F, dt, Counter.n); FRONTS[(no, k)] = F; XFR[(no, k)] = X; line(no, name, k, s); ents.append((f"{name} on {PROB[k].name}", PROB[k], F))
    show(ents)
run_classical(1, 'Weighted Sum', weighted_sum, ['P1', 'P3', 'P4'])
run_classical(2, 'ε-constraint', eps_constraint, ['P2', 'P3', 'P5'])

# %% [markdown]
# ### Alg 3 — Benson · Alg 4 — Weight metric (p = 1, 2, ∞)

# %%
run_classical(3, 'Benson', benson, ['P1', 'P7'])
ents = []
for pp in (1, 2, np.inf):
    for k in ('P1', 'P3'):
        Counter.n = 0; t = time.time(); X, F = weight_metric(PROB[k], pp); dt = time.time() - t; kk = k + ('_p%s' % ('inf' if np.isinf(pp) else int(pp)))
        s = log(4, kk, PROB[k], F, dt, Counter.n); FRONTS[(4, kk)] = F; line(4, f"Wmetric p={pp}", kk, s)
        if k == 'P3': ents.append((f"p={pp} on P3", PROB[k], F))
show(ents)
RES[(4, 'P1')] = RES[(4, 'P1_p2')]; RES[(4, 'P3')] = RES[(4, 'P3_pinf')]

# %% [markdown]
# ### Alg 5 — Value function (P4 portfolio and P14 laptop) · Alg 6 — Goal programming · Alg 7 — Lexicographic

# %%
# P4: DM utility = Cobb-Douglas of normalised risk and return, weight a on risk (conservative a=.8, balanced .5, aggressive .2)
rows = []; fr4 = []
for a, lab in ((.8, 'conservative'), (.5, 'balanced'), (.2, 'aggressive')):
    Counter.n = 0; x, F, u = value_function(PROB['P4'], cobb_douglas(a)); w = PROB['P4'].weights(x[None])[0]; fr4.append(F)
    rows.append([lab, a, np.round(w, 3).tolist(), F[0], -F[1], u])
md_table(["investor", "weight a on risk", "weights w", "risk", "return", "utility"], rows)
# P14: DM wants ~1.8 kg and 12 h -> utility = -(normalised squared distance to the DM point)
dm_pt = np.array([1.8, -12.]); Upref = lambda Fn: -(((Fn - PROB['P14'].norm(dm_pt[None])[0]) ** 2).sum(1))
Counter.n = 0; x14, F14, u14 = value_function(PROB['P14'], Upref); print("P14 value-function solution: screen %.2f in, battery %.1f Wh -> %.3f kg, %.2f h" % (x14[0], x14[1], F14[0], -F14[1]))
s = log(5, 'P4', PROB['P4'], np.array(fr4), 0, Counter.n); s5 = log(5, 'P14', PROB['P14'], F14[None], 0, Counter.n)
show([("Value fn on P4 (3 investors)", PROB['P4'], np.array(fr4)), ("Value fn on P14 (DM point 1.8kg,12h)", PROB['P14'], F14[None])])

# %%
# Goal programming and Lexicographic on P11 (profit, units) and P7 (water)
md("**Goal programming, P11** (goals: profit ≥ 2400, units ≥ 600). Unit weights vs weights normalised by the goal values:")
rows = []
for lab, wt in (("unit weights", [1, 1]), ("normalised (1/goal)", [1 / 2400, 1 / 600]), ("priority-like (1,100)", [1, 100])):
    x, nat, dp, dm, obj = goal_programming(p11, [2400, 600], wt, ['>=', '>='])
    rows.append([lab, np.round(x, 2).tolist(), np.round(nat, 2).tolist(), np.round(dm, 2).tolist(), np.round(dp, 2).tolist(), obj])
md_table(["weights", "(xA,xB)", "(profit,units)", "d-", "d+", "objective"], rows)
md("**Lexicographic, P11** (profit first, units second) and **why the tie must be broken**: profit = 2400 on the whole edge from (300,200) to (0,400), units are 500 vs 400 there, so only the 2nd priority picks (300,200).")
x1, F1_, b1 = lexicographic(p11, [0]); x2, F2_, b2 = lexicographic(p11, [0, 1])
md_table(["priority", "(xA,xB)", "(profit,units)"], [["profit only", np.round(x1, 1).tolist(), (-F1_).round(1).tolist()], ["profit then units", np.round(x2, 1).tolist(), (-F2_).round(1).tolist()]])
md("**P7 water allocation** (goals: yield ≥ 40, output ≥ 50; reference point of the DM = (40, 50)). Both goals together would need 91.7 > 90 million m³, so a shortfall is unavoidable:")
rows = []
for lab, wt in (("unit weights", [1, 1]), ("normalised", [1 / 40, 1 / 50]), ("agri-first (10,1)", [10, 1])):
    x, nat, dp, dm, obj = goal_programming(p7, [40, 50], wt, ['>=', '>='])
    rows.append([lab, np.round(x, 2).tolist(), np.round(nat, 2).tolist(), np.round(dm, 2).tolist()])
md_table(["weights", "(x1,x2)", "(yield,output)", "d-"], rows)
xa, Fa, ba = lexicographic(p7, [0, 1]); xb, Fb_, bb = lexicographic(p7, [1, 0])
md_table(["priority", "(x1,x2)", "(yield,output)"], [["agri first", np.round(xa, 1).tolist(), (-Fa).round(1).tolist()], ["industry first", np.round(xb, 1).tolist(), (-Fb_).round(1).tolist()]])
for no, k in ((6, 'P11'), (6, 'P7'), (7, 'P11'), (7, 'P7')):
    RES[(no, k)] = dict(hv=np.nan, igd=np.nan, sp=np.nan, n=1, time=0., nfe=0, status='works')

# %% [markdown]
# ### Alg 8 — TOPSIS · Alg 9 — VIKOR on P12 (supplier selection)

# %%
crit = ["Cost", "Quality", "Delivery", "Reliability"]; benefit = [False, True, False, True]; wts = np.array([.3, .3, .2, .2])
minform = SUP * np.where(benefit, -1, 1); nd12 = nds(minform)[0]
print("Pareto-optimal suppliers:", ["S%d" % (i + 1) for i in sorted(nd12)])
C, o = topsis(SUP, wts, benefit); print("TOPSIS closeness:", np.round(C, 4), "ranking:", ["S%d" % (i + 1) for i in o])
rows = []
for v in (.25, .5, .75):
    r = vikor(SUP, wts, benefit, v); rows.append([v, np.round(r['Q'], 3).tolist(), ["S%d" % (i + 1) for i in r['order']], r['C1'], r['C2']])
md_table(["v", "Q", "ranking", "C1", "C2"], rows)
rows = []
for dw, lab in (([.1, -.1, 0, 0], "+0.1 cost / -0.1 quality"), ([-.1, .1, 0, 0], "-0.1 cost / +0.1 quality"), ([0, 0, .1, -.1], "+0.1 delivery / -0.1 reliability")):
    w2 = wts + np.array(dw); rows.append([lab, ["S%d" % (i + 1) for i in topsis(SUP, w2, benefit)[1]], ["S%d" % (i + 1) for i in vikor(SUP, w2, benefit, .5)['order']]])
md_table(["weight shift", "TOPSIS ranking", "VIKOR (v=.5) ranking"], rows)
# protection against one very bad criterion: add S6 with an excellent profile except reliability 40 %
SUP6 = np.vstack([SUP, [80, 9, 6, 40]])
t6 = topsis(SUP6, wts, benefit)[1]; v6 = vikor(SUP6, wts, benefit, 0.0)['order']; v65 = vikor(SUP6, wts, benefit, 0.5)['order']
print("With bad-criterion supplier S6 (reliability 40%): TOPSIS rank of S6 =", list(t6).index(5) + 1, "| VIKOR v=0 (max regret) rank of S6 =", list(v6).index(5) + 1, "| VIKOR v=.5 rank =", list(v65).index(5) + 1)
for no in (8, 9): RES[(no, 'P12')] = dict(hv=np.nan, igd=np.nan, sp=np.nan, n=len(SUP), time=0., nfe=0, status='works')
