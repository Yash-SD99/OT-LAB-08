# %% [markdown]
# ## A2 · Unit tests for algorithms 10–37 (each checks the key rule of the algorithm)

# %%
def _t():
    U = dict(UNIT); r = np.random.default_rng(0)
    # 10 VEGA: sub-population m is selected on objective m only
    Fp = r.random((200, 2)); w = vega_select(Fp, 0, 2000, r); U[10] = Fp[w, 0].mean() < Fp[:, 0].mean() - .1 and abs(Fp[w, 1].mean() - Fp[:, 1].mean()) < .05
    # 11 MOGA: rank = 1 + number of dominators
    U[11] = list(moga_rank(np.array([[1, 1], [2, 2], [3, 3], [1, 3.]]))) == [1, 2, 4, 2]
    # 12 NSGA: dummy fitness of front 1 > front 2 after sharing
    fit, fr = nsga_fitness(np.array([[1, 3], [3, 1], [3, 3.]])); U[12] = len(fr) == 2 and fit[:2].min() > fit[2]
    # 13 NPGA: candidate dominated by the comparison set loses
    F3 = np.array([[1, 1], [2, 2], [3, .5]]); U[13] = npga_winner(F3, 1, 2, np.array([0]), np.zeros(3)) == 2
    # 14 NSGA-II: boundary crowding inf, interior known value, lower rank wins tournaments
    cd = crowding(np.array([[0, 1], [.5, .5], [1, 0.]])); wn = tournament(np.array([0, 1]), np.array([0., 9.]), 400, r)
    U[14] = np.isinf(cd[[0, 2]]).all() and abs(cd[1] - 2) < 1e-12 and (wn == 0).mean() > .7
    # 15 NSGA-III: association to the nearest reference line
    ref, d = associate(np.array([[.9, .1], [.1, .9], [.5, .5]]), np.array([[1, 0], [0, 1], [.5, .5]])); U[15] = list(ref) == [0, 1, 2]
    # 16 SPEA2: non-dominated fitness < 1, dominated >= 1
    f, _ = spea2_fitness(np.array([[1, 3], [3, 1], [3, 3.]])); U[16] = (f[:2] < 1).all() and f[2] >= 1
    # 17 PESA-II: less-crowded hyper-box preferred
    a = PESA2(PROB['P3']); a.arc = Arc(100); Fq = np.vstack([np.full((9, 2), .05) + r.random((9, 2)) * .01, [[.95, .95]]]); a.arc.add(np.zeros((10, 10)), Fq, np.zeros(10), Fq)
    a.arc.Fp = Fq; a.arc.X = np.zeros((10, 10)); U[17] = np.mean([a.pick() == 9 for _ in range(400)]) > .5
    # 18 MOEA/D: Tchebycheff value and neighbourhood
    W = np.c_[np.linspace(0, 1, 11), 1 - np.linspace(0, 1, 11)]; B = neighbours(W, 3)
    U[18] = abs(tcheb(np.array([2., 4.]), np.array([.5, .5]), np.zeros(2), np.array([4., 4.])) - .5) < 1e-12 and set(B[0]) == {0, 1, 2}
    # 19 MOEA/D-DE: DE mutant v = x1 + F (x2 - x3)
    U[19] = abs(de_mutant(1., 2., 3., .5) - .5) < 1e-12
    # 20 MOEA/D-ACO: construction follows a dominant pheromone trail; result is a permutation
    t0 = np.random.default_rng(3).permutation(15); Wt = edge_matrix(t0) * 1000 + 1e-6; np.fill_diagonal(Wt, 0); tt = aco_tour(Wt, r)
    U[20] = sorted(tt) == list(range(15)) and (edge_matrix(tt) * edge_matrix(t0)).sum() >= 2 * 15 - 4
    # 21 Constrained NSGA-II: feasible beats infeasible with better objectives, then smaller violation
    c = CNSGA2(PROB['P5'], N=3); sel = c.survive(np.array([[5, 5], [1, 1], [2, 2.]]), np.array([0, 1, .5]), np.array([[5, 5], [1, 1], [2, 2.]]), 2); U[21] = list(sel) == [0, 2]
    # 22 MOPSO velocity rule
    U[22] = abs(pso_velocity(.5, 1, 1, 2., 0., 1., 3., 1., 1.) - 5) < 1e-12
    # 23 MOGWO: with a=0 the position is the mean of the three leaders
    Xa, Xb, Xd = np.full((1, 2), 1.), np.full((1, 2), 2.), np.full((1, 2), 6.); U[23] = np.allclose(gwo_pos(np.zeros((1, 2)), Xa, Xb, Xd, 0., r), 3.)
    # 24 MOABC: greedy dominance selection
    U[24] = dom_pair(np.array([1, 1.]), np.array([2, 2.])) and not dom_pair(np.array([2, 2.]), np.array([1, 1.])) and not dom_pair(np.array([1, 3.]), np.array([3, 1.]))
    # 25 MOACO: lambda=1 uses the distance pheromone/heuristic only; tours valid
    m = MOACO(PROB['P8']); m.max_nfe = 100; m.initialize(); m.iterate(); U[25] = np.allclose(m.weights(1.), (m.tau[0] ** m.alpha) * (m.eta[0] ** m.beta)) and all(sorted(x) == list(range(15)) for x in m.X)
    # 26 MOSA: Boltzmann acceptance
    U[26] = sa_accept_prob(-1, .1) == 1 and abs(sa_accept_prob(.1, .1) - np.exp(-1)) < 1e-12 and sa_accept_prob(.1, .01) < sa_accept_prob(.1, .1)
    # 27 MODE: after one generation every member is unchanged or was DOMINATED by its replacement
    md_ = MODE(PROB['P3'], N=30); md_.max_nfe = 1e9; md_.initialize(); old = md_.Fp.copy(); md_.iterate(); eq = (md_.Fp == old).all(1); dm_ = np.all(md_.Fp <= old, 1) & np.any(md_.Fp < old, 1); U[27] = bool(np.all(eq | dm_))
    # 28 MO-TLBO: teacher phase moves towards the teacher (mean of results = x + .5 (T - TF mean))
    Xs = np.ones((20000, 1)); tp = teacher_phase(Xs, np.array([3.]), 1., np.random.default_rng(1)); U[28] = abs(tp.mean() - 2.) < .02 and tp.min() >= 1 and tp.max() <= 3
    # 29 MOBA: frequency range and loudness/pulse schedule
    mb = MOBA(PROB['P3'], N=20); mb.max_nfe = 1e9; mb.initialize(); mb.iterate(); U[29] = bat_freq(0, 2, .5) == 1 and (mb.A <= 1).all() and (mb.r >= 0).all() and (mb.r < mb.r0).all()
    # 30 MOMA: local search never worsens the scalarised value
    mm = MOMA(PROB['P5'], N=20); mm.max_nfe = 1e9; mm.initialize(); w = np.array([[.5, .5]] * 20); before = mm.tcheb(mm.Fp, w); X2, F2, C2, Fp2 = mm.local_search(mm.X.copy(), mm.F.copy(), mm.CV.copy(), mm.Fp.copy(), w, 5); U[30] = bool((mm.tcheb(Fp2, w) <= before + 1e-12).all() and (mm.tcheb(Fp2, w) < before).any())
    # 31 RVEA: larger angle => larger APD (same norm direction family)
    V = np.array([[1, 0.], [0, 1.]]); ap, aa = apd_values(np.array([[1, .01], [1, .5]]), V, np.array([np.pi / 2] * 2), np.zeros(2), 1., 2); U[31] = list(aa) == [0, 0] and ap[1] > ap[0] * 1.2
    # 32 IBEA: I_eps+ values and fitness order
    I = eps_indicator(np.array([[0, 0], [1, 1.]])); fi, _ = ibea_fitness(I); U[32] = I[0, 1] == -1 and I[1, 0] == 1 and fi[0] > fi[1]
    # 33 SMS-EMOA: hypervolume contributions of a 3-point front
    hc = hv_contributions(np.array([[0, 1], [.5, .5], [1, 0.]]), np.array([1.1, 1.1])); U[33] = abs(hc[1] - .25) < 1e-9 and np.argmin(hc) in (0, 2)
    # 34 HypE: MC fitness (k=1) approximates the exact exclusive contribution
    Fn = np.array([[0, 1], [.5, .5], [1, 0.]]); S = np.random.default_rng(0).random((60000, 2)) * 1.1; hf = hype_fitness(Fn, 1, S, 1.21); U[34] = np.allclose(hf, hv_contributions(Fn, np.array([1.1, 1.1])), atol=.015)
    # 35 R-NSGA-II: preference order with ε-clearing
    U[35] = list(pref_order(np.array([[.1, .9], [.11, .89], [.9, .1]]), np.array([[0, 1.]]), .05)) == [0, 2, 1]
    # 36 MGDA
    U[36] = abs(mgda_alpha(np.array([1., 0]), np.array([0., 1])) - .5) < 1e-12 and mgda_alpha(np.array([1., 0]), np.array([2., 0])) == 1. and abs(min_norm(np.eye(3)).sum() - 1) < 1e-9
    # 37 GradNorm: the task with the larger gradient gets a smaller weight, sum(w)=M
    w2 = gradnorm_update(np.ones(2), np.array([10., 1.]), np.ones(2), np.ones(2)); U[37] = w2[0] < 1 < w2[1] and abs(w2.sum() - 2) < 1e-9
    return U
UNIT = _t(); NAMES = {1: 'Weighted Sum', 2: 'ε-Constraint', 3: 'Benson', 4: 'Weight Metric', 5: 'Value Function', 6: 'Goal Programming', 7: 'Lexicographic', 8: 'TOPSIS', 9: 'VIKOR', 10: 'VEGA', 11: 'MOGA', 12: 'NSGA', 13: 'NPGA', 14: 'NSGA-II',
 15: 'NSGA-III', 16: 'SPEA2', 17: 'PESA-II', 18: 'MOEA/D', 19: 'MOEA/D-DE', 20: 'MOEA/D-ACO', 21: 'Constrained NSGA-II', 22: 'MOPSO', 23: 'MOGWO', 24: 'MOABC', 25: 'MOACO', 26: 'MOSA', 27: 'MODE', 28: 'MO-TLBO', 29: 'MOBA', 30: 'MOMA',
 31: 'RVEA', 32: 'IBEA', 33: 'SMS-EMOA', 34: 'HypE', 35: 'R-NSGA-II', 36: 'MGDA', 37: 'GradNorm'}
md_table(["#", "algorithm", "unit test"], [[k, NAMES[k], "PASS" if UNIT.get(k) else "FAIL"] for k in range(1, 38)])
assert all(UNIT.get(k) for k in range(1, 38)), [k for k in range(1, 38) if not UNIT.get(k)]

# %% [markdown]
# ## A2 · Runs with a fixed seed (seed 0, 10,000 evaluations; P9 uses 3,000 because every evaluation is a 5-fold 5-NN cross-validation)
# Each algorithm is run on the problems listed for it in the catalogue. Plots show the obtained front (red) over the true / reference front (grey). Budget = number of function evaluations (the actual count can exceed 10,000 by at most one generation).

# %%
MANY = dict(eta_c=30., pc=1.)
def ref_pt(prob, frac=.25): z = prob.Z[np.argsort(prob.Z[:, 0])]; return z[int(len(z) * frac)] * np.where(z[int(len(z) * frac)] > 0, .92, 1.08)
REFP = {'P5': None, 'P14': np.array([1.8, -12.])}
def run_family(items):
    for no, cls, plist in items:
        ents = []
        for key, kw, nf in plist:
            kw = dict(kw)
            if cls is RNSGA2: kw['ref'] = ref_pt(PROB[key]) if key == 'P5' else REFP[key]
            X, F, dt, a = run_alg(cls, PROB[key], seed=0, nfe=nf or BUDGET, **kw); s = log(no, key, PROB[key], F, dt, a.nfe); FRONTS[(no, key)] = F; XFR[(no, key)] = X; line(no, cls.name, key, s)
            ents.append((f"{cls.name} on {PROB[key].name}", PROB[key], F))
            if cls is RNSGA2: ents[-1] = (ents[-1][0] + ' (pref. pt ★)', PROB[key], F)
        show(ents)

P9N = 3000 if not QUICK else 1000
run_family([(10, VEGA, [('P2', {}, None), ('P3', {}, None)]), (11, MOGA, [('P3', {}, None), ('P5', {}, None)]), (12, NSGA, [('P5', {}, None), ('P7', {}, None)]), (13, NPGA, [('P3', {}, None), ('P5', {}, None)])])

# %%
run_family([(14, NSGA2, [('P3', {}, None), ('P5', {}, None), ('P6', {}, None), ('P10_5', MANY, None)]), (15, NSGA3, [('P10_3', MANY, None), ('P10_5', MANY, None)]), (16, SPEA2, [('P4', {}, None), ('P5', {}, None)]), (17, PESA2, [('P3', {}, None), ('P5', {}, None)])])

# %%
run_family([(18, MOEAD, [('P3', {}, None), ('P10_3', MANY, None)]), (19, MOEADDE, [('P3', {}, None), ('P5', {}, None)]), (20, MOEADACO, [('P8', {}, None)]), (21, CNSGA2, [('P5', {}, None), ('P6', {}, None), ('P7', {}, None)])])

# %%
run_family([(22, MOPSO, [('P4', {}, None), ('P5', {}, None)]), (23, MOGWO, [('P5', {}, None), ('P9', {}, P9N)]), (24, MOABC, [('P3', {}, None), ('P6', {}, None)]), (25, MOACO, [('P8', {}, None)])])

# %%
run_family([(26, MOSA, [('P3', {}, None), ('P5', {}, None)]), (27, MODE, [('P3', {}, None), ('P5', {}, None)]), (28, MOTLBO, [('P3', {}, None), ('P5', {}, None)]), (29, MOBA, [('P3', {}, None), ('P4', {}, None)])])

# %%
run_family([(30, MOMA, [('P5', {}, None), ('P9', {}, P9N)]), (31, RVEA, [('P10_3', MANY, None), ('P10_5', MANY, None)]), (32, IBEA, [('P3', {}, None), ('P14', {}, None)]), (33, SMSEMOA, [('P3', {}, None), ('P14', {}, None)])])

# %%
run_family([(34, HypE, [('P10_3', MANY, None), ('P10_5', MANY, None)]), (35, RNSGA2, [('P5', {}, None), ('P14', {}, None)])])

# %% [markdown]
# ## A3 · Implementation log

# %%
def fmt(v): return '–' if (v is None or (isinstance(v, float) and not np.isfinite(v))) else f"{v:.3f}"
rows = []
for k in range(1, 38):
    keys = [kk for (n, kk) in RES if n == k]; keys = [x for x in keys if '_p' not in x] or keys
    if not keys: rows.append([k, NAMES[k], '–', '–', '–', '–', '–', 'not done']); continue
    R = [RES[(k, x)] for x in keys]; ok = UNIT.get(k) and all(r['n'] >= 1 for r in R)
    note = {27: 'works (collapses on P3, see Task B)', 12: 'works', 26: 'works'}.get(k, 'works')
    rows.append([k, NAMES[k], ", ".join(keys), "/".join(str(int(r['nfe'])) if r['nfe'] else 'sweep' for r in R), " / ".join(fmt(r['hv']) for r in R), " / ".join(fmt(r['igd']) for r in R), " / ".join(f"{r['time']:.1f}s" for r in R), note if ok else 'partly works'])
md_table(["#", "algorithm", "problems", "evaluations", "HV", "IGD", "runtime", "status"], rows)
