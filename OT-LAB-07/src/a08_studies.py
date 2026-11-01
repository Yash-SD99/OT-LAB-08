# %% [markdown]
# ## Problem studies — the "Investigate" questions of P1–P14
# Short experiments that answer the questions printed under each problem (they also supply numbers used as evidence in Task B).

# %%
import copy
def nocon(p): q = copy.copy(p); q.fn = lambda X, f=p.fn: (f(X)[0], np.zeros(len(X))); return q
def front_stats(prob, F): s = score(prob, F); return s['n'], s['hv'], s['igd'], s['sp']

# ---------- P1
md("### P1 — Factory speed (convex)")
F1 = FRONTS[(1, 'P1')]; Fn = PROB['P1'].norm(F1); gaps = np.sqrt((np.diff(Fn[np.argsort(Fn[:, 0])], axis=0) ** 2).sum(1))
print(f"Weighted sum, 21 equally spaced weights -> {len(F1)} distinct points; consecutive gaps on the front: min {gaps.min():.3f}, max {gaps.max():.3f} (ratio {gaps.max() / gaps.min():.1f}) -> on the convex P1 equal weight steps give nearly even spacing, but on P4 only {RES[(1,'P4')]['n']} of 21 weights give distinct points (Spacing {RES[(1,'P4')]['sp']:.3f} vs {RES[(1,'P1')]['sp']:.3f} on P1): weights bunch the points near the ends of the front.")
STUDY['ws_gap_ratio'] = gaps.max() / gaps.min()
md_table(["weight metric", "points", "HV", "IGD", "Spacing"], [[f"p={p}", *[RES[(4, f'P1_p{p}')][k] for k in ('n', 'hv', 'igd', 'sp')]] for p in ('1', '2', 'inf')])

# ---------- P2
md("### P2 — Fleet sizing: real vs integer x, and spread of the points")
rows = []
for key in ('P2', 'P2i'):
    X, F, dt, a = run_alg(NSGA2, PROB[key], nfe=BUDGET); Fn = PROB[key].norm(F); o = np.argsort(Fn[:, 0]); d = np.sqrt((np.diff(Fn[o], axis=0) ** 2).sum(1)); xs = np.round(X[:, 0] if key == 'P2' else np.round(X[:, 0]), 3)
    rows.append([PROB[key].name, len(F), len(np.unique(np.round(F[:, 0], 6))), d.max() if len(d) else np.nan, np.sort(xs)[:3].tolist(), np.sort(xs)[-3:].tolist()])
    STUDY['p2_' + key] = len(np.unique(np.round(F[:, 0], 6)))
md_table(["variant", "front points", "distinct", "largest gap (norm.)", "smallest x", "largest x"], rows)
Xv, Fv, _, _ = run_alg(VEGA, PROB['P2'], nfe=BUDGET); share = (Fv[:, 0] < 40).mean(); print(f"VEGA on P2: {len(Fv)} points, {share:.0%} of them in the cheap-cost half; NSGA-II keeps the spread by crowding distance.")

# ---------- P3
md("### P3 — which methods miss the middle, and IGD vs population size / generations (NSGA-II)")
mid = lambda F: int(((F[:, 0] > .25) & (F[:, 0] < .75)).sum())
rows = [[nm, len(FRONTS[k]), mid(FRONTS[k])] for nm, k in (("Weighted sum (21 w)", (1, 'P3')), ("Weight metric p=1", (4, 'P3_p1')), ("Weight metric p=2", (4, 'P3_p2')), ("Weight metric p=inf", (4, 'P3_pinf')), ("ε-constraint", (2, 'P3')), ("VEGA", (10, 'P3')), ("MODE", (27, 'P3')), ("NSGA-II", (14, 'P3')))]
md_table(["method", "points", "points with 0.25<f1<0.75"], rows); STUDY['ws_mid'] = rows[0][2]; STUDY['p1_mid'] = rows[1][2]; STUDY['pinf_mid'] = rows[3][2]
pops, gens = (20, 50, 100, 200), (25, 50, 100, 200); T = np.zeros((4, 4))
for i, N in enumerate(pops):
    for j, G in enumerate(gens): T[i, j] = np.mean([score(PROB['P3'], run_alg(NSGA2, PROB['P3'], seed=s, nfe=N * G, N=N)[1])['igd'] for s in range(3 if not QUICK else 1)])
md("IGD (mean of 3 seeds); rows = population size, columns = generations:"); md_table(["N \\ gens"] + [str(g) for g in gens], [[str(n)] + list(T[i]) for i, n in enumerate(pops)]); STUDY['igd_20_25'], STUDY['igd_200_200'] = T[0, 0], T[3, 3]

# ---------- P4
md("### P4 — Portfolio: repair vs penalty for Σw = 1, and investor profiles")
rows = []
for lab, key, cls in (("repair (w/Σw) + NSGA-II", 'P4', NSGA2), ("penalty |Σw-1|>1e-3 + NSGA-II", 'P4p', NSGA2), ("penalty + Deb rules (C-NSGA-II)", 'P4p', CNSGA2)):
    if key == 'P4p' and key not in PROB: PROB['P4p'] = P4('penalty'); PROB['P4p'].Z, PROB['P4p'].ideal, PROB['P4p'].nadir = PROB['P4'].Z, PROB['P4'].ideal, PROB['P4'].nadir
    X, F, dt, a = run_alg(cls, PROB[key], nfe=BUDGET); s = score(PROB['P4'], F); rows.append([lab, s['n'], s['hv'], s['igd'], 100 * (a.CV <= 0).mean(), (np.abs(X.sum(1) - 1) if key == 'P4p' else np.zeros(1)).max()])
    if key == 'P4': Xr, Fr = X, F
md_table(["scheme", "front pts", "HV", "IGD", "% feasible in final pop", "max |Σw-1|"], rows); STUDY['p4_repair_hv'], STUDY['p4_pen_hv'] = rows[0][2], rows[1][2]
W = PROB['P4'].weights(Xr); o = np.argsort(Fr[:, 0]); th = np.array_split(o, 3)
md_table(["profile", "risk range", "return range", "mean weights (A1..A4)"], [[n, f"{Fr[t, 0].min():.4f}–{Fr[t, 0].max():.4f}", f"{-Fr[t, 1].max():.3f}–{-Fr[t, 1].min():.3f}", np.round(W[t].mean(0), 2).tolist()] for n, t in zip(("conservative", "balanced", "aggressive"), th)])
print("Minimum-risk end of the front is a mixture:", np.round(W[o[0]], 3).tolist(), "| maximum-return end:", np.round(W[o[-1]], 3).tolist())

# ---------- P5
md("### P5 — Cantilever beam: infeasible fraction, constraint-handling rule, ignoring the stress limit")
r = np.random.default_rng(0); Xs = PROB['P5'].lb + r.random((100000, 2)) * (PROB['P5'].ub - PROB['P5'].lb); cv = PROB['P5'].evaluate(Xs)[1]; STUDY['p5_infeas'] = (cv > 0).mean()
print(f"Fraction of uniformly random designs that violate the stress limit: {(cv > 0).mean():.1%}")
p5u = nocon(PROB['P5']); Xu, Fu, _, _ = run_alg(NSGA2, p5u, nfe=BUDGET); cvu = PROB['P5'].evaluate(Xu)[1]; STUDY['p5_ignore_infeas'] = (cvu > 0).mean()
Xc_, Fc, _, _ = run_alg(CNSGA2, PROB['P5'], nfe=BUDGET)
print(f"Ignoring the constraint: {(cvu > 0).mean():.0%} of the 'front' designs violate stress (lightest design {Fu[:, 0].min():.2f} kg vs {Fc[:, 0].min():.2f} kg with the constraint). Rule used elsewhere: Deb's feasibility rules (C-NSGA-II) / penalty 10·CV·range (others).")
fig, ax = plt.subplots(figsize=(5, 3.6)); ax.loglog(Fu[:, 0], Fu[:, 1], 'o', ms=3, c='C1', label='stress limit ignored'); ax.loglog(Fc[:, 0], Fc[:, 1], 'o', ms=3, c='C3', label='with Deb rules'); ax.set_xlabel('mass (kg)'); ax.set_ylabel('deflection (m)'); ax.legend(); ax.set_title('P5 front with / without the stress constraint', fontsize=9); plt.show()

# ---------- P6
md("### P6 — Binh–Korn: Deb's three rules on four candidates, and an algorithm without feasibility rules")
cands = {"a (2,2)": (2, 2), "b (1,1)": (1, 1), "c (0,3)": (0, 3), "d (.5,3)": (.5, 3)}
Fc4, cv4 = PROB['P6'].evaluate(np.array(list(cands.values()), float)); rows = []
for (na, i), (nb, j) in itertools.combinations(list(zip(cands, range(4))), 2):
    wn = na if cdominates(Fc4[i], cv4[i], Fc4[j], cv4[j]) else (nb if cdominates(Fc4[j], cv4[j], Fc4[i], cv4[i]) else 'neither')
    rule = "1 feasible>infeasible" if (cv4[i] <= 0) != (cv4[j] <= 0) else ("2 smaller violation" if cv4[i] > 0 else "3 dominance")
    rows.append([na, nb, f"({Fc4[i][0]:.1f},{Fc4[i][1]:.1f}) cv={cv4[i]:.2f}", f"({Fc4[j][0]:.1f},{Fc4[j][1]:.1f}) cv={cv4[j]:.2f}", wn, rule])
md_table(["a", "b", "F(a), cv", "F(b), cv", "winner", "rule"], rows)
cs = CNSGA2(PROB['P6'], N=4); sel = cs.survive(Fc4, cv4, Fc4, 3); print("Algorithm check: C-NSGA-II keeps (N=3) candidates", [list(cands)[i] for i in sel], "-> the worst (largest violation) is dropped")
pn = nocon(PROB['P6']); Xn_, Fn_, _, an = run_alg(NSGA2, pn, nfe=BUDGET); cvn = PROB['P6'].evaluate(Xn_)[1]; Xp_, Fp_, _, ap = run_alg(NSGA2, PROB['P6'], nfe=BUDGET); Xd_, Fd_, _, ad = run_alg(CNSGA2, PROB['P6'], nfe=BUDGET)
md_table(["algorithm", "% of final population feasible", "front points returned (feasible)"], [["NSGA-II ignoring constraints", 100 * (PROB['P6'].evaluate(an.X)[1] <= 0).mean(), int((cvn <= 0).sum())], ["NSGA-II + penalty", 100 * (ap.CV <= 0).mean(), len(Fp_)], ["C-NSGA-II (Deb)", 100 * (ad.CV <= 0).mean(), len(Fd_)]])
STUDY['p6_nocon_feas'] = 100 * (PROB['P6'].evaluate(an.X)[1] <= 0).mean()

# ---------- P7
md("### P7 — Water allocation: front on the line, and a decision maker with reference point (40, 50)")
Xp7, Fp7 = XFR[(21, 'P7')], FRONTS[(21, 'P7')]; res = np.abs(Xp7.sum(1) - 90).max(); nat = -Fp7; d = np.sqrt(((nat - np.array([40, 50])) ** 2).sum(1)); b = np.argmin(d)
print(f"C-NSGA-II front: max |x1+x2-90| = {res:.2e} (lies on the line); {len(Fp7)} points. Closest point to the DM's reference (40,50): x=({Xp7[b,0]:.1f},{Xp7[b,1]:.1f}) -> (yield, output)=({nat[b,0]:.1f},{nat[b,1]:.1f}), distance {d[b]:.1f}.")
print("Reference point lies above the line? 0.8*x1=40 & 1.2*x2=50 need x1+x2 = %.1f > 90 -> unattainable, so the preferred point is a compromise on the line." % (50 + 50 / 1.2))
STUDY['p7_resid'] = res

# ---------- P8
md("### P8 — Routing: permutation operators and the two single-objective optimal tours")
P8p = PROB['P8']; rows = []
for lab, cm, mu in (("OX + swap", 'ox', 'swap'), ("OX + inversion", 'ox', 'inversion'), ("OX + both", 'ox', 'both'), ("mutation only (no crossover)", 'none', 'both'), ("crossover only (no mutation)", 'ox', 'none')):
    hv = []
    for sd in range(5 if not QUICK else 2):
        a = NSGA2(P8p, N=40, seed=sd); a.var.perm_cross, a.var.perm_mut = cm, mu; X, F = a.run(1200); hv.append(score(P8p, F)['hv'])
    rows.append([lab, np.mean(hv), np.std(hv)])
md_table(["operators (NSGA-II, 1200 evals, 5 seeds)", "HV mean", "HV std"], rows); STUDY['p8_ops'] = rows
Fd_, Ft_ = P8p.evaluate(P8p.tour_d[None])[0][0], P8p.evaluate(P8p.tour_t[None])[0][0]
edges = lambda t: {frozenset((int(a), int(b))) for a, b in zip(t, np.roll(t, -1))}; sh = len(edges(P8p.tour_d) & edges(P8p.tour_t))
print(f"Shortest-distance tour: dist {Fd_[0]:.1f}, time {Fd_[1]:.1f}.  Shortest-time tour: dist {Ft_[0]:.1f}, time {Ft_[1]:.1f}.  They share {sh} of 15 edges. A nearest-neighbour tour from city 0 for comparison:")
def nn_tour(Cm):
    t = [0]; left = set(range(1, 15))
    while left: nx = min(left, key=lambda j: Cm[t[-1], j]); t.append(nx); left.remove(nx)
    return np.array(t)
nd_, nt_ = nn_tour(P8p.D), nn_tour(P8p.T); rnd = P8p.init(1, np.random.default_rng(0))
md_table(["tour", "distance", "time"], [["random", *P8p.evaluate(rnd)[0][0].round(1)], ["nearest neighbour (dist)", *P8p.evaluate(nd_[None])[0][0].round(1)], ["nearest neighbour (time)", *P8p.evaluate(nt_[None])[0][0].round(1)], ["2-opt best distance", *Fd_.round(1)], ["2-opt best time", *Ft_.round(1)]])
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
for a_, t, ttl in ((ax[0], P8p.tour_d, 'shortest-distance tour'), (ax[1], P8p.tour_t, 'shortest-time tour')):
    c = P8p.C[np.r_[t, t[0]]]; a_.plot(c[:, 0], c[:, 1], '-o', ms=4); a_.set_title(ttl, fontsize=9)
plt.show()

# ---------- P9
md("### P9 — Feature selection: smallest feature set within 1 percentage point of the all-features baseline")
p9 = PROB['P9']; base_acc = p9.acc(np.ones(30)); Xa, Fa, _, _ = run_alg(NSGA2, p9, nfe=P9N)
XX = np.vstack([Xa, XFR[(23, 'P9')], XFR[(30, 'P9')]]).astype(int); FF = np.vstack([Fa, FRONTS[(23, 'P9')], FRONTS[(30, 'P9')]]); nd = nds(FF)[0]; XX, FF = XX[nd], FF[nd]
ok = FF[:, 0] <= (1 - base_acc) + .01; best = np.argmin(np.where(ok, FF[:, 1], 99)); freq = XX.sum(0)
print(f"Baseline (all 30 features): accuracy {base_acc:.4f}. Smallest set within 1 pp: {int(FF[best,1])} features, accuracy {1-FF[best,0]:.4f}.")
print("Most frequent features over the", len(XX), "front solutions:", [(p9.feat[i], int(freq[i])) for i in np.argsort(-freq)[:6]]); STUDY['p9_k'] = int(FF[best, 1]); STUDY['p9_base'] = base_acc

# ---------- P10
md("### P10 — DTLZ2: fraction of a random population that is non-dominated")
rows = []
for M in (3, 5, 10):
    pr = PROB.get(f'P10_{M}') or P10(M); fr_ = [len(nds(pr.evaluate(pr.init(100, np.random.default_rng(s)))[0])[0]) / 100 for s in range(30)]; rows.append([M, np.mean(fr_)]); STUDY[f'nd_frac_{M}'] = np.mean(fr_)
md_table(["M", "mean non-dominated fraction of 100 random solutions"], rows)
print("With M=5 about half the random population is already 'rank 1', so Pareto ranking cannot discriminate and selection pressure vanishes; at M=10 almost everything is non-dominated.")

# ---------- P14
md("### P14 — Laptop: is x1 = 13 discovered, and where is the DM's point (1.8 kg, 12 h)?")
rows = []
for no in (32, 33, 35):
    X = XFR[(no, 'P14')]; rows.append([NAMES[no], len(X), float(np.abs(X[:, 0] - 13).max()), float((np.abs(X[:, 0] - 13) < .05).mean())])
md_table(["algorithm", "front pts", "max |x1-13|", "share with x1≈13"], rows); STUDY['p14_x1'] = rows
Zf = PROB['P14'].Z; k = np.argmin(np.abs(Zf[:, 0] - 1.8)); print(f"On the true front, weight 1.8 kg gives runtime {-Zf[k,1]:.2f} h (> 12 h): the DM point (1.8, 12) is feasible but DOMINATED by the front, so a preference method should converge to the front part near (1.8, {-Zf[k,1]:.1f}).")
Fr = FRONTS[(35, 'P14')]; print(f"R-NSGA-II front spans weights {Fr[:,0].min():.2f}–{Fr[:,0].max():.2f} kg and runtimes {-Fr[:,1].max():.1f}–{-Fr[:,1].min():.1f} h; points closest to the DM point: {np.round(Fr[np.argsort(np.abs(Fr[:,0]-1.8)+np.abs(Fr[:,1]+12)/10)[:3]],2).tolist()}")
Fi = FRONTS[(32, 'P14')]; print(f"For comparison IBEA (no preference) spans {Fi[:,0].min():.2f}–{Fi[:,0].max():.2f} kg; the preference run is {'narrower' if np.ptp(Fr[:,0])<np.ptp(Fi[:,0]) else 'not narrower'}.")
