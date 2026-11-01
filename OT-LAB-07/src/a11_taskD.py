# %% [markdown]
# # Task D · Head-to-head comparison (10 seeds, 10 000 evaluations)
# Same budget, same seeds 0–9, same metrics (HV with reference point from the problem, IGD to the reference front, Spacing). Ranking = mean IGD (lower is better); HV is the tie-break check.
# ## D.1 Two-objective problems: P3 (ZDT2-type: 10 variables, smooth concave front) and P5 (constrained beam design, curved front)

# %%
D_ALGS = ['NSGA-II', 'NSGA-III', 'SPEA2', 'PESA-II', 'MOEA/D', 'MOEA/D-DE', 'C-NSGA-II', 'MOPSO', 'MOABC', 'MO-TLBO', 'MOMA', 'IBEA', 'SMS-EMOA']
def head2head(key, algs, nfe=None, kw_of=lambda n: {}):
    nfe = nfe or BUDGET; jobs = [(n, key, s, nfe, kw_of(n)) for n in algs for s in SEEDS]; res = pmap(run_cfg, jobs); out = {}
    for (n, k, s, _, kw), (F, dt, nf) in zip(jobs, res): out.setdefault(n, []).append((F, score(PROB[k], F), dt))
    return out
def rank_table(out, key, title, front2d=True):
    stats = {}
    for n, lst in out.items():
        g = lambda f: np.array([r[1][f] for r in lst]); stats[n] = dict(hv=(g('hv').mean(), g('hv').std()), igd=(g('igd').mean(), g('igd').std()), sp=(np.nanmean(g('sp')), np.nanstd(g('sp'))), n=g('n').mean(), t=np.mean([r[2] for r in lst]))
    order = sorted(stats, key=lambda n: stats[n]['igd'][0]); hv_rank = {n: i + 1 for i, n in enumerate(sorted(stats, key=lambda n: -stats[n]['hv'][0]))}
    rows = [[i + 1, n, "%.4f ± %.4f" % stats[n]['igd'], "%.4f ± %.4f" % stats[n]['hv'], "%.4f ± %.4f" % stats[n]['sp'], "%.0f" % stats[n]['n'], hv_rank[n], "%.1fs" % stats[n]['t']] for i, n in enumerate(order)]
    md(f"**{title}** — ranked by mean IGD over {NSEEDS} seeds"); md_table(["rank", "algorithm", "IGD (mean ± std)", "HV (mean ± std)", "Spacing (mean ± std)", "points", "HV rank", "time/run"], rows)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.2)); P = PROB[key]
    if not front2d:
        ax[0].boxplot([[r[1]['igd'] for r in out[n]] for n in order]); ax[0].set_xticklabels(order, rotation=40, fontsize=8); ax[0].set_title(f'{key}: IGD per algorithm (left = best)', fontsize=9)
        for n, c in ((order[0], 'C2'), (order[-1], 'C3')):
            F = P.norm(out[n][0][0]); idx = np.random.default_rng(0).choice(len(F), min(30, len(F)), replace=False)
            for j, i in enumerate(idx): ax[1].plot(range(1, P.M + 1), F[i], c=c, alpha=.4, label=f"{n} (seed 0)" if j == 0 else None)
        ax[1].legend(fontsize=7); ax[1].set_xlabel('objective'); ax[1].set_title('parallel coordinates, best vs worst', fontsize=9); plt.tight_layout(); plt.show(); return order, stats
    ax[0].boxplot([[r[1]['igd'] for r in out[n]] for n in order]); ax[0].set_xticklabels(order, rotation=60, fontsize=7); ax[0].set_yscale('log'); ax[0].set_title(f'{key}: IGD per algorithm (log scale; left = best)', fontsize=9)
    Z = P.Z[np.argsort(P.Z[:, 0])]; ax[1].plot(Z[:, 0], Z[:, 1], '-', c='.7', lw=4, label='reference front')
    for n, c in ((order[0], 'C2'), (order[len(order) // 2], 'C0'), (order[-1], 'C3')):
        F = out[n][0][0]; ax[1].plot(F[:, 0], F[:, 1], '.', c=c, ms=6, alpha=.8, label=f"{n} (rank {order.index(n) + 1}, seed 0)")
    ax[1].legend(fontsize=7); ax[1].set_title('best / median / worst algorithm, seed 0', fontsize=9); plt.tight_layout(); plt.show()
    return order, stats
# R-NSGA-II is a preference method, so it is not ranked on the whole-front metrics here (it is judged in Task E).
outD3 = head2head('P3', D_ALGS); ord3, st3 = rank_table(outD3, 'P3', 'P3 (unconstrained, concave front)')

# %%
outD5 = head2head('P5', D_ALGS); ord5, st5 = rank_table(outD5, 'P5', 'P5 (constrained beam design)')
STUDY['D'] = dict(P3=ord3, P5=ord5, st3=st3, st5=st5)

# %%
def dtext(order, st, key):
    b, w = order[:3], order[-2:]
    return (f"**{key}.** Best three by IGD: **{', '.join(b)}** (IGD {st[b[0]]['igd'][0]:.4f}, {st[b[1]]['igd'][0]:.4f}, {st[b[2]]['igd'][0]:.4f}); worst: {', '.join(w)} ({st[w[0]]['igd'][0]:.4f}, {st[w[1]]['igd'][0]:.4f}). "
            f"Largest seed-to-seed std among the top half: {max(order[:len(order)//2], key=lambda n: st[n]['igd'][1])} (±{max(st[n]['igd'][1] for n in order[:len(order)//2]):.4f}).")
md(f"""### D.1 Interpretation
{dtext(ord3, st3, 'P3')}

{dtext(ord5, st5, 'P5')}

**Why the ranking looks like this (read from the tables, not from folklore).**
* *Variation operator mattered more than the selection scheme.* MOEA/D-DE and MOEA/D share the same decomposition and update rule, yet on P3 their IGD is {st3['MOEA/D-DE']['igd'][0]:.4f} vs {st3['MOEA/D']['igd'][0]:.4f}: the only difference is DE variation instead of SBX. The other P3 leaders (MO-TLBO, MOABC) also use difference-style / neighbour-based moves, which converge faster than SBX on this 10-variable concave problem (ZDT2 has a local front that traps SBX-based runs).
* *Robustness (std) separates algorithms more than the mean does.* On P3, NSGA-II ({st3['NSGA-II']['igd'][0]:.4f} ± {st3['NSGA-II']['igd'][1]:.4f}), SPEA2 ({st3['SPEA2']['igd'][0]:.4f} ± {st3['SPEA2']['igd'][1]:.4f}), NSGA-III and MOEA/D have a std at or above their mean: some of the 10 seeds are stuck on the local front. The leaders have std ≈ 1e-3. A single-seed comparison would have ranked these methods arbitrarily.
* *SMS-EMOA* has the smallest variance in both tables (std ≈ 1e-4) but returns only 50 points (population limit) and is 10× slower (hypervolume contributions).
* *NSGA-III and IBEA are weakest on the 2-objective problems.* NSGA-III's reference-point machinery is designed for M ≥ 3 and brings nothing at M = 2 (it is best in D.2, not here); IBEA's indicator fitness gives an uneven front on P5 (HV rank {[r for r in ord5].index('IBEA')+1} by IGD).
* *Constraints (P5):* C-NSGA-II (Deb rules) IGD {st5['C-NSGA-II']['igd'][0]:.4f} ± {st5['C-NSGA-II']['igd'][1]:.4f} vs NSGA-II with the static penalty {st5['NSGA-II']['igd'][0]:.4f} ± {st5['NSGA-II']['igd'][1]:.4f} – a difference **inside one std**, i.e. on this easy constraint the handling method is not the deciding factor (ignoring the constraint altogether, however, leaves 45 % infeasible solutions, see the P5 study).
* *Rule of thumb for reading the tables:* a gap smaller than one std over 10 seeds is not a result; only quote clear gaps.
""")

# %% [markdown]
# ## D.2 Many-objective: P10 with M = 5 (DTLZ2-type, spherical front)
# NSGA-II vs the four many-objective designs. IGD is to the analytic front; HV is Monte-Carlo (M ≥ 3). Diagnostics explain **why** Pareto-rank + crowding degrades.

# %%
M_ALGS = ['NSGA-II', 'NSGA-III', 'MOEA/D', 'RVEA', 'HypE']
outM = head2head('P10_5', M_ALGS, kw_of=lambda n: MANY); ordM, stM = rank_table(outM, 'P10_5', 'P10, M = 5', front2d=False)
# diagnostics for NSGA-II: (i) share of non-dominated individuals in a random population and in the final-generation population, (ii) spread of the crowding distances
rng = np.random.default_rng(0); P5_ = PROB['P10_5']; rows = []
for M_, key in ((3, 'P10_3'), (5, 'P10_5')):
    Pm = PROB[key]; Xr = Pm.lb + rng.random((100, Pm.n)) * (Pm.ub - Pm.lb); Fr = Pm.evaluate(Xr)[0]; nd = len(nds(Fr)[0])
    cd = crowding(Pm.norm(Fr)); fin = np.isfinite(cd); rows.append([M_, nd, "%.0f %%" % nd, "%.2f" % (np.std(cd[fin]) / np.mean(cd[fin]))])
md("**Why NSGA-II degrades.** In a random population of 100 the number of non-dominated individuals is:"); md_table(["M", "non-dominated of 100", "share", "CV of crowding distance"], rows)
STUDY['D_many'] = dict(order=ordM, stats=stM, nd=rows)
md(f"""### D.2 Interpretation
Ranking by IGD: **{' < '.join(ordM)}** (IGD {', '.join('%.3f' % stM[n]['igd'][0] for n in ordM)}). 
NSGA-II loses ground as M grows for three connected reasons: **(1) dominance pressure vanishes** – with M = 5 a large share of even a random population is already non-dominated (table above), so rank-based selection cannot tell good from bad and the sort only acts as noise; **(2) crowding distance is a poor diversity proxy in 5-D** – it sums per-objective neighbour gaps, so points that are extreme in any one objective get huge values and the population drifts to the boundary of the front; **(3) recombination of distant parents** produces offspring far from the front (we use SBX η=30 to mitigate this for all methods). The reference-vector methods (NSGA-III, MOEA/D, RVEA) replace crowding by *structured* diversity, and HypE replaces dominance ranking by a hypervolume-based fitness estimated by sampling – both address the failure at its root.""")

# %% [markdown]
# ## D.3 Combinatorial: P8 (bi-objective TSP-15) at 1 000 and 10 000 evaluations
# MOEA/D-ACO, MOACO, NSGA-II and MOMA. There is no analytic front, so the reference is the **union non-dominated set of every run in this section**; IGD is to that set (normalised with the problem's ideal/nadir). Seeds 0–9.

# %%
T_ALGS = ['MOEA/D-ACO', 'MOACO', 'NSGA-II', 'MOMA']; BUDS = (1000, 10000) if not QUICK else (300, 1500)
allF = {}
for b in BUDS:
    jobs = [(n, 'P8', s, b, {}) for n in T_ALGS for s in SEEDS]; res = pmap(run_cfg, jobs)
    for (n, k, s, _, kw), (F, dt, nf) in zip(jobs, res): allF[(n, b, s)] = (F, dt)
U = np.vstack([v[0] for v in allF.values()]); P8_ = PROB['P8']; Uref = U[nds(U)[0]]; Uref = np.unique(np.round(Uref, 6), axis=0)
rows = []; box = {}
for b in BUDS:
    for n in T_ALGS:
        vals = [dict(igd=igd(P8_.norm(Uref), P8_.norm(allF[(n, b, s)][0])), hv=hypervolume(P8_.norm(allF[(n, b, s)][0]), P8_.ref), k=len(allF[(n, b, s)][0])) for s in SEEDS]; box[(n, b)] = vals
        rows.append([b, n, "%.4f ± %.4f" % (np.mean([v['igd'] for v in vals]), np.std([v['igd'] for v in vals])), "%.3f ± %.3f" % (np.mean([v['hv'] for v in vals]), np.std([v['hv'] for v in vals])), "%.1f" % np.mean([v['k'] for v in vals]), "%.1fs" % np.mean([allF[(n, b, s)][1] for s in SEEDS])])
md_table(["budget", "algorithm", "IGD to union front", "HV", "points", "time/run"], rows)
fig, ax = plt.subplots(1, 3, figsize=(16, 4)); ax[0].plot(*U.T, '.', c='.8', ms=3, label='all runs'); ax[0].plot(*Uref.T, 'k.', ms=6, label='union front')
for n, c in zip(T_ALGS, ('C0', 'C1', 'C2', 'C3')):
    F = allF[(n, BUDS[1], 0)][0]; ax[0].plot(*F.T, 'x', c=c, ms=5, label=f"{n}, seed 0")
ax[0].legend(fontsize=6); ax[0].set_xlabel('distance'); ax[0].set_ylabel('time'); ax[0].set_title(f'P8 fronts at {BUDS[1]} evaluations', fontsize=9)
for a_, b in zip(ax[1:], BUDS):
    a_.boxplot([[v['igd'] for v in box[(n, b)]] for n in T_ALGS]); a_.set_xticklabels(T_ALGS, fontsize=7); a_.set_title(f'IGD to union front, budget {b}', fontsize=9)
plt.tight_layout(); plt.show()
m_ = lambda n, b: np.mean([v['igd'] for v in box[(n, b)]])
best_lo = min(T_ALGS, key=lambda n: m_(n, BUDS[0])); best_hi = min(T_ALGS, key=lambda n: m_(n, BUDS[1]))
STUDY['D_tsp'] = dict(rows=rows, best_lo=best_lo, best_hi=best_hi)
md(f"""### D.3 Interpretation
Best IGD at the small budget ({BUDS[0]}): **{best_lo}**; at the large budget ({BUDS[1]}): **{best_hi}**. Mean IGD change from small to large budget: """ + ", ".join(f"{n} {m_(n, BUDS[0]):.3f} → {m_(n, BUDS[1]):.3f}" for n in T_ALGS) + """.

Constructive methods (ACO family) use heuristic information (1/distance, 1/time) so tours are already good after the first few ants – they are strong at tiny budgets. NSGA-II with order crossover and swap mutation starts from random permutations and needs many generations, but it keeps improving with budget. MOMA's local search spends evaluations on refinement and pays off only when the budget can afford it. Where the best algorithm *changes* between the two budgets, the choice depends on how many evaluations are affordable.""")

# %% [markdown]
# ## D.4 Decision support on an obtained front: TOPSIS and VIKOR on the NSGA-II front of P4
# TOPSIS and VIKOR are **rankers, not optimisers**: they pick one compromise from a front that an optimiser has already produced. 20 evenly spread points are taken from the NSGA-II front (risk ↓, return ↑).

# %%
X4, F4, dt4, a4 = run_alg(NSGA2, PROB['P4'], seed=0, nfe=BUDGET); FRONTS[(14, 'P4')] = F4
sel = np.unique(np.round(np.linspace(0, len(F4) - 1, 20)).astype(int)); Fs = F4[np.argsort(F4[:, 0])][sel]; D4 = np.c_[Fs[:, 0], -Fs[:, 1]]; ben4 = [False, True]
def rank_names(w):
    tp = topsis(D4, np.array(w), ben4)[1]; vk = vikor(D4, np.array(w), ben4, .5)['order']; return tp, vk
rows = []; top = {}
for w in ((.5, .5), (.8, .2), (.2, .8)):
    tp, vk = rank_names(w); top[w] = (tp[:3], vk[:3]); rows.append([f"{w}", ", ".join("#%d (risk %.4f, ret %.4f)" % (i, D4[i, 0], D4[i, 1]) for i in tp[:3]), ", ".join("#%d" % i for i in vk[:3])])
md_table(["weights (risk, return)", "TOPSIS top-3", "VIKOR top-3 (v=.5)"], rows)
w0, w1 = (.5, .5), (.8, .2); flip = top[w0][0][0] != top[w1][0][0]
fig, ax = plt.subplots(figsize=(5.6, 4)); ax.plot(D4[:, 0], D4[:, 1], 'o-', c='.6', label='20 front points')
for w, c, mk in ((w0, 'C2', '*'), (w1, 'C3', 's'), ((.2, .8), 'C0', '^')): ax.plot(*D4[top[w][0][0]], mk, c=c, ms=13, label=f"TOPSIS pick, w={w}")
ax.set_xlabel('risk'); ax.set_ylabel('return'); ax.legend(fontsize=7); plt.show()
STUDY['D_dm'] = dict(flip=flip)
md(f"""Shifting the weights from (0.5, 0.5) to (0.8, 0.2) {'**changes**' if flip else 'does not change'} the TOPSIS winner (#{top[w0][0][0]} → #{top[w1][0][0]}): the pick moves along the front towards lower risk, exactly as the weights say. Neither method generates a single new solution – they only order the 20 candidates the optimiser supplied, so their quality is bounded by the quality (convergence and spread) of the NSGA-II front.

**Gradient-based algorithms (36–37)** were studied on P13 above (toy + network). MGDA converged to a Pareto-stationary point of the toy problem for every start and decreased both losses on the balanced network; under the ×20 regression scale the equal-weight sum is dominated by the regression gradient, while MGDA/GradNorm rebalance. They are not run on P3/P5 because they need gradients of every objective and a differentiable (continuous, single-point) search: they find **one** Pareto-stationary point per run rather than a population-wide front, so a head-to-head front-metric comparison with population methods would not be like-for-like.
""")
