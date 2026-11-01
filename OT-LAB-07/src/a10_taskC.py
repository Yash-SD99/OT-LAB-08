# %% [markdown]
# # Task C — Deep dive on my three algorithms
# Roll number **241070018 → R = 18**:  Algorithm 1 = ((18−1) mod 37)+1 = **18 (MOEA/D)**, Algorithm 2 = ((18+11) mod 37)+1 = **30 (MOMA)**, Algorithm 3 = ((18+23) mod 37)+1 = **5 (Value Function)**.
#
# ---
# ## C.1 Study cards
# ### Algorithm 18 — MOEA/D (Multiobjective Evolutionary Algorithm based on Decomposition)
# * **Year / authors / problem fixed.** Q. Zhang & H. Li, *IEEE Trans. Evolutionary Computation* 11(6), 2007. Pareto-ranking methods (NSGA-II, SPEA2) spend O(MN²) per generation and need a separate diversity step; MOEA/D replaces both by decomposing the problem into N scalar sub-problems that are solved *cooperatively* with neighbouring sub-problems.
# * **Core idea (≤ 5 lines).** (1) Choose N weight vectors λⁱ; sub-problem *i* is min g(x|λⁱ,z*) = max_k λᵢₖ|f_k(x) − z*_k|. (2) Each sub-problem keeps ONE solution. (3) A child is bred from solutions of neighbouring sub-problems (the T closest weights). (4) A child that is better than a neighbour's solution *for that neighbour's weight* replaces it (at most nr times). (5) The N surviving solutions form the front.
#
# ```
#  weights λ¹..λᴺ ─► neighbourhoods B(i) ─► init x¹..xᴺ, ideal z*
#        ┌──────────────────────────────────────────────────────┐
#        ▼                                                      │
#  for each i: pick 2 parents from B(i) ─► crossover+mutation ─► y ─► evaluate
#        ▼                                                      │
#  z* = min(z*, f(y)) ─► for j in B(i): if g(y|λʲ) ≤ g(xʲ|λʲ): xʲ = y (≤ nr times) ─┘
#        ▼ (budget spent)
#  output the N solutions (non-dominated, feasible)
# ```
# * **Numbered steps and formulas.** 1) λⁱ for i=1..N (M=2: (i/(N−1), 1−i/(N−1))); B(i) = indices of the T nearest λ. 2) x_i random, F_i = f(x_i), z*_k = min_i f_k(x_i). 3) For each i (random order): with prob δ take P = B(i) else P = all; draw two indices from P; y = SBX + polynomial mutation. 4) z*_k ← min(z*_k, f_k(y)). 5) For j ∈ P (random order): if g(y|λʲ,z*) ≤ g(xʲ|λʲ,z*) then xʲ ← y, count++; stop after nr replacements. In the code g uses objectives normalised by the current ideal z* and nadir estimate z_n = max f over the population: g = max_k λ_k |f_k − z*_k|/(z_n,k − z*_k).
# * **Parameters (too small / too large).** N: sparse front / wasted evaluations. **T**: too small → neighbourhoods barely overlap, search is local and copies spread slowly (diversity ok, convergence slow); too large → every child competes everywhere, neighbours lose individuality and the front collapses onto few solutions (see sensitivity study). **nr**: too small → slow progress; too large → one good child overwrites a whole neighbourhood (duplicates). **δ**: small → mostly global mating, large → local mating. SBX η_c, mutation rate as usual. Scalarisation: weighted sum (convex only), **Tchebycheff** (used), PBI.
# * **How it handles.** *Convergence*: each sub-problem is a scalar problem, children replace several neighbours so good genes spread fast. *Diversity*: given by the weight vectors (no density estimator). *Constraints*: penalty added to the objectives before g (or Deb rules inside g). *Non-convex fronts*: yes with Tchebycheff. *> 3 objectives*: works (no dominance), but N (= number of weights) grows combinatorially, C(p+M−1, M−1).
# * **Complexity.** O(M·T) per child (g evaluated for ≤ T neighbours) → **O(M·N·T) per generation** vs O(M·N²) for NSGA-II.
#
# ### Algorithm 30 — MOMA (Multiobjective Memetic Algorithm = NSGA-II + local hill-climbing)
# * **Year / authors / problem fixed.** Memetic multiobjective GAs go back to Ishibuchi & Murata (MOGLS, *IEEE Trans. SMC-C* 28(3), 1998) and Jaszkiewicz (2002); the version asked in the lecture combines **NSGA-II (Deb et al., 2002)** with hill climbing. (I could not find a single canonical paper for the name "MOMA"; the exact slide reference should be added by the student.) Problem fixed: population-based global search is slow to polish solutions near the front; adding local search speeds up convergence at a small evaluation cost.
# * **Core idea.** Run NSGA-II; before survival, improve a share p_ls of the offspring by L hill-climbing steps on a random-weight Tchebycheff scalarisation (accept a neighbour only if the scalar value improves) and write the improved solution back (Lamarckian); then NSGA-II survival keeps the best N.
#
# ```
#  P_t ─► tournament (rank, crowding) ─► SBX+PM ─► Q_t ─► evaluate
#   ▲                                              │
#   │                       pick p_ls of Q_t ─► random weight w ─► L × hill-climb step (accept if Tcheb(w) improves)
#   │                                              ▼
#   └── fronts + crowding ◄── R_t = P_t ∪ Q_t' ◄──┘
# ```
# * **Steps / formulas.** 1–4 as NSGA-II. 5) for each chosen offspring draw w ~ Dirichlet(1) and repeat L times: y' = neighbour(y) (Gaussian 5 % of range / swap / bit flip), accept if s(y'|w) < s(y|w) with s(f|w) = max_k w_k·f̂_k (f̂ normalised). 6) R = P ∪ Q'; sort into fronts; fill N slots; truncate last front by crowding distance.
# * **Parameters.** p_ls: 0 → plain NSGA-II; too large → evaluations are spent on local search, diversity can drop (all offspring pulled to the same scalar optimum). L: too small → no effect; too large → cost. Neighbour step: too small → stuck, too large → random search. Plus NSGA-II parameters.
# * **How it handles.** *Convergence*: faster than NSGA-II thanks to local search. *Diversity*: crowding distance plus random weights spread the polishing. *Constraints*: penalty (or Deb's rules) in the scalar value. *Non-convex*: Tchebycheff local search works. *> 3 objectives*: inherits NSGA-II's loss of selection pressure.
# * **Complexity.** NSGA-II O(M·N²) per generation + O(p_ls·N·L) extra evaluations.
#
# ### Algorithm 5 — Value Function (utility optimisation)
# * **Year / authors / problem fixed.** Utility theory: von Neumann & Morgenstern (1944); multi-attribute value/utility functions: Keeney & Raiffa, *Decisions with Multiple Objectives* (1976). Problem fixed: a decision maker (DM) needs ONE answer, not a front – encode preferences as a function U(f) and solve a single-objective problem.
# * **Core idea.** Ask the DM for U(f₁,…,f_M); maximise U(f(x)) s.t. the constraints; the optimum (for monotone U) is a Pareto-optimal solution.
#
# ```
#  DM preferences ─► utility U(f) ─► max_x U(f(x)) s.t. constraints ─► ONE preferred solution
# ```
# * **Steps.** 1) normalise f̂_k ∈ [0,1] (0 = best). 2) choose U, here Cobb–Douglas **U = (1−f̂₁)^a (1−f̂₂)^(1−a)** (a = importance of objective 1) or a target-distance utility. 3) solve max U with a constrained scalar optimiser (own DE + pattern search). 4) report x*, f(x*), U*.
# * **Parameters.** a (or the target): decides which part of the front is returned; too extreme (a→0 or 1) → an end point. Solver budget: too small → inaccurate optimum.
# * **How it handles.** *Convergence*: one scalar problem → fast. *Diversity*: none (single point). *Constraints*: inside the solver. *Non-convex fronts*: yes (U is non-linear). *> 3 objectives*: trivial in dimension but the DM must define U. **Risk:** a non-monotone U (e.g. distance to a target) can return a *dominated* design (shown on P14).
# * **Complexity.** One scalar solve: O(pop·gens) evaluations (here 40 × 200 + polish ≈ 8,000).
#
# ---
# ## C.2 Hand traces (two full iterations on a tiny instance) — and the code reproduces every number
# Each trace is computed below with a **plain-arithmetic script** (no algorithm class). The class is then run on the same instance with the same scripted offspring and the numbers are compared (assertions).

# %% [markdown]
# ### Trace of Algorithm 18 — MOEA/D on P1 (f₁=(x−3)², f₂=(x−7)²), N = 4 sub-problems, T = 2, nr = 2, δ = 1
# Weights λ = (0,1), (⅓,⅔), (⅔,⅓), (1,0) (a weight of 0 is replaced by 10⁻⁶), initial x = (1, 4, 6, 9). Offspring are *given* (as if produced by SBX): iteration 1 y = (3.0, 4.5, 5.5, 7.5); iteration 2 y = (3.5, 5.0, 6.5, 8.0).

# %%
def moead_hand_trace(x0, kids, T=2, nr=2):
    f = lambda x: ((x - 3) ** 2, (x - 7) ** 2)
    W = [(max(i / 3, 1e-6), max(1 - i / 3, 1e-6)) for i in range(4)]
    dist = lambda a, b: ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** .5
    B = [sorted(range(4), key=lambda j: dist(W[i], W[j]))[:T] for i in range(4)]       # stable sort: ties -> lower index
    x = list(x0); F = [f(v) for v in x]; z = [min(F[i][0] for i in range(4)), min(F[i][1] for i in range(4))]; zn = [max(F[i][0] for i in range(4)), max(F[i][1] for i in range(4))]
    g = lambda fv, w: max(w[0] * abs(fv[0] - z[0]) / max(zn[0] - z[0], 1e-12), w[1] * abs(fv[1] - z[1]) / max(zn[1] - z[1], 1e-12))
    log_ = [f"neighbourhoods B = {B}", f"initial x = {x}, F = {[tuple(round(a, 3) for a in v) for v in F]}, z* = {z}, z_n = {zn}"]; hist = []
    for it, ys in enumerate(kids, 1):
        for i in range(4):
            y = ys[i]; fy = f(y); z = [min(z[0], fy[0]), min(z[1], fy[1])]; cnt = 0; rows = []
            for j in B[i]:
                gy, gx = g(fy, W[j]), g(F[j], W[j]); rep = gy <= gx and cnt < nr
                rows.append(f"   j={j}: g(y|λ{j})={gy:.4f}  vs  g(x{j}|λ{j})={gx:.4f}  → {'REPLACE' if rep else 'keep'}")
                if rep: x[j], F[j] = y, fy; cnt += 1
            log_.append(f"iteration {it}, sub-problem {i}: y={y}  F(y)=({fy[0]:.3f},{fy[1]:.3f})  z*={tuple(round(a, 3) for a in z)}\n" + "\n".join(rows))
        zn = [max(F[i][0] for i in range(4)), max(F[i][1] for i in range(4))]; hist.append((list(x), [tuple(v) for v in F], list(z), list(zn)))
        log_.append(f"end of iteration {it}: x = {x}, F = {[tuple(round(a, 3) for a in v) for v in F]}, z* = {tuple(round(a, 3) for a in z)}, z_n = {tuple(round(a, 3) for a in zn)}")
    return "\n".join(log_), hist
KIDS = [[3.0, 4.5, 5.5, 7.5], [3.5, 5.0, 6.5, 8.0]]
txt, hist = moead_hand_trace([1., 4., 6., 9.], KIDS); print(txt)

# %%
# --- the algorithm class on the same instance with the same scripted offspring
a = MOEAD(PROB['P1'], N=4, T=2, nr=2, delta=1.0, seed=0); a.max_nfe = 10 ** 9
a.X = np.array([[1.], [4.], [6.], [9.]]); a.F, a.CV, a.Fp = a.evaluate(a.X); a.z = a.Fp.min(0); a.zn = a.Fp.max(0)
for it, ys in enumerate(KIDS):
    for i in range(4): a.update(i, np.array([ys[i]]), a.B[i])
    a.zn = a.Fp.max(0)
    assert np.allclose(a.X[:, 0], hist[it][0]) and np.allclose(a.F, np.array(hist[it][1])) and np.allclose(a.z, hist[it][2]) and np.allclose(a.zn, hist[it][3])
print("MOEA/D class reproduces the hand trace after iteration 1 and 2: x =", a.X[:, 0].tolist(), "| z* =", a.z.tolist(), "| MATCH ✔")

# %% [markdown]
# ### Trace of Algorithm 30 — MOMA on P1, population 4
# Parents x = (1, 4, 6, 9). Offspring (given): iteration 1 Q = (2.0, 5.5, 7.5, 3.5); local search (L = 1, w = (½,½)) on offspring #0 with step +0.5 and on offspring #2 with step −0.5. Iteration 2 Q = (4.5, 2.5, 6.0, 5.0); local search on #1 (step +0.9, w=(0.2,0.8)) and #3 (step −0.5, w=(0.2,0.8)). Objectives are normalised by 16 for the Tchebycheff value s(f|w)=max_k w_k f_k/16.

# %%
def dom_(a, b): return all(u <= v for u, v in zip(a, b)) and any(u < v for u, v in zip(a, b))
def crowd_(pts):
    n = len(pts); d = [0.] * n
    if n <= 2: return [float('inf')] * n
    for k in range(2):
        o = sorted(range(n), key=lambda i: pts[i][k]); rg = pts[o[-1]][k] - pts[o[0]][k]; d[o[0]] = d[o[-1]] = float('inf')
        for r in range(1, n - 1):
            if rg > 0: d[o[r]] += (pts[o[r + 1]][k] - pts[o[r - 1]][k]) / rg
    return d
def moma_hand_trace(x0, iters):
    f = lambda x: ((x - 3) ** 2, (x - 7) ** 2); s_ = lambda fv, w: max(w[0] * fv[0] / 16, w[1] * fv[1] / 16); P = list(x0); out = []; hist = []
    for it, (Q, ls) in enumerate(iters, 1):
        out.append(f"=== iteration {it}: parents {P}, offspring Q = {Q}"); Q = list(Q)
        for idx, step, w in ls:
            fq = f(Q[idx]); y = min(10, max(1, Q[idx] + step)); fy = f(y); acc = s_(fy, w) < s_(fq, w)
            out.append(f"  local search on Q[{idx}]={Q[idx]}: s={s_(fq, w):.4f}; neighbour y={y}: F=({fy[0]:.3f},{fy[1]:.3f}) s={s_(fy, w):.4f} → {'ACCEPT' if acc else 'reject'}")
            if acc: Q[idx] = y
        R = P + Q; FR = [f(v) for v in R]; rank = [0] * 8; left = set(range(8)); fr_ = []
        while left:
            cur = [i for i in left if not any(dom_(FR[j], FR[i]) for j in left if j != i)]; fr_.append(sorted(cur)); left -= set(cur)
        sel = []; cd_all = {}
        for r, fr in enumerate(fr_):
            cd = crowd_([FR[i] for i in fr]); cd_all.update({i: c for i, c in zip(fr, cd)})
            for i in fr: rank[i] = r + 1
        for r, fr in enumerate(fr_):
            if len(sel) + len(fr) <= 4: sel += fr
            else:
                cd = [cd_all[i] for i in fr]; order = sorted(range(len(fr)), key=lambda k: -cd[k]); sel += [fr[k] for k in order[:4 - len(sel)]]; break
        out.append("  union R (x, F, rank, crowding): " + "; ".join(f"{R[i]:g}:({FR[i][0]:.2f},{FR[i][1]:.2f}) rank {rank[i]} cd={'inf' if cd_all.get(i, 0) == float('inf') else round(cd_all.get(i, 0), 3)}" for i in range(8)))
        P = [R[i] for i in sel]; out.append(f"  selected (indices {sel}) → new parents {P}"); hist.append(list(P))
    return "\n".join(out), hist
ITERS = [([2.0, 5.5, 7.5, 3.5], [(0, +.5, (.5, .5)), (2, -.5, (.5, .5))]), ([4.5, 2.5, 6.0, 5.0], [(1, +.9, (.2, .8)), (3, -.5, (.2, .8))])]
txt, hist = moma_hand_trace([1., 4., 6., 9.], ITERS); print(txt)

# %%
m = MOMA(PROB['P1'], N=4, seed=0); m.max_nfe = 10 ** 9; m.X = np.array([[1.], [4.], [6.], [9.]]); m.F, m.CV, m.Fp = m.evaluate(m.X); m.rank_crowd()
for it, (Q, ls) in enumerate(ITERS):
    Qa = np.array(Q, float)[:, None]; QF, QCV, QFp = m.evaluate(Qa)
    for idx, step, w in ls:
        m.var.neighbor = lambda X, scale=None, st=step: np.clip(X + st, 1, 10)
        r_ = slice(idx, idx + 1); Qa[r_], QF[r_], QCV[r_], QFp[r_] = m.local_search(Qa[r_], QF[r_], QCV[r_], QFp[r_], np.array([w]), 1)
    m.merge(Qa, QF, QCV, QFp); assert sorted(m.X[:, 0].tolist()) == sorted(hist[it]), (m.X[:, 0], hist[it])
print("MOMA class reproduces the hand trace after iteration 1 and 2: parents =", sorted(m.X[:, 0].tolist()), "MATCH ✔")

# %% [markdown]
# ### Trace of Algorithm 5 — Value function on P1, two utility settings (a = 0.5 and a = 0.8)
# U(x) = (1 − f₁/16)^a (1 − f₂/16)^(1−a). By hand, tabulate U on x = 3, 3.5, …, 7 (the Pareto set) and pick the best; then refine with the derivative condition d ln U/dx = 0.

# %%
xs = np.arange(3, 7.01, .5); f1 = (xs - 3) ** 2; f2 = (xs - 7) ** 2; best = {}
for a_ in (.5, .8):
    Uv = (1 - f1 / 16) ** a_ * (1 - f2 / 16) ** (1 - a_); best[a_] = xs[np.argmax(Uv)]
    md(f"**a = {a_}**"); md_table(["x", "f₁", "f₂", "1−f₁/16", "1−f₂/16", "U"], [[x, p, q, 1 - p / 16, 1 - q / 16, u] for x, p, q, u in zip(xs, f1, f2, Uv)])
    print(f"a={a_}: best grid point x = {best[a_]}  (U = {Uv.max():.4f})")
for a_ in (.5, .8):
    x, F, u = value_function(PROB['P1'], cobb_douglas(a_)); dl = lambda t: a_ * (-2 * (t - 3) / 16) / (1 - (t - 3) ** 2 / 16) + (1 - a_) * (-2 * (t - 7) / 16) / (1 - (t - 7) ** 2 / 16)
    print(f"a={a_}: code solution x* = {x[0]:.4f}, F = ({F[0]:.3f},{F[1]:.3f}), U* = {u:.5f}, d lnU/dx at x* = {dl(x[0]):.2e}  (grid best {best[a_]})")
    assert abs(x[0] - best[a_]) <= .5 and abs(dl(x[0])) < 1e-3 and u >= (((1 - f1 / 16) ** a_) * ((1 - f2 / 16) ** (1 - a_))).max() - 1e-9
print("Value-function code reproduces the hand-computed optimum for both settings ✔")

# %% [markdown]
# ## C.3 Sensitivity study (most important parameter, 3 values, 10 seeds each)
# * MOEA/D: neighbourhood size **T ∈ {5, 20, 50}** on P3.   * MOMA: local-search probability **p_ls ∈ {0, 0.2, 0.6}** on P5 (p_ls = 0 is plain NSGA-II).   * Value function: importance **a ∈ {0.2, 0.5, 0.8}** on P4 (10 solver seeds).

# %%
ALGS = {c.name: c for c in (VEGA, MOGA, NSGA, NPGA, NSGA2, NSGA3, SPEA2, PESA2, MOEAD, MOEADDE, MOEADACO, CNSGA2, MOPSO, MOGWO, MOABC, MOACO, MOSA, MODE, MOTLBO, MOBA, MOMA, RVEA, IBEA, SMSEMOA, HypE, RNSGA2)}
def run_cfg(name, key, seed, nfe, kw):
    t = time.time(); a = ALGS[name](PROB[key], seed=seed, **kw); X, F = a.run(nfe); return F, time.time() - t, a.nfe
def study(name, key, param, values, fixed={}, nfe=None):
    nfe = nfe or BUDGET; jobs = [(name, key, s, nfe, {**fixed, param: v}) for v in values for s in SEEDS]; res = pmap(run_cfg, jobs); out = {}
    for (n, k, s, _, kw), (F, dt, nf) in zip(jobs, res): out.setdefault(kw[param], []).append((F, score(PROB[key], F), dt))
    return out
def report(out, key, param, title):
    rows = []; fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for v, lst in out.items():
        m_ = lambda f: (np.nanmean([r[1][f] for r in lst]), np.nanstd([r[1][f] for r in lst])); rows.append([f"{param} = {v}", "%.3f ± %.3f" % m_('hv'), "%.4f ± %.4f" % m_('igd'), "%.4f ± %.4f" % m_('sp'), "%.1f ± %.1f" % m_('n'), "%.1fs" % np.mean([r[2] for r in lst])])
        F = lst[0][0]; Z = PROB[key].Z[np.argsort(PROB[key].Z[:, 0])]; ax[0].plot(F[np.argsort(F[:, 0]), 0], F[np.argsort(F[:, 0]), 1], '.', ms=5, label=f"{param}={v}")
    ax[0].plot(Z[:, 0], Z[:, 1], '-', c='.6', lw=3, alpha=.5, zorder=0); ax[0].legend(fontsize=7); ax[0].set_title(f"{title}: seed-0 fronts", fontsize=9)
    ax[1].boxplot([[r[1]['igd'] for r in lst] for lst in out.values()]); ax[1].set_xticklabels([str(v) for v in out]); ax[1].set_title('IGD over %d seeds (lower = better)' % NSEEDS, fontsize=9); ax[1].set_xlabel(param)
    ax[2].boxplot([[r[1]['hv'] for r in lst] for lst in out.values()]); ax[2].set_xticklabels([str(v) for v in out]); ax[2].set_title('HV over %d seeds (higher = better)' % NSEEDS, fontsize=9); ax[2].set_xlabel(param)
    plt.tight_layout(); plt.show(); md_table([param, "HV", "IGD", "Spacing", "points", "time/run"], rows); return rows
md("**MOEA/D — neighbourhood size T on P3**"); outT = study('MOEA/D', 'P3', 'T', [5, 20, 50]); rowsT = report(outT, 'P3', 'T', 'MOEA/D')

# %%
md("**MOMA — local-search probability p_ls on P5** (p_ls = 0 ≡ NSGA-II)"); outL = study('MOMA', 'P5', 'p_ls', [0., .2, .6]); rowsL = report(outL, 'P5', 'p_ls', 'MOMA')

# %%
md("**Value function — importance a of objective 1 (risk) on P4**"); rows = []; fig, ax = plt.subplots(figsize=(5.5, 3.8)); Z = PROB['P4'].Z[np.argsort(PROB['P4'].Z[:, 0])]; ax.plot(Z[:, 0], -Z[:, 1], '-', c='.6', lw=3, alpha=.5, label='reference front')
for a_ in (.2, .5, .8):
    sol = [value_function(PROB['P4'], cobb_douglas(a_), seed=s)[1] for s in SEEDS]; sol = np.array(sol); ax.plot(sol[:, 0], -sol[:, 1], 'o', label=f"a = {a_}")
    rows.append([a_, "%.5f ± %.5f" % (sol[:, 0].mean(), sol[:, 0].std()), "%.5f ± %.5f" % (-sol[:, 1].mean(), sol[:, 1].std())])
ax.set_xlabel('risk'); ax.set_ylabel('return'); ax.legend(); ax.set_title('Value function: 10 solver seeds per a', fontsize=9); plt.show(); md_table(["a", "risk (mean ± std)", "return (mean ± std)"], rows)
STUDY['sens'] = dict(T=rowsT, L=rowsL, VF=rows)
_trap = lambda lst: sum(r[1]['igd'] > .05 for r in lst); _m = lambda lst, f: np.mean([r[1][f] for r in lst]); _s = lambda lst, f: np.std([r[1][f] for r in lst])
md(f"""**Reading the plots (numbers are from the tables above).**
* *MOEA/D (P3, 10 seeds)*: mean IGD is {_m(outT[5], 'igd'):.4f} for T=5, {_m(outT[20], 'igd'):.4f} for T=20 and {_m(outT[50], 'igd'):.4f} for T=50, but the seed-to-seed std ({_s(outT[5], 'igd'):.3f}, {_s(outT[20], 'igd'):.3f}, {_s(outT[50], 'igd'):.3f}) is as large as or larger than the means: the number of seeds that end far from the front (IGD > 0.05) is **{_trap(outT[5])}, {_trap(outT[20])}, {_trap(outT[50])} of 10** for T = 5, 20, 50. So on P3 the dominant effect is whether a run gets trapped on a local front, not T itself; the differences between the three T values are **not statistically meaningful** with 10 seeds. What T does change reliably is cost: {np.mean([r[2] for r in outT[5]]):.1f}s → {np.mean([r[2] for r in outT[20]]):.1f}s → {np.mean([r[2] for r in outT[50]]):.1f}s per run (more neighbours = more replacement checks), and the textbook trade-off (small T = local, slow information flow; large T = fast mixing, risk of neighbours overwriting each other) is only weakly visible here.
* *MOMA (P5)*: IGD {_m(outL[0.], 'igd'):.4f} (p_ls=0, i.e. NSGA-II), {_m(outL[.2], 'igd'):.4f} (p_ls=.2), {_m(outL[.6], 'igd'):.4f} (p_ls=.6) with std ≈ {_s(outL[.2], 'igd'):.4f}: differences are within one std, so local search gives **no reliable gain on this smooth 2-variable problem** (NSGA-II already converges). Higher p_ls spends evaluations on local search (5 per selected child), hence fewer generations and shorter run time ({np.mean([r[2] for r in outL[.6]]):.1f}s vs {np.mean([r[2] for r in outL[0.]]):.1f}s).
* *Value function*: a only changes *where* on the front the single answer lies (a=0.2 → high return/high risk, a=0.8 → low risk); the solver seed has essentially no effect (std below 1e-5, see table).""")

