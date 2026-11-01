# %% [markdown]
# # Task F · Viva questions with short answers and demos

# %%
md("""**Q1. Why does the weighted sum miss part of a front, and how do you show it?** Minimising w·f finds only points where a line of slope −w/(1−w) touches the front from outside, i.e. *supported* points of the **convex hull**. On a concave or non-convex front the interior is never the minimiser of any weighted sum. Demo below: P3 (concave) – the weighted sum returns only the end points, whereas a population method returns the whole curve.""")
Wf = FRONTS[(1, 'P3')] if (1, 'P3') in FRONTS else None
fig, ax = plt.subplots(figsize=(5, 3.6)); P3_ = PROB['P3']; Z = P3_.Z[np.argsort(P3_.Z[:, 0])]; ax.plot(Z[:, 0], Z[:, 1], c='.7', lw=4, label='true front (concave)')
if Wf is not None: ax.plot(Wf[:, 0], Wf[:, 1], 'rs', ms=8, label='weighted sum (%d pts)' % len(Wf))
F14 = FRONTS[(14, 'P3')]; ax.plot(F14[:, 0], F14[:, 1], 'b.', ms=4, label='NSGA-II (%d pts)' % len(F14)); ax.legend(fontsize=7); plt.show()

# %%
md("""**Q2. Why can VEGA collapse and what fixes it?** VEGA fills the mating pool with sub-populations selected on *one objective at a time*, so it favours solutions that are extreme in an objective (the "speciation" bias) and ignores compromise solutions. Demo: VEGA vs NSGA-II on P3.""")
Fv, Fn = FRONTS[(10, 'P3')], FRONTS[(14, 'P3')]; fig, ax = plt.subplots(figsize=(5, 3.6)); ax.plot(Z[:, 0], Z[:, 1], c='.7', lw=4); ax.plot(Fv[:, 0], Fv[:, 1], 'r.', label='VEGA (%d pts)' % len(Fv)); ax.plot(Fn[:, 0], Fn[:, 1], 'b.', ms=3, label='NSGA-II'); ax.legend(fontsize=7); plt.show()
print("IGD  VEGA = %.4f   NSGA-II = %.4f  (seed 0, same budget).  Fix: use Pareto ranking (MOGA/NSGA/NSGA-II) so compromise solutions can win." % (r_(10, 'P3', 'igd'), r_(14, 'P3', 'igd')))

# %%
md("""**Q3. What does elitism buy you?** Non-elitist algorithms (NSGA, MOGA) can lose non-dominated solutions that were already found, because parents are replaced by offspring. NSGA-II merges parents and offspring, so a found Pareto solution is only displaced by one that dominates it or is less crowded. Demo: same problem, NSGA vs NSGA-II.""")
rows = [[NAMES[n], "%.4f" % r_(n, 'P5', 'hv'), "%.4f" % r_(n, 'P5', 'igd'), r_(n, 'P5', 'n')] for n in (12, 14)]; md_table(["algorithm (P5, seed 0)", "HV", "IGD", "points"], rows)
# direct monotonicity check of the elitist archive: HV of NSGA-II population over generations never decreases (by construction) – verify
a_ = NSGA2(PROB['P3'], seed=1); a_.max_nfe = 6000; a_.initialize(); hvs = []
while a_.nfe < 6000:
    a_.iterate(); hvs.append(hypervolume(PROB['P3'].norm(a_.F[nds(a_.F)[0]]), PROB['P3'].ref))
print("NSGA-II: front hypervolume over %d generations is non-decreasing: %s (min step %.2e)" % (len(hvs), bool(np.all(np.diff(hvs) >= -1e-12)), np.min(np.diff(hvs))))

# %%
md("""**Q4. Why does crowding distance stop working with many objectives, and what replaces it?** In M ≥ 4 almost all individuals are mutually non-dominated, so the front-based ranking is flat and crowding distance (a sum of per-objective neighbour gaps) is the only selector; it prefers points that are extreme in one objective and is unrelated to Euclidean spread. NSGA-III (reference points), MOEA/D (weight vectors), RVEA (angle-penalised reference vectors), HypE (sampled hypervolume contribution) give *structured* diversity.""")
rows = [[int(m), STUDY.get(f'nd_frac_{m}', np.nan)] for m in (3, 5, 10)]; md_table(["M", "non-dominated share of a random population"], rows)
print("D.2 ranking by IGD on M=5:", " < ".join(STUDY['D_many']['order']))

# %%
md("""**Q5. How does a preference-based EA (R-NSGA-II) differ from NSGA-II?** Within a front, R-NSGA-II ranks individuals by their distance to the decision-maker's reference point (with ε-clearing for diversity) instead of by crowding distance. So the population is *pulled* toward the region of interest instead of being spread over the whole front.""")
Fr, Zp = FRONTS[(35, 'P14')], PROB['P14'].Z; fig, ax = plt.subplots(figsize=(5, 3.6)); ax.plot(Zp[:, 0], Zp[:, 1], c='.7', lw=4, label='true front'); ax.plot(Fr[:, 0], Fr[:, 1], 'r.', label='R-NSGA-II'); ax.plot(*REFP['P14'], 'k*', ms=14, label='reference (1.8 kg, 12 h)'); ax.legend(fontsize=7); plt.show()

# %%
md("""**Q6. Show Deb's constrained-domination rules on an example.** Minimise both; A = (2,3), CV 0; B = (1,1), CV 2; C = (5,5), CV 0.5; D = (4,4), CV 0.""")
pts = {'A': ((2, 3), 0.), 'B': ((1, 1), 2.), 'C': ((5, 5), .5), 'D': ((4, 4), 0.)}; rows = []
for x, y in (('A', 'B'), ('B', 'C'), ('A', 'D'), ('D', 'A'), ('C', 'D')):
    rows.append([f"{x} vs {y}", cdominates(np.array(pts[x][0], float), pts[x][1], np.array(pts[y][0], float), pts[y][1]), {('A', 'B'): 'rule 1: feasible beats infeasible, although B has better objectives', ('B', 'C'): 'rule 2: both infeasible, smaller CV (0.5) wins ⇒ C beats B so this is False', ('A', 'D'): 'rule 3: both feasible, A dominates D', ('D', 'A'): 'neither dominates the other in rule 3 sense ⇒ False', ('C', 'D'): 'rule 1: D feasible, C is not ⇒ False'}[(x, y)]])
md_table(["pair", "first constrained-dominates second?", "reason"], rows)
assert rows[0][1] and not rows[1][1] and rows[2][1] and not rows[3][1] and not rows[4][1]

# %%
md("""**Q7. How can you detect premature convergence in practice?** (i) the hypervolume of the archive stops growing while population diversity keeps falling; (ii) spacing/spread collapses while the number of distinct objective vectors drops; (iii) different seeds converge to *different* fronts (large across-seed std of IGD/HV); (iv) a restart or higher mutation rate immediately finds dominating points. Demo: NSGA-II HV trace on P3 and the seed-to-seed spread of MOEA/D from the sensitivity study.""")
fig, ax = plt.subplots(figsize=(5, 3.4)); ax.plot(hvs); ax.set_xlabel('generation'); ax.set_ylabel('HV'); ax.set_title('NSGA-II on P3: HV plateau = converged / stalled'); plt.show()
_i = [round(float(r[1]["igd"]), 4) for r in outT[20]]; print("MOEA/D (T=20) on P3, IGD of the 10 seeds:", _i, "-> %d seeds ended far from the front (IGD > 0.05) while the others converged: criterion (iii) in action." % sum(v > .05 for v in _i))

# %%
md("""**Q8. Why are TOPSIS and VIKOR not optimisers?** They take a *finite list of alternatives* (here the NSGA-II front of P4, D.4) and only rank it – TOPSIS by relative closeness to the ideal/anti-ideal point, VIKOR by a compromise between group utility S and individual regret R. They create no new solution and cannot improve a front; a poor front gives a poor "best" compromise. Their role is the *decision* stage after the optimiser. In D.4 the TOPSIS pick moved when the weights changed: %s.""" % ('yes' if STUDY['D_dm']['flip'] else 'no (the pick was stable for these weights)'))

# %%
md("""**Q9. How does the number of evaluations/seeds change conclusions?** A single seed can reverse a ranking: in the head-to-head tables a gap smaller than one standard deviation over 10 seeds is not a result. At the TSP (D.3) the ranking depends on the budget. Always report mean ± std, the budget, and the seed set.""")
