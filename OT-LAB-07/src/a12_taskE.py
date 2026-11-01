# %% [markdown]
# # Task E · Algorithm selection reasoning
# For each scenario: the **choice**, the **properties of the problem that drive it**, a **runner-up and why it loses**, and **evidence** (✔ = backed by a number measured in this notebook, ○ = reasoning only). At least three are evidence-backed, including one many-objective and one constrained problem.

# %%
def r_(no, key, f): return RES[(no, key)][f]
def g_(a): return f"{a:.4f}" if np.isfinite(a) else "n/a"
st3_, st5_ = STUDY['D']['st3'], STUDY['D']['st5']; sm_ = STUDY['D_many']['stats']; om_ = STUDY['D_many']['order']
E = [
 ("1 · P3 — ZDT2-type, 10 variables, smooth *concave* front, unconstrained, 2 objectives, cheap evaluations",
  "**MOEA/D-DE** (best IGD in D.1); NSGA-II as the safe, parameter-light default",
  "Concave front ⇒ weighted-sum methods can never reach the interior (✔ in the P3 study the weighted sum returned only 2 points, %s of them with 0.25<f1<0.75, vs %s for the Tchebycheff-type weight metric (p=∞)); population methods need no convexity. Two objectives ⇒ Pareto ranking still has discriminating power, evaluations are cheap ⇒ a population method with a good variation operator is enough." % (STUDY.get('ws_mid', '?'), STUDY.get('p1_mid', '?')),
  "NSGA-II: ✔ rank %d (IGD %s ± %s) – a few of 10 seeds get trapped on ZDT2's local front (std > mean). Note also that plain SBX-based MOEA/D ranked %d/13 (IGD %s): the DE operator, not decomposition alone, is what makes MOEA/D-DE win." % (STUDY['D']['P3'].index('NSGA-II') + 1, g_(st3_['NSGA-II']['igd'][0]), g_(st3_['NSGA-II']['igd'][1]), STUDY['D']['P3'].index('MOEA/D') + 1, g_(st3_['MOEA/D']['igd'][0])),
  "✔ D.1 table: MOEA/D-DE rank 1, IGD %s ± %s over 10 seeds." % (g_(st3_['MOEA/D-DE']['igd'][0]), g_(st3_['MOEA/D-DE']['igd'][1]))),
 ("2 · P5 — beam design: **constrained** (the constraint is active on the front), 2 variables, 2 objectives, curved front",
  "**C-NSGA-II** (Deb's feasibility rules)",
  "Only %.0f %% of random P5 points are infeasible, yet the constraint is *active*: ✔ an NSGA-II that ignores it ends with %.0f %% infeasible solutions. A penalty method needs a tuned penalty weight (too small → infeasible 'winners', too big → premature convergence at the boundary); constrained domination needs no parameter and always prefers feasible solutions. Honest caveat: with only one mild constraint the measured advantage over penalty-NSGA-II is small." % (100 * STUDY.get('p5_infeas', np.nan), 100 * STUDY.get('p5_ignore_infeas', np.nan)),
  "NSGA-II with static penalty: ✔ rank %d vs C-NSGA-II rank %d on P5 (IGD %s vs %s – inside one std, so a tie; MOABC and SMS-EMOA had the lowest IGD)." % (STUDY['D']['P5'].index('NSGA-II') + 1, STUDY['D']['P5'].index('C-NSGA-II') + 1, g_(st5_['NSGA-II']['igd'][0]), g_(st5_['C-NSGA-II']['igd'][0])),
  "✔ D.1 (P5) head-to-head + infeasible-share numbers; ○ the rest is reasoning."),
 ("3 · P10 (M = 5) — **many-objective**, spherical front, 14 variables, plenty of evaluations",
  "**%s** (best IGD in D.2); NSGA-III is the default textbook choice" % om_[0],
  "With M = 5 Pareto dominance stops discriminating (✔ %.0f %% of a random population is already non-dominated at M=5 vs %.0f %% at M=3) and crowding distance fails; reference-point/reference-vector designs keep diversity structurally." % (100 * STUDY.get('nd_frac_5', np.nan), 100 * STUDY.get('nd_frac_3', np.nan) if 'nd_frac_3' in STUDY else np.nan),
  "NSGA-II: ✔ IGD %s vs %s for the winner (mean of 10 seeds)." % (g_(sm_['NSGA-II']['igd'][0]), g_(sm_[om_[0]]['igd'][0])),
  "✔ D.2 ranking: " + " < ".join(om_)),
 ("4 · P8 — bi-objective TSP-15 (permutations), cheap evaluation, **tight budget** (~1 000 evaluations)",
  "**%s** at the small budget; %s when the budget grows" % (STUDY['D_tsp']['best_lo'], STUDY['D_tsp']['best_hi']),
  "Permutation encoding makes SBX/polynomial mutation useless; the problem has heuristic information (1/distance, 1/time) that constructive ACO methods exploit from the first iteration, a decisive advantage when only ~1 000 evaluations are allowed.",
  "NSGA-II with order crossover/swap: needs the larger budget to catch up (see the D.3 IGD table).",
  "✔ D.3 table at budgets 1 000 / 10 000."),
 ("5 · P9 — feature selection (30 binary variables, **each evaluation = 5-fold model training**, budget ≈ 3 000)",
  "**MOMA** or NSGA-II with bit-flip operators; if evaluation were much more expensive, add a surrogate",
  "Binary encoding and an expensive objective ⇒ need operators that exploit structure cheaply: bit-flip local search around good subsets (memetic) uses evaluations where they matter. The problem is also *discrete*, so gradient or classical scalarisation (needs a single-objective solver for a binary programme) is unattractive.",
  "MOGWO (continuous swarm, mapped to binary): works (✔ logged in the implementation table) but its continuous update is not designed for subsets.",
  "○ reasoning plus ✔ the P9 runs for algorithms 23 and 30 in the implementation log (A3)."),
 ("6 · P14 — laptop choice where the decision maker already has an **aspiration point (1.8 kg, 12 h)**",
  "**R-NSGA-II** (reference-point preference)",
  "The DM does not need the whole front; concentrating the budget near the aspiration point gives a denser set of relevant trade-offs. Because the aspiration is *dominated* by the true front, R-NSGA-II is expected to converge onto the front region closest to it.",
  "NSGA-II + a posteriori TOPSIS/VIKOR: works, but spends most of the population on irrelevant regions.",
  "✔ the P14 R-NSGA-II run (algorithm 35): front spans %.2f–%.2f kg, whereas the full front spans %.2f–%.2f kg." % (FRONTS[(35, 'P14')][:, 0].min(), FRONTS[(35, 'P14')][:, 0].max(), PROB['P14'].Z[:, 0].min(), PROB['P14'].Z[:, 0].max())),
]
rows = [[a, b, c, d, e] for a, b, c, d, e in E]
md_table(["scenario", "recommended algorithm", "why (problem properties)", "runner-up and why it loses", "evidence"], rows)
STUDY['E'] = E
