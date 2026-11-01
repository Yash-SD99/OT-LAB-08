# %% [markdown]
# # Task B — Master comparison table with evidence
# One row per algorithm, grouped by family. Every row lists ≥ 4 advantages and ≥ 4 disadvantages. Items marked **†** quote a number measured in *my own runs above* (Task A, single seed 0, 10,000 evaluations unless stated; HV/IGD/Spacing on normalised objectives, reference point 1.1). Unmarked items are justified by the algorithm's steps.

# %%
def v(no, key, f): return RES[(no, key)][f]
H = lambda n, k: f"{v(n, k, 'hv'):.3f}"; I = lambda n, k: f"{v(n, k, 'igd'):.3f}"; S = lambda n, k: f"{v(n, k, 'sp'):.3f}"; N = lambda n, k: f"{int(v(n, k, 'n'))}"
T = lambda n, k: f"{v(n, k, 'time'):.1f}s"; E = lambda n, k: f"{int(v(n, k, 'nfe')):,}"
mid_ = lambda F: int(((F[:, 0] > .25) & (F[:, 0] < .75)).sum())
tp = topsis(SUP, wts, benefit); tp_flip = [list(topsis(SUP, wts + np.array(d), benefit)[1]) != list(tp[1]) for d in ([.1, -.1, 0, 0], [-.1, .1, 0, 0], [0, 0, .1, -.1], [0, 0, -.1, .1])]
vk = [vikor(SUP, wts, benefit, vv) for vv in (.25, .5, .75)]; vk_flip = [list(vikor(SUP, wts + np.array(d), benefit, .5)['order']) != list(vk[1]['order']) for d in ([.1, -.1, 0, 0], [-.1, .1, 0, 0], [0, 0, .1, -.1], [0, 0, -.1, .1])]
gp7a = goal_programming(PROB['P7'], [40, 50], [10, 1], ['>=', '>=']); gp7b = goal_programming(PROB['P7'], [40, 50], [1, 10], ['>=', '>=']); gp11 = goal_programming(PROB['P11'], [2400, 600], [1, 1], ['>=', '>='])
lexA = lexicographic(PROB['P7'], [0, 1]); lexB = lexicographic(PROB['P7'], [1, 0]); p13 = {(r[0], r[1]): r for r in STUDY['p13_rows']}
nsga2_m5_nd = len(FRONTS[(14, 'P10_5')])
gp7u = goal_programming(PROB['P7'], [40, 50], [1, 1], ['>=', '>=']); gp7n = goal_programming(PROB['P7'], [40, 50], [1 / 40, 1 / 50], ['>=', '>='])

Xq, Fq, dq, aq = run_alg(NSGA2, PROB['P8']); log(14, 'P8', PROB['P8'], Fq, dq, aq.nfe)      # NSGA-II baseline on the routing problem
B = {}
B[1] = dict(
 what="Normalise the objectives, minimise w₁f₁+w₂f₂ with a scalar solver for 21 weights between 0 and 1, keep the non-dominated results.",
 adv=["Extremely simple: one scalarisation + any single-objective solver.", f"Accurate on convex fronts: IGD {I(1,'P1')} on P1 with {N(1,'P1')} distinct points †.", "Each solution is Pareto-optimal for w>0 (guaranteed by theory).", "Scales to many objectives (longer weight vector) and the weights are easy for a decision maker to read.",
      f"Cheap per point (all 21 solves took {T(1,'P1')} on P1 †)."],
 dis=[f"Cannot reach dented regions: only {N(1,'P3')} distinct points on non-convex P3 and {mid_(FRONTS[(1,'P3')])} with 0.25<f₁<0.75 †.", f"Equal weight steps do not give equal spacing: on P4 only {N(1,'P4')} of 21 weights gave distinct points (Spacing {S(1,'P4')}) †.", f"Each point needs a whole solver run ({E(1,'P1')} evaluations for 21 points on P1 †) – far more than the 10,000 budget of the population methods.", "Sensitive to objective scaling (needs normalisation) and to weights.", "No handling of integer/discrete fronts unless the solver supports them."],
 best="Convex continuous problems with a fast scalar solver (P1, P4 portfolio).", avoid="Non-convex or discontinuous fronts (P3), or when evenly spread points are needed.")
B[2] = dict(
 what="Minimise one objective while the others are bounded by ε (swept from ideal to nadir in 21 steps); each solve gives one Pareto point.",
 adv=[f"Reaches non-convex fronts: {N(2,'P3')} points, IGD {I(2,'P3')} on P3 (weighted sum got {N(1,'P3')}) †.", f"Handles constraints naturally (P5: HV {H(2,'P5')}, IGD {I(2,'P5')}) †.", "Spread is controlled directly by the ε grid.", "Works for convex, concave and mixed fronts; solutions are Pareto-optimal when the ε-constraint is active.", "Uses a plain single-objective constrained solver."],
 dis=[f"ε grid is uniform in one objective, so points are uneven in objective space (Spacing {S(2,'P2')} on P2) †.", f"One constrained solve per point: {E(2,'P3')} evaluations on P3 †.", "Needs ideal and nadir values to choose ε; an ε that is too tight is infeasible.", "Grid size grows as 21^(M−1) for M objectives – impractical beyond 3 objectives.", "May return weakly Pareto-optimal points if the bound is inactive."],
 best="2–3 objectives, non-convex or constrained, with a reliable scalar solver (P3, P5).", avoid="Many objectives (M ≥ 5) or when the nadir point is unknown.")
B[3] = dict(
 what="From a feasible starting design x⁰ maximise Σwᵢ(fᵢ(x⁰)−fᵢ(x)) subject to f(x) ≤ f(x⁰); the result is a Pareto point that dominates x⁰.",
 adv=["Guarantees an improvement: the output dominates the start (checked in the unit test).", f"Works on linear P7 and convex P1 (HV {H(3,'P7')} and {H(3,'P1')}) †.", "Needs no ideal/nadir point and no weights to be tuned – the start design plays the role of a reference point.", "Natural tool to improve an existing design.", f"Cheap: {E(3,'P1')} evaluations for 12 starts †."],
 dis=[f"Coverage depends on the starts: {N(3,'P1')} distinct points from 12 starts on P1, Spacing {S(3,'P1')} (uneven) †.", f"IGD {I(3,'P1')} on P1 is worse than weighted sum's {I(1,'P1')} † because starts cluster.", "Needs a feasible start (P7 required rejection sampling).", "One constrained nonlinear solve per start; the feasible set f(x) ≤ f(x⁰) can be thin and hard for the solver.", "No control of the spread of the final set."],
 best="Improving a given feasible design (what-can-I-gain-for-free questions).", avoid="When a uniform picture of the whole front is required.")
B[4] = dict(
 what="Minimise the weighted distance to the ideal point, (Σ(wᵢfᵢ)^p)^(1/p) for p = 1, 2 or max(wᵢfᵢ) for p = ∞ (Tchebycheff), over a weight sweep.",
 adv=[f"p = ∞ recovers the whole non-convex front: {N(4,'P3_pinf')} points, IGD {I(4,'P3_pinf')} on P3 †.", f"p = 1 reproduces weighted sum, so one family covers a range of behaviours ({N(4,'P3_p1')} points on P3 for p=1, {N(4,'P3_p2')} for p=2) †.", "Ideal point gives an intuitive 'closest to perfect' reading of preferences.", "Normalised version is scale-free.", f"Good spacing on P1 for p=1 (Spacing {S(4,'P1_p1')}) †."],
 dis=[f"p = 2 only partly reaches non-convex parts ({N(4,'P3_p2')} points, IGD {I(4,'P3_p2')} on P3) †.", "Needs the ideal point first, i.e. M extra single-objective solves.", f"Spacing depends on p: {S(4,'P1_p1')} for p=1 vs {S(4,'P1_pinf')} for p=∞ on P1 †.", "p = ∞ is non-smooth and can return weakly Pareto-optimal points.", f"Whole sweep costs {E(4,'P3_pinf')} evaluations on P3 †."],
 best="Non-convex fronts with a known ideal point when a scalar solver is available (p = ∞).", avoid="p = 1 on non-convex fronts; very high-dimensional objective spaces (sweep size).")
B[5] = dict(
 what="The decision maker supplies a utility U(f); one scalar optimisation maximises U and returns the single preferred solution.",
 adv=[f"Returns the answer directly, no post-selection: one solve ({E(5,'P14')} evaluations on P14) †.", f"Handles any shape of preference: on P14 it found exactly (1.800 kg, 12.00 h) †.", "Works with constraints and any number of objectives.", "Changing the utility parameter moves the solution along the front (sensitivity study in Task C)."],
 dis=["Needs the DM to know the utility in advance (hard in practice).", "A badly chosen utility returns a dominated design: the target-distance utility on P14 chose screen 14.5″ although every Pareto point has x₁ = 13″ †.", "One solution only – no picture of the trade-off.", "Non-monotone utilities create local optima and need global solvers.", f"For P4 three investor profiles needed three separate solves ({E(5,'P4')} evaluations) †."],
 best="A priori preference is known (target design such as P14).", avoid="Exploratory design when the DM wants to see the front.")
B[6] = dict(
 what="Targets for each objective, deviation variables d⁺, d⁻, minimise the weighted undesirable deviations subject to the constraints.",
 adv=[f"Works directly in natural units and reports how much each goal is missed: P11 shortfall of the units goal = {gp11[3][1]:.0f} †.", f"Exact on linear problems: P11 solution {np.round(gp11[0], 0).tolist()} = the analytic vertex (300, 200) †.", "Goals that are over-ambitious still give a best compromise instead of 'infeasible'.", "Weights express priorities; one solve gives one answer."],
 dis=[f"Result depends on weights and goal values: on P7 weights (10,1) give outputs {np.round(gp7a[1], 1).tolist()} but (1,10) give {np.round(gp7b[1], 1).tolist()} †.", f"Unit weights mix incommensurable units: on P7 unit weights give outputs {np.round(gp7u[1], 1).tolist()} but weights normalised by the goals give {np.round(gp7n[1], 1).tolist()} †.", "If the goals are pessimistic the solution may be dominated (goal attained, but better designs exist).", "Gives one point, not a front.", "Goal values require prior knowledge of what is attainable."],
 best="Planning problems with explicit targets (production, budgets – P11, P7).", avoid="When goals are unknown or when a Pareto front is the deliverable.")
B[7] = dict(
 what="Optimise objectives strictly in priority order; each later problem keeps the earlier optimum within a tolerance.",
 adv=[f"Needs no weights; breaks ties exactly: on P11 profit 2400 is kept and units rise to {-lexicographic(PROB['P11'], [0, 1])[1][1]:.0f} at (300,200) †.", f"Exact priority: P7 agriculture-first returns x₁ = {lexA[0][0]:.1f}, the maximum possible 70 †.", "Matches how some organisations really decide (safety first, cost second).", "Handles constraints through the underlying solver."],
 dis=[f"Lower priorities are ignored unless there are ties: on P7 agriculture-first gives {np.round(-lexA[1], 1).tolist()}, industry-first {np.round(-lexB[1], 1).tolist()} (two extreme points) †.", "The tolerance (here 0.5 %) is a hidden sensitive parameter: too small gives a thin feasible slab the solver cannot find, too large lets priority 1 leak.", "Gives only one point, an extreme with respect to priority 1.", "An exact tie is needed for later objectives to matter (P11: the profit ridge) †.", "Priority order must be known."],
 best="Clearly ranked objectives or tie-breaking (P11).", avoid="Objectives of comparable importance – you get extreme solutions.")
B[8] = dict(
 what="Vector-normalise and weight the decision matrix, compute distance to the ideal and anti-ideal alternatives, rank by closeness C = d⁻/(d⁺+d⁻).",
 adv=["Closed-form, no optimisation, runs in microseconds.", f"Ranks any finite set, e.g. P12 (S3 > S1 > S5 > S2 > S4, C = {np.round(tp[0], 3).tolist()}) †.", f"All {len(nd12)} suppliers of P12 are Pareto-optimal, yet TOPSIS still gives a complete order (S{tp[1][0] + 1} first) †.", "Easy to explain: closest to ideal and farthest from worst.", "Works on a front produced by NSGA-II (Task D)."],
 dis=[f"Fully compensatory: a supplier with reliability 40 % still ranks #{list(topsis(SUP6, wts, benefit)[1]).index(5) + 1} when it is excellent elsewhere †.", f"Ranking flips under a 0.1 weight shift in {sum(tp_flip)} of 4 tested shifts on P12 †.", "Needs weights from the decision maker and a normalisation choice that changes the answer.", "Not an optimiser: it cannot create alternatives, only rank given ones.", "Rank reversal when alternatives are added."],
 best="Selecting from a finite list or from a Pareto front with agreed weights.", avoid="When one criterion must never be very bad (use VIKOR with v small).")
B[9] = dict(
 what="Group utility S, individual regret R, Q = vS+(1−v)R; rank by smallest Q and test acceptable-advantage (C1) and stability (C2).",
 adv=[f"Parameter v trades average performance against worst-criterion regret: with v = 0 supplier S6 (40 % reliability) is ranked #{list(vikor(SUP6, wts, benefit, 0.0)['order']).index(5) + 1} but with v = 0.5 it is #{list(vikor(SUP6, wts, benefit, .5)['order']).index(5) + 1} †.", f"C1/C2 tell you whether a single winner is justified (P12: C1={vk[1]['C1']}, C2={vk[1]['C2']}) †.", "Returns a compromise set rather than forcing one winner.", "Closed form, fast.", "Less compensatory than TOPSIS."],
 dis=[f"Ranking flips under a 0.1 weight shift in {sum(vk_flip)} of 4 tested shifts (v = .5) †.", f"Winner changes with v on P12: {['S%d' % (vv['order'][0] + 1) for vv in vk]} for v = .25/.5/.75 †.", "Needs weights and v from the DM.", "When C1 fails the answer is a set of alternatives, not one.", "Not an optimiser; only ranks given alternatives."],
 best="Group decisions where a very bad criterion is unacceptable.", avoid="Large alternative sets where C1's 1/(J−1) threshold is meaningless.")
B[10] = dict(
 what="Population split into M sub-populations; sub-population m is selected by tournaments on objective m only, then everything is mixed and varied.",
 adv=["Simplest multiobjective GA: one line changes the selection.", f"Very fast ({T(10,'P3')} for 10,000 evaluations) †.", "No ranking, no sorting, no sharing parameters.", f"Finds the extreme solutions quickly: {N(10,'P2')} points on P2 with IGD {I(10,'P2')} †.", "Works with any variation operator."],
 dis=[f"Collapses to the extremes: a single point (n = {N(10,'P3')}, HV {H(10,'P3')}) on P3 †.", "On P2 about 96 % of the final points sit in the low-cost half (study) †.", "Middle compromise solutions are mediocre in every single objective, so tournaments never favour them ('speciation').", "No elitism: a good non-dominated set can be lost in the next generation.", "No constraint handling or diversity preservation."],
 best="Baseline / teaching; problems whose interesting solutions are the extremes.", avoid="Anything where the compromise region matters.")
B[11] = dict(
 what="Rank = 1 + number of dominators; fitness interpolated by rank; fitness sharing in objective space; fitness-proportionate selection.",
 adv=[f"Dominance-based, so it finds the middle of the front: P5 n = {N(11,'P5')}, HV {H(11,'P5')} †.", "Sharing keeps points spread.", "Conceptually simple, objective-scale-free after min-max.", "No external archive needed.", f"Cheap: {T(11,'P5')} †."],
 dis=[f"Non-elitist: on P3 only {N(11,'P3')} points, HV {H(11,'P3')} vs NSGA-II's {H(14,'P3')} †.", f"Sharing radius σ must be tuned; wrong σ gives clusters (Spacing {S(11,'P5')} on P5 vs NSGA-II {S(14,'P5')}) †.", "Rank-to-fitness mapping and scaling are arbitrary.", "O(N²M) dominance counting.", "Fitness-proportionate selection is sensitive to scaling."],
 best="Historical/baseline Pareto-ranking GA on small 2-objective problems.", avoid="Hard fronts such as ZDT2 within 10,000 evaluations.")
B[12] = dict(
 what="Fronts from non-dominated sorting; front k gets dummy fitness slightly below front k−1's minimum shared fitness; sharing inside each front.",
 adv=[f"Strong pressure towards the front through layered fitness; P5 HV {H(12,'P5')} †.", "Sharing inside each front spreads points.", f"Constraints by penalty: {N(12,'P5')} points on P5 and {N(12,'P7')} on P7 †.", "Natural predecessor of NSGA-II, easy to follow.", "Needs no archive."],
 dis=[f"Non-elitist, and on P7 returns only {N(12,'P7')} points (HV {H(12,'P7')}) vs {N(21,'P7')} points (HV {H(21,'P7')}) for Constrained NSGA-II †.", "O(M N³) sorting and O(N²) sharing – the slowest of the early methods.", "Sharing radius σ must be tuned.", f"IGD {I(12,'P5')} on P5 is worse than NSGA-II's {I(14,'P5')} †.", "Fitness-proportionate selection is scale sensitive."],
 best="Learning the history of Pareto-based GAs.", avoid="Anything where NSGA-II is available.")
B[13] = dict(
 what="Binary tournament: both candidates are compared with a random comparison set (10 % of N); a dominated candidate loses; ties go to the smaller niche count.",
 adv=["No global ranking or sorting – selection cost is O(N·t_dom).", f"Selection pressure is tunable through t_dom; P5 HV {H(13,'P5')} †.", "Niche counts keep diversity.", f"Cheap selection without sorting: {T(13,'P5')} per 10,000 evaluations †.", "Works with any variation operator."],
 dis=[f"Weak on P3: HV {H(13,'P3')}, IGD {I(13,'P3')} after 10,000 evaluations †.", "t_dom too small → almost random selection, too large → premature convergence.", "Needs both t_dom and the niche radius σ.", "Non-elitist.", f"Slower than NSGA-II here ({T(13,'P3')} vs {T(14,'P3')}) †, because of the Python tournament loop."],
 best="Large populations where full sorting is too expensive.", avoid="Small budgets (10,000 evaluations) on hard fronts.")
B[14] = dict(
 what="Parent+offspring pool sorted by non-dominated rank, ties broken by crowding distance, binary tournament mating (elitist).",
 adv=[f"Elitist and nearly parameter-free: P3 HV {H(14,'P3')}, IGD {I(14,'P3')} †.", f"Very even spread through crowding distance: Spacing {S(14,'P3')} (P3) and {S(14,'P6')} (P6) †.", f"Fast: {T(14,'P3')} for 10,000 evaluations †.", "Handles real, permutation and binary variables with suitable operators (P8, P9).", "Excellent two- and three-objective baseline."],
 dis=[f"Collapses on 5 objectives: P10 (M=5) HV {H(14,'P10_5')}, IGD {I(14,'P10_5')} vs NSGA-III's HV {H(15,'P10_5')} †.", f"At M=5 a random population is already {STUDY['nd_frac_5']:.0%} non-dominated (study), so ranking cannot select †.", "Crowding distance is meaningless when most points are non-dominated in many dimensions.", "O(M N²) sorting per generation.", "Constraints only through penalty unless the constrained variant is used."],
 best="2–3 objective problems with up to a few thousand evaluations (P3, P5, P6).", avoid="M ≥ 4 objectives.")
B[15] = dict(
 what="NSGA-II-style survival with crowding replaced by reference points: normalise, associate to reference lines, fill the least crowded niches.",
 adv=[f"Keeps spread in many objectives: P10 (M=5) HV {H(15,'P10_5')} vs NSGA-II {H(14,'P10_5')} †.", f"On 3 objectives gets {N(15,'P10_3')} points, IGD {I(15,'P10_3')} †.", "Reference points give a predefined well-spread target distribution.", "Adaptive normalisation copes with differently scaled objectives.", "Same operators as NSGA-II."],
 dis=[f"Slow convergence at M = 5 in 10,000 evaluations: IGD {I(15,'P10_5')}, mean solution still outside the unit sphere †.", "Number of reference points (and hence N) grows combinatorially with M and divisions p.", "Assumes a simplex-like front; irregular fronts leave empty reference lines.", "Normalisation (extreme points, intercepts) is fragile – fallbacks needed.", f"Slower per generation than NSGA-II ({T(15,'P10_3')} on M=3) †."],
 best="Many-objective problems with regular fronts (P10).", avoid="Disconnected or strongly degenerate fronts, tiny budgets.")
B[16] = dict(
 what="Fitness = raw (sum of strengths of dominators) + density from the k-th nearest neighbour; fixed-size archive truncated by nearest-neighbour distance.",
 adv=[f"Even, dense fronts: P4 Spacing {S(16,'P4')}, IGD {I(16,'P4')} †.", "Fitness distinguishes dominated individuals as well (fine-grained).", "Archive guarantees elitism and constant size.", "Density from k-th neighbour scales with the population.", f"Works unchanged with penalty constraints (P5: {N(16,'P5')} points, HV {H(16,'P5')}) †."],
 dis=[f"Heavy: {T(16,'P5')} for 10,000 evaluations vs {T(14,'P5')} for NSGA-II †.", "O(n³) archive truncation in the worst case.", f"P5 IGD {I(16,'P5')} is worse than NSGA-II's {I(14,'P5')} †.", "Fitness needs full strength/raw computation each generation (O(n²M)).", "Density estimator also degrades with many objectives."],
 best="2–3 objective problems where the archive quality matters (P4).", avoid="Tight time budgets or many objectives.")
B[17] = dict(
 what="External archive on an adaptive grid; selection picks a hyper-box (less crowded preferred) then a member in it.",
 adv=[f"Region-based selection is cheap to maintain (grid index) and gives {N(17,'P3')} points on P3 †.", f"Good IGD {I(17,'P3')} on P3 †.", "Archive is bounded; crowded boxes are thinned automatically.", "Few evaluations per generation allow fast adaptation.", "Insensitive to objective scales (grid is rescaled)."],
 dis=[f"Grid resolution is a sensitive parameter (10 divisions here); too fine → many empty boxes, too coarse → no pressure.", f"Slower than NSGA-II in Python ({T(17,'P3')} vs {T(14,'P3')}) because of per-generation grid/archive updates †.", f"P5 IGD {I(17,'P5')} vs NSGA-II {I(14,'P5')} †.", "Grid size grows as div^M – unusable for many objectives.", "Small internal population (20) limits exploration."],
 best="Bi-objective problems with limited memory.", avoid="M ≥ 4 or poorly-bounded objective ranges.")
B[18] = dict(
 what="N weight vectors define N Tchebycheff sub-problems; each is improved with children bred from neighbouring sub-problems, and solved children replace neighbours.",
 adv=[f"Accurate and even: P3 IGD {I(18,'P3')}, Spacing {S(18,'P3')} †.", f"Handles three objectives (P10: HV {H(18,'P10_3')}) †.", "O(MNT) per generation – no sorting.", "Tchebycheff reaches non-convex fronts.", "Weights give direct control of where points lie."],
 dis=[f"Returned only {N(18,'P3')} of 100 sub-problem solutions on P3 (several sub-problems share a solution) †.", f"Needs weights/neighbourhood T/nr/δ; slow in Python ({T(18,'P3')}) †.", "Uniform weights do not give uniform points on irregular fronts.", "Needs ideal/nadir estimates (here the current population) that can mislead early on.", f"Weaker than NSGA-III on 3-objective P10 (IGD {I(18,'P10_3')} vs {I(15,'P10_3')} with fewer points) †."],
 best="Smooth fronts with many objectives; when preferences map to weights.", avoid="Discontinuous fronts without weight adaptation.")
B[19] = dict(
 what="MOEA/D whose variation is differential evolution (x_r1 + F(x_r2 − x_r3)) with polynomial mutation and a limit nr on replacements.",
 adv=[f"Best IGD on P3 among the decomposition runs: {I(19,'P3')}, Spacing {S(19,'P3')} †.", f"Dense front on P3: {N(19,'P3')} points †.", "DE steps adapt to the scale of the neighbourhood.", "nr limit protects diversity.", "No crossover distribution index to tune."],
 dis=[f"On P5 worse than NSGA-II: IGD {I(19,'P5')} vs {I(14,'P5')}, Spacing {S(19,'P5')} †.", "Real-valued only (not for permutations/bits).", f"Python cost {T(19,'P3')} for 10,000 evaluations †.", "F and CR matter; CR=1 destroys separability.", "Same weight/neighbourhood parameters as MOEA/D."],
 best="Continuous problems with complicated Pareto sets (P3).", avoid="Discrete variables.")
B[20] = dict(
 what="MOEA/D where each sub-problem has its own pheromone matrix; ants build tours, neighbouring sub-problems share information.",
 adv=[f"Finds the whole 10-point front of P8 (HV {H(20,'P8')}) †.", "Decomposition gives each ant a clear scalar target.", "Heuristic information specific to each weight vector.", "Pheromone sharing among neighbours speeds up search.", f"Cheap on a 15-city instance ({T(20,'P8')}) †."],
 dis=[f"Slower than NSGA-II in wall-clock ({T(20,'P8')} vs about 0.8 s for NSGA-II on P8) †.", "Parameters: α, β, ρ, T, number of sub-problems.", "One pheromone matrix per sub-problem: O(N n²) memory.", "Permutation problems only.", "No benefit shown over simple operators on only 15 cities (same front as MOACO and NSGA-II) †."],
 best="Larger routing problems where good heuristics exist.", avoid="Continuous problems.")
B[21] = dict(
 what="NSGA-II in which dominance is replaced by Deb's rules: feasible beats infeasible, smaller violation beats larger, otherwise Pareto dominance.",
 adv=[f"No penalty parameter; P6 HV {H(21,'P6')} †.", f"On P7 returns {N(21,'P7')} points (HV {H(21,'P7')}) – far better than non-elitist NSGA's {N(12,'P7')} †.", "Infeasible solutions still guide the search through their violation.", "Same cost and complexity as NSGA-II.", f"P5 results identical to penalty NSGA-II here (HV {H(21,'P5')}) – the rule never hurts †."],
 dis=[f"P6 IGD {I(21,'P6')} is slightly worse than penalty NSGA-II's {I(14,'P6')} †.", "If the feasible region is tiny or disconnected, early populations contain no feasible points and diversity is lost.", "Violation measures of different constraints must be aggregated and scaled.", "Equality constraints need a tolerance.", f"P7: only {N(21,'P7')} of 100 individuals are non-dominated after {E(21,'P7')} evaluations †."],
 best="Constrained problems (P5, P6, P7).", avoid="Problems where infeasible regions must be crossed to reach disconnected parts.")
B[22] = dict(
 what="Particles move by v = wv + c₁r₁(pbest−x) + c₂r₂(gbest−x); leaders come from a grid-based archive; turbulence mutation decays.",
 adv=[f"Fast: {T(22,'P5')} for 10,000 evaluations †.", f"Good convergence on P5 (HV {H(22,'P5')}, IGD {I(22,'P5')} vs MOGWO's {I(23,'P5')}) †.", "Few parameters, easy for real variables.", f"Dense, even front on P4 (Spacing {S(22,'P4')}) †.", "Archive gives elitism."],
 dis=[f"IGD on P4 ({I(22,'P4')}) is worse than SPEA2's ({I(16,'P4')}) †.", "Velocity can leave the bounds (needs reflection).", "Real variables only (binary/perm need other variants).", "Premature convergence if w and the archive leader pressure are high.", f"The archive cap (100) limits the output: {N(22,'P5')} points on P5 †."],
 best="Continuous, smooth problems with a modest budget (P4, P5).", avoid="Discrete or highly multimodal problems.")
B[23] = dict(
 what="Three leaders (α, β, δ) from the archive guide every wolf by encircling; the control factor a falls from 2 to 0.",
 adv=[f"Few parameters; P5 HV {H(23,'P5')} in {T(23,'P5')} †.", "Exploration → exploitation schedule built in.", "Works on binary problems via transfer function (P9).", "Archive-based leaders provide elitism.", f"Handles binary P9 via a sigmoid transfer (found {N(23,'P9')} subsets) †."],
 dis=[f"P5 IGD {I(23,'P5')} is worse than NSGA-II's {I(14,'P5')} and MOPSO's {I(22,'P5')} †.", f"Binary P9 run found only {N(23,'P9')} non-dominated subsets and cost {T(23,'P9')} for 3,000 evaluations (5-NN CV) †.", "Mean of leaders drags solutions to the centre of the archive → weak spread at the ends.", "Sigmoid binary version loses the geometric meaning of the update.", "Time-based schedule needs the budget in advance."],
 best="Continuous constrained design problems with limited tuning effort.", avoid="Discrete combinatorial problems.")
B[24] = dict(
 what="Employed bees perturb one dimension toward a neighbour, onlookers favour better ranks, scouts restart stagnant sources; dominance-based greedy selection.",
 adv=[f"Strong results: P3 HV {H(24,'P3')}, IGD {I(24,'P3')} † and P6 IGD {I(24,'P6')} †.", f"Very fast ({T(24,'P3')}) †.", "Only the limit and colony size to tune.", "Scouts prevent stagnation.", "Dense archive."],
 dis=[f"One-dimension moves converge slowly on rotated problems.", "Dominance-only greedy rule loses information when solutions are non-dominated (random 50 % acceptance).", f"Overshoots the budget slightly ({E(24,'P3')} evaluations) because a cycle evaluates 2–3 N points †.", "Real variables only.", f"P3 IGD {I(24,'P3')} trails MO-TLBO ({I(28,'P3')}) †."],
 best="Separable continuous problems (P3, P6).", avoid="Strongly correlated variables.")
B[25] = dict(
 what="Two pheromone matrices (distance, time) and heuristics; each ant uses its own weight λ; archive-based pheromone deposit.",
 adv=[f"Finds the 10-point front of P8 (HV {H(25,'P8')}) †.", f"Cheapest of the routing algorithms here ({T(25,'P8')}) †.", "Heuristic information speeds up tour construction.", "Different λ per ant spreads the search along the front.", "Natural for permutation problems."],
 dis=["Permutation/graph problems only.", "Several parameters (α, β, ρ, ant count, bounds).", "Pheromone matrices cost O(n²) each and the deposit rule is heuristic.", f"No advantage on 15 cities: NSGA-II with order crossover gets the same HV ({H(14,'P8')}) †.", f"Python tour construction makes it slower than NSGA-II ({T(25,'P8')} vs {T(14,'P8')}) †."],
 best="Routing problems with good heuristic information.", avoid="Continuous design problems.")
B[26] = dict(
 what="One current solution moves to neighbours; Boltzmann acceptance with a cooling schedule; archive of non-dominated solutions with return-to-base.",
 adv=[f"Spacing {S(26,'P5')} on P5 comparable to NSGA-II's {S(14,'P5')} with a single trajectory †.", f"Surprisingly good on P5: HV {H(26,'P5')}, IGD {I(26,'P5')} †.", "Escapes local optima by accepting worse moves.", "Works for any representation that has a neighbourhood move.", "Archive keeps all non-dominated solutions found."],
 dis=[f"Sequential: {T(26,'P5')} for 10,000 Python iterations (slowest of the singles) †.", f"Only {N(26,'P3')} points and IGD {I(26,'P3')} on P3 †.", "Cooling schedule and move size must be tuned to the budget.", "Single trajectory = limited exploration of the whole front.", "Archive dominance tests grow with archive size."],
 best="Problems with a cheap neighbourhood and small memory.", avoid="Parallel hardware / very smooth fronts where population methods are faster.")
B[27] = dict(
 what="DE/rand/1/bin; the trial replaces its target only if it dominates it; external archive stores non-dominated trials.",
 adv=[f"Excellent on P5 (HV {H(27,'P5')}, IGD {I(27,'P5')}) †.", f"Fast ({T(27,'P5')}) †.", "Only F and CR.", "Self-adapting step sizes.", "Archive keeps best solutions seen."],
 dis=[f"Collapsed to {N(27,'P3')} point on P3 (HV {H(27,'P3')}, IGD {I(27,'P3')}) †.", "'Only if dominates' keeps the target in every incomparable case, so diversity is not rewarded and the population drifts to one end of the front.", "Real variables only.", f"Archive after 10,000 evaluations: {N(27,'P3')} point on P3 vs {N(28,'P3')} for MO-TLBO †.", "Sensitive to F, CR."],
 best="Problems where one converged region is enough or with diversity added.", avoid="Fronts that need spread (P3).")
B[28] = dict(
 what="Teacher phase moves learners toward the best (first-front) teacher, learner phase interacts pairwise; NSGA-II-style ranking/crowding keeps N.",
 adv=[f"Algorithm-specific parameter-free (only N): P3 IGD {I(28,'P3')} †.", f"Dense fronts: {N(28,'P3')} of 100 points on P3 †.", f"Fast ({T(28,'P3')}) †.", "No crossover/mutation parameters.", "Elitist through ranking."],
 dis=[f"Uses two evaluations per learner, so overshoots the budget ({E(28,'P3')}) †.", "Real variables only.", "TF (shrinking factor) is a fixed heuristic.", f"P5 IGD {I(28,'P5')} is worse than MODE/MOSA ({I(27,'P5')}/{I(26,'P5')}) †.", "Teacher selection from the first front can bias towards the knee."],
 best="Continuous problems when no tuning is desired.", avoid="Discrete variables.")
B[29] = dict(
 what="Bats with frequency-driven velocity toward archive leaders, loudness decreasing and pulse rate increasing, local walk around leaders.",
 adv=[f"Very fast ({T(29,'P3')}) †.", f"Excellent on P4: Spacing {S(29,'P4')}, IGD {I(29,'P4')} †.", "Automatic switch from exploration to exploitation.", "Elitist archive.", "Few evaluations wasted."],
 dis=[f"Weaker on P3: HV {H(29,'P3')}, IGD {I(29,'P3')} (vs MO-TLBO {I(28,'P3')}) †.", "Many parameters (fmin, fmax, α, γ, r₀, A₀).", "Acceptance by dominance only; incomparable moves accepted randomly.", "Real variables only.", f"P3 front has {N(29,'P3')} points vs {N(24,'P3')} for MOABC †."],
 best="Smooth, low-dimensional continuous problems (P4).", avoid="Rugged fronts like ZDT2 with small budgets.")
B[30] = dict(
 what="NSGA-II global search plus Tchebycheff hill-climbing applied to a share of offspring (Lamarckian).",
 adv=[f"Better than plain NSGA-II on P5: IGD {I(30,'P5')} vs {I(14,'P5')} †.", f"Cheap local search: only {int(v(30,'P5','nfe')) - 10000} extra evaluations on P5 †.", "Works for binary (P9) and permutation (P8) problems via swap/flip moves.", "Keeps all NSGA-II advantages (elitism, crowding).", "Local step can repair infeasible designs."],
 dis=[f"Extra parameters: p_ls and L (see sensitivity study).", f"On P9 only {N(30,'P9')} non-dominated subsets and {T(30,'P9')} for 3,000 evaluations †.", "Local search can reduce diversity if applied too often.", "Random weights bias towards the middle of the front.", f"Local steps cost evaluations: {E(30,'P5')} used on P5 †."],
 best="Mixed global/local problems such as P5 and feature selection.", avoid="Very expensive evaluations (local steps multiply cost).")
B[31] = dict(
 what="Reference vectors partition the objective space; each vector keeps the solution with the smallest angle-penalised distance; vectors adapt to the objective range.",
 adv=[f"Strong on M = 5: HV {H(31,'P10_5')} (vs NSGA-II {H(14,'P10_5')}) †.", f"Very cheap ({T(31,'P10_3')} for 10,000 evaluations) †.", "Angle penalty trades convergence against diversity over time.", "Reference-vector adaptation handles scaled objectives.", "No sorting."],
 dis=[f"Population shrinks to the number of occupied vectors ({N(31,'P10_3')} points on M=3, {N(31,'P10_5')} on M=5) †.", f"IGD {I(31,'P10_3')} on M=3 is worse than NSGA-III's {I(15,'P10_3')} †.", "Needs the budget (t_max) for the penalty schedule.", "α and f_r are extra parameters.", "Designed for regular fronts."],
 best="Many-objective problems with a regular front (P10).", avoid="Few objectives where simpler methods are enough.")
B[32] = dict(
 what="Fitness = −Σexp(−I_ε+(y,x)/(cκ)) from pairwise additive-ε indicators; remove the worst until N remain; no explicit diversity step.",
 adv=[f"Simple, no sorting, no density estimator; P14 IGD {I(32,'P14')} †.", f"Fast ({T(32,'P14')}) †.", "Indicator directly encodes the preferred set quality.", f"Spread is good: Spacing {S(32,'P14')} on P14 †.", "Handles any number of objectives."],
 dis=[f"Fewer points on P3 ({N(32,'P3')}) with IGD {I(32,'P3')} †.", "κ must be tuned and the objective scaling matters (min-max of union).", "O(N²M) pairwise indicator matrix each generation.", "Biased toward the knee and boundaries of the ε-indicator.", f"P3 Spacing {S(32,'P3')} is worse than MO-TLBO's {S(28,'P3')} †."],
 best="Preference-free approximation of the front for 2–5 objectives (P14).", avoid="Large populations (memory N²).")
B[33] = dict(
 what="Steady-state (μ+1): one child per step; drop the individual with the smallest exclusive hypervolume contribution in the worst front.",
 adv=[f"Hypervolume-optimal subsets: HV {H(33,'P3')} on P3 and {H(33,'P14')} on P14 †.", f"Even spacing (Spacing {S(33,'P14')} on P14) †.", "No diversity parameter at all.", "Steady-state allows fine control of the budget.", "Theoretical link to HV maximisation."],
 dis=[f"Slow in Python: {T(33,'P3')} for 10,000 iterations †.", f"Only 50 points (population size) — {N(33,'P3')} on P3 †.", "Exact HV contribution is exponential for M ≥ 4 (needs Monte Carlo).", "Reference point choice affects the result.", "Sequential: hard to parallelise."],
 best="2–3 objectives where hypervolume is the target quality.", avoid="M ≥ 5 without an approximation.")
B[34] = dict(
 what="Monte-Carlo estimate of hypervolume-based fitness (weights α_j for k individuals to remove); tournament and survival by this fitness.",
 adv=[f"Works for many objectives where exact HV is impossible: M=5 HV {H(34,'P10_5')} †.", f"Large HV on M=3 too ({H(34,'P10_3')}) †.", "Sample count controls the accuracy/time trade-off.", "No diversity parameter.", "Fine-grained fitness for dominated points."],
 dis=[f"Noisy fitness; slow ({T(34,'P10_5')} for 10,000 evaluations at M=5) †.", f"IGD {I(34,'P10_5')} on M=5 shows HV-optimality ≠ uniform coverage (solutions pushed toward the boundary) †.", "Needs a reference point and sample budget.", "Results vary with the random samples.", "Cost grows with the number of samples × population."],
 best="Many-objective problems when HV is the quality measure.", avoid="Very tight time limits.")
B[35] = dict(
 what="NSGA-II where crowding distance is replaced by the preference rank (distance to the DM reference point) with ε-clearing.",
 adv=[f"Concentrates the population near the DM's point: P14 front spans {FRONTS[(35,'P14')][:,0].min():.2f}–{FRONTS[(35,'P14')][:,0].max():.2f} kg vs {FRONTS[(32,'P14')][:,0].min():.2f}–{FRONTS[(32,'P14')][:,0].max():.2f} kg for IBEA †.", f"Very even inside the focus (Spacing {S(35,'P5')} on P5) †.", "Saves evaluations in regions the DM does not care about.", "Several reference points allowed.", "Inherits NSGA-II's elitism."],
 dis=[f"If the reference point is badly placed the focus region is wrong (P14 point is a dominated, attainable point) †.", f"Worse IGD against the full front on P5 ({I(35,'P5')}) – by design †.", "ε must be tuned (too large → no focus, too small → clustering).", "Does not reveal the rest of the front.", f"Slower than NSGA-II ({T(35,'P5')}) † because of the Python ε-clearing loop."],
 best="DM has an aspiration level (P14).", avoid="Exploratory phase when the DM has no preference yet.")
B[36] = dict(
 what="Find α in the simplex minimising ‖Σαᵢgᵢ‖; step along −Σαᵢgᵢ, a common descent direction for all tasks.",
 adv=[f"Converges to the Pareto set: from (0,5) it reached {np.round(STUDY['p13_toy']['mgda'], 2).tolist()} (x₁=x₂) †.", "No weights to tune; closed form for 2 tasks.", "Guarantees non-increase of all objectives to first order.", f"Both network losses fell in the balanced case: cls {p13[(1.0,'mgda')][2]:.2f}→{p13[(1.0,'mgda')][3]:.2f}, reg {p13[(1.0,'mgda')][4]:.2f}→{p13[(1.0,'mgda')][5]:.2f} †.", "Stops exactly at Pareto-stationary points."],
 dis=["Needs differentiable objectives and gradients of every task at every step.", f"With regression targets ×20 the verdict was '{p13[(20.0,'mgda')][6]}' (cls {p13[(20.0,'mgda')][2]:.2f}→{p13[(20.0,'mgda')][3]:.2f}, reg {p13[(20.0,'mgda')][4]:.1f}→{p13[(20.0,'mgda')][5]:.1f}) †.", "Large-gradient tasks dominate the min-norm direction unless gradients are normalised.", "Returns one Pareto-stationary point per run, not a whole front.", f"On the toy it stops at the nearest Pareto point {np.round(STUDY['p13_toy']['mgda'], 2).tolist()} rather than a chosen trade-off †."],
 best="Multi-task learning with comparable gradient scales.", avoid="Non-differentiable objectives.")
B[37] = dict(
 what="Learn task weights wᵢ so that weighted gradient norms match a target set by the relative inverse training rates (restoring force α).",
 adv=[f"Balanced both tasks when gradients were 20× imbalanced: cls {p13[(20.0,'gradnorm')][2]:.2f}→{p13[(20.0,'gradnorm')][3]:.2f}, reg {p13[(20.0,'gradnorm')][4]:.1f}→{p13[(20.0,'gradnorm')][5]:.1f} †.", f"Equal-weight baseline on the same data: '{p13[(20.0,'equal')][6]}' †.", "Cheap: only gradient norms at the shared layer.", "Learns weights online; works with any number of tasks.", "Weights stay normalised (Σw = M)."],
 dis=[f"Toy P13a ended at {np.round(STUDY['p13_toy']['gradnorm'], 2).tolist()} not at the mid-point — weights depend on the history †.", "Extra hyper-parameters α and the weight learning rate.", "Needs per-task gradient norms every iteration.", "Does not guarantee Pareto-stationarity (unlike MGDA).", f"Balanced-scale network: regression loss ends at {p13[(1.0,'gradnorm')][5]:.2f} vs MGDA {p13[(1.0,'mgda')][5]:.2f} †."],
 best="Multi-task networks with very different loss scales.", avoid="When a descent guarantee is required.")
B[20]['dis'][0] = f"Slower than NSGA-II in wall-clock ({T(20,'P8')} vs {T(14,'P8')} for NSGA-II on P8) †."
def n_ev(b): return sum('†' in x for x in b['adv']), sum('†' in x for x in b['dis'])
FAM = [("Classical / MCDM", range(1, 10)), ("Non-elitist evolutionary", range(10, 14)), ("Elitist evolutionary & decomposition", range(14, 21)), ("Constrained & swarm", range(21, 26)), ("Physics / evolutionary / human-inspired", range(26, 30)), ("Hybrid, many-objective, indicator, preference", range(30, 36)), ("Gradient-based", range(36, 38))]
for fam, rng_ in FAM:
    md(f"### {fam}")
    rows = [[no, NAMES[no], B[no]['what'], "<br>".join("• " + a for a in B[no]['adv']), "<br>".join("• " + a for a in B[no]['dis']), B[no]['best'], B[no]['avoid']] for no in rng_]
    md_table(["#", "algorithm", "what it does (my words)", "advantages", "disadvantages", "best choice when", "avoid when"], rows)
bad = [(no, n_ev(B[no])) for no in B if min(n_ev(B[no])) < 2 or len(B[no]['adv']) < 4 or len(B[no]['dis']) < 4]
print("Rows violating the rule (>=4 each, >=2 evidenced each):", bad); assert not bad

# %% [markdown]
# **How the families differ (half-page summary).**
#
# *Diversity.* The classical scalarisation methods get diversity only from the weight/ε grid, and weighted sum bunches points at the ends (21 weights gave 17 distinct points on P4, 2 on P3). Non-elitist GAs use sharing (MOGA, NSGA, NPGA) or nothing at all (VEGA collapsed to one point on P3). Elitist methods keep diversity explicitly: crowding distance (NSGA-II, Spacing ≈ 0.01), nearest-neighbour density (SPEA2), grid boxes (PESA-II, MOPSO, MOGWO), weight vectors (MOEA/D), reference points/vectors (NSGA-III, RVEA) or the indicator itself (IBEA, SMS-EMOA, HypE). Crowding and grids stop working beyond three objectives, which is why the reference-point and indicator families exist.
#
# *Convergence.* Population methods with elitism converge reliably in 10,000 evaluations on two objectives; the simplest swarm/DE/TLBO methods are often the fastest per evaluation. Non-elitist methods lose the best front between generations. Gradient methods converge only to Pareto-stationary points and need gradients.
#
# *Constraints.* Classical methods pass constraints to the scalar solver; Constrained NSGA-II uses Deb's rules; most others fall back on a static penalty. In our P5/P6/P7 runs the penalty worked, but it needs a scale factor, whereas Deb's rules do not.
#
# *Non-convex fronts.* Weighted sum (and p = 1) cannot reach them; ε-constraint, Tchebycheff (p = ∞), and every dominance-, decomposition- or indicator-based population method can. *Many objectives.* Pareto ranking (NSGA-II) fails at M = 5 because about half of a random population is already non-dominated; NSGA-III, RVEA, MOEA/D and HypE keep selection pressure with reference points, angles, scalarisation or hypervolume.
