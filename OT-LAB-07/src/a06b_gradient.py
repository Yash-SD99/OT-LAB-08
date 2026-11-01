# %% [markdown]
# ## Family: gradient-based multi-task methods (algorithms 36–37) on P13
# **(a) Toy:** f1 = ‖x−a‖², f2 = ‖x−b‖², a=(3,3), b=(−3,−3), start (0,5); Pareto set = segment a–b.
# **(b) Network:** a shared body (tanh layer) with a classification head (logistic loss) and a regression head (MSE), forward/backward written by hand in NumPy.

# %%
def mgda_alpha(g1, g2):
    """Alg 36 core: closed-form min-norm point of the segment [g2, g1]:  alpha* = clip( (g2-g1).g2 / ||g1-g2||^2 , 0, 1 ),  d = alpha g1 + (1-alpha) g2."""
    d = g1 - g2; den = float(d @ d); return 0.5 if den < 1e-18 else float(np.clip(((g2 - g1) @ g2) / den, 0, 1))
def min_norm(G, iters=200):
    """Frank-Wolfe min-norm point in the convex hull of the rows of G (any number of tasks)."""
    M = len(G); a = np.ones(M) / M; K = G @ G.T
    for _ in range(iters):
        t = int(np.argmin(K @ a)); e = np.zeros(M); e[t] = 1; d = e - a; den = d @ K @ d
        if den < 1e-18: break
        a = a + float(np.clip(-(a @ K @ d) / den, 0, 1)) * d
    return a
def mgda_step(grads):
    """Alg 36 MGDA (Désidéri 2012; Sener & Koltun 2018). Steps: (1) gradient g_i of every objective; (2) find weights alpha in the simplex minimising ||sum alpha_i g_i||; (3) step along -d, d=sum alpha_i g_i
    (a common descent direction, d=0 <=> Pareto-stationary)."""
    G = np.asarray(grads); a = np.array([mgda_alpha(G[0], G[1]), 1 - mgda_alpha(G[0], G[1])]) if len(G) == 2 else min_norm(G); return a, a @ G

def gradnorm_update(w, G, L, L0, alpha=1.5, lr=.025):
    """Alg 37 core. G_i = ||w_i grad_W L_i||, r_i = (L_i/L_i(0)) / mean(.), target_i = mean(G) r_i^alpha; w_i <- w_i - lr * sign(G_i - target_i) * ||grad_W L_i|| ; renormalise sum w = M.
    A task whose gradient is too large (G_i > target) has its weight decreased."""
    G = np.asarray(G); gn = G / np.maximum(w, 1e-12); Gw = w * gn; Lt = np.asarray(L) / np.asarray(L0); r = Lt / Lt.mean(); tgt = Gw.mean() * r ** alpha
    w = np.maximum(w - lr * np.sign(Gw - tgt) * gn, 1e-3); return w * len(w) / w.sum()

# ---- (a) toy
A_, B_ = np.array([3., 3.]), np.array([-3., -3.])
def toy_grads(x): return np.array([2 * (x - A_), 2 * (x - B_)])
def toy_run(method, x0=(0., 5.), eta=.05, iters=200):
    x = np.array(x0, float); w = np.ones(2); L0 = None; path = [x.copy()]
    for _ in range(iters):
        g = toy_grads(x); L = np.array([((x - A_) ** 2).sum(), ((x - B_) ** 2).sum()]); L0 = L if L0 is None else L0
        if method == 'equal': d = .5 * g[0] + .5 * g[1]
        elif method == 'mgda': d = mgda_step(g)[1]
        else: w = gradnorm_update(w, np.linalg.norm(w[:, None] * g, axis=1), L, L0); d = (w[:, None] * g).sum(0) / 2
        x = x - eta * d; path.append(x.copy())
    return np.array(path)

# ---- (b) network
class Net:
    def __init__(s, d=8, h=16, seed=0, reg_scale=1., n=400):
        r = np.random.default_rng(seed); s.X = r.normal(size=(n, d)); s.yc = ((s.X[:, 0] * s.X[:, 1] + np.sin(s.X[:, 2])) > 0).astype(float)
        s.yr = reg_scale * (s.X[:, 3] + .5 * s.X[:, 4] ** 2 - s.X[:, 5]) + .1 * r.normal(size=n)
        s.W1 = r.normal(size=(d, h)) * .3; s.b1 = np.zeros(h); s.wc = r.normal(size=h) * .3; s.bc = 0.; s.wr = r.normal(size=h) * .3; s.br = 0.
    def forward(s):
        s.H = np.tanh(s.X @ s.W1 + s.b1); z = s.H @ s.wc + s.bc; s.p = 1 / (1 + np.exp(-z)); s.r = s.H @ s.wr + s.br; n = len(s.yc)
        L1 = -np.mean(s.yc * np.log(s.p + 1e-12) + (1 - s.yc) * np.log(1 - s.p + 1e-12)); L2 = np.mean((s.r - s.yr) ** 2); return np.array([L1, L2])
    def backward(s):
        """Returns (shared-parameter gradients [2 x P], head gradients (wc,bc),(wr,br)) -- hand-coded backprop."""
        n = len(s.yc); dz = (s.p - s.yc) / n; dr = 2 * (s.r - s.yr) / n; gh = [];
        for dh in (np.outer(dz, s.wc), np.outer(dr, s.wr)):
            dpre = dh * (1 - s.H ** 2); gh.append(np.r_[(s.X.T @ dpre).ravel(), dpre.sum(0)])
        return np.array(gh), (s.H.T @ dz, dz.sum()), (s.H.T @ dr, dr.sum())
    def apply(s, d_shared, wts, hc, hr, lr):
        d, h = s.W1.shape; s.W1 -= lr * d_shared[:d * h].reshape(d, h); s.b1 -= lr * d_shared[d * h:]
        s.wc -= lr * wts[0] * hc[0]; s.bc -= lr * wts[0] * hc[1]; s.wr -= lr * wts[1] * hr[0]; s.br -= lr * wts[1] * hr[1]
def net_train(method, reg_scale=1., iters=500, lr=.05, seed=0):
    net = Net(seed=seed, reg_scale=reg_scale); w = np.ones(2); L0 = None; hist = []
    for it in range(iters):
        L = net.forward(); L0 = L.copy() if L0 is None else L0; hist.append(L.copy()); G, hc, hr = net.backward()
        if not np.all(np.isfinite(L)): break
        if method == 'equal': d = G[0] + G[1]; wts = (1., 1.)
        elif method == 'mgda': a, d = mgda_step(G); d = 2 * d; wts = (1., 1.)           # alpha-combination (scaled by M) for the shared body; heads follow their own losses
        else:
            w = gradnorm_update(w, np.linalg.norm(w[:, None] * G, axis=1), L, L0); d = (w[:, None] * G).sum(0); wts = tuple(w)
        net.apply(d, wts, hc, hr, lr)
    return np.array(hist)

# %% [markdown]
# ### Run: toy (a) and network (b), balanced and imbalanced (regression targets ×20 ⇒ much larger gradients)

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8)); ends = {}
for m, c in (('equal', 'C0'), ('mgda', 'C3'), ('gradnorm', 'C2')):
    for x0, ls in (((0., 5.), '-'), ((5., -2.), '--'), ((-4., 1.), ':')):
        pth = toy_run(m, x0); ax[0].plot(pth[:, 0], pth[:, 1], ls, c=c, label=m if x0 == (0., 5.) else None); ax[0].plot(*pth[-1], 'o', c=c)
    ends[m] = toy_run(m)[-1]
ax[0].plot([3, -3], [3, -3], 'k-', lw=4, alpha=.3, label='Pareto set'); ax[0].set_title('P13a toy: paths from 3 starts'); ax[0].legend(fontsize=7)
for sc, a_ in ((1., ax[1]), (20., ax[2])):
    for m, c in (('equal', 'C0'), ('mgda', 'C3'), ('gradnorm', 'C2')):
        h = net_train(m, sc); a_.plot(h[:, 0] / h[0, 0], c=c, ls='-', label=m + ' cls'); a_.plot(h[:, 1] / h[0, 1], c=c, ls='--', label=m + ' reg')
    a_.set_yscale('log'); a_.set_title('P13b network, regression scale ×%g\n(loss / initial loss; solid=cls, dashed=reg)' % sc, fontsize=9); a_.legend(fontsize=6, ncol=2); a_.set_xlabel('iteration')
plt.tight_layout(); plt.show()
rows = []
for sc in (1., 20.):
    for m in ('equal', 'mgda', 'gradnorm'):
        h = net_train(m, sc); rows.append([sc, m, h[0, 0], h[-1, 0], h[0, 1], h[-1, 1], 'both fall' if (h[-1] < h[0]).all() else 'NOT both'])
STUDY['p13_rows'] = rows; STUDY['p13_toy'] = {m: v.tolist() for m, v in ends.items()}
md_table(["reg scale", "method", "cls loss start", "cls loss end", "reg loss start", "reg loss end", "verdict"], rows)
print("Toy end points (start (0,5)):", {m: np.round(v, 3).tolist() for m, v in ends.items()}, "-> Pareto set is x1 = x2")
for no in (36, 37): RES[(no, 'P13')] = dict(hv=np.nan, igd=np.nan, sp=np.nan, n=1, time=0., nfe=0, status='works')
