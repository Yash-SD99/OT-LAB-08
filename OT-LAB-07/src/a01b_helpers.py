# %% [markdown]
# ## A2 · Shared run / plot / log helpers

# %%
FRONTS, XFR, STUDY = {}, {}, {}   # obtained fronts / decision vectors / numbers measured in the problem studies (used as evidence in Task B)
RES = {}     # (alg_no, problem key) -> metrics dict, filled by every run; used for the implementation log (A3) and Task B evidence

def score(prob, F):
    if len(F) == 0: return dict(hv=0., igd=np.inf, sp=np.nan, n=0)
    Fn = prob.norm(F); Zn = prob.Zn
    return dict(hv=hypervolume(Fn, prob.ref), igd=(igd(Zn, Fn) if Zn is not None else np.nan), sp=spacing(Fn), n=len(F))

def draw(ax, prob, F, title, color='C3'):
    if prob.M == 2:
        if prob.Z is not None:
            Z = prob.Z[np.argsort(prob.Z[:, 0])]; ax.plot(Z[:, 0], Z[:, 1], '-', c='0.6', lw=3, alpha=.6, label='true/ref front', zorder=1)
        if len(F): ax.scatter(F[:, 0], F[:, 1], s=14, c=color, zorder=2, label='obtained (%d)' % len(F))
        ax.set_xlabel('f1'); ax.set_ylabel('f2'); ax.legend(fontsize=7)
    elif prob.M == 3:
        if len(F): ax.scatter(F[:, 0], F[:, 1], F[:, 2], s=8, c=color)
        ax.set_xlabel('f1'); ax.set_ylabel('f2'); ax.set_zlabel('f3')
    else:
        idx = np.random.default_rng(0).choice(len(F), min(40, len(F)), replace=False) if len(F) else []
        for i in idx: ax.plot(range(1, prob.M + 1), F[i], c='C0', alpha=.35)
        ax.set_xlabel('objective'); ax.set_ylabel('value')
    ax.set_title(title, fontsize=9)

def show(entries, suptitle=None):
    """entries: list of (title, prob, F). One row of subplots."""
    k = len(entries); fig = plt.figure(figsize=(4.6 * k, 3.6))
    for i, (t, p, F) in enumerate(entries):
        ax = fig.add_subplot(1, k, i + 1, projection='3d' if p.M == 3 else None); draw(ax, p, F, t)
    if suptitle: fig.suptitle(suptitle, fontsize=10)
    plt.tight_layout(); plt.show()

def log(no, key, prob, F, t, nfe, status='works'):
    s = score(prob, F); s.update(time=t, nfe=nfe, status=status); RES[(no, key)] = s; return s

def line(no, name, key, s): print(f"[{no:>2}] {name:<14} {key:<7} n={s['n']:<4} HV={s['hv']:.4f}  IGD={s['igd']:.4f}  Spacing={s['sp']:.4f}  nfe={s['nfe']}  t={s['time']:.1f}s")


def pmap(fn, arglist, procs=2, timeout=3600):
    """Run fn(*args) for every args in arglist on `procs` forked worker processes (falls back to a serial loop). Results come back in order."""
    arglist = list(arglist)
    if QUICK or procs <= 1 or len(arglist) < 2 or not hasattr(__import__('os'), 'fork'): return [fn(*a) for a in arglist]
    try:
        import multiprocessing as mp; ctx = mp.get_context('fork'); q = ctx.Queue()
        def work(ids):
            for i in ids:
                try: q.put((i, fn(*arglist[i])))
                except Exception as e: q.put((i, e))
        ps = [ctx.Process(target=work, args=(list(range(k, len(arglist), procs)),)) for k in range(procs)]; [p.start() for p in ps]
        out = [None] * len(arglist)
        for _ in arglist: i, r = q.get(timeout=timeout); out[i] = r
        [p.join() for p in ps]
        if any(isinstance(r, Exception) for r in out): raise RuntimeError('worker failed')
        return out
    except Exception: return [fn(*a) for a in arglist]
