import sys, re, time
# execute source files in order inside one namespace (cells separated by '# %%')
files = sys.argv[1:]
ns = {'__name__': '__main__'}
import matplotlib; matplotlib.use('Agg')
for f in files:
    src = open(f).read()
    import os
    if os.environ.get('LAB7_QUICK') and 'QUICK = False' in src: src = src.replace('QUICK = False','QUICK = True',1)
    t = time.time()
    exec(compile(src, f, 'exec'), ns)
    print(f"-- {f} done in {time.time()-t:.1f}s")
