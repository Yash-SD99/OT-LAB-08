import nbformat as nbf, time
from nbclient import NotebookClient
nb = nbf.read('../notebooks/Lab7_MOO.ipynb', 4); t = time.time()
c = NotebookClient(nb, timeout=7200, kernel_name='python3', resources={'metadata': {'path': '.'}}, allow_errors=True)
try: c.execute()
finally:
    nbf.write(nb, '../notebooks/Lab7_MOO_executed.ipynb')
    errs = [(i, o.ename, o.evalue[:200]) for i, cl in enumerate(nb.cells) if cl.cell_type == 'code' for o in cl.get('outputs', []) if o.output_type == 'error']
    print('done in %.0fs; errors:' % (time.time() - t), errs)
