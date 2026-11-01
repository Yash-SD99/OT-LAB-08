import re, glob, json, nbformat as nbf
order = [l for l in open('run_all.sh').read().split() if l.startswith('../src/')]
cells = []
for f in order:
    txt = open(f).read(); parts = re.split(r'^# %%(.*)$', txt, flags=re.M)
    head = parts[0].strip()
    if head: cells.append(nbf.v4.new_code_cell(head))
    for tag, body in zip(parts[1::2], parts[2::2]):
        if 'markdown' in tag:
            lines = [l[2:] if l.startswith('# ') else (l[1:] if l.startswith('#') else l) for l in body.strip('\n').split('\n')]
            cells.append(nbf.v4.new_markdown_cell('\n'.join(lines)))
        elif body.strip(): cells.append(nbf.v4.new_code_cell(body.strip('\n')))
nb = nbf.v4.new_notebook(); nb.cells = cells; nb.metadata['kernelspec'] = dict(name='python3', display_name='Python 3', language='python')
nbf.write(nb, '../notebooks/Lab7_MOO.ipynb'); print(len(cells), 'cells')
