"""For every case whose output moved by more than 1e-6 between consecutive releases, follow the
most-changed element through all 16 releases. Writes tracks.csv."""
import glob, re, numpy as np, pandas as pd
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def vkey(p): return tuple(int(x) for x in re.findall(r"scipy-([\d.]+)\.npz", p)[0].split("."))
files = sorted(glob.glob("out/scipy-*.npz"), key=vkey)
vers = [re.findall(r"scipy-([\d.]+)\.npz", f)[0] for f in files]
D = [np.load(f) for f in files]
cases = sorted(set.intersection(*[set(d.files) for d in D]) - {"__meta__"})
rows = []
for c in cases:
    arrs = [d[c] for d in D]
    if any(a.dtype.kind == "U" for a in arrs) or len({a.shape for a in arrs}) > 1:
        continue
    A = np.stack([a.ravel() for a in arrs]).astype(complex if np.iscomplexobj(arrs[0]) else float)
    with np.errstate(all="ignore"):
        scale = np.maximum(np.abs(A[1:]), np.abs(A[:-1]))
        d = np.abs(A[1:] - A[:-1]) / np.maximum(scale, 1e-300)
        d[~np.isfinite(d)] = 0
        both_inf_change = (np.isfinite(A[1:]) != np.isfinite(A[:-1]))
        d[both_inf_change] = np.inf
    if d.max() <= 1e-6:
        continue
    j = int(np.unravel_index(np.argmax(d), d.shape)[1])
    rows.append(dict(case=c, index=j, **{v: A[k, j] for k, v in enumerate(vers)}))
T = pd.DataFrame(rows); T.to_csv("tracks.csv", index=False)
pd.set_option("display.width", 400, "display.max_rows", 300, "display.float_format", "{:.4g}".format)
print(len(T), "cases"); print(T.to_string())
