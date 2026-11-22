"""Diff the replay outputs of consecutive SciPy releases.

For each case and each pair of consecutive releases, report the largest relative change among
finite values, plus changes in where the output is NaN, infinite or an error. Writes
changes.csv (every change above 1e-12 relative) and prints the large ones.
"""
import glob, re, sys
import numpy as np
import pandas as pd


def vkey(path):
    return tuple(int(x) for x in re.findall(r"scipy-([\d.]+)\.npz", path)[0].split("."))


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
files = sorted(glob.glob("out/scipy-*.npz"), key=vkey)
data = [(re.findall(r"scipy-([\d.]+)\.npz", f)[0], np.load(f)) for f in files]


def compare(a, b):
    if a.dtype.kind == "U" or b.dtype.kind == "U":
        sa, sb = str(a), str(b)
        if sa == sb:
            return None
        return dict(kind="error-state", detail=f"{sa[:80]} -> {sb[:80]}", rel=np.inf)
    if a.shape != b.shape:
        return dict(kind="shape", detail=f"{a.shape} -> {b.shape}", rel=np.inf)
    a, b = a.ravel(), b.ravel()
    na, nb = ~np.isfinite(a), ~np.isfinite(b)
    detail = []
    if (np.isnan(a) != np.isnan(b)).any():
        detail.append(f"NaN pattern changed at {int((np.isnan(a) != np.isnan(b)).sum())} points")
    if (np.isinf(a) != np.isinf(b)).any() or (np.isinf(a) & np.isinf(b) & (np.sign(a) != np.sign(b))).any():
        detail.append(f"inf pattern changed at {int((np.isinf(a) != np.isinf(b)).sum())} points")
    both = ~na & ~nb
    rel = 0.0
    worst = None
    if both.any():
        d = np.abs(a[both] - b[both]) / np.maximum(np.maximum(np.abs(a[both]), np.abs(b[both])), 1e-300)
        i = int(np.argmax(d))
        rel = float(d[i])
        worst = (complex(a[both][i]) if np.iscomplexobj(a) else float(a[both][i]),
                 complex(b[both][i]) if np.iscomplexobj(b) else float(b[both][i]))
    if rel <= 1e-12 and not detail:
        return None
    return dict(kind="value" if not detail else "pattern", rel=rel, worst=worst, detail="; ".join(detail),
                n_changed=int(((np.abs(a - b) > 1e-12 * np.maximum(np.abs(a), np.abs(b))) & np.isfinite(a)
                               & np.isfinite(b)).sum()))


rows = []
for (v0, d0), (v1, d1) in zip(data, data[1:]):
    for case in sorted(set(d0.files) | set(d1.files)):
        if case == "__meta__":
            continue
        if case not in d0.files or case not in d1.files:
            continue
        c = compare(d0[case], d1[case])
        if c:
            rows.append(dict(case=case, frm=v0, to=v1, **c))
df = pd.DataFrame(rows)
df.to_csv("changes.csv", index=False)
print(f"{len(data)} releases, {len(df)} case-transitions changed above 1e-12")
if df.empty:
    sys.exit()
big = df[(df.rel > 1e-6) & (df.kind != "error-state")]
pd.set_option("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 300)
print(big[["case", "frm", "to", "kind", "rel", "worst", "detail"]].to_string())
