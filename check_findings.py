"""Reproduce the findings in README.md on the installed SciPy, against mpmath references.
Run under SciPy 1.17 or later to see the current behaviour; under 1.16.3 for the earlier one."""
import warnings
import numpy as np, scipy, mpmath as mp
from scipy import integrate as si, stats as st
warnings.filterwarnings("ignore")
mp.mp.dps = 50
print("SciPy", scipy.__version__)

# 1. integrate.ode with LSODA: integrate() returns the same array object each call
r = si.ode(lambda t, y: [-y[0]]).set_integrator("lsoda")
r.set_initial_value([1.0], 0.0)
a = r.integrate(1.0); b = r.integrate(2.0)
print(f"1. lsoda: same object returned twice: {a is b}; first result now {a[0]:.6f} "
      f"(should be exp(-1) = {np.exp(-1):.6f})")

# 2. t.ppf for tiny p (df = 3.5): sign of the answer
nu = mp.mpf("3.5")
tcdf = lambda x: 0.5 * mp.betainc(nu / 2, mp.mpf(1) / 2, 0, nu / (nu + mp.mpf(x) ** 2), regularized=True)
for p in (1e-200, 1e-250, 1e-300):
    lo, hi = mp.mpf(1), mp.mpf("1e200")
    for _ in range(400):
        mid = mp.sqrt(lo * hi)
        lo, hi = (mid, hi) if tcdf(-mid) > p else (lo, mid)
    print(f"2. t.ppf({p:g}, 3.5) = {st.t.ppf(p, 3.5):.4g}; reference {-float(lo):.4g}")

# 3. invgauss.isf far in the tail (mu = 0.6)
mu = mp.mpf("0.6")
igsf = lambda x: mp.ncdf(-mp.sqrt(1 / x) * (x / mu - 1)) - mp.exp(2 / mu) * mp.ncdf(-mp.sqrt(1 / x) * (x / mu + 1))
for p in (1e-50, 1e-100, 1e-300):
    lo, hi = mp.mpf(1), mp.mpf(10000)
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if igsf(mid) > p else (lo, mid)
    print(f"3. invgauss.isf({p:g}, 0.6) = {st.invgauss.isf(p, 0.6):.4g}; reference {float(lo):.4g}")

# 4. rice.sf in the right tail (b = 1.5)
b = mp.mpf("1.5")
pdf = lambda t: t * mp.exp(-(t * t + b * b) / 2) * mp.besseli(0, t * b)
for x in (8.0, 10.0, 14.38449888287663):
    ref = mp.quad(pdf, [x, x + 5, x + 20, mp.inf])
    print(f"4. rice.sf({x:.4g}, 1.5) = {st.rice.sf(x, 1.5):.4g}; reference {float(ref):.4g}")
