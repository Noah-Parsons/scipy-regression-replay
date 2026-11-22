# SciPy regression replay

Noah Parsons. Results as of 21 November 2026.

## Question

When SciPy changes, through rewrites, the replacement of old Fortran with C, or new back ends such as Boost, which answers change, and which of those changes are silent regressions and which are fixes?

## Method

- **`replay.py`** runs a fixed corpus of 351 cases through the installed SciPy and saves every output. The corpus covers:
  - special functions over wide argument ranges;
  - distribution tails (cdf, sf, logcdf, logsf, ppf, isf) for 30 distributions;
  - statistical tests;
  - ODE solvers (`solve_ivp`, `odeint`, `ode`) on stiff and non-stiff problems;
  - quadrature, optimisation, root finding and linear algebra.

  A case that raises an exception records the exception instead of stopping the run.
- **`run_all.sh`** builds one isolated environment per release, with a NumPy of the same period, and runs the corpus on 16 releases from 1.3.3 (2019) to 1.18.1. All 16 ran. 20 to 25 cases raise an exception in each release, mostly deliberate skips and functions not yet present in older releases.
- **`diff.py`** compares consecutive releases case by case (`changes.csv`), and **`track.py`** follows the most-changed element of each case through all 16 releases (`tracks.csv`).
- **`check_findings.py`** checks the main findings against independent references computed in mpmath at 50 digits. It runs on any installed SciPy.

## Results

226 case-to-release transitions change by more than 1e-12, and 100 cases move by more than 1e-6 at some point. Most of these are improvements, especially in distribution tails: values that used to underflow to 0 or overflow to infinity are now right. The changes that are not improvements are below.

### 1. New, apparently unreported: `integrate.ode` with LSODA returns the same array from every call (SciPy 1.17.0 and later)

Since 1.17.0, `ode(...).set_integrator("lsoda").integrate(t)` returns the integrator's internal state array itself, not a copy. Any result a program keeps is silently overwritten by later calls:

```python
from scipy.integrate import ode
r = ode(lambda t, y: [-y[0]]).set_integrator("lsoda")
r.set_initial_value([1.0], 0.0)
a = r.integrate(1.0)       # exp(-1) = 0.3679
b = r.integrate(2.0)
print(a is b, a)           # 1.16.3: False [0.36788]   1.17.0 and 1.18.1: True [0.13534]
```

`vode`, `zvode`, `dopri5` and `dop853` still return fresh arrays. The usual pattern, a loop that appends `r.integrate(t)` to a list, therefore gives a trajectory in which every row is the final state. In the replay, this turned the Robertson problem's first output, y₃(10⁻⁵) = 4.8 × 10⁻¹¹, into 0.982. The change arrives with 1.17.0, the release in which another `ode` integrator also changed behaviour: scipy/scipy#24933, a transposed Jacobian in `vode`.

A search of SciPy's issue tracker on 21 November 2026 found no report of it.

### 2. Already reported: `stats.t.ppf` returns +∞ for very small p (1.17 and later)

`t.ppf(1e-300, 3.5)` gives +∞; the reference is −6.1 × 10⁸⁵, so the answer has the wrong sign. Releases up to 1.16 gave −1.6 × 10⁸⁶. This is open as scipy/scipy#25968.

### 3. Long-standing: `stats.invgauss.isf` fails beyond about p = 10⁻⁶⁰ (μ = 0.6)

| p | SciPy (1.10 to 1.18) | Reference |
|---|---|---|
| 10⁻⁵⁰ | 78.47 | 78.47 |
| 10⁻¹⁰⁰ | 2.3 × 10³⁴ | 160.6 |
| 10⁻³⁰⁰ | 2.5 × 10²³³ | 491 |

Boost's root finder fails here and SciPy returns its last guess. A RuntimeWarning is raised, but the value returned is wrong by about 230 orders of magnitude. Releases before 1.10 returned 100.

### 4. `stats.rice.sf` in the right tail (b = 1.5): wrong before 1.17, zero since

| x | 1.16.3 | 1.17 and later | Reference |
|---|---|---|---|
| 8 | 9.46e-11 | 9.46e-11 | 9.46e-11 |
| 10 | 4.33e-15 | 0 | 2.48e-17 |
| 14.4 | 4.33e-15 | 0 | 8.59e-38 |

Before 1.17 it lost all precision, from computing 1 − cdf. Since 1.17 it underflows to 0, and so `logsf` gives −∞, even though the true values are well within double precision.

### 5. A regression already fixed: `stats.skewnorm.ppf` and `isf` (1.10 to 1.13)

From 1.10 to 1.13, `skewnorm.ppf` returned values such as −1.0 × 10⁸⁰ in the far left tail, and `isf` returned 2.3 × 10³⁴. Release 1.14 corrected both.

### Expected changes

- **`integrate.romberg` was removed in 1.15,** so it shows as NaN, as expected.
- **Solver step and evaluation counts change between releases,** for example `solve_ivp` LSODA and BDF on van der Pol in 1.13. These are counts, not answers.
- **`odeint` on the stiff van der Pol problem (μ = 1000) changes by 14% at one time point in 1.13.** That point lies on a near-vertical relaxation jump, where tolerance-sized differences in the jump's timing give large differences in value. It is not a defect.

## Status

As of 21 November 2026, findings 1, 3 and 4 have not been reported to SciPy. Finding 2 is open as scipy/scipy#25968.

## Reproducing

The replay needs Linux and [uv](https://docs.astral.sh/uv/). `run_all.sh` builds a temporary environment for each release, runs the corpus and deletes the environment. Allow about an hour.

```
bash run_all.sh            # writes out/scipy-<version>.npz for 16 releases
python diff.py             # changes.csv: changes between consecutive releases
python track.py            # tracks.csv: each changed case through all releases
python check_findings.py   # re-checks the findings below on the installed SciPy, against mpmath
```

Python packages: numpy, scipy, pandas, mpmath.

## Files

| File | Contents |
|---|---|
| `replay.py`, `run_all.sh`, `run_all.log` | Corpus and runner |
| `out/scipy-*.npz` | Outputs from each release |
| `diff.py`, `changes.csv`, `track.py`, `tracks.csv` | Comparisons |
| `check_findings.py` | Independent checks of the main findings |
