"""Run a fixed corpus of numerical inputs through the installed SciPy and save every output.

Usage: python replay.py OUT.npz
The same script runs unchanged on SciPy 1.3 to 1.17. Each case is isolated: a case that
raises records the exception class instead of an array, so API gaps don't stop the run.
"""
import sys, warnings, traceback
import numpy as np
import scipy
warnings.filterwarnings("ignore")

OUT = {}
import os
VERBOSE = bool(os.environ.get('REPLAY_VERBOSE'))


def case(name):
    def deco(f):
        if VERBOSE:
            print(name, flush=True)
        try:
            r = f()
            OUT[name] = np.asarray(r, dtype=np.complex128 if np.iscomplexobj(r) else np.float64)
        except Exception as e:  # recorded, not fatal
            OUT[name] = np.array(f"ERROR {type(e).__name__}: {str(e)[:120]}")
        return f
    return deco


from scipy import special as sc, stats as st, integrate as si, optimize as so, linalg as sl
from scipy import interpolate as sp_int

# ---------------------------------------------------------------- special functions
x_pos = np.concatenate([np.logspace(-300, -1, 40), np.linspace(0.1, 50, 60), np.logspace(2, 300, 40)])
x_sym = np.concatenate([-x_pos[::-1], [0.0], x_pos])
x_mod = np.concatenate([np.logspace(-8, -1, 15), np.linspace(0.1, 40, 80), np.logspace(1.7, 3, 15)])
orders = [0.0, 0.5, 1.0, 2.5, 10.0, 50.0, 100.5]

for fname in ["gamma", "gammaln", "digamma", "erf", "erfc", "erfcx", "expi", "exp1", "zeta_1arg",
              "lambertw", "ndtr", "ndtri", "log_ndtr", "expit", "logit", "spence", "dawsn",
              "ellipk", "ellipe", "i0", "i1", "k0", "k1", "i0e", "k0e", "psi", "sinc", "cbrt"]:
    @case(f"special.{fname}")
    def _(fname=fname):
        if fname == "zeta_1arg":
            return sc.zeta(np.linspace(1.01, 60, 80), 1.0)
        f = getattr(sc, fname)
        if fname in ("ndtri", "logit"):
            return f(np.concatenate([np.logspace(-300, -1, 40), np.linspace(0.11, 0.89, 20),
                                     1 - np.logspace(-16, -1, 30)]))
        if fname in ("ellipk", "ellipe"):
            return f(np.concatenate([-np.logspace(3, -3, 20), np.linspace(0, 0.999, 40),
                                     1 - np.logspace(-15, -4, 20)]))
        if fname == "lambertw":
            return f(np.concatenate([[-np.exp(-1) + 1e-12], -np.logspace(-0.44, -20, 20), x_pos]))
        if fname == "spence":
            return f(x_pos)
        return f(x_sym if fname in ("erf", "erfc", "erfcx", "ndtr", "log_ndtr", "expit", "dawsn",
                                     "sinc", "cbrt", "gamma", "digamma", "psi", "expi") else x_pos)

for fname in ["jv", "yv", "iv", "kv", "ive", "kve", "struve", "modstruve"]:
    @case(f"special.{fname}")
    def _(fname=fname):
        f = getattr(sc, fname)
        return np.array([f(v, x_mod) for v in orders])

for fname, args in [("hyp2f1", [(0.5, 1.5, 2.0), (1, 1, 2), (-3, 2.5, 1.5), (10, 10, 1.5), (0.3, 0.7, 1.0)]),
                    ("hyp1f1", [(0.5, 1.5), (1, 2), (-10.5, 3), (50, 2), (2, 50)])]:
    @case(f"special.{fname}")
    def _(fname=fname, args=args):
        f = getattr(sc, fname)
        z2 = np.concatenate([np.linspace(-20, 0.99, 50), [0.999, 0.9999, 1 - 1e-8]])
        z1 = np.linspace(-200, 200, 81)
        return np.array([f(*a, z2 if fname == "hyp2f1" else z1) for a in args])

for fname in ["gammainc", "gammaincc", "betainc", "betaincinv", "gammaincinv", "gammainccinv",
              "chdtr", "chdtrc", "stdtr", "bdtr", "pdtr", "pdtrc", "expn", "kolmogorov", "smirnov"]:
    @case(f"special.{fname}")
    def _(fname=fname):
        f = getattr(sc, fname)
        a = np.array([0.1, 0.5, 1, 2.5, 10, 100, 1e4])
        if fname in ("gammainc", "gammaincc", "chdtr", "chdtrc"):
            x = np.concatenate([np.logspace(-10, 1, 20), np.linspace(10, 1e4, 20)])
            return np.array([f(ai, x) for ai in a])
        if fname in ("gammaincinv", "gammainccinv"):
            p = np.concatenate([np.logspace(-300, -1, 20), [0.5], 1 - np.logspace(-1, -15, 10)])
            return np.array([f(ai, p) for ai in a])
        if fname == "betainc":
            x = np.concatenate([np.logspace(-10, -1, 10), np.linspace(0.1, 0.9, 9), 1 - np.logspace(-1, -10, 10)])
            return np.array([[f(ai, bi, x) for bi in a] for ai in a])
        if fname == "betaincinv":
            p = np.concatenate([np.logspace(-100, -1, 10), [0.5], 1 - np.logspace(-1, -12, 8)])
            return np.array([[f(ai, bi, p) for bi in a] for ai in a])
        if fname == "stdtr":
            return np.array([f(ai, np.linspace(-1e3, 1e3, 41)) for ai in a])
        if fname in ("bdtr",):
            return np.array([f(k, 100, np.linspace(0.001, 0.999, 30)) for k in [0, 1, 10, 50, 99]])
        if fname in ("pdtr", "pdtrc"):
            return np.array([f(k, np.logspace(-3, 3, 30)) for k in [0, 1, 10, 100, 1000]])
        if fname == "expn":
            return np.array([f(n, x_pos[:100]) for n in [0, 1, 2, 5, 20]])
        return f(np.concatenate([np.linspace(0.01, 5, 50), np.linspace(5, 40, 20)]) if fname == "kolmogorov"
                 else np.linspace(0.001, 0.999, 30)) if fname != "smirnov" else \
            np.array([f(n, np.linspace(0.001, 0.999, 30)) for n in [1, 5, 50, 500]])

# ---------------------------------------------------------------- distributions (tails)
x_tail = np.concatenate([-np.logspace(3, -2, 20), [0.0], np.logspace(-2, 3, 20)])
p_tail = np.concatenate([np.logspace(-300, -2, 20), [0.5], 1 - np.logspace(-2, -16, 15)])
dists = {"norm": (), "t": (3.5,), "cauchy": (), "logistic": (), "laplace": (), "gumbel_r": (),
         "chi2": (4.5,), "gamma": (2.5,), "beta": (2.5, 0.7), "f": (3, 7.5), "lognorm": (0.9,),
         "weibull_min": (1.7,), "invgauss": (0.6,), "ncx2": (4, 7.5), "nct": (5, 2.0),
         "ncf": (3, 8, 2.5), "skewnorm": (4.0,), "genextreme": (-0.3,), "levy_stable": (1.5, 0.5),
         "kstwobign": (), "vonmises": (2.0,), "rice": (1.5,), "johnsonsu": (1.0, 2.0)}
for d, shp in dists.items():
    for meth in ("cdf", "sf", "pdf", "logcdf", "logsf", "ppf", "isf"):
        @case(f"stats.{d}.{meth}")
        def _(d=d, shp=shp, meth=meth):
            dist = getattr(st, d)
            if d == "levy_stable" and meth in ("ppf", "isf", "logcdf", "logsf"):
                raise RuntimeError("skipped: slow")
            arg = p_tail if meth in ("ppf", "isf") else x_tail
            return getattr(dist, meth)(arg, *shp)
for d, shp in {"binom": (60, 0.3), "poisson": (12.5,), "nbinom": (5, 0.4), "hypergeom": (100, 30, 20),
               "betabinom": (40, 2.3, 4.1), "skellam": (3.0, 5.5), "zipf": (2.5,)}.items():
    for meth in ("cdf", "sf", "pmf", "ppf"):
        @case(f"stats.{d}.{meth}")
        def _(d=d, shp=shp, meth=meth):
            dist = getattr(st, d)
            if meth == "ppf" and d == "zipf":
                raise RuntimeError("skipped: unbounded ppf search")
            p_disc = np.concatenate([np.logspace(-30, -2, 10), [0.5], 1 - np.logspace(-2, -9, 8)])
            return getattr(dist, meth)(p_disc if meth == "ppf" else np.arange(-2, 120), *shp)


# ---------------------------------------------------------------- statistical tests (fixed data)
rng = np.random.RandomState(12345)            # legacy generator: identical across NumPy versions
A, B = rng.standard_normal(40), rng.standard_normal(55) + 0.3
C = rng.exponential(size=30)
for name, f in {"ttest_ind": lambda: st.ttest_ind(A, B),
                "ttest_welch": lambda: st.ttest_ind(A, B, equal_var=False),
                "mannwhitneyu": lambda: st.mannwhitneyu(A, B, alternative="two-sided"),
                "wilcoxon": lambda: st.wilcoxon(A[:30], C),
                "ks_2samp": lambda: st.ks_2samp(A, B),
                "kstest_norm": lambda: st.kstest(A, "norm"),
                "shapiro": lambda: st.shapiro(A),
                "anderson_stat": lambda: [st.anderson(A).statistic],
                "kruskal": lambda: st.kruskal(A, B, C),
                "pearsonr": lambda: st.pearsonr(A[:30], C),
                "spearmanr": lambda: st.spearmanr(A[:30], C),
                "kendalltau": lambda: st.kendalltau(A[:30], C),
                "fisher_exact": lambda: st.fisher_exact([[8, 2], [1, 5]]),
                "chi2_contingency": lambda: st.chi2_contingency([[10, 20, 30], [6, 9, 17]])[:3],
                "binom_test": lambda: [st.binomtest(7, 30, 0.1).pvalue] if hasattr(st, "binomtest")
                else [st.binom_test(7, 30, 0.1)],
                "linregress": lambda: st.linregress(A[:30], C)[:5],
                "skew_kurt": lambda: [st.skew(C), st.kurtosis(C), st.skew(C, bias=False)],
                }.items():
    @case(f"stats.test.{name}")
    def _(f=f):
        return np.array([float(v) for v in f()])


# ---------------------------------------------------------------- integration
def vdp(t, y, mu=1000.0):
    return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]


def rober(t, y):
    return [-0.04 * y[0] + 1e4 * y[1] * y[2], 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
            3e7 * y[1] ** 2]


def kepler(t, y):
    r3 = (y[0] ** 2 + y[1] ** 2) ** 1.5
    return [y[2], y[3], -y[0] / r3, -y[1] / r3]


problems = {"vdp1000": (vdp, (0, 3000), [2, 0], np.linspace(0, 3000, 31)),
            "robertson": (rober, (0, 1e5), [1, 0, 0], np.logspace(-5, 5, 21)),
            "kepler_e09": (kepler, (0, 20), [0.1, 0, 0, np.sqrt(1.9 / 0.1)], np.linspace(0, 20, 41))}
for pname, (fun, span, y0, te) in problems.items():
    for meth in ("RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"):
        @case(f"integrate.solve_ivp.{pname}.{meth}")
        def _(fun=fun, span=span, y0=y0, te=te, meth=meth, pname=pname):
            if pname != "kepler_e09" and meth in ("RK45", "RK23", "DOP853"):
                raise RuntimeError("skipped: stiff")
            if pname == "robertson":
                te = np.concatenate([[0], te])
            r = si.solve_ivp(fun, span, y0, method=meth, t_eval=te, rtol=1e-8, atol=1e-10)
            return np.concatenate([r.y.ravel(), [r.nfev, r.status]])
    @case(f"integrate.odeint.{pname}")
    def _(fun=fun, y0=y0, te=te, pname=pname):
        if pname == "robertson":
            te = np.concatenate([[0], te])
        return si.odeint(lambda y, t: fun(t, y), y0, te, rtol=1e-8, atol=1e-10, tfirst=False).ravel()
    for integ in ("vode", "lsoda", "dopri5", "dop853"):
        @case(f"integrate.ode.{pname}.{integ}")
        def _(fun=fun, y0=y0, te=te, integ=integ, pname=pname):
            if pname != "kepler_e09" and integ in ("dopri5", "dop853"):
                raise RuntimeError("skipped: stiff")
            kw = {"method": "bdf"} if integ == "vode" and pname != "kepler_e09" else {}
            r = si.ode(fun).set_integrator(integ, rtol=1e-8, atol=1e-10, nsteps=100000, **kw)
            r.set_initial_value(y0, te[0] if pname != "robertson" else 0.0)
            out = []
            for t in (te[1:] if pname != "robertson" else te):
                out.append(r.integrate(t))
            return np.array(out).ravel()

quad_cases = {"sqrt_log": (lambda x: np.sqrt(x) * np.log(x), 0, 1), "osc": (lambda x: np.cos(50 * x) * np.exp(-x), 0, np.inf),
              "peak": (lambda x: 1 / (1e-4 + (x - 0.3) ** 2), 0, 1), "endsing": (lambda x: x ** -0.9, 0, 1),
              "gauss_inf": (lambda x: np.exp(-x * x / 2), -np.inf, np.inf), "sinc_long": (lambda x: np.sinc(x), 0, 200)}
for qn, (f, a, b) in quad_cases.items():
    @case(f"integrate.quad.{qn}")
    def _(f=f, a=a, b=b):
        v, e = si.quad(f, a, b, limit=200)
        return [v, e]
    @case(f"integrate.romberg_like.{qn}")
    def _(f=f, a=a, b=b):
        if not (np.isfinite(a) and np.isfinite(b)):
            raise RuntimeError("skipped: infinite")
        return [si.fixed_quad(f, a, b, n=50)[0], si.quadrature(f, a, b, maxiter=200)[0]
                if hasattr(si, "quadrature") else np.nan]


# ---------------------------------------------------------------- optimisation and roots
def rosen(x):
    return so.rosen(x)


for meth in ("Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "SLSQP", "trust-constr"):
    @case(f"optimize.minimize.rosen5.{meth}")
    def _(meth=meth):
        r = so.minimize(rosen, np.array([-1.2, 1, -0.5, 0.8, 1.5]), method=meth,
                        jac=None if meth in ("Nelder-Mead", "Powell") else so.rosen_der,
                        options={"maxiter": 20000})
        return np.concatenate([r.x, [r.fun, r.nfev]])
for rn, f, br in [("cubic", lambda x: x ** 3 - 2 * x - 5, (2, 3)), ("flat", lambda x: (x - 1) ** 7, (0, 3)),
                  ("tiny", lambda x: np.exp(-1 / x ** 2) - 1e-300 if x != 0 else -1e-300, (0.01, 1))]:
    for meth in ("brentq", "brenth", "ridder", "bisect", "toms748"):
        @case(f"optimize.root_scalar.{rn}.{meth}")
        def _(f=f, br=br, meth=meth):
            fn = getattr(so, meth)
            x, r = fn(f, *br, full_output=True)
            return [x, r.iterations, r.function_calls]
@case("optimize.least_squares.exp_fit")
def _():
    t = np.linspace(0, 4, 30); y = 2.5 * np.exp(-1.3 * t) + 0.3 + 0.01 * np.sin(17 * t)
    return so.least_squares(lambda p: p[0] * np.exp(-p[1] * t) + p[2] - y, [1, 1, 0]).x
@case("optimize.root.hybr")
def _():
    return so.root(lambda v: [v[0] + 0.5 * (v[0] - v[1]) ** 3 - 1, 0.5 * (v[1] - v[0]) ** 3 + v[1]], [0, 0]).x
@case("optimize.linprog")
def _():
    return so.linprog([-1, 4], A_ub=[[-3, 1], [1, 2]], b_ub=[6, 4], bounds=[(None, None), (-3, None)]).x

# ---------------------------------------------------------------- linear algebra
M = rng.standard_normal((6, 6)); S = M @ M.T + 6 * np.eye(6)
for name, f in {"expm": lambda: sl.expm(M), "expm_big": lambda: sl.expm(30 * M),
                "logm": lambda: sl.logm(S), "sqrtm": lambda: sl.sqrtm(S), "inv": lambda: sl.inv(S),
                "det": lambda: [sl.det(M)], "eigvalsh": lambda: sl.eigvalsh(S), "svdvals": lambda: sl.svdvals(M),
                "solve_sylv": lambda: sl.solve_sylvester(S, -S.T + np.eye(6), M),
                "solve_lyap": lambda: sl.solve_continuous_lyapunov(-S, M + M.T),
                "pinv": lambda: sl.pinv(M[:, :4]), "hilbert_solve": lambda: sl.solve(sl.hilbert(10), np.ones(10)),
                "cond_hilbert": lambda: [np.linalg.cond(sl.hilbert(10))]}.items():
    @case(f"linalg.{name}")
    def _(f=f):
        return f()

# ---------------------------------------------------------------- interpolation
xk = np.linspace(0, 10, 15); yk = np.sin(xk) + 0.1 * xk ** 2; xq = np.linspace(-0.5, 10.5, 101)
for name, f in {"CubicSpline_nat": lambda: sp_int.CubicSpline(xk, yk, bc_type="natural")(xq),
                "CubicSpline_nak": lambda: sp_int.CubicSpline(xk, yk)(xq),
                "splrep_s0": lambda: sp_int.splev(xq, sp_int.splrep(xk, yk)),
                "splrep_s1": lambda: sp_int.splev(xq, sp_int.splrep(xk, yk, s=1.0)),
                "pchip": lambda: sp_int.PchipInterpolator(xk, yk)(xq),
                "akima": lambda: sp_int.Akima1DInterpolator(xk, yk)(xq),
                "interp1d_cubic": lambda: sp_int.interp1d(xk, yk, kind="cubic", bounds_error=False)(xq),
                "rbf_tps": lambda: sp_int.Rbf(xk, yk, function="thin_plate")(xq)}.items():
    @case(f"interpolate.{name}")
    def _(f=f):
        return f()

meta = {"scipy": scipy.__version__, "numpy": np.__version__, "python": sys.version.split()[0]}
np.savez_compressed(sys.argv[1], __meta__=np.array(str(meta)), **OUT)
print(meta, len(OUT), "cases,", sum(1 for v in OUT.values() if v.dtype.kind == "U"), "errors")
