"""Reproduce every computed claim in "Resurgence of the squeezed Kerr oscillator: tunnel splittings, Kummer connection,
and a perturbative/non-perturbative relation" (Shankar Sivarajan).

Each check recomputes a quantity with the `resurgence` package in this repository and compares it with the value or
bound stated in the paper, printing PASS or FAIL. Labels follow the paper (Sections 2-6, Results 3.1-6.1, Tables 1-3).
Expensive intermediate results (exact splittings, perturbative series) are cached in reproduce/.cache_kerr_cat.json and
reproduce/.series_kerr_cat/, so an interrupted run resumes where it stopped.

Run:  python reproduce/kerr_cat.py            (about an hour on one core; resumable)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sympy as sp  # noqa: E402
from flint import fmpq  # noqa: E402
from mpmath import digamma, e as E, exp, factorial, gamma, log, mp, mpf, nstr, pi, sin, sqrt  # noqa: E402

from resurgence import borel  # noqa: E402
from resurgence import largeorder as lo  # noqa: E402
from resurgence.checkpoint import Checkpoint  # noqa: E402
from resurgence.models import kerr_cat as kc  # noqa: E402

HERE = Path(__file__).resolve().parent
SER = HERE / ".series_kerr_cat"
SER.mkdir(exist_ok=True)
CACHE = Checkpoint(HERE / ".cache_kerr_cat.json", meta={"version": 1}, every=60)
T0 = time.time()
RESULTS = []


def log_(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def check(label, ok, detail=""):
    RESULTS.append((label, bool(ok)))
    log_(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"   ({detail})" if detail else ""))


def section(name):
    log_(f"===== {name} =====")


def series(n, num, den, kmax=300):
    return kc.rs_series(n, fmpq(num, den), kmax, checkpoint=SER / f"rs_n{n}_D{num}_{den}_K{kmax}.json.gz")


def coeffs(n, num, den):
    e = series(n, num, den)
    return [mpf(int(e[2 * j].p)) / mpf(int(e[2 * j].q)) for j in range(151)]


def splitting(e2, D, n, dps):
    """S_n = E_odd,n - E_even,n (cached by parameters and dps)."""
    key = f"S|{n}|{mp.nstr(mpf(e2), 30)}|{mp.nstr(mpf(D), 30)}|{dps}"
    if CACHE.has(key):
        return CACHE.get_mpf(key)
    with mp.workdps(dps):
        S = kc.exact_levels(mpf(e2), mpf(D), 1, nlev=n + 1)[n] - kc.exact_levels(mpf(e2), mpf(D), 0, nlev=n + 1)[n]
    CACHE.set(key, S)
    return S


def lead(e2, D, n):
    return 8 * e2 * (4 * e2) ** (2 * n - D / 2) * exp(-2 * e2) / (factorial(n) * gamma(n - D / 2))


def frac(s):
    a, b = (s.split("/") + ["1"])[:2]
    return int(a), int(b)


# =====================================================================================================================
def sec2():
    section("Section 2: perturbative coefficients (eq. c1, c2), Delta = 0 factorization")
    mp.dps = 60
    for n in range(4):
        for num, den in [(1, 2), (-1, 1), (3, 2)]:
            e = series(n, num, den, 12)
            D = sp.Rational(num, den)
            c1 = sp.Rational(-1, 4) * n * (2 * n + D)
            c2 = sp.Rational(-1, 64) * (4 * n - D) * (4 * n**2 - 2 * D * n - D)
            ok = sp.Rational(int(e[2].p), int(e[2].q)) == c1 and sp.Rational(int(e[4].p), int(e[4].q)) == c2
            check(f"c1, c2 exact (n={n}, Delta={num}/{den})", ok)
    e0 = kc.rs_series(0, fmpq(0), 60)
    check("Delta=0, n=0: c_j = 0 for 1 <= j <= 30", all(x == 0 for x in e0[1:]))
    check("Delta=0: exact ground energy -eps2^2 (eps2=3), both parities",
          all(abs(kc.exact_levels(3, 0, p)[0] + 9) < mpf(10) ** -50 for p in (0, 1)))


# =====================================================================================================================
def sec3():
    section("Result 3.1 / Table 1: Richardson limit of R_n (alpha^2 = 10..36), Delta in {-1, 1/2, 1, 3}")
    mp.dps = 60
    bounds = {0: 2e-11, 1: 6e-9, 2: 5e-7, 3: 1e-5}
    for n in range(4):
        worst = mpf(0)
        for D in (mpf(-1), mpf(1) / 2, mpf(1), mpf(3)):
            R = []
            for i in range(14):
                a2 = mpf(10 + 2 * i)
                e2 = a2 - D / 2
                R.append(splitting(e2, D, n, 60) / lead(e2, D, n))
            with mp.workdps(60):
                lim = lo.richardson(R, 7, start=5)[-1]
            worst = max(worst, abs(lim - 1))
        check(f"Table 1, n={n}: max |lim R_n - 1| < {bounds[n]:.0e}", worst < bounds[n], f"{nstr(worst, 3)}")

    section("Section 3: Delta = 2m with m < n")
    for n, Dv in [(1, 0), (2, 0), (2, 2), (3, 2)]:
        D = mpf(Dv)
        E2 = [16 + 2 * i for i in range(13)]
        R = [splitting(e2, D, n, 100) / lead(mpf(e2), D, n) for e2 in E2]
        with mp.workdps(100):
            lim = lo.richardson(R, 7, start=8)[-1]
            k1e = lo.richardson([E2[i] * (R[i] - 1) for i in range(13)], 6, start=8)[-1]
        k1 = -mpf(3) / 2 * n**2 - n + mpf(3) / 8 * (2 * n + 1) * D - D**2 / 16
        check(f"(n,Delta)=({n},{Dv}): |lim R - 1| < 4e-7 and eps2(R-1) -> k1",
              abs(lim - 1) < 4e-7 and abs(k1e - k1) < 1e-4,
              f"{nstr(abs(lim - 1), 3)}, k1 {nstr(k1e, 8)} vs {nstr(k1, 8)}")

    section("Section 3: Marthaler-Dykman vs Stirling (14% at Delta=-1, 5% at Delta=1)")
    for Dv, pct in ((-1, 14), (1, 5)):
        v = 1 + mpf(Dv) / 2
        r = sqrt(2 * pi / v) * (v / E) ** v / gamma(v)
        check(f"Stirling/Gamma(1+Delta/2) at Delta={Dv}: {pct}%", round(float((1 - r) * 100)) == pct, nstr(r, 6))


# =====================================================================================================================
def sec4():
    section("Result 4.1 / Section 4: action, exponent, prefactor")
    mp.dps = 120
    A_b = {0: 1e-15, 1: 3e-12, 2: 4e-10, 3: 3e-8}
    b_b = {0: 5e-15, 1: 2e-11, 2: 4e-9, 3: 3e-7}
    f_b = {0: 2e-17, 1: 9e-14, 2: 2e-11}
    for n in range(4):
        wa = wb = wf = mpf(0)
        for num, den in [(1, 2), (1, 1), (-1, 1), (3, 2)]:
            D = mpf(num) / den
            c = coeffs(n, num, den)
            A, _ = lo.estimate_action(c, order=12, start=30)
            b, _ = lo.estimate_exponent(c, 4, order=12, start=30)
            wa = max(wa, abs(A - 4))
            wb = max(wb, abs(b - (4 * n - D)))
            if n <= 2:
                f0, _ = lo.fit_large_order(c, 4, 4 * n - D, nterms=1, order=14, start=30)[0]
                pred = -exp(2 * D) / (factorial(n) * gamma(n - D / 2)) ** 2
                wf = max(wf, abs(f0 / pred - 1))
        check(f"n={n}: |A-4| < {A_b[n]:.0e}", wa < A_b[n], nstr(wa, 3))
        check(f"n={n}: |b-(4n-Delta)| < {b_b[n]:.0e}", wb < b_b[n], nstr(wb, 3))
        if n <= 2:
            check(f"n={n}: prefactor (eq. lo) rel. error < {f_b[n]:.0e}", wf < f_b[n], nstr(wf, 3))
    worst = mpf(0)
    for s in ["1/8", "1/4", "1/2", "3/4", "1", "5/4", "3/2", "5/2", "3", "7/2", "9/2", "5", "-1/2", "-3/2"]:
        num, den = frac(s)
        D = mpf(num) / den
        c = coeffs(0, num, den)
        f0, _ = lo.fit_large_order(c, 4, -D, nterms=1, order=14, start=20)[0]
        worst = max(worst, abs(f0 / (-exp(2 * D) / gamma(-D / 2) ** 2) - 1))
    check("n=0, fourteen Delta in [-3/2, 5]: prefactor rel. error < 1e-10", worst < 1e-10, nstr(worst, 3))
    c = coeffs(0, 2, 1)
    rt = {j: abs(c[j]) ** (mpf(1) / j) for j in (30, 90, 120, 150)}
    mx = max(abs(c[j]) ** (mpf(1) / j) for j in range(1, 151) if c[j] != 0)
    check("Delta=2: |c_j|^(1/j) <= 1.06 (j <= 150); 0.87, 1.00, 1.04, 1.05 at j=30,90,120,150",
          mx <= 1.06 and [round(float(rt[j]), 2) for j in (30, 90, 120, 150)] == [0.87, 1.00, 1.04, 1.05],
          ", ".join(nstr(rt[j], 4) for j in rt))


# =====================================================================================================================
def sec5():
    section("Result 5.1 / Table 2: median Borel-Pade sum")
    mp.dps = 80
    table2 = {
        "1/2": [(6, "1.86e-6", "3.47e-12", "3.995", "3.934"), (8, "3.18e-8", "1.16e-15", "4.586", "4.531"),
                (10, "5.51e-10", "3.82e-19", "5.040", "4.990"), (12, "9.64e-12", "1.26e-22", "5.409", "5.363")],
        "-1/2": [(6, "4.63e-6", "5.82e-11", "10.876", "11.242"), (8, "9.07e-8", "2.37e-14", "11.512", "11.798"),
                 (10, "1.75e-9", "9.21e-18", "11.997", "12.232"), (12, "3.35e-11", "3.48e-21", "12.388", "12.588")],
    }
    for s, rows in table2.items():
        num, den = frac(s)
        D = mpf(num) / den
        d = coeffs(0, num, den)[2:]
        for a2, dh_s, res_s, r_s, b_s in rows:
            A2 = mpf(a2)
            e2 = A2 - D / 2
            x = 1 / A2
            with mp.workdps(80):
                Ee = kc.exact_levels(e2, D, 0)[0]
                Eo = kc.exact_levels(e2, D, 1)[0]
            hc = ((Ee + Eo) / 2 + A2**2) / (4 * A2)
            dh = (Eo - Ee) / (4 * A2)
            key = f"med|{s}|{a2}"
            if CACHE.has(key + "_re"):
                mre = CACHE.get_mpf(key + "_re")
                mim = CACHE.get_mpf(key + "_im")
            else:
                med, jump = borel.median_sum(d, x, beta=1 - D, eps=mpf("0.1"), L=74, M=73)
                mre = x**D * med.real
                mim = x**D * jump.imag
                CACHE.set(key + "_re", mre)
                CACHE.set(key + "_im", mim)
            r = (hc - mre) / (dh / 2) ** 2
            brk = 2 * log(4 * e2) - digamma(-D / 2) - digamma(1)
            ok = (nstr(abs(dh), 3) == nstr(mpf(dh_s), 3) and nstr(abs(hc - mre), 3) == nstr(mpf(res_s), 3)
                  and f"{float(r):.3f}" == r_s and f"{float(brk):.3f}" == b_s)
            check(f"Table 2, Delta={s}, alpha^2={a2}", ok,
                  f"{nstr(abs(dh), 3)}, {nstr(abs(hc - mre), 3)}, {float(r):.3f}, {float(brk):.3f}")
            if a2 == 12:
                jr = mim / (-pi * (dh / 2) ** 2)
                check(f"half-jump/(pi (delta_h/2)^2) -> 1 (Delta={s}, alpha^2=12: within 1.1%)",
                      abs(abs(jr) - 1) < 0.011, nstr(jr, 6))
    br = {s: 2 * log(4 * (12 - mpf(frac(s)[0]) / frac(s)[1] / 2)) - digamma(-(mpf(frac(s)[0]) / frac(s)[1]) / 2) - digamma(1)
          for s in ("1/2", "-1/2")}
    check("psi term: 7.22 predicted vs 6.98 observed (alpha^2 = 12)",
          f"{float(br['-1/2'] - br['1/2']):.2f}" == "7.22" and f"{12.388 - 5.409:.2f}" == "6.98")


# =====================================================================================================================
def _pert_sym(nn, Ds, k_ord=8):
    """Exact RS with symbolic Delta (orders <= k_ord)."""
    D = Ds

    def V1(psi):
        out = {}
        for m, cc in psi.items():
            if m:
                out[m + 1] = out.get(m + 1, 0) + cc * m / 2
            if m >= 2:
                out[m - 1] = out.get(m - 1, 0) + cc * m * (m - 1) / 2
        return out

    def V2(psi):
        out = {}
        for m, cc in psi.items():
            q = cc / 4
            out[m + 2] = out.get(m + 2, 0) + q * D / 2
            if m >= 2:
                out[m - 2] = out.get(m - 2, 0) + q * D / 2 * m * (m - 1)
            out[m] = out.get(m, 0) + q * (-D * m + m * (m - 1))
        return out

    psis = [{nn: sp.Integer(1)}]
    Ek = [sp.Integer(nn)]
    for k in range(1, k_ord + 1):
        rhs = V1(psis[k - 1])
        if k >= 2:
            for m, cc in V2(psis[k - 2]).items():
                rhs[m] = rhs.get(m, 0) + cc
        val = sp.expand(rhs.get(nn, 0))
        new = {}
        for m, cc in rhs.items():
            if m != nn:
                new[m] = -cc
        for j in range(1, k):
            if Ek[j] != 0:
                for m, cc in psis[k - j].items():
                    new[m] = new.get(m, 0) + Ek[j] * cc
        psis.append({m: sp.expand(cc / (m - nn)) for m, cc in new.items() if m != nn})
        Ek.append(val)
    return Ek


def _md(e2, D):
    lam = 1 / (2 * e2)
    mu = (D + 2) / (2 * e2)
    A = sqrt(mu + 1) + mu * log((1 + sqrt(mu + 1)) / sqrt(mu))
    return -(64 * e2**2 / sqrt(pi * (D + 2))) * (1 + mu) ** (mpf(5) / 4) * sin(pi * D / 2) * exp(-A / lam)


def sec6():
    section("Result 6.1: perturbative energies F_n, the P-NP relation, k1, k2, k3")
    Ds, ns = sp.symbols("Delta n")
    k_ord = 8
    data = {nn: _pert_sym(nn, Ds, k_ord) for nn in range(7)}
    ek = {}
    for k in range(0, k_ord + 1, 2):
        deg = k // 2 + 2
        cs = sp.symbols(f"c0:{deg + 1}")
        P = sum(cs[i] * ns**i for i in range(deg + 1))
        sol = sp.solve([P.subs(ns, nn) - data[nn][k] for nn in range(deg + 1)], cs)
        ek[k] = sp.expand(P.subs(sol))
        check(f"e_{k}(n) polynomial in n (fit on n<=deg, verified n<=6)",
              all(sp.expand(ek[k].subs(ns, nn) - data[nn][k]) == 0 for nn in range(7)))
    g = sp.symbols("g")
    eps = 1 / g
    a2 = eps + Ds / 2
    h = sum(ek[k] * a2 ** (-sp.Rational(k, 2)) for k in ek)
    F = sp.expand(sp.series(-(Ds**2 / (16 * eps) - (a2 / eps) * h), g, 0, 5).removeO())
    Fpaper = (ns + g * (-ns**2 / 2 + Ds * ns / 4 - Ds**2 / 16)
              + g**2 * (-ns**3 / 4 + 3 * Ds * ns**2 / 16 + Ds * ns / 16 - Ds**2 * ns / 32 - Ds**2 / 64))
    check("eq. (F): F_n through g^2", sp.expand(sum(F.coeff(g, i) * g**i for i in range(3)) - Fpaper) == 0)
    dF = sp.expand(sp.diff(F, ns))
    f = [dF.coeff(g, i) for i in range(5)]
    B = 2 * ns - Ds / 2
    check("order g of (du) is f_1 = -B/2", sp.expand(f[1] + B / 2) == 0)
    Kp = sp.expand(sp.series(dF * sp.exp(2 * f[2] * g + f[3] * g**2 + sp.Rational(2, 3) * f[4] * g**3), g, 0, 4).removeO())
    kp = [sp.expand(Kp.coeff(g, i)) for i in range(4)]
    A_fun = 4 / g - 2 * sp.log(sp.series(Kp / dF, g, 0, 4).removeO())
    du_res = sp.expand(sp.series(dF + (g / 4) * (2 * B + g * sp.diff(A_fun, g)), g, 0, 4).removeO())
    check("eq. (Kpred) satisfies eq. (du) through g^3", du_res == 0)
    k1p = -sp.Rational(3, 2) * ns**2 - ns + sp.Rational(3, 8) * (2 * ns + 1) * Ds - Ds**2 / 16
    k2p = (Ds**4 - 8 * (3 * ns + 1) * Ds**3 + 4 * (48 * ns**2 + 20 * ns - 5) * Ds**2 - 16 * (36 * ns**3 + 12 * ns**2 - 14 * ns - 3) * Ds
           + 64 * ns * (9 * ns**3 + 2 * ns**2 - 6 * ns - 1)) / 512
    k3p0 = (-Ds**6 / 24576 + Ds**5 / 4096 + sp.Rational(23, 6144) * Ds**4 + Ds**3 / 1024 - sp.Rational(25, 384) * Ds**2
            + sp.Rational(3, 64) * Ds)
    check("eq. (k1)", sp.expand(kp[1] - k1p) == 0)
    check("k2 (general n, displayed)", sp.expand(kp[2] - k2p) == 0)
    check("k3 at n=0", sp.expand(kp[3].subs(ns, 0) - k3p0) == 0)
    check("k3 has total degree 6", sp.Poly(kp[3], ns, Ds).total_degree() == 6)
    check("k2(n=2) predicted values", [f"{float(kp[2].subs({ns: 2, Ds: d})):.6f}" for d in
          (-1, sp.Rational(-1, 2), sp.Rational(1, 2), 1, sp.Rational(3, 2))] == ["28.166016", "21.972778", "12.414185", "8.884766", "6.084106"])
    k3vals = ["-0.1094970703", "-0.03960990906", "0.007524490356", "-0.01330566406", "-0.05253410339"]
    check("k3(n=0) predicted values", [nstr(mpf(sp.N(kp[3].subs({ns: 0, Ds: d}), 30)), 10) for d in
          (-1, sp.Rational(-1, 2), sp.Rational(1, 2), 1, sp.Rational(3, 2))] == k3vals)

    section("Section 6: extraction of k1, k2, k3 from exact splittings (eps2 = 16..50)")
    E2 = [16 + 2 * i for i in range(18)]
    Dlist = ["-1", "-1/2", "1/2", "1", "3/2", "3", "5"]
    kb = {0: (1e-13, 1e-10), 1: (3e-10, 2e-7), 2: (3e-8, None)}
    for n in range(3):
        dps = 110 if n == 0 else 120
        w1 = w2 = mpf(0)
        k2n2 = []
        for s in Dlist:
            num, den = frac(s)
            D = mpf(num) / den
            Kn = [splitting(e2, D, n, dps) / lead(mpf(e2), D, n) for e2 in E2]
            with mp.workdps(dps):
                s1 = [E2[i] * (Kn[i] - 1) for i in range(18)]
                k1 = lo.richardson(s1, 8, start=8)[-1]
                s2 = [E2[i] * (s1[i] - k1) for i in range(18)]
                k2 = lo.richardson(s2, 7, start=8)[-1]
            k1t = mpf(sp.N(k1p.subs({ns: n, Ds: sp.Rational(num, den)}), 40))
            k2t = mpf(sp.N(k2p.subs({ns: n, Ds: sp.Rational(num, den)}), 40))
            w1 = max(w1, abs(k1 - k1t))
            if n < 2:
                w2 = max(w2, abs(k2 - k2t))
            elif s in ("-1", "-1/2", "1/2", "1", "3/2"):
                k2n2.append(abs(k2 - k2t))
            if n == 0 and s in ("-1", "-1/2", "1/2", "1", "3/2"):
                s3 = [E2[i] ** 3 * (Kn[i] - 1 - k1t / E2[i] - k2t / E2[i] ** 2) for i in range(18)]
                with mp.workdps(dps):
                    k3 = lo.richardson(s3, 8, start=8)[-1]
                k3t = mpf(sp.N(k3p0.subs(Ds, sp.Rational(num, den)), 40))
                check(f"k3(n=0, Delta={s}): extraction = prediction to 10 digits", nstr(k3, 10) == nstr(k3t, 10),
                      f"{nstr(k3, 12)} vs {nstr(k3t, 12)}")
        check(f"k1(n={n}) within {kb[n][0]:.0e}", w1 < kb[n][0], nstr(w1, 3))
        if n < 2:
            check(f"k2(n={n}) within {kb[n][1]:.0e}", w2 < kb[n][1], nstr(w2, 3))
        else:
            check("k2(n=2) prediction within 1.1e-5 of extraction", max(k2n2) < 1.1e-5, nstr(max(k2n2), 3))

    section("Table 3 and abstract accuracy claims")
    mp.dps = 60
    table3 = [(-1, 10, "1.04715", "1.8e-5", "0.948818", "1.10615"), (-1, 30, "1.01494", "1.9e-7", "0.886908", "1.03398"),
              (1, 10, "0.969325", "1.4e-6", "1.03507", "1.09339"), (3, 10, "0.94981", "-1.8e-5", "1.048", "1.08334"),
              (7, 30, "1.0168", "1.1e-5", "1.01014", "1.02899"), (11, 10, "1.41943", "0.012", "1.04479", "1.05826"),
              (11, 30, "1.12305", "0.00015", "1.01392", "1.02699")]
    worst_lo = worst_hi = mpf(0)
    for Dv, e2v, Ls, LKs, MDs, Us in table3:
        D, e2 = mpf(Dv), mpf(e2v)
        ex = splitting(e2, D, 0, 60)
        l_ = lead(e2, D, 0)
        g_ = 1 / e2
        k1 = mpf(sp.N(k1p.subs({ns: 0, Ds: Dv}), 30))
        k2 = mpf(sp.N(k2p.subs({ns: 0, Ds: Dv}), 30))
        k3 = mpf(sp.N(k3p0.subs(Ds, Dv), 30))
        lk = l_ * (1 + k1 * g_ + k2 * g_**2 + k3 * g_**3) / ex - 1
        v = 1 + D / 2
        u = _md(e2, D) * gamma(v) / (sqrt(2 * pi / v) * (v / E) ** v) / ex
        ok = (nstr(l_ / ex, 6) == Ls and nstr(lk, 2) == nstr(mpf(LKs), 2) and nstr(_md(e2, D) / ex, 6) == MDs
              and nstr(u, 6) == Us)
        check(f"Table 3 row Delta={Dv}, eps2={e2v}", ok,
              f"{nstr(l_ / ex, 6)}, {nstr(lk, 2)}, {nstr(_md(e2, D) / ex, 6)}, {nstr(u, 6)}")
    for e2v in (10, 20, 30):
        for i in range(17):
            Dv = mpf(-1) + mpf(i) / 4
            if Dv in (0, 2):
                continue
            e2 = mpf(e2v)
            ex = splitting(e2, Dv, 0, 80)
            k1 = 3 * Dv / 8 - Dv**2 / 16
            k2 = Dv**4 / 512 - Dv**3 / 64 - 5 * Dv**2 / 128 + 3 * Dv / 32
            k3 = -Dv**6 / 24576 + Dv**5 / 4096 + 23 * Dv**4 / 6144 + Dv**3 / 1024 - 25 * Dv**2 / 384 + 3 * Dv / 64
            g_ = 1 / e2
            worst_lo = max(worst_lo, abs(lead(e2, Dv, 0) * (1 + k1 * g_ + k2 * g_**2 + k3 * g_**3) / ex - 1))
    for i in range(33):
        Dv = mpf(3) + mpf(i) / 4
        if Dv in (4, 6, 8, 10):
            continue
        e2 = mpf(30)
        ex = splitting(e2, Dv, 0, 80)
        k1 = 3 * Dv / 8 - Dv**2 / 16
        k2 = Dv**4 / 512 - Dv**3 / 64 - 5 * Dv**2 / 128 + 3 * Dv / 32
        k3 = -Dv**6 / 24576 + Dv**5 / 4096 + 23 * Dv**4 / 6144 + Dv**3 / 1024 - 25 * Dv**2 / 384 + 3 * Dv / 64
        worst_hi = max(worst_hi, abs(lead(e2, Dv, 0) * (1 + k1 / 30 + k2 / 900 + k3 / 27000) / ex - 1))
    check("abstract: with k1..k3, rel. error < 2e-5 for -1 <= Delta <= 3 at eps2 = 10, 20, 30 (step 1/4)",
          worst_lo < 2e-5, nstr(worst_lo, 3))
    check("abstract: rel. error < 2e-4 for 3 <= Delta <= 11 at eps2 = 30 (step 1/4)",
          worst_hi < 2e-4, nstr(worst_hi, 3))


if __name__ == "__main__":
    for part in (sec2, sec3, sec4, sec5, sec6):
        part()
    CACHE.save()
    npass = sum(ok for _, ok in RESULTS)
    log_(f"===== SUMMARY: {npass}/{len(RESULTS)} PASS =====")
    for label, ok in RESULTS:
        if not ok:
            log_(f"FAILED: {label}")
    sys.exit(0 if npass == len(RESULTS) else 1)
