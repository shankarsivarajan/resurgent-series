"""Reproduce every computed claim in "Resurgence of the finite-N Curie-Weiss model: metastability, the spinodal, and
Lee-Yang zeros" (Shankar Sivarajan).

Each check recomputes a quantity from scratch with the `resurgence` package in this repository and compares it with
the value stated in the paper, printing PASS or FAIL. Labels follow the paper's numbering (Sections 2-5, Result 3.1,
Remark 5.1, equations (7), (8), (13), (14), Tables 1-2, Figures 1-2). Statements that are cited rather than recomputed
are printed as NOTE lines and listed at the end. Expensive intermediate results are cached in
reproduce/.cache_curie_weiss.json, so an interrupted run resumes where it stopped.

Run:  python reproduce/curie_weiss.py          (a few minutes on one core; resumable)
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpmath import (airyai, airybi, atanh, diff, exp, findroot, im, log, loggamma, mp, mpc, mpf,  # noqa: E402
                    nstr, pi, quad, re, sqrt, tanh)

from resurgence import borel  # noqa: E402
from resurgence import largeorder as lo  # noqa: E402
from resurgence.checkpoint import Checkpoint  # noqa: E402
from resurgence.models import curie_weiss as cw  # noqa: E402
from resurgence.models import curie_weiss_uniform as U  # noqa: E402
from resurgence.models import curie_weiss_zeros as Z  # noqa: E402

T0 = time.time()
RESULTS, NOTES = [], []
CACHE = Checkpoint(Path(__file__).resolve().parent / ".cache_curie_weiss.json", meta={"version": 1})
K = mpf("1.5")


def log_(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def check(label, ok, detail=""):
    RESULTS.append((label, bool(ok)))
    log_(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"   ({detail})" if detail else ""))


def note(label):
    NOTES.append(label)
    log_(f"[NOTE] {label}")


def section(name):
    log_(f"===== {name} =====")


def cached(key, fn):
    """Cache an mpmath number (real or complex) under key."""
    if CACHE.has(key + "_r"):
        r, i = CACHE.get_mpf(key + "_r"), CACHE.get_mpf(key + "_i")
        return mpc(r, i) if i != 0 else r
    v = fn()
    CACHE.set(key + "_r", re(v))
    CACHE.set(key + "_i", im(v))
    CACHE.save()
    return v


def rounds_to(x, s):
    """True if |x| rounds to the decimal string s (e.g. '1.6e-11', '0.938') at the precision of s."""
    from decimal import Decimal
    d = Decimal(s)
    q = Decimal(1).scaleb(d.as_tuple().exponent)
    return abs(Decimal(str(float(abs(x)))) - d) <= q / 2 * Decimal("1.0000001")


def spinodal():
    ms = -sqrt(1 - 1 / K)
    xs = (1 + ms) / 2
    return atanh(ms) - K * ms, (1 / xs**2 - 1 / (1 - xs) ** 2) / 6, xs


def H_lin(zl, N):
    Hsp, g, _ = spinodal()
    return Hsp - mpf(zl) * (3 * g) ** (mpf(1) / 3) / (2 * mpf(N) ** (mpf(2) / 3))


def lateral_sums(q, N, ang=0.1):
    up = borel.borel_pade_sum(q, mpf(1) / N, -mpf(1) / 2, ang)
    dn = borel.borel_pade_sum(q, mpf(1) / N, -mpf(1) / 2, -ang)
    return up, dn


# =====================================================================================================================
def sec2():
    section("Section 2: model and exact representations")
    mp.dps = 60
    N, n = 1000, 300
    x = mpf(n) / N
    exact = loggamma(N + 1) - loggamma(n + 1) - loggamma(N - n + 1)
    logP = loggamma(N + 1) + N - N * log(N) - log(2 * pi)
    Hh = mpf("0.05")
    phi0 = cw.Phi0(x, K, Hh) - (K * (2 * x - 1) ** 2 / 2 + Hh * (2 * x - 1))  # entropy part only
    stir = logP - log(N) + N * phi0 - log(x * (1 - x)) / 2
    for j in range(1, 8, 2):
        B = __import__("mpmath").bernoulli(j + 1)
        stir += -B / (j * (j + 1)) * (x ** -j + (1 - x) ** -j) * mpf(N) ** -j
    check("Sec 2 Stirling form: summand = P_N N^{-1} exp(N Phi0 + Phi1 + sum_j N^{-j} Phi_{j+1}) (N = 1000, n = 300, "
          "terms through j = 7)", abs(stir - exact) < mpf(10) ** -24, f"diff {nstr(abs(stir - exact), 3)}")
    mp.dps = 40
    H = mpf("0.05")
    xp, x0, xm = cw.critical_points(K, H)
    for nm, xc in (("+", xp), ("-", xm)):
        m = 2 * xc - 1
        q0 = cw.series(xc, K, H, 1)[0]
        check(f"Sec 2: q_0^{nm} = sqrt(2 pi / (1 - K(1 - m^2)))", abs(q0 - sqrt(2 * pi / (1 - K * (1 - m**2)))) < mpf(10) ** -30)
    for N in (20, 60):
        f = lambda y, N=N: exp(N * (-K * y * y / 2 + log(2 * __import__("mpmath").cosh(K * y + H))))  # noqa: E731
        hs = sqrt(N * K / (2 * pi)) * quad(f, [-12, -1, 0, 1, 12])
        exact_Z = sum(__import__("mpmath").binomial(N, k) * exp(N * (K * (2 * mpf(k) / N - 1) ** 2 / 2 + H * (2 * mpf(k) / N - 1)))
                      for k in range(N + 1))
        check(f"Eq. (7): Hubbard-Stratonovich integral = Z_N (N = {N})", abs(hs / exact_Z - 1) < mpf(10) ** -25,
              f"rel diff {nstr(abs(hs / exact_Z - 1), 3)}")
    for xc in (xp, x0, xm):
        y = 2 * xc - 1
        psi = -K * y * y / 2 + log(2 * __import__("mpmath").cosh(K * y + H))
        check(f"Sec 2: Psi(m_i) = Phi0(x_i) at m = {nstr(y, 6)}", abs(psi - cw.Phi0(xc, K, H)) < mpf(10) ** -30)
    vals = [K / 2 - H, K / 2 + H, cw.Phi0(x0, K, H), cw.Phi0(xm, K, H)]
    check("Sec 2 (K=1.5, H=0.05): Phi0(0) = 0.70, Phi0(1) = 0.80, Phi0(x0) = 0.69, Phi0(x-) = 0.77",
          all(rounds_to(v, s) for v, s in zip(vals, ("0.70", "0.80", "0.69", "0.77"))), ", ".join(nstr(v, 5) for v in vals))


# =====================================================================================================================
def sec3():
    section("Section 3: two-phase transseries, Result 3.1, eq. (8)")
    mp.dps = 80
    H = mpf("0.05")
    xp, x0, xm = cw.critical_points(K, H)
    qp, qm = cw.series(xp, K, H, 60), cw.series(xm, K, H, 60)
    for nm, q, xc, bound in (("stable", qp, xp, mpf("2e-11")), ("metastable", qm, xm, mpf("1e-13"))):
        A = lo.estimate_action(q, order=8, start=5)[0]
        bar = cw.Phi0(xc, K, H) - cw.Phi0(x0, K, H)
        check(f"Sec 3: large-order action of the {nm} series = barrier height {nstr(bar, 12)} (to better than {nstr(bound, 1)})",
              abs(A - bar) < bound, f"|diff| {nstr(abs(A - bar), 2)}")
    bounds = {20: "3e-12", 40: "9e-17", 60: "3e-19", 100: "2e-25", 150: "1e-28"}
    ratios = {20: "0.454", 40: "0.473", 60: "0.481", 100: "0.488", 150: "0.492"}
    imf = {20: "0.905", 40: "0.947", 60: "0.962", 100: "0.976"}
    for N in (20, 40, 60, 100, 150):
        mpp, hp = borel.median_sum(qp, mpf(1) / N, beta=-mpf(1) / 2, eps=0.1)
        mmm, hm = borel.median_sum(qm, mpf(1) / N, beta=-mpf(1) / 2, eps=0.1)
        rec = exp(N * cw.Phi0(xp, K, H)) * mpp.real + exp(N * cw.Phi0(xm, K, H)) * mmm.real
        Zx = cw.Z_over_P(N, K, H)
        check(f"Eq. (8): median reconstruction = Z_N/P_N to better than {bounds[N]} (N = {N}, 60-term Borel-Pade)",
              abs(Zx / rec - 1) < mpf(bounds[N]), f"{nstr(abs(Zx / rec - 1), 3)}")
        wp, wm = exp(N * cw.Phi0(xp, K, H)) * abs(hp), exp(N * cw.Phi0(xm, K, H)) * abs(hm)
        r = abs(hm) / cw.barrier_thimble(N, K, H, "-")
        check(f"Sec 3: weighted half-jumps of the two phases agree to >= 8 digits; ratio to the Gaussian thimble = "
              f"{ratios[N]} (N = {N})", abs(wp / wm - 1) < mpf("1e-8") and rounds_to(r, ratios[N]), f"{nstr(r, 6)}")
        if N in imf:
            lead = cw.barrier_thimble(N, K, H, "-") / 2 / (N * mmm.real)
            rr = (mp.atan(abs(hm) / mmm.real) / N) / lead
            check(f"Sec 3: Im f_- / leading estimate = {imf[N]} (N = {N})", rounds_to(rr, imf[N]), nstr(rr, 6))
        if N == 20:
            amb = wm / Zx
            check("Sec 3: canceled ambiguity = 1.65% of Z_N at N = 20", rounds_to(amb * 100, "1.65"), f"{nstr(amb * 100, 4)}%")
    # Result 3.1 sign structure (N = 40, 40 terms, rays at +-0.1)
    mp.dps = 60
    for H in (mpf("0.05"), mpf("0.001")):
        xp, x0, xm = cw.critical_points(K, H)
        N = 40
        out = {}
        for nm, xc in (("+", xp), ("-", xm)):
            w = exp(N * cw.Phi0(xc, K, H))
            u, d = lateral_sums(cw.series(xc, K, H, 40), N)
            out[nm] = (w * u, w * d)
        Zx = cw.Z_over_P(N, K, H)
        hjp, hjm = (out["+"][0] - out["+"][1]) / 2, (out["-"][0] - out["-"][1]) / 2
        same = out["+"][0] + out["-"][0]
        opp = out["+"][0] + out["-"][1]
        check(f"Result 3.1: both phases' weighted jumps are equal (same sign, Im S_up > 0); Z = S_tau[A+] + S_-tau[A-] "
              f"(H = {nstr(H, 3)}, N = 40)",
              abs(hjp / hjm - 1) < 1e-8 and im(hjp) > 0 and abs(opp / Zx - 1) < 1e-12 and abs(im(same / Zx)) > 1e-4,
              f"half-jump {nstr(hjp, 8)}; jump ratio - 1 = {nstr(abs(hjp / hjm - 1), 2)}; same-side Im/Z {nstr(im(same / Zx), 3)}; opposite |rel| {nstr(abs(opp / Zx - 1), 2)}")
        if H == mpf("0.05"):
            ystar = mpc(-H / K, pi / (2 * K))
            y0 = 2 * x0 - 1
            P = exp(loggamma(N + 1) + N - N * log(N) - log(2 * pi))
            f = lambda y: exp(N * (-K * y * y / 2 + log(2 * __import__("mpmath").cosh(K * y + H))))  # noqa: E731
            Thalf = sqrt(N * K / (2 * pi)) / P * im(quad(f, [y0, ystar]))
            check("Result 3.1: weighted half-jump = T/2 with T the HS barrier-thimble integral (N = 40, H = 0.05)",
                  abs(Thalf / im(hjp) - 1) < 1e-10, f"T/2 = {nstr(Thalf, 10)}")
    # thimble endpoint
    for label, H in (("H = 0.05", mpf("0.05")), ("zeta_lin = 0.5, N = 100", H_lin("0.5", 100))):
        xp, x0, xm = cw.critical_points(K, H)
        h = float(H)
        y = complex(float(2 * x0 - 1), 1e-6)
        ys = complex(-h / 1.5, np.pi / 3)
        for _ in range(800000):
            g = -1.5 * y + 1.5 * np.tanh(1.5 * y + h)
            y = y - 2.5e-5 * np.conj(g) / abs(g)
            if abs(np.cosh(1.5 * y + h)) < 1e-6:
                break
        check(f"Proof of Result 3.1: steepest-descent path from y0 ends at y* = (i pi/2 - H)/K ({label})", abs(y - ys) < 1e-4,
              f"|end - y*| = {abs(y - ys):.1e}")
    note("Result 3.1: the identification of lateral thimble integrals with lateral Borel sums is the standard step, not "
         "recomputed beyond the numerical checks above")


# =====================================================================================================================
def sec4():
    section("Section 4: approaching the spinodal, eq. (13), Table 1, Figure 1")
    mp.dps = 60
    Hsp, g, xs = spinodal()
    check("Sec 2/4: H_sp = 0.2075 at K = 1.5", rounds_to(Hsp, "0.2075"), nstr(Hsp, 8))
    lim = 4 / (3 * K * (1 - 1 / K) ** (mpf(1) / 4))
    check("Sec 4: fold limit 16/(3 sqrt(Phi0'''(x_s))) = 4/(3K(1-1/K)^{1/4}) = 1.1698",
          abs(lim - 16 / (3 * sqrt(6 * g))) < mpf(10) ** -40 and rounds_to(lim, "1.1698"), nstr(lim, 8))
    for d, s in (("0.157546", "1.2052"), ("0.0045", "1.1708")):
        xp, x0, xm = cw.critical_points(K, Hsp - mpf(d))
        r = (cw.Phi0(xm, K, Hsp - mpf(d)) - cw.Phi0(x0, K, Hsp - mpf(d))) / mpf(d) ** mpf(1.5)
        check(f"Sec 4: Delta Phi_- / delta^(3/2) = {s} at delta = {d}", rounds_to(r, s), nstr(r, 6))
    mp.dps = 80
    for N in (50, 100):
        H = H_lin("0.5", N)
        xp, x0, xm = cw.critical_points(K, H)
        mre, mim = cw.metastable_sector_exact(N, K, H)
        medp, hdp = borel.median_sum(cw.series(xp, K, H, 60), mpf(1) / N, beta=-mpf(1) / 2, eps=0.1)
        d1 = abs(mre / (cw.Z_over_P(N, K, H) - exp(N * cw.Phi0(xp, K, H)) * medp.real) - 1)
        d2 = abs(mim / (exp(N * cw.Phi0(xp, K, H)) * abs(hdp)) - 1)
        check(f"Sec 4: exact HS sector = independent subtraction route, median and half-jump, to better than 1e-14 "
              f"(N = {N}, zeta_lin = 0.5)", d1 < 1e-14 and d2 < 1e-14, f"{nstr(d1, 2)}, {nstr(d2, 2)}")
    table1 = {("0.5", 50): ("9.8e-3", "4.1e-7"), ("0.5", 400): ("1.0e-3", "8.5e-14"), ("0.5", 3200): ("1.2e-4", "2.9e-20"),
              ("1", 50): ("1.2e-2", "1.5e-6"), ("1", 400): ("1.1e-3", "1.2e-13"), ("1", 3200): ("1.2e-4", "3.2e-20"),
              ("2", 50): ("1.7e-2", "2.9e-5"), ("2", 400): ("1.2e-3", "2.4e-13"), ("2", 3200): ("1.3e-4", "4.0e-20"),
              ("3", 400): ("1.3e-3", "4.8e-13"), ("3", 3200): ("1.3e-4", "4.8e-20"),
              ("5", 400): ("1.6e-3", "2.0e-12"), ("5", 3200): ("1.4e-4", "6.9e-20")}
    err = {}
    for zl in ("0.5", "1", "2", "3", "5"):
        for N in (50, 400, 800, 1600, 3200):
            H = H_lin(zl, N)
            if H <= 0:
                if N == 50:
                    check(f"Table 1: n/a entry has H < 0 (zeta_lin = {zl}, N = 50)", True, f"H = {nstr(H, 3)}")
                continue

            def cell(zl=zl, N=N, H=H):
                mre, mim = cw.metastable_sector_exact(N, K, H)
                co = U.cfu_coefficients(K, H, 6)
                out = []
                for t in (0, 6):
                    r, i, _ = U.cfu_sector(N, co, t)
                    out.append(max(abs(mre / r - 1), abs(mim / i - 1)))
                return mpc(out[0], out[1])

            v = cached(f"T1_{zl}_{N}", cell)
            err[(zl, N)] = (re(v), im(v))
            if (zl, N) in table1:
                s0, s6 = table1[(zl, N)]
                check(f"Table 1 / eq. (13): zeta_lin = {zl}, N = {N}: order 0 -> 6 error {s0} -> {s6}",
                      rounds_to(re(v), s0) and rounds_to(im(v), s6), f"{nstr(re(v), 3)} -> {nstr(im(v), 3)}")
    f16 = [err[(z, 1600)][1] / err[(z, 3200)][1] for z in ("0.5", "1", "2", "3", "5")]
    f8 = [err[(z, 800)][1] / err[(z, 1600)][1] for z in ("0.5", "1", "2", "3", "5")]
    check("Sec 4: order-6 error falls by 139 to 215 (N 1600 -> 3200) and about 143 to 285 (N 800 -> 1600)",
          rounds_to(min(f16), "139") and rounds_to(max(f16), "215") and rounds_to(min(f8), "143") and abs(max(f8) - 285) < 1.5,
          f"{nstr(min(f16), 4)}-{nstr(max(f16), 4)}; {nstr(min(f8), 4)}-{nstr(max(f8), 4)}")
    H = H_lin("1", 400)
    mre, mim = cw.metastable_sector_exact(400, K, H)
    r, i, _ = U.cfu_sector(400, U.cfu_coefficients(K, H, 6), 6)
    check("Eq. (13): signed real part = median and signed imaginary part = S_up half-jump (zeta_lin = 1, N = 400)",
          r > 0 and i > 0 and abs(r / mre - 1) < 1e-12 and abs(i / mim - 1) < 1e-12)
    for N, s in ((400, "0.938"), (3200, "0.983")):
        H = H_lin("3", N)
        mre, mim = cw.metastable_sector_exact(N, K, H)
        rr = (mim / mre) / (airyai(mpf(3)) / airybi(mpf(3)))
        check(f"Sec 4: (|Im|/Re)/(Ai/Bi)(zeta_lin) = {s} at zeta_lin = 3, N = {N}", rounds_to(rr, s), nstr(rr, 5))
    mp.dps = 50
    zmax = None
    gmax = mpf(0)
    for zl in ("0.75", "0.8", "0.85", "0.9"):
        H = H_lin(zl, 200)
        xp, x0, xm = cw.critical_points(K, H)
        zeta = mpf(200) ** (mpf(2) / 3) * (mpf(3) / 4 * (cw.Phi0(xm, K, H) - cw.Phi0(x0, K, H))) ** (mpf(2) / 3)
        gr = cw.barrier_thimble(200, K, H, "-") / 2 / (cw.series(xm, K, H, 1)[0] * mpf(200) ** (-mpf(1) / 2))
        gr = gr / (airyai(zeta) / airybi(zeta))
        if gr > gmax:
            gmax, zmax = gr, zeta
    check("Fig. 1(b) / Sec 4: Gaussian half-thimble overshoots Ai/Bi by about 19% at most, near zeta = 0.85 (N = 200)",
          rounds_to((gmax - 1) * 100, "19") and abs(zmax - mpf("0.85")) < 0.1, f"max ratio {nstr(gmax, 5)} at zeta {nstr(zmax, 3)}")
    zeta200 = None
    for N, tol in ((200, 0.06), (1600, 0.007)):
        H = H_lin("5", N)
        xp, x0, xm = cw.critical_points(K, H)
        zeta = mpf(N) ** (mpf(2) / 3) * (mpf(3) / 4 * (cw.Phi0(xm, K, H) - cw.Phi0(x0, K, H))) ** (mpf(2) / 3)
        mre, mim = cw.metastable_sector_exact(N, K, H)
        rr = (mim / mre) / (airyai(zeta) / airybi(zeta))
        if N == 200:
            zeta200 = zeta
        check(f"Fig. 1(b): exact ratio / (Ai/Bi) approaches 1 as N grows (zeta_lin = 5, N = {N}: within {tol})",
              abs(rr - 1) < tol, f"{nstr(rr, 5)} at zeta = {nstr(zeta, 4)}")
    check("Sec 4: largest tested exact zeta (zeta_lin = 5, N = 200) is below 5.11", zeta200 < mpf("5.11"), nstr(zeta200, 5))
    note("Eq. (13): Bleistein's integration-by-parts recursion and the injectivity of the cubic map are assumed; support is "
         "the agreement in Table 1")


# =====================================================================================================================
def ly_trans(eta, N=40):
    H = mpc(0, 1) * eta
    ms = findroot(lambda m: m - tanh(K * m), mpf("0.85"))
    x = cw.track_roots(K, [H * t / 60 for t in range(61)], ms)[-1]
    S = borel.borel_pade_sum(cw.series(x, K, H, 40), mpf(1) / N, -mpf(1) / 2, 0.0)
    return 2 * re(exp(N * cw.Phi0(x, K, H)) * S)


def sec5():
    section("Section 5: Lee-Yang zeros, eq. (14), Remark 5.1, Table 2, Figure 2")
    mp.dps = 60
    for eta, b in (("0.1", "1e-18"), ("0.3", "1e-24"), ("0.5", "1e-24")):
        v = cached(f"LYm_{eta}", lambda eta=eta: abs(ly_trans(mpf(eta)) / re(cw.Z_over_P(40, K, mpc(0, 1) * mpf(eta))) - 1))
        check(f"Eq. (14): two-phase transseries = exact Z_N(i eta) to better than {b} (N = 40, eta = {eta}, 40 terms)",
              re(v) < mpf(b), nstr(re(v), 3))
    zeros = ["0.0474759392", "0.1400296650", "0.2294970478", "0.3161807643", "0.4004983984", "0.4828217947", "0.5634587992"]
    bounds = ["2.5e-18", "1.6e-22", "2e-25", "3e-28", "3e-28", "3e-28", "3e-28"]
    for k, (zs, b) in enumerate(zip(zeros, bounds)):
        ex = findroot(lambda e: re(cw.Z_over_P(40, K, mpc(0, 1) * e)) / 1e13, mpf(zs) + mpf("5e-11"), tol=mpf(10) ** -50)
        check(f"Sec 5: exact Lee-Yang zero {k + 1} at N = 40 begins {zs}...", nstr(ex, 15)[: len(zs)] == zs, nstr(ex, 15))

        def off(ex=ex):
            h = mpf(10) ** -12
            return abs(ly_trans(ex) / ((ly_trans(ex + h) - ly_trans(ex - h)) / (2 * h)))
        v = cached(f"LYz_{k}", off)
        check(f"Sec 5: transseries reproduces zero {k + 1} to better than {b}", re(v) < mpf(b), nstr(re(v), 3))
    ms = findroot(lambda m: m - tanh(K * m), mpf("0.85"))
    x = (1 + ms) / 2
    med, hd = borel.median_sum(cw.series(x, K, mpf(0), 40), mpf(1) / 40, beta=-mpf(1) / 2, eps=0.1)
    pref, Zx = 2 * exp(40 * cw.Phi0(x, K, 0)), cw.Z_over_P(40, K, 0)
    check("Sec 5 (H = 0, N = 40): lateral sums carry imaginary parts of relative size 5e-3; median gives Z_N to 5e-15",
          rounds_to(abs(pref * hd) / Zx, "5e-3") and rounds_to(abs(pref * med.real / Zx - 1), "5e-15"),
          f"{nstr(abs(pref * hd) / Zx, 3)}, {nstr(abs(pref * med.real / Zx - 1), 3)}")
    # zero series
    mp.dps = 80
    a, m0, A = Z.zero_series(K, pi / 16, checkpoint=Path(__file__).resolve().parent / ".cache_zero_series.json")
    stated = ["0.2249856434", "0.1610133589", "0.7499786", "2.0878070", "-29.43", "-586.1"]
    check("Sec 5: a_0..a_5 = " + ", ".join(stated), all(rounds_to(abs(a[n]), st.lstrip("-")) and (a[n] < 0) == st.startswith("-")
                                                      for n, st in enumerate(stated)), ", ".join(nstr(a[n], 8) for n in range(6)))
    kap = 1 - K * (1 - m0**2)
    check("Sec 5: a_1 = arg(kappa) / (2 Re m)", abs(a[1] - im(log(kap)) / (2 * re(m0))) < mpf(10) ** -30)
    q = cw.series((1 + m0) / 2, K, mpc(0, 1) * a[0], 2)
    F0pp = -im((1 - m0**2) / kap)
    F1p = -re(K * m0 * (1 - m0**2) / kap**2)
    a2 = -(F0pp / 2 * a[1] ** 2 + F1p * a[1] + im(q[1] / q[0])) / re(m0)
    check("Sec 5: a_2 closed form (F0'', F1', Im(q1/q0)) = Cauchy-sampled a_2", abs(a2 - a[2]) < mpf(10) ** -25, nstr(a2, 15))

    def F1(e):
        mm = 2 * cw.track_roots(K, [mpc(0, e) * t / 40 for t in range(41)], ms)[-1] - 1
        return -im(log(1 - K * (1 - mm**2))) / 2
    check("Sec 5: F_1' = -Re[K m (1-m^2)/kappa^2] (numerical derivative)", abs(diff(F1, a[0]) - F1p) < mpf(10) ** -20)
    s = pi / 16
    check("Sec 5: barrier action A = 0.05750 + 0.19635 i with Im A = s = pi/16", abs(im(A) - s) < mpf(10) ** -40 and
          rounds_to(re(A), "0.05750") and rounds_to(im(A), "0.19635"), nstr(A, 10))
    theta = a[0] - mpc(0, 1) * K * (-(m0.conjugate()))  # theta = eta - i K m_-,  m_- = -conj(m_+)
    check("Remark 5.1: theta = eta + K tan(theta) and Im theta / K = Re m_+ (Kabluchko's density)",
          abs(theta - a[0] - K * __import__("mpmath").tan(theta)) < mpf(10) ** -40 and abs(im(theta) / K - re(m0)) < mpf(10) ** -40)
    note("Remark 5.1: that Kabluchko's limiting density has support R for K >= 1 is cited from his Remark 2.8")
    for M, st in ((8, "0.0617+0.2028j"), (14, "0.0587+0.1988j")):
        p = min(borel.borel_pade_poles(a[1:], 0, L=M, M=M), key=lambda z: abs(z - A))
        check(f"Sec 5: nearest [{M}/{M}] Pade pole = {st}", rounds_to(re(p), st.split("+")[0]) and rounds_to(im(p), st.split("+")[1][:-1]), nstr(p, 6))
    p = min(borel.borel_pade_poles(a[1:], 0), key=lambda z: abs(z - A))
    check("Fig. 2(a): nearest [15/14] pole at distance 0.0026 from A", rounds_to(abs(p - A), "0.0026"), nstr(abs(p - A), 3))
    table2 = {24: ("0.233029526069293597", "4.4e-5", 6, "4.1e-9"), 40: ("0.229497047871485927", "9.0e-7", 11, "1.6e-11"),
              56: ("0.228108088420070792", "1.2e-8", 13, "1.5e-14"), 88: ("0.226914660426714395", "1.7e-11", 23, "1.1e-18"),
              136: ("0.226210845282279990", "7.5e-16", 23, "2.4e-23"), 200: ("0.225809700417462746", None, None, "5.7e-28"),
              328: ("0.225483565311543875", None, None, "1.8e-34")}
    x = (1 + m0) / 2
    bp_all = {}
    for k in range(1, 41):
        N = 8 * (2 * k + 1)
        eps = mpf(1) / N

        def zf(e, N=N):
            return re(cw.Z_over_P(N, K, mpc(0, e))) * exp(-N * re(cw.Phi0(x, K, mpc(0, e))))
        g = a[0] + a[1] * eps
        ex = findroot(zf, (g - mpf("0.3") * eps, g + mpf("0.3") * eps), solver="anderson", tol=mpf(10) ** -70, verify=False)
        bp = abs(re(a[0] + borel.borel_pade_sum(a[1:], eps, 0, 0.0)) - ex)
        bp_all[N] = bp
        if N in table2:
            exs, ots, otn, bps = table2[N]
            ok = abs(ex - mpf(exs)) <= mpf(10) ** -18 / 2 and rounds_to(bp, bps)
            det = f"exact {nstr(ex, 20)}, BP {nstr(bp, 5)}"
            if ots:
                terms = [abs(a[n] * eps**n) for n in range(1, len(a))]
                nstar = 1 + min(range(len(terms)), key=lambda i: terms[i])
                ot = abs(sum(a[n] * eps**n for n in range(nstar)) - ex)
                ok = ok and nstar == otn and rounds_to(ot, ots) and N * abs(A) < 30
                det += f", optimal truncation {nstr(ot, 2)} ({nstar})"
            else:
                ok = ok and N * abs(A) >= 30
            check(f"Table 2: N = {N} (s = pi/16)", ok, det)
    check("Fig. 2(b): Borel-Pade error decreases monotonically for N = 8(2k+1), k = 1..40",
          all(bp_all[8 * (2 * k + 1)] > bp_all[8 * (2 * k + 3)] for k in range(1, 40)))
    import math
    e1 = math.log(float(bp_all[200] / bp_all[328])) / math.log(328 / 200)
    e2 = math.log(float(bp_all[328] / bp_all[648])) / math.log(648 / 328)
    check("Table 2 caption: Borel-Pade error falls roughly like N^-31 (local exponents 30 to 32, N = 200..648)",
          30 <= e1 <= 32 and 30 <= e2 <= 32, f"{e1:.1f}, {e2:.1f}")


if __name__ == "__main__":
    for part in (sec2, sec3, sec4, sec5):
        part()
    CACHE.save()
    npass = sum(ok for _, ok in RESULTS)
    log_(f"===== SUMMARY: {npass}/{len(RESULTS)} checks passed =====")
    for lab, ok in RESULTS:
        if not ok:
            log_(f"  FAILED: {lab}")
    log_(f"{len(NOTES)} NOTE(s) (cited or assumed, not recomputed):")
    for n_ in NOTES:
        log_(f"  - {n_}")
    sys.exit(0 if npass == len(RESULTS) else 1)
