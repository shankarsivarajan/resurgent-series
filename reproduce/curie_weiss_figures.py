"""Figures 1 and 2 of the Curie-Weiss note (written to reproduce/figures/).

Fig. 1  spinodal:  (a) error of the uniform (CFU) expansion vs N, orders t = 0, 2, 4, 6, at zeta_lin = 1;
                   (b) |Im|/Re of the metastable sector vs the exact Airy variable at N = 200, against Ai/Bi and the
                       Gaussian half-thimble.
Fig. 2  Lee-Yang:  (a) Borel plane of the 1/N series of the zero at filling s = pi/16 (Pade poles, actions A, conj A);
                   (b) error of that series vs N = 8(2k+1): best partial sum (30 terms) and Borel-Pade sum.

Data are computed once and cached (reproduce/.cache_figures.json); `python reproduce/curie_weiss_figures.py plot` only
redraws.
Palette: dataviz reference palette, validated (categorical slots 1-3 all-pairs; 4-step blue ordinal ramp).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpmath import airyai, airybi, atanh, exp, findroot, mp, mpc, mpf, re, sqrt  # noqa: E402

from resurgence.checkpoint import Checkpoint  # noqa: E402
from resurgence.models import curie_weiss as cw  # noqa: E402

K = mpf("1.5")
OUT = ROOT / "reproduce" / "figures"
CK = ROOT / "reproduce" / ".cache_figures.json"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # orders t = 0, 2, 4, 6
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e8e7e3", "#9a9893"


def spinodal_consts():
    ms = -sqrt(1 - 1 / K)
    xs = (1 + ms) / 2
    return atanh(ms) - K * ms, (1 / xs**2 - 1 / (1 - xs) ** 2) / 6


def H_of(zl, N):
    Hsp, c = spinodal_consts()
    return Hsp - zl * (3 * c) ** (mpf(1) / 3) / (2 * mpf(N) ** (mpf(2) / 3))


# ------------------------------------------------------------------------------------------- data
def compute():
    from resurgence.models import curie_weiss_uniform as U

    ck = Checkpoint(CK, meta={"K": "1.5", "version": 1})
    # Fig 1a
    mp.dps = 80
    Ns = sorted({int(round(50 * 64 ** (i / 29))) for i in range(30)})
    for N in Ns:
        if ck.has(f"1a_{N}_t6"):
            continue
        H = H_of(mpf(1), N)
        mre, mim = cw.metastable_sector_exact(N, K, H)
        co = U.cfu_coefficients(K, H, 6)
        for t in (0, 2, 4, 6):
            r, i, _ = U.cfu_sector(N, co, t)
            ck.set(f"1a_{N}_t{t}", max(abs(mre / abs(r) - 1), abs(mim / abs(i) - 1)))
        ck.save()
        print("1a", N, flush=True)
    # Fig 1b: exact points
    mp.dps = 50
    N = 200
    for j, zl in enumerate([mpf(z) / 20 for z in (1, 2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 36, 42, 50, 60, 70, 80, 90, 100)]):
        if ck.has(f"1b_{j}_r"):
            continue
        H = H_of(zl, N)
        xp, x0, xm = cw.critical_points(K, H)
        rho = (mpf(3) / 4 * (cw.Phi0(xm, K, H) - cw.Phi0(x0, K, H))) ** (mpf(2) / 3)
        mre, mim = cw.metastable_sector_exact(N, K, H)
        ck.set(f"1b_{j}_z", mpf(N) ** (mpf(2) / 3) * rho)
        ck.set(f"1b_{j}_r", mim / mre)
        ck.save()
        print("1b", j, flush=True)
    # Fig 1b: exact points at N = 1600 (same zeta_lin grid)
    for j, zl in enumerate([mpf(z) / 20 for z in (1, 2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 36, 42, 50, 60, 70, 80, 90, 100)]):
        if ck.has(f"1c_{j}_r"):
            continue
        H = H_of(zl, 1600)
        xp, x0, xm = cw.critical_points(K, H)
        rho = (mpf(3) / 4 * (cw.Phi0(xm, K, H) - cw.Phi0(x0, K, H))) ** (mpf(2) / 3)
        mre, mim = cw.metastable_sector_exact(1600, K, H)
        ck.set(f"1c_{j}_z", mpf(1600) ** (mpf(2) / 3) * rho)
        ck.set(f"1c_{j}_r", mim / mre)
        ck.save()
        print("1c", j, flush=True)
    # Fig 1b: Gaussian half-thimble curve (dense)
    for j in range(60):
        if ck.has(f"1g_{j}_g"):
            continue
        zl = mpf(0.03) * (mpf(6) / mpf(0.03)) ** (mpf(j) / 59)
        H = H_of(zl, N)
        xp, x0, xm = cw.critical_points(K, H)
        rho = (mpf(3) / 4 * (cw.Phi0(xm, K, H) - cw.Phi0(x0, K, H))) ** (mpf(2) / 3)
        q0 = cw.series(xm, K, H, 1)[0]
        g = cw.barrier_thimble(N, K, H, "-") / 2 / (q0 * mpf(N) ** (-mpf(1) / 2))
        ck.set(f"1g_{j}_z", mpf(N) ** (mpf(2) / 3) * rho)
        ck.set(f"1g_{j}_g", g)
    ck.save()
    # Fig 2b: exact zeros at s = pi/16, N = 8(2k+1)
    from mpmath import pi as PI

    from resurgence.models import curie_weiss_zeros as Z

    mp.dps = 80
    a, m0, _ = Z.zero_series(K, PI / 16, checkpoint=ROOT / "reproduce" / ".cache_zero_series.json")
    eta0 = a[0]
    x = (1 + m0) / 2
    for k in range(1, 41):
        if ck.has(f"2_{k}"):
            continue
        N = 8 * (2 * k + 1)
        eps = mpf(1) / N

        def zf(e, N=N):
            return re(cw.Z_over_P(N, K, mpc(0, e))) * exp(-N * re(cw.Phi0(x, K, mpc(0, e))))

        g = eta0 + a[1] * eps
        ck.set(f"2_{k}", findroot(zf, (g - mpf("0.3") * eps, g + mpf("0.3") * eps), solver="anderson",
                                   tol=mpf(10) ** -70, verify=False))
        ck.save()
        print("2", k, flush=True)
    ck.save()


# ------------------------------------------------------------------------------------------- plotting
def style():
    import matplotlib as mpl
    from matplotlib import font_manager

    lm = Path("/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf")  # Latin Modern = Computer Modern
    if lm.exists():
        font_manager.fontManager.addfont(str(lm))

    mpl.rcParams.update({
        "text.usetex": False, "font.family": "serif", "font.serif": ["Latin Modern Roman", "cmr10"], "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True, "axes.unicode_minus": False, "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK, "xtick.color": AXIS,
        "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.size": 3,
        "ytick.major.size": 3, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.which": "major",
        "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
        "lines.linewidth": 1.3, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 3.2, "lines.markeredgewidth": 0.5, "lines.markeredgecolor": "white",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    })


def panel_label(ax, s):
    ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontsize=9, color=INK, va="bottom")


def plot():
    OUT.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    import numpy as np
    from mpmath import mpf as M
    from mpmath import nstr

    from mpmath import pi as PI

    from resurgence import borel
    from resurgence.models import curie_weiss_zeros as Z

    style()
    ck = Checkpoint(CK)
    f = lambda key: float(ck.get_mpf(key))  # noqa: E731

    # ---------------- Fig. 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 2.55), constrained_layout=True)
    Ns = sorted({int(k.split("_")[1]) for k in ck.data if k.startswith("1a_")})
    for t, col in zip((0, 2, 4, 6), RAMP):
        y = [f(f"1a_{N}_t{t}") for N in Ns]
        ax1.plot(Ns, y, color=col)
        ax1.text(Ns[-1] * 1.08, y[-1], rf"$t={t}$", color=INK, fontsize=8, va="center")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(45, 5200)
    ax1.set_xlabel(r"$N$")
    ax1.set_ylabel(r"relative error of uniform expansion")
    ax1.set_yticks([1e-2, 1e-6, 1e-10, 1e-14, 1e-18])
    ax1.minorticks_off()
    panel_label(ax1, "(a)")

    jb = sorted({int(k.split("_")[1]) for k in ck.data if k.startswith("1b_") and k.endswith("_r")})
    zb = np.array([f(f"1b_{j}_z") for j in jb])
    rb = np.array([f(f"1b_{j}_r") for j in jb])
    aibi = lambda z: float(airyai(M(z)) / airybi(M(z)))  # noqa: E731
    jg = sorted({int(k.split("_")[1]) for k in ck.data if k.startswith("1g_") and k.endswith("_g")})
    zg = np.array([f(f"1g_{j}_z") for j in jg])
    gg = np.array([f(f"1g_{j}_g") for j in jg])
    keep = (zg > 0.0) & (zg < 5.2)
    ax2.axhline(1.0, color=BLUE, lw=1.3, zorder=2)
    ax2.plot(zg[keep], gg[keep] / np.array([aibi(z) for z in zg[keep]]), color=ORANGE)
    ax2.plot(zb, rb / np.array([aibi(z) for z in zb]), "o", color=INK, markerfacecolor="white", markeredgecolor=INK,
             markeredgewidth=0.6, markersize=3.2, zorder=3)
    jc = sorted({int(k.split("_")[1]) for k in ck.data if k.startswith("1c_") and k.endswith("_r")})
    if jc:
        zc = np.array([f(f"1c_{j}_z") for j in jc])
        rc = np.array([f(f"1c_{j}_r") for j in jc])
        ax2.plot(zc, rc / np.array([aibi(z) for z in zc]), "o", color=INK, markersize=3.0, zorder=3)
    ax2.set_xlim(0, 5.2)
    ax2.set_yticks([0.9, 1.0, 1.1, 1.2])
    ax2.set_ylim(0.86, 1.22)
    ax2.set_xlabel(r"$\zeta=N^{2/3}\rho$")
    ax2.set_ylabel(r"$(|\mathrm{Im}|/\mathrm{Re})\,/\,[\mathrm{Ai}(\zeta)/\mathrm{Bi}(\zeta)]$")
    ax2.text(3.55, 0.975, r"Airy, $\mathrm{Ai}/\mathrm{Bi}$", color=BLUE, fontsize=8)
    ax2.text(1.35, 1.17, r"Gaussian half-thimble, $N=200$", color=ORANGE, fontsize=8)
    ax2.text(3.0, 1.122, r"exact, $N=200$ (open)", color=INK, fontsize=8)
    ax2.text(3.0, 1.102, r"exact, $N=1600$ (filled)", color=INK, fontsize=8)
    panel_label(ax2, "(b)")
    fig.savefig(OUT / "fig_spinodal.pdf")
    plt.close(fig)

    # ---------------- Fig. 2
    mp.dps = 80
    a, m0, _ = Z.zero_series(K, PI / 16, checkpoint=ROOT / "reproduce" / ".cache_zero_series.json")
    eta0 = a[0]
    xp = (1 + m0) / 2
    H0 = mpc(0, 1) * eta0
    xb = cw.track_roots(K, [H0 * t / 40 for t in range(41)], mpf(0))[-1]
    A = cw.Phi0(xp, K, H0) - cw.Phi0(xb, K, H0)
    poles = borel.borel_pade_poles(a[1:], 0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 2.55), constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1, 1.75]})
    pr = np.array([float(re(z)) for z in poles])
    pi_ = np.array([float(z.imag) for z in poles])
    sel = (pr > -0.1) & (pr < 0.3) & (np.abs(pi_) < 0.42)
    ax1.axhline(0, color=AXIS, lw=0.6, zorder=1)
    ax1.axvline(0, color=AXIS, lw=0.6, zorder=1)
    ax1.plot(pr[sel], pi_[sel], "o", color=BLUE, markersize=3.0, zorder=3)
    Ar, Ai_ = float(re(A)), float(A.imag)
    for sgn, lab in ((1, r"$\mathcal{A}$"), (-1, r"$\bar{\mathcal{A}}$")):
        ax1.plot([Ar], [sgn * Ai_], marker="x", color=ORANGE, markeredgecolor=ORANGE, markersize=5.5, markeredgewidth=1.3, zorder=4)
        ax1.text(Ar - 0.06, sgn * Ai_ + (0.0 if sgn > 0 else -0.035), lab, color=INK, fontsize=9, ha="right")
    ax1.set_xlim(-0.1, 0.3)
    ax1.set_ylim(-0.42, 0.42)
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"$\mathrm{Re}\,\xi$")
    ax1.set_ylabel(r"$\mathrm{Im}\,\xi$")
    ax1.set_xticks([0, 0.1, 0.2])
    ax1.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
    panel_label(ax1, "(a)")

    ks = sorted(int(k.split("_")[1]) for k in ck.data if k.startswith("2_"))
    Nz, best, bp = [], [], []
    for k in ks:
        N = 8 * (2 * k + 1)
        eps = M(1) / N
        ex = ck.get_mpf(f"2_{k}")
        terms = [abs(a[n] * eps**n) for n in range(1, len(a))]
        nstar = 1 + min(range(len(terms)), key=lambda i: terms[i])
        b = abs(sum(a[n] * eps**n for n in range(nstar)) - ex) if N * abs(A) < 30 else None
        s = re(eta0 + borel.borel_pade_sum(a[1:], eps, 0, 0.0))
        Nz.append(N)
        best.append(float(b) if b is not None else np.nan)
        bp.append(float(abs(s - ex)) if abs(s - ex) > 0 else 1e-60)
    ax2.plot(Nz, best, "-o", color=ORANGE, markersize=2.6)
    ax2.plot(Nz, bp, "-o", color=BLUE, markersize=2.6)
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$N$")
    ax2.set_ylabel(r"$|\eta_k-\eta_k^{\mathrm{exact}}|$")
    ax2.set_xlim(0, 680)
    ax2.minorticks_off()
    i2 = int(len(Nz) * 0.62)
    ax2.text(150, 3e-12, "optimal truncation", color=ORANGE, fontsize=8, ha="left")
    ax2.text(Nz[i2], bp[i2] / 3e5, "Borel–Padé sum", color=BLUE, fontsize=8, ha="center")
    panel_label(ax2, "(b)")
    fig.savefig(OUT / "fig_leeyang.pdf")
    plt.close(fig)
    print("A =", nstr(A, 10), " poles shown:", int(sel.sum()))


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "compute":
        compute()
    if len(sys.argv) < 2 or sys.argv[1] == "plot":
        plot()
