"""Figures 1 and 2 of the Kerr-cat note (written to reproduce/figures/).

Fig. 1  split:   (a) normalized ground splitting S_0 (4 eps2)^{Delta/2} e^{2 eps2} / (8 eps2) vs Delta:
                     exact at eps2 = 10, 30; leading 1/Gamma(-Delta/2); Marthaler-Dykman;
                 (b) large order: c_j^{(n)} / prediction (eq. lo) vs j after order-6 Richardson
                     acceleration, n = 0, 1, 2 at Delta = 1.
Fig. 2  median:  |h_c - S_med|, |delta_h|, (delta_h/2)^2 vs alpha^2 for Delta = 1/2 (the method of Table 2).

Data are computed once and cached (reproduce/.cache_kerr_cat_figures.json);
`python reproduce/kerr_cat_figures.py plot` only redraws.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flint import fmpq  # noqa: E402
from mpmath import exp, factorial, gamma, log, mp, mpf, pi, sin, sqrt  # noqa: E402

from resurgence import borel  # noqa: E402
from resurgence.checkpoint import Checkpoint  # noqa: E402
from resurgence.largeorder import richardson  # noqa: E402
from resurgence.models import kerr_cat as kc  # noqa: E402

HERE = ROOT / "reproduce"
OUT = HERE / "figures"
CK = HERE / ".cache_kerr_cat_figures.json"
SER = HERE / ".series_kerr_cat"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e8e7e3", "#9a9893"


def md_norm(e2, D):
    """Marthaler-Dykman splitting divided by 8 eps2 (4 eps2)^{-Delta/2} e^{-2 eps2}."""
    lam = 1 / (2 * e2)
    mu = (D + 2) / (2 * e2)
    A = sqrt(mu + 1) + mu * log((1 + sqrt(mu + 1)) / sqrt(mu))
    md = -(64 * e2**2 / sqrt(pi * (D + 2))) * (1 + mu) ** (mpf(5) / 4) * sin(pi * D / 2) * exp(-A / lam)
    return md / (8 * e2 * (4 * e2) ** (-D / 2) * exp(-2 * e2))


# ------------------------------------------------------------------------------------------- data
def compute():
    SER.mkdir(exist_ok=True)
    mp.dps = 60
    ck = Checkpoint(CK, every=60)
    Ds = [mpf(-2) + mpf(i) / 10 for i in range(101)]  # -2 .. 8
    for e2 in (10, 30):
        for D in Ds:
            key = f"S|{e2}|{float(D):.1f}"
            if not ck.has(key):
                e = mpf(e2)
                S = kc.exact_levels(e, D, 1)[0] - kc.exact_levels(e, D, 0)[0]
                ck.set(key, S / (8 * e * (4 * e) ** (-D / 2) * exp(-2 * e)))
    mp.dps = 80
    D = mpf(1) / 2
    e = kc.rs_series(0, fmpq(1, 2), 300, checkpoint=SER / "rs_n0_D1_2_K300.json.gz")
    d = [mpf(int(e[2 * j].p)) / mpf(int(e[2 * j].q)) for j in range(2, 151)]
    for a2 in range(4, 15):
        key = f"med|{a2}"
        if ck.has(key):
            continue
        A2 = mpf(a2)
        eps2 = A2 - D / 2
        x = 1 / A2
        Ee = kc.exact_levels(eps2, D, 0)[0]
        Eo = kc.exact_levels(eps2, D, 1)[0]
        hc = ((Ee + Eo) / 2 + A2**2) / (4 * A2)
        dh = (Eo - Ee) / (4 * A2)
        med, _ = borel.median_sum(d, x, beta=1 - D, eps=mpf("0.1"), L=74, M=73)
        ck.set(key, f"{mp.nstr(abs(x**D * med.real - hc), 20)},{mp.nstr(abs(dh), 20)}")
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
    SER.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt
    import numpy as np

    style()
    ck = Checkpoint(CK)
    mp.dps = 60

    # ---------------- Fig. 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={"wspace": 0.38})
    Dg = np.linspace(-2, 8, 801)
    lead = [float(1 / gamma(-mpf(d) / 2)) if abs(d / 2 - round(d / 2)) > 1e-12 or d < 0 else 0.0 for d in Dg]
    ax1.axhline(0, color=AXIS, lw=0.6, zorder=1)
    ax1.plot(Dg, lead, color=BLUE, zorder=2)
    for e2, mk in ((10, "o"), (30, "s")):
        xs = [-2 + i / 10 for i in range(101)][::4]
        ys = [float(ck.get_mpf(f"S|{e2}|{x:.1f}")) for x in xs]
        ax1.plot(xs, ys, mk, color=INK, markerfacecolor="white" if e2 == 10 else INK, markeredgecolor=INK,
                 markersize=3.4, zorder=3)
    mdx = [d for d in Dg if d > -1.95]
    ax1.plot(mdx, [float(md_norm(mpf(10), mpf(d))) for d in mdx], color=ORANGE, lw=1.0, zorder=2)
    ax1.set_xlabel(r"detuning $\Delta$")
    ax1.set_ylabel(r"$S_0\,(4\epsilon_2)^{\Delta/2}\,\mathrm{e}^{2\epsilon_2}/(8\epsilon_2)$")
    ax1.set_ylim(-1.3, 2.4)
    ax1.text(3.3, 1.2, r"$1/\Gamma(-\Delta/2)$", color=BLUE, fontsize=8)
    ax1.text(-1.9, 2.15, "Marthaler–Dykman, " + r"$\epsilon_2=10$", color=ORANGE, fontsize=8, va="center")
    ax1.text(-1.9, 1.75, r"exact: $\epsilon_2=10$ (open), $30$ (filled)", color=INK, fontsize=7.5, va="center")
    panel_label(ax1, "(a)")

    mp.dps = 120
    cols = {0: BLUE, 1: ORANGE, 2: AQUA}
    D = mpf(1)
    for n in (0, 1, 2):
        e = kc.rs_series(n, fmpq(1), 300, checkpoint=SER / f"rs_n{n}_D1_1_K300.json.gz")
        pref = -exp(2 * D) / (factorial(n) * gamma(n - D / 2)) ** 2
        js = list(range(2, 151))
        r = [mpf(int(e[2 * j].p)) / mpf(int(e[2 * j].q)) / (pref * gamma(j + 4 * n - D) * mpf(4) ** (-j)) for j in js]
        acc = richardson(r, 6, start=2)          # acc[i] uses r at j = i+2 .. i+8
        jj = [i + 2 for i in range(len(acc))]
        sel = [(j, abs(float(a - 1))) for j, a in zip(jj, acc) if j >= 10]
        ax2.semilogy([t[0] for t in sel], [t[1] for t in sel], color=cols[n])
        ax2.text(sel[-1][0] + 2, sel[-1][1], rf"$n={n}$", color=INK, fontsize=8, va="center")
    ax2.set_xlabel(r"order $j$")
    ax2.set_ylabel(r"$|\,$accelerated ratio $-\,1\,|$")
    ax2.set_xlim(0, 160)
    panel_label(ax2, "(b)")
    fig.savefig(OUT / "fig_split.pdf")
    plt.close(fig)

    # ---------------- Fig. 2
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    a2s = list(range(4, 15))
    med = [float(mpf(ck.get(f"med|{a}").split(",")[0])) for a in a2s]
    dh = [float(mpf(ck.get(f"med|{a}").split(",")[1])) for a in a2s]
    ax.semilogy(a2s, dh, color=BLUE)
    ax.semilogy(a2s, [(d / 2) ** 2 for d in dh], color=ORANGE)
    ax.semilogy(a2s, med, "o", color=INK, markerfacecolor=INK, markeredgecolor="white", markersize=3.8)
    ax.text(9.6, dh[6] * 30, r"splitting $|\delta_h|$", color=BLUE, fontsize=8)
    ax.text(4.4, (dh[4] / 2) ** 2 / 3e3, r"$(\delta_h/2)^2$", color=ORANGE, fontsize=8)
    ax.text(9.3, med[7] * 3e3, r"$|h_{\mathrm{c}}-\mathcal{S}_{\mathrm{med}}|$ (points)", color=INK, fontsize=8)
    ax.set_xlabel(r"$\alpha^2$")
    ax.set_ylabel("magnitude")
    fig.savefig(OUT / "fig_median.pdf")
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "compute":
        compute()
    if len(sys.argv) < 2 or sys.argv[1] == "plot":
        plot()
