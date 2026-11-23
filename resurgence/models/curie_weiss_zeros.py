r"""Lee-Yang zeros of the finite-N Curie-Weiss model: the 1/N series of each zero location.

On H = i eta the zero condition is  N Im Phi0(x+(i eta)) + arg A+(N, i eta) = (k + 1/2) pi.  With eps = 1/N,
s = (k + 1/2) pi eps and  log sum_m q_m eps^m = sum_j l_j eps^j,
    F(eta, eps) = Im Phi0 + sum_{j>=0} eps^{j+1} Im l_j(eta) = s,   so   eta_k = sum_n a_n(s) eps^n.
Closed forms:  int_0^{a_0} Re m(i eta) d eta = s,   a_1 = arg(1 - K(1 - m^2)) / (2 Re m).
Higher orders: Taylor coefficients in eta of the F_j by Cauchy sampling of Im G(eta) = (G(eta) - conj G(conj eta))/2i
(analytic in eta) on a circle around a_0, then formal inversion.
"""
from __future__ import annotations

from mpmath import exp, findroot, im, log, mpc, mpf, pi, re, tanh

from .. import series as ps
from ..checkpoint import Checkpoint
from . import curie_weiss as cw


def _m_of(K, eta, mstar):
    path = [mpc(0, eta * t / 40) for t in range(41)]
    return cw.track_roots(K, path, mstar)[-1] * 2 - 1


def _G_at(K, eta, m_start, J):
    H = mpc(0, 1) * eta
    m = findroot(lambda mm: mm - tanh(K * mm + H), m_start)
    x = (1 + m) / 2
    return [cw.Phi0(x, K, H)] + ps.log_series(cw.series(x, K, H, J + 1), J)


def zero_series(K, s, J=30, P=80, R="0.04", checkpoint=None, eta_guess="0.23", verbose=False):
    """1/N series of the Lee-Yang zero at filling s on the imaginary-field axis.

    Returns (a, m0, A): a[0..J] with eta = sum a_n N^-n; m0 = m_+(i a_0); A = complex barrier action at a_0.
    `checkpoint`: path for resumable circle samples (each sample is one all-orders Laplace series)."""
    from mpmath import mp

    K, s, R = mpf(K), mpf(s), mpf(R)
    mstar = findroot(lambda m: m - tanh(K * m), mpf("0.9"))
    eta0 = findroot(lambda e: im(cw.Phi0((1 + _m_of(K, e, mstar)) / 2, K, mpc(0, 1) * e)) - s, mpf(eta_guess))
    m0 = _m_of(K, eta0, mstar)
    ck = None
    if checkpoint is not None:
        ck = Checkpoint(checkpoint, meta={"K": str(K), "J": J, "P": P, "R": str(R), "dps": mp.dps})
    G = []
    for p in range(P):
        key = f"p{p}"
        if ck is not None and ck.has(key + "_0r"):
            G.append([mpc(ck.get_mpf(f"{key}_{j}r"), ck.get_mpf(f"{key}_{j}i")) for j in range(J + 1)])
            continue
        g = _G_at(K, eta0 + R * exp(2j * pi * p / P), m0, J)
        if ck is not None:
            for j, v in enumerate(g):
                ck.set(f"{key}_{j}r", re(v))
                ck.set(f"{key}_{j}i", im(v))
        G.append(g)
        if verbose:
            print(f"  sample {p + 1}/{P}", flush=True)
    if ck is not None:
        ck.save()
    Gc = _G_at(K, eta0, m0, J)
    for g in G:  # unwrap the logarithmic branches (Phi0, log q0) against the centre
        for j in (0, 1):
            while im(g[j] - Gc[j]) > pi:
                g[j] -= 2j * pi
            while im(g[j] - Gc[j]) < -pi:
                g[j] += 2j * pi
    F = [[(G[p][j] - G[(P - p) % P][j].conjugate()) / 2j for p in range(P)] for j in range(J + 1)]
    c = [[re(sum(F[j][p] * exp(-2j * pi * n * p / P) for p in range(P)) / P) / R**n for n in range(J + 1)]
         for j in range(J + 1)]
    a = [mpf(0)] * (J + 1)
    for n in range(1, J + 1):  # formal inversion of sum_j eps^j sum_k c[j][k] delta^k = s
        pw = [[mpf(1)] + [mpf(0)] * J]
        for _ in range(J):
            pw.append(ps.mul(pw[-1], a, J + 1))
        lhs = [mpf(0)] * (J + 1)
        for j in range(J + 1):
            for k in range(J + 1 - j):
                if c[j][k] != 0:
                    for t in range(J + 1 - j):
                        lhs[t + j] += c[j][k] * pw[k][t]
        a[n] = -lhs[n] / c[0][1]
    a[0] = eta0
    H0 = mpc(0, 1) * eta0
    xb = cw.track_roots(K, [H0 * t / 40 for t in range(41)], mpf(0))[-1]  # barrier, continued from x = 1/2
    A = cw.Phi0((1 + m0) / 2, K, H0) - cw.Phi0(xb, K, H0)
    return a, m0, A


def zero_first_orders(K, m):
    """Closed form a_1 = arg(1 - K(1 - m^2)) / (2 Re m) at m = m_+(i a_0)."""
    return im(log(1 - K * (1 - m**2))) / (2 * re(m))
