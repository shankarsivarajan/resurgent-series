"""Borel transforms, Borel-Pade singularity maps and (lateral / median) resummation.

Convention: an asymptotic series in a small parameter z,

    f(z) ~ sum_{n>=0} c_n z^{n + beta + 1},

has Borel transform  B(zeta) = zeta^beta * sum_n c_n zeta^n / Gamma(n + beta + 1)
and Borel sum  S_theta f(z) = int_0^{infinity e^{i theta}} e^{-zeta/z} B(zeta) d zeta.

Examples: a series sum c_n N^{-n-1/2} (Laplace-method output) is beta = -1/2
with z = 1/N; a series sum d_n N^{-n-1} is beta = 0.

The polynomial part sum_n c_n zeta^n / Gamma(n+beta+1) is continued with a
diagonal-ish Pade approximant. Poles of the Pade denominator accumulating on a
ray mark branch cuts of the true Borel transform; isolated stable poles mark
genuine poles.
"""
from __future__ import annotations

from collections.abc import Sequence

from mpmath import exp, gamma, inf, mp, mpc, mpf, pade, polyroots, polyval, quad


def borel_coefficients(c: Sequence, beta=0) -> list:
    """b_n = c_n / Gamma(n + beta + 1)."""
    return [c[n] / gamma(n + beta + 1) for n in range(len(c))]


def pade_approximant(a: Sequence, L: int | None = None, M: int | None = None):
    """[L/M] Pade of the power series sum a_n x^n. Returns (p, q), ascending."""
    n = len(a)
    if L is None and M is None:
        M = (n - 1) // 2
        L = n - 1 - M
    elif L is None:
        L = n - 1 - M
    elif M is None:
        M = n - 1 - L
    # Degenerate (near-rational) inputs make high-order Pade singular: back off.
    while True:
        try:
            return pade(list(a[: L + M + 1]), L, M)
        except ZeroDivisionError:
            if M <= 1:
                raise
            M -= 1
            L -= 1


def pade_eval(p, q, x):
    return polyval(p[::-1], x) / polyval(q[::-1], x)


def pade_poles(a: Sequence, L: int | None = None, M: int | None = None) -> list:
    """Poles (zeros of the denominator) of the [L/M] Pade of sum a_n x^n."""
    _, q = pade_approximant(a, L, M)
    # strip trailing (highest-order) zeros
    while len(q) > 1 and q[-1] == 0:
        q = q[:-1]
    return polyroots(q[::-1], maxsteps=200, extraprec=4 * mp.dps)


def borel_pade_poles(c: Sequence, beta=0, L=None, M=None) -> list:
    """Singularity map: Pade poles of the Borel transform's regular part."""
    return pade_poles(borel_coefficients(c, beta), L, M)


def borel_pade_sum(c: Sequence, z, beta=0, theta=0.0, L=None, M=None):
    """Lateral Borel-Pade sum along the ray arg(zeta) = theta.

    Returns approx  sum c_n z^{n+beta+1}  resummed. For a real series with
    singularities on the positive axis use small theta = +-eps and/or
    ``median_sum``.
    """
    p, q = pade_approximant(borel_coefficients(c, beta), L, M)
    e = exp(mpc(0, theta))

    def integrand(u):
        zeta = u * e
        return exp(-zeta / z) * zeta**beta * pade_eval(p, q, zeta) * e

    return quad(integrand, [0, mpf(1) * abs(z), 10 * abs(z), inf])


def median_sum(c: Sequence, z, beta=0, eps=0.15, L=None, M=None):
    """Average of the two lateral sums at angles +-eps.

    For real coefficients this equals the principal-value (median) Borel sum,
    provided there are no Borel singularities with 0 < |arg zeta| < eps.
    """
    up = borel_pade_sum(c, z, beta, +eps, L, M)
    dn = borel_pade_sum(c, z, beta, -eps, L, M)
    return (up + dn) / 2, (up - dn) / 2  # (median, half-discontinuity)


def optimal_truncation(c: Sequence, z, beta=0):
    """Sum up to the smallest term. Returns (value, n_stop, smallest_term)."""
    terms = [c[n] * z ** (n + beta + 1) for n in range(len(c))]
    mags = [abs(t) for t in terms]
    n_stop = min(range(len(mags)), key=lambda i: mags[i])
    return sum(terms[:n_stop]), n_stop, mags[n_stop]
