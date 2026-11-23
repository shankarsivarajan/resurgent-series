r"""All-orders Laplace method with an N-dependent amplitude.

Computes q_m in

    int e^{N [phi(x) - phi(x0)]} exp( sum_{j>=0} N^{-j} psi_j(x) ) dx  ~  sum_m q_m N^{-m-1/2}

around a non-degenerate maximum x0 of phi, given Taylor coefficients of phi and
psi_j at x0. Method: substitute phi(x0) - phi(x0 + d) = w^2/2, revert d(w), and
use  int e^{-N w^2/2} w^{2n} dw = Gamma(n+1/2) 2^{n+1/2} N^{-n-1/2}.

Cost ~ O(M^4) mp operations for M output coefficients (fine up to M ~ 80).
"""
from __future__ import annotations

from mpmath import gamma, mpf, mpmathify

from . import series as ps


def laplace_coefficients(phi, psis, M: int):
    """phi: Taylor coeffs of the phase at its maximum (phi[1] == 0, phi[2] < 0).
    psis: list of Taylor-coefficient lists; psis[j] multiplies N^{-j} in the amplitude exponent.
    Returns q_0..q_{M-1}."""
    D = 2 * M + 1  # w-degree needed (+1 margin)
    # w = d * g(d),  g = sqrt(-2 phi_2) * sqrt(1 + sum_{p>=3} (phi_p/phi_2) d^{p-2})
    inner = [mpf(1)] + [phi[p] / phi[2] for p in range(3, D + 2)]
    g = ps.sqrt_series(inner, D)
    s2 = (-2 * phi[2]) ** mpf(0.5)
    w_of_d = [0] + [s2 * x for x in g]
    d_of_w = ps.revert(w_of_d, D + 1)
    dd = ps.deriv(d_of_w)  # d'(w)
    # amplitude: E_j(d) for j < M, degree <= D - 2j
    psi0 = psis[0]
    E0 = ps.exp_series(psi0, D + 1)
    G = [[mpf(1)] + [0] * D]
    for j in range(1, M):
        n_j = D + 1 - 2 * j
        acc = [0] * n_j
        for i in range(1, j + 1):
            if i >= len(psis):
                break
            prod = ps.mul(psis[i], G[j - i], n_j)
            for t in range(n_j):
                acc[t] += i * prod[t]
        G.append([x / j for x in acc])
    # powers of d(w)
    powers = [[mpf(1)] + [0] * D]
    for _ in range(D):
        powers.append(ps.mul(powers[-1], d_of_w, D + 1))
    q = [mpf(0)] * M
    for j in range(M):
        n_j = D + 1 - 2 * j
        Ej = ps.mul(E0, G[j], n_j)
        F = [0] * (D + 1)
        for p, e in enumerate(Ej):
            if e == 0:
                continue
            Pp = powers[p]
            for t in range(p, D + 1):
                F[t] += e * Pp[t]
        F = ps.mul(F, dd, D + 1)
        for n in range(M - j):
            q[n + j] += F[2 * n] * gamma(n + mpf(1) / 2) * 2 ** (n + mpf(1) / 2)
    return q


# ----------------------------------------------------------- Taylor helpers
def taylor_log(c, x0, n):
    """log(x - c) around x0."""
    from mpmath import log

    w0 = mpmathify(x0) - mpmathify(c)
    return [log(w0)] + [(-1) ** (p + 1) / (p * w0**p) for p in range(1, n)]


def taylor_xlogx(c, x0, n):
    """(x - c) log(x - c) around x0."""
    from mpmath import log

    w0 = mpmathify(x0) - mpmathify(c)
    out = [w0 * log(w0), log(w0) + 1]
    out += [(-1) ** p / (p * (p - 1) * w0 ** (p - 1)) for p in range(2, n)]
    return out[:n]


def taylor_invpow(c, x0, k, n):
    """(x - c)^{-k} around x0."""
    from mpmath import binomial

    w0 = mpmathify(x0) - mpmathify(c)
    return [(-1) ** p * binomial(k + p - 1, p) * w0 ** (-k - p) for p in range(n)]
