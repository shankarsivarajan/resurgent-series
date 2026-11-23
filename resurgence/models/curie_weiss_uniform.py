r"""Uniform (Chester-Friedman-Ursell) expansion of the Curie-Weiss metastable sector through the spinodal.

Map Phi0(x) = B + rho u - u^3/3 (u = +a <-> metastable max x-, u = -a <-> barrier x0, a = sqrt(rho)), with
    rho^{3/2} = (3/4)(Phi0(x-) - Phi0(x0)),   B = (Phi0(x-) + Phi0(x0))/2.
With G(u) = g(x(u)) |dx/du| and g = exp(Phi1 + sum_k N^-k Phi_{k+1}), integration by parts
    G_k = p_k + q_k u + (u^2 - rho) H_k,   G_{k+1} = H_k'
gives, for either lateral contour,
    e^{N Phi0(x-)} S_pm[A-] = pi e^{N B} N^{-1/3} [ P(N) (Bi pm i Ai)(zeta) + N^{-1/3} Q(N) (Bi' pm i Ai')(zeta) ],
    zeta = N^{2/3} rho,  P = sum p_k N^-k,  Q = sum q_k N^-k  (contour independent).
Two-point implementation: only Taylor data at u = +-a are used (local square-root maps and series reversion).
"""
from __future__ import annotations

from mpmath import airyai, airybi, exp, mpf, pi, sqrt

from .. import series as ps
from . import curie_weiss as cw


def _compose(f, d, n):
    """f(d(v)) for a Taylor list f in d and a series d(v) with d[0] = 0."""
    out = [mpf(0)] * n
    pw = [mpf(1)] + [mpf(0)] * (n - 1)
    for p in range(n):
        if p < len(f):
            for t in range(n):
                out[t] += f[p] * pw[t]
        pw = ps.mul(pw, d, n)
    return out


def _G_series(K, xc, a_sign, a, H, sigma, nj, D):
    """Taylor coefficients in v (u = a_sign*a + v) of G^{(j)}(u), j < nj, around the critical point xc."""
    phi, phi1, higher = cw._taylor(xc, K, H, max(nj, D // 2 + 2))
    n = D + 2
    # Phi0(xc + d) - Phi0(xc) = sum_{p>=2} phi_p d^p  ==  -a_sign a v^2 (1 + a_sign v / (3a))
    lhs = ps.sqrt_series([mpf(1)] + [phi[p] / phi[2] for p in range(3, n + 2)], n)
    rhs = ps.sqrt_series([mpf(1), a_sign / (3 * a)] + [mpf(0)] * (n - 2), n)
    d_of_w = ps.revert([mpf(0)] + [sqrt(abs(phi[2])) * c for c in lhs[: n - 1]], n)
    w_of_v = [mpf(0)] + [sigma * sqrt(a) * c for c in rhs[: n - 1]]
    d_v = _compose(d_of_w, w_of_v, n)
    dd_v = ps.deriv(d_v) + [mpf(0)]
    e1 = ps.exp_series(phi1, n)
    E = [[mpf(1)] + [mpf(0)] * (n - 1)]
    for j in range(1, nj):
        acc = [mpf(0)] * n
        for i in range(1, j + 1):
            prod = ps.mul(higher[i - 1], E[j - i], n)
            for t in range(n):
                acc[t] += i * prod[t]
        E.append([x / j for x in acc])
    return [ps.mul(_compose(ps.mul(e1, E[j], n), d_v, n), dd_v, n) for j in range(nj)]


def cfu_coefficients(K, H, T=6):
    """(B, rho, p, q) with p[j][k], q[j][k] the order-N^{-(j+k)} coefficients (j: Stirling, k: CFU), j + k <= T."""
    K, H = mpf(K), mpf(H)
    D = 2 * T + 6
    xp, x0, xm = cw.critical_points(K, H)
    Pm, Pb = cw.Phi0(xm, K, H), cw.Phi0(x0, K, H)
    rho = (mpf(3) / 4 * (Pm - Pb)) ** (mpf(2) / 3)
    a = sqrt(rho)
    sigma = 1 if xm > x0 else -1  # orientation dx/du (u = +a <-> xm)
    Gp = _G_series(K, xm, +1, a, H, sigma, T + 1, D)
    Gm = _G_series(K, x0, -1, a, H, sigma, T + 1, D)
    p = [[mpf(0)] * (T + 1) for _ in range(T + 1)]
    q = [[mpf(0)] * (T + 1) for _ in range(T + 1)]
    for j in range(T + 1):
        A_, Bm = Gp[j], Gm[j]
        for k in range(T + 1 - j):
            pk, qk = (A_[0] + Bm[0]) / 2, (A_[0] - Bm[0]) / (2 * a)
            p[j][k], q[j][k] = pk, qk
            n = len(A_)
            numA = [mpf(0), A_[1] - qk] + A_[2:]
            numB = [mpf(0), Bm[1] - qk] + Bm[2:]
            hA = ps.mul(numA[1:], [(-1) ** t / (2 * a) ** (t + 1) for t in range(n)], n - 1)  # / (u + a) at u = +a
            hB = ps.mul(numB[1:], [-1 / (2 * a) ** (t + 1) for t in range(n)], n - 1)  # / (u - a) at u = -a
            A_, Bm = ps.deriv(hA), ps.deriv(hB)
    # G = g |dx/du|: with x- < x0 mapped to u = +a > -a, dx/du < 0, so flip the sign (sigma = -1) to make c_00 > 0
    p = [[sigma * c for c in row] for row in p]
    q = [[sigma * c for c in row] for row in q]
    return (Pm + Pb) / 2, rho, p, q


def cfu_sector(N, coeffs, order):
    """(Re, Im, zeta) of e^{N Phi0(x-)} S_up[A-] (signed; Re = median, Im = +half-jump) at total order `order`."""
    B, rho, p, q = coeffs
    T = len(p) - 1
    zeta = mpf(N) ** (mpf(2) / 3) * rho
    P = sum(p[j][k] * mpf(N) ** -(j + k) for j in range(T + 1) for k in range(T + 1 - j) if j + k <= order)
    Q = sum(q[j][k] * mpf(N) ** -(j + k) for j in range(T + 1) for k in range(T + 1 - j) if j + k <= order)
    pref = exp(N * B) * mpf(N) ** (-mpf(1) / 3) * pi
    s = mpf(N) ** (-mpf(1) / 3)
    re_ = pref * (P * airybi(zeta) + s * Q * airybi(zeta, 1))
    im_ = pref * (P * airyai(zeta) + s * Q * airyai(zeta, 1))
    return re_, im_, zeta
