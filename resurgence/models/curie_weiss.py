r"""Curie-Weiss (mean-field) Ising model at finite N: stable/metastable phases as a transseries.

Z_N = sum_{k=0}^{N} C(N,k) exp(N [K m^2/2 + H m]),  m = (2k - N)/N,  K = beta J, H = beta h.

With k = N x and the generalized Stirling series,
  log T(k) = log Gamma(N+1) - (N+1) log N + N - log(2 pi) + N Phi0(x) + Phi1(x) + sum_k N^{-k} Phi_{k+1}(x),
  Phi0 = -x log x - (1-x) log(1-x) + K (2x-1)^2/2 + H (2x-1),
  Phi1 = -(1/2) log(x (1-x)),
  Phi_{k+1} = -(-1)^{k+1} B_{k+1} / (k(k+1)) [x^{-k} + (1-x)^{-k}].
Sum -> integral is exact to all orders, so with P_N = Gamma(N+1) e^N N^{-N} / (2 pi):
  Z_N / P_N  ~  e^{N Phi0(x+)} A_+(N) + e^{N Phi0(x-)} A_-(N),   A_pm = sum_m q_m^pm N^{-m-1/2},
x_pm the stable / metastable maxima (m = tanh(K m + H)), x_0 the barrier (minimum) between them.
"""
from __future__ import annotations

from mpmath import bernoulli, exp, findroot, log, loggamma, mp, mpf, mpmathify, re, tanh

from ..laplace import laplace_coefficients, taylor_invpow, taylor_log


def Z_over_P(N: int, K, H):
    """Exact Z_N / P_N (finite sum), computed in logs for stability."""
    K, H = mpf(K), mpmathify(H)
    logP = loggamma(N + 1) + N - N * log(N) - log(2 * mp.pi)
    terms = []
    for k in range(N + 1):
        m = mpf(2 * k - N) / N
        terms.append(loggamma(N + 1) - loggamma(k + 1) - loggamma(N - k + 1) + N * (K * m * m / 2 + H * m))
    mx = max(re(t) for t in terms)
    return exp(mx - logP) * sum(exp(t - mx) for t in terms)


def Phi0(x, K, H):
    x, K, H = mpmathify(x), mpf(K), mpmathify(H)
    m = 2 * x - 1
    return -x * log(x) - (1 - x) * log(1 - x) + K * m * m / 2 + H * m


def critical_points(K, H):
    """(x_plus, x_0, x_minus): stable max, barrier min, metastable max (needs K > 1, |H| < H_sp)."""
    K, H = mpf(K), mpf(H)
    f = lambda m: m - tanh(K * m + H)  # noqa: E731
    # bracket sign changes on a fine grid (robust near the spinodal, where two roots nearly merge)
    grid = [mpf(-1) + mpf(2) * i / 4000 for i in range(1, 4000)]
    vals = [f(m) for m in grid]
    roots = []
    for a, b, fa, fb in zip(grid, grid[1:], vals, vals[1:]):
        if fa == 0:
            roots.append(a)
        elif fa * fb < 0:
            roots.append(findroot(f, (a, b), solver="anderson"))
    roots.sort(reverse=True)
    if len(roots) != 3:
        raise ValueError(f"need three roots (below T_c, inside spinodal), got {roots}")
    mp_, m0, mm = roots
    if H < 0:
        mp_, mm = mm, mp_
    return tuple((1 + r) / 2 for r in (mp_, m0, mm))


def _taylor(x0, K, H, M):
    """Taylor coefficients at x0 of Phi0 (derivative-integrated), Phi1 and Phi_{k+1} (k=1..M)."""
    x0, K, H = mpmathify(x0), mpf(K), mpmathify(H)
    n = 2 * M + 6
    lx = taylor_log(0, x0, n)
    l1x = taylor_log(1, x0, n)  # log(x - 1): same derivative coefficients as log(1 - x)
    l1x[0] = log(1 - x0)
    # Phi0' = -log x + log(1-x) + 2K(2x-1) + 2H
    d = [-lx[p] + l1x[p] for p in range(n)]
    d[0] += 2 * K * (2 * x0 - 1) + 2 * H
    d[1] += 4 * K
    phi = [Phi0(x0, K, H)] + [d[p - 1] / p for p in range(1, n)]
    phi1 = [-(lx[p] + l1x[p]) / 2 for p in range(n)]
    higher = []
    for k in range(1, M + 1):
        B = bernoulli(k + 1)
        if B == 0:
            higher.append([mpf(0)] * n)
            continue
        a = taylor_invpow(0, x0, k, n)
        b = taylor_invpow(1, x0, k, n)
        c = -((-1) ** (k + 1)) * B / (k * (k + 1))
        higher.append([c * (a[p] + (-1) ** k * b[p]) for p in range(n)])
    return phi, phi1, higher


def series(x0, K, H, M: int):
    """q_0..q_{M-1} of A(N) = sum_m q_m N^{-m-1/2} for the maximum x0 (stable or metastable)."""
    phi, phi1, higher = _taylor(x0, K, H, M)
    ph = [mpf(0)] + [mpf(0)] + phi[2:]
    return laplace_coefficients(ph, [phi1] + higher, M)


def barrier_thimble(N, K, H, which):
    """Leading full imaginary thimble through the barrier, normalised to A_which:
       i e^{-N (Phi0(x_which) - Phi0(x_0))} e^{Phi1(x_0)} sqrt(2 pi / (N Phi0''(x_0)))."""
    xp, x0, xm = critical_points(K, H)
    xw = xp if which == "+" else xm
    phi, phi1, _ = _taylor(x0, K, H, 1)
    return exp(-N * (Phi0(xw, K, H) - Phi0(x0, K, H))) * exp(phi1[0]) * (2 * mp.pi / (N * 2 * phi[2])) ** mpf(0.5)


def track_roots(K, H_path, start_m):
    """Continue a root of m = tanh(K m + H) along a list of (complex) fields, from start_m at H_path[0].
    Returns the list of x = (1+m)/2."""
    K = mpf(K)
    m = mpmathify(start_m)
    out = []
    for H in H_path:
        H = mpmathify(H)
        m = findroot(lambda mm: mm - tanh(K * mm + H), m)
        out.append((1 + m) / 2)
    return out


def lee_yang_leading(K, s):
    """Leading terms of the 1/N series of the Lee-Yang zero at filling s = (k + 1/2) pi / N on H = i eta:
        eta_k = eta0(s) + a1(s)/N + O(1/N^2),
        int_0^{eta0} Re m(i eta') d eta' = s,   a1 = arg(1 - K(1 - m^2)) / (2 Re m),   m = m_+(i eta0).
    Returns (eta0, a1, m)."""
    from mpmath import arg, im, mpc

    K, s = mpf(K), mpf(s)
    mstar = findroot(lambda m: m - tanh(K * m), mpf("0.9"))

    def m_at(eta):
        return 2 * track_roots(K, [mpc(0, eta) * t / 40 for t in range(41)], mstar)[-1] - 1

    def F0(eta):  # Im Phi0 along the stable branch; d/d eta = Re m
        m = m_at(eta)
        return im(Phi0((1 + m) / 2, K, mpc(0, eta)))

    eta0 = findroot(lambda e: F0(e) - s, s / mstar)
    m = m_at(eta0)
    return eta0, arg(1 - K * (1 - m**2)) / (2 * re(m)), m


def metastable_sector_exact(N, K, H):
    """Exact metastable sector e^{N Phi0(x-)} S[A-] in Z/P units, from the Hubbard-Stratonovich form
        Z = sqrt(N K / 2 pi) int dy (2 cosh(K y + H))^N e^{-N K y^2 / 2}   (entire integrand).
    Median = int_{-inf}^{y0} + Re int_{y0}^{y*},  |half Stokes jump| = |Im int_{y0}^{y*}|,
    y0 = barrier, y* = (i pi/2 - H)/K (a zero of the integrand, where the barrier thimble ends).
    Returns (median, half_jump). Assumes real H > 0 with x- < x0 (metastable on the left)."""
    from mpmath import cosh, im, mpc, quad, sqrt

    K, H = mpf(K), mpf(H)
    xp, x0, xm = critical_points(K, H)
    y0, ym = 2 * x0 - 1, 2 * xm - 1
    Pm = Phi0(xm, K, H)
    logP = loggamma(N + 1) + N - N * log(N) - log(2 * mp.pi)

    def f(y):
        return exp(N * (-K * y * y / 2 + log(2 * cosh(K * y + H))) - N * Pm)

    ystar = mpc(-H / K, mp.pi / (2 * K))
    w = 1 / sqrt(mpf(N))
    left = quad(f, [ym - 40 * w - 2, ym - 8 * w, ym, y0])
    th = quad(f, [y0, ystar])
    norm = sqrt(N * K / (2 * mp.pi)) * exp(N * Pm - logP)
    return norm * (left + re(th)), norm * abs(im(th))
