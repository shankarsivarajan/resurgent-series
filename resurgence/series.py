"""Minimal truncated power-series arithmetic on lists of mpmath numbers (ascending order)."""
from __future__ import annotations

from mpmath import exp, mpf, sqrt


def mul(a, b, n):
    out = [0] * n
    for i, ai in enumerate(a[:n]):
        if ai == 0:
            continue
        for j in range(min(len(b), n - i)):
            out[i + j] += ai * b[j]
    return out


def inv(a, n):
    out = [0] * n
    out[0] = 1 / a[0]
    for k in range(1, n):
        acc = 0
        for j in range(1, min(k, len(a) - 1) + 1):
            acc += a[j] * out[k - j]
        out[k] = -acc / a[0]
    return out


def exp_series(a, n):
    """exp of a power series (a[0] allowed nonzero)."""
    e0 = exp(a[0])
    out = [0] * n
    out[0] = mpf(1)
    for k in range(1, n):
        acc = 0
        for j in range(1, min(k, len(a) - 1) + 1):
            acc += j * a[j] * out[k - j]
        out[k] = acc / k
    return [e0 * x for x in out]


def sqrt_series(a, n):
    """Square root with s[0] = sqrt(a[0])."""
    s = [0] * n
    s[0] = sqrt(a[0])
    for k in range(1, n):
        acc = a[k] if k < len(a) else 0
        for j in range(1, k):
            acc -= s[j] * s[k - j]
        s[k] = acc / (2 * s[0])
    return s


def deriv(a):
    return [k * a[k] for k in range(1, len(a))]


def revert(w, n):
    """Series reversion: given w(d) = sum_{p>=1} w_p d^p (w_1 != 0), return d(w) to n terms.

    Lagrange inversion: d = sum r_k w^k,  r_k = (1/k) [d^{k-1}] (d/w(d))^k.
    """
    g = w[1 : n + 1]  # w = d * g(d)
    h = inv(g, n)  # d / w(d)
    out = [0] * n
    pw = [mpf(1)] + [0] * (n - 1)
    for k in range(1, n):
        pw = mul(pw, h, n)
        out[k] = pw[k - 1] / k
    return out


def log_series(a, n):
    """log of a power series (a[0] != 0; constant term log a[0], principal branch).
    From r L' = r' with r = a / a[0]:  k l_k = k r_k - sum_{i=1}^{k-1} i l_i r_{k-i}."""
    from mpmath import log

    r = [x / a[0] for x in a[:n]] + [0] * max(0, n - len(a))
    out = [log(a[0])] + [0] * (n - 1)
    for k in range(1, n):
        acc = k * r[k]
        for i in range(1, k):
            acc -= i * out[i] * r[k - i]
        out[k] = acc / k
    return out
