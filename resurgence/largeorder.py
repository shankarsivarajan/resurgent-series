"""Large-order analysis of factorially divergent series.

Conventions
-----------
We study coefficients c_n (n = 0, 1, 2, ...) assumed to behave as

    c_n ~ A^{-n} * sum_k f_k A^k Gamma(n + b - k)           (n -> infinity)

i.e. a Borel singularity at zeta = A with "exponent" b. In resurgence language
the f_k are (up to the Stokes constant / 2 pi i) the coefficients of the
fluctuation series around the sector with action A. ``fit_large_order``
extracts f_0, f_1, ... directly in this form so they can be compared against
an independent computation of the subleading sector.

Everything works with mpmath numbers; raise ``mp.dps`` generously (Richardson
extrapolation of order K loses roughly K * log10(n) digits).
"""
from __future__ import annotations

from collections.abc import Sequence

from mpmath import factorial, gamma, mp, mpf


def richardson(seq: Sequence, order: int, start: int = 0) -> list:
    """Richardson-accelerate a sequence s_n ~ L + a_1/n + a_2/n^2 + ...

    ``seq[i]`` is interpreted as s_n with n = start + i (n must be >= 1).
    Returns the list of order-``order`` estimates R(n), one per window
    [n, n + order] that fits in the data. The last element is usually best.
    """
    out = []
    for i in range(len(seq) - order):
        n = start + i
        if n <= 0:
            continue
        acc = mpf(0)
        for k in range(order + 1):
            acc += (
                seq[i + k]
                * mpf(n + k) ** order
                * (-1) ** (k + order)
                / (factorial(k) * factorial(order - k))
            )
        out.append(acc)
    return out


def richardson_limit(seq: Sequence, order: int, start: int = 0):
    """Best (last-window) Richardson estimate of lim s_n, plus a crude error.

    The error is the difference between the order and order-1 estimates.
    """
    hi = richardson(seq, order, start)[-1]
    lo = richardson(seq, order - 1, start)[-1] if order > 1 else seq[-1]
    return hi, abs(hi - lo)


def ratios(c: Sequence) -> list:
    """r_n = c_{n+1} / c_n."""
    return [c[n + 1] / c[n] for n in range(len(c) - 1)]


def estimate_action(c: Sequence, order: int = 10, start: int = 1):
    """Estimate A from c_{n+1}/(n c_n) -> 1/A (works for either sign of A).

    Returns (A, error_estimate). Requires c_n != 0 (compress even/odd series
    first). ``start`` drops the first few low-order coefficients.
    """
    s = [c[n + 1] / (n * c[n]) for n in range(start, len(c) - 1)]
    inv, err = richardson_limit(s, order, start)
    return 1 / inv, err / abs(inv) ** 2


def estimate_exponent(c: Sequence, A, order: int = 10, start: int = 1):
    """Given A, estimate b from A c_{n+1}/c_n - n -> b."""
    s = [A * c[n + 1] / c[n] - n for n in range(start, len(c) - 1)]
    return richardson_limit(s, order, start)


def fit_large_order(c: Sequence, A, b, nterms: int = 4, order: int = 10, start: int = 1):
    """Extract f_0..f_{nterms-1} in  c_n ~ A^{-n} sum_k f_k A^k Gamma(n+b-k).

    Uses successive Richardson extrapolation. Returns a list of (f_k, err_k).
    Later f_k are progressively less accurate.
    """
    s = [c[n] * A**n / gamma(n + b) for n in range(start, len(c))]
    out = []
    for k in range(nterms):
        fk, err = richardson_limit(s, order, start)
        out.append((fk, err))
        # Peel off f_k and rescale so the next term becomes the limit.
        s = [(s[i] - fk) * (start + i + b - k - 1) / A for i in range(len(s))]
    return out


def compress(c: Sequence, step: int = 2, offset: int = 0) -> list:
    """Keep every ``step``-th coefficient (e.g. drop the zero odd terms)."""
    return list(c[offset::step])


def with_precision(dps: int):
    """Convenience context manager: ``with with_precision(100): ...``."""
    return mp.workdps(dps)


# ------------------------------------------------------------ two-sided (+A and -A) singularities
def estimate_exponent_parity(c: Sequence, A, parity: int, order: int = 10, start: int = 10):
    """b for c_k ~ A^{-k} Gamma(k+b) [F + (-1)^k G], using only k = parity mod 2.

    From A^2 c_{k+2}/c_k = (k+b)(k+b+1) (1 + O(1/k^2)), solved for b, then Richardson in j.
    """
    from mpmath import sqrt

    s, ks = [], []
    k = start + ((parity - start) % 2)
    while k + 2 < len(c):
        t = A**2 * c[k + 2] / c[k]
        s.append((-(2 * k + 1) + sqrt((2 * k + 1) ** 2 - 4 * (k * k + k - t))) / 2)
        ks.append(k)
        k += 2
    return richardson_limit(s, order, start=max(1, ks[0] // 2))


def fit_parity(c: Sequence, A, b, parity: int, order: int = 10, start: int = 10):
    """lim_{k = parity mod 2} c_k A^k / Gamma(k+b)  (= F + (-1)^parity G)."""
    k0 = start + ((parity - start) % 2)
    s = [c[k] * A**k / gamma(k + b) for k in range(k0, len(c), 2)]
    return richardson_limit(s, order, start=max(1, k0 // 2))
