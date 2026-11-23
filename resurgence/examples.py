"""Textbook divergent series with known large-order behaviour (for validation)."""
from __future__ import annotations

from fractions import Fraction

from mpmath import bernoulli, factorial, mpf


def euler_series(n: int) -> list:
    """sum (-1)^n n! z^n  ->  A = -1, b = 1, f_0 = 1, f_k>0 = 0."""
    return [(-1) ** k * factorial(k) for k in range(n)]


def stirling_series(n: int) -> list:
    """log Gamma(z) - [(z-1/2)log z - z + log(2pi)/2] = sum_k a_k z^{-(2k-1)}.

    a_k = B_{2k} / (2k (2k-1)). Borel singularities at +-2 pi i m (m >= 1).
    """
    return [bernoulli(2 * k) / (2 * k * (2 * k - 1)) for k in range(1, n + 1)]


def quartic_aho(nmax: int) -> list[Fraction]:
    """Ground-state Rayleigh-Schrodinger series of H = p^2/2 + x^2/2 + g x^4.

    Exact rationals E_0..E_nmax via the Bender-Wu polynomial recursion
    (psi = e^{-x^2/2} sum_n g^n phi_n(x), phi_n(0) = delta_{n0}).
    Known: E = 1/2 + 3/4 g - 21/8 g^2 + 333/16 g^3 - ...,
    E_n ~ -(-3)^n sqrt(6/pi^3) Gamma(n+1/2) (1 - 95/(72 n) + ...), i.e.
    A = -1/3, b = 1/2, f_0 = -sqrt(6)/pi^{3/2}.
    """
    # A[n][j] = coefficient of x^{2j} in phi_n
    A: list[list[Fraction]] = [[Fraction(1)]]
    E = [Fraction(1, 2)]
    for n in range(1, nmax + 1):
        row = [Fraction(0)] * (2 * n + 2)
        prev = A[n - 1]
        for j in range(2 * n, 0, -1):
            val = (j + 1) * (2 * j + 1) * row[j + 1]
            if 0 <= j - 2 < len(prev):
                val -= prev[j - 2]
            for k in range(1, n):
                Ak = A[n - k]
                if j < len(Ak):
                    val += E[k] * Ak[j]
            row[j] = val / (2 * j)
        E.append(-row[1])
        A.append(row[: 2 * n + 1])
    return E


def to_mp(seq) -> list:
    return [mpf(x.numerator) / x.denominator if isinstance(x, Fraction) else mpf(x) for x in seq]
