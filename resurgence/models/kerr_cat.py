"""Squeezed (two-photon-driven) Kerr oscillator, the Kerr-cat qubit.

Convention (K = 1, sign flipped so the cat states are the *ground* doublet):

    H = a†² a² − ε₂ (a†² + a²) − Δ a†a.

Classical wells at a = ±α with α² = ε₂ + Δ/2. Displacing a = α + b gives exactly

    H / (4α²) = −α²/4 + b†b + g V₁ + g² V₂,      g = 1/α,
    V₁ = ½ (b†² b + b† b²),
    V₂ = ¼ [ (Δ/2)(b†² + b²) − Δ b†b + b†² b² ].

so E_n = −α⁴ + 4α² (n + Σ_k e_k g^k), with e_k = 0 for odd k (g → −g with b → −b).

Rayleigh–Schrödinger is done in the *unnormalised* basis |m) = b†^m |0>, where every matrix element
is an integer polynomial in m, so the coefficients are exact rationals (python-flint fmpq) for
rational Δ:
    b†²b |m) = m |m+1),   b†b² |m) = m(m−1) |m−1),   b†² |m) = |m+2),
    b² |m) = m(m−1) |m−2), b†b |m) = m |m),          b†²b² |m) = m(m−1) |m).
The basis is orthogonal, so the |n)-component extraction of intermediate-normalised RS is exact.

The exact spectrum: in the Fock basis, H couples n ↔ n±2 only, so each parity sector is a
symmetric tridiagonal matrix (even sector ↔ the even cat |C₊>, odd ↔ |C₋>).
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

from flint import fmpq
from mpmath import mp, mpf, sqrt


# ---------------------------------------------------------------- perturbative series

def _apply_V1(psi: dict) -> dict:
    out: dict = {}
    for m, c in psi.items():
        if m:
            out[m + 1] = out.get(m + 1, 0) + c * m / 2  # ½ b†²b
            if m >= 2:
                out[m - 1] = out.get(m - 1, 0) + c * (m * (m - 1)) / 2  # ½ b†b²
    return out


def _apply_V2(psi: dict, delta) -> dict:
    out: dict = {}
    q = fmpq(1, 4)
    for m, c in psi.items():
        cq = c * q
        out[m + 2] = out.get(m + 2, 0) + cq * delta / 2
        if m >= 2:
            out[m - 2] = out.get(m - 2, 0) + cq * delta / 2 * (m * (m - 1))
        diag = -delta * m + m * (m - 1)
        if diag:
            out[m] = out.get(m, 0) + cq * diag
    return out


def rs_series(n: int, delta, kmax: int, checkpoint: str | Path | None = None,
              save_every: float = 120.0, log=None) -> list:
    """Exact coefficients e_0..e_kmax (fmpq) of n + Σ e_k g^k for level n of the displaced well.

    `delta` must be an fmpq (or int). Resumable: state (all ψ_k and e_k) is gzip-JSON checkpointed.
    """
    delta = fmpq(delta)
    psis: list[dict] = [{n: fmpq(1)}]
    E: list = [fmpq(n)]
    ck = Path(checkpoint) if checkpoint else None
    if ck and ck.exists():
        blob = json.loads(gzip.decompress(ck.read_bytes()))
        assert blob["n"] == n and blob["delta"] == str(delta), "checkpoint parameter mismatch"
        E = [fmpq(*map(int, s.split("/"))) if "/" in s else fmpq(int(s)) for s in blob["E"]]
        psis = [{int(m): (fmpq(*map(int, s.split("/"))) if "/" in s else fmpq(int(s))) for m, s in p.items()}
                for p in blob["psis"]]
    last = time.monotonic()

    def save():
        if not ck:
            return
        blob = {"n": n, "delta": str(delta), "E": [str(e) for e in E],
                "psis": [{str(m): str(c) for m, c in p.items()} for p in psis]}
        tmp = ck.with_suffix(ck.suffix + ".tmp")
        tmp.write_bytes(gzip.compress(json.dumps(blob).encode(), 6))
        tmp.replace(ck)

    for k in range(len(E), kmax + 1):
        rhs = _apply_V1(psis[k - 1])
        if k >= 2:
            for m, c in _apply_V2(psis[k - 2], delta).items():
                rhs[m] = rhs.get(m, 0) + c
        Ek = rhs.get(n, fmpq(0))
        new: dict = {}
        for m, c in rhs.items():
            if m != n:
                new[m] = -c
        for j in range(1, k):  # Σ_{j=1}^{k-1} E_j ψ_{k-j}  (j = k term only feeds the n component)
            if E[j] == 0:
                continue
            for m, c in psis[k - j].items():
                new[m] = new.get(m, 0) + E[j] * c
        psi_k = {m: c / (m - n) for m, c in new.items() if m != n and c != 0}
        psis.append(psi_k)
        E.append(Ek)
        if log and k % 10 == 0:
            log(f"order {k}: e_k ≈ {float(Ek):.6e}")
        if ck and time.monotonic() - last > save_every:
            save()
            last = time.monotonic()
    save()
    return E


def energy_from_series(e: list, alpha, kmax: int | None = None):
    """E = −α⁴ + 4α² Σ_k e_k α^{−k} (partial sum through kmax)."""
    kmax = len(e) - 1 if kmax is None else kmax
    s = mpf(0)
    g = 1 / mpf(alpha)
    for k in range(kmax + 1):
        if e[k] != 0:
            s += mpf(int(e[k].p)) / mpf(int(e[k].q)) * g**k
    return -mpf(alpha) ** 4 + 4 * mpf(alpha) ** 2 * s


# ---------------------------------------------------------------- exact spectrum

def _tridiag_sector(eps2, delta, parity: int, size: int):
    """Diagonal d and off-diagonal o of H restricted to Fock states n = parity, parity+2, ..."""
    ns = [parity + 2 * i for i in range(size)]
    d = [mpf(nn * (nn - 1)) - delta * nn for nn in ns]
    o = [-eps2 * sqrt(mpf((nn + 1) * (nn + 2))) for nn in ns[:-1]]
    return d, o


def _count_below(d, o, x):
    """Sturm count: number of eigenvalues of the symmetric tridiagonal (d, o) below x."""
    cnt = 0
    q = d[0] - x
    tiny = mpf(10) ** (-(mp.dps + 20))
    if q < 0:
        cnt += 1
    for i in range(1, len(d)):
        if q == 0:
            q = tiny
        q = d[i] - x - o[i - 1] ** 2 / q
        if q < 0:
            cnt += 1
    return cnt


def exact_levels(eps2, delta, parity: int, nlev: int = 1, size: int | None = None):
    """Lowest `nlev` eigenvalues in a parity sector by Sturm bisection at the current mp.dps."""
    eps2, delta = mpf(eps2), mpf(delta)
    alpha2 = eps2 + delta / 2
    if size is None:
        size = int(4 * alpha2 + 4 * mp.dps + 60)
    d, o = _tridiag_sector(eps2, delta, parity, size)
    lo = -alpha2**2 - 10 * (abs(delta) + 1) * (alpha2 + 1)
    hi = -alpha2**2 + 4 * alpha2 * (2 * nlev + 4) + 10 * (abs(delta) + 1) ** 2 + 10
    while _count_below(d, o, lo) > 0:
        lo -= abs(lo) + 1
    while _count_below(d, o, hi) < nlev:
        hi += abs(hi) + 1
    out = []
    for j in range(nlev):
        a, b = lo, hi
        tol = mpf(10) ** (-mp.dps + 5) * (1 + abs(a))
        while b - a > tol:
            c = (a + b) / 2
            if _count_below(d, o, c) > j:
                b = c
            else:
                a = c
        out.append((a + b) / 2)
    return out
