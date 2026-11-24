# Resurgent Series

Companion code for a series of notes applying resurgence and transseries analysis to tractable model systems.

> Shankar Sivarajan, *Resurgence of the finite-N Curie–Weiss model: metastability, the spinodal, and Lee–Yang zeros*, [hexagon:2611.00950](https://hexagonmath.org/2611.00950) (2026).
> 
> Shankar Sivarajan, *Resurgence of the squeezed Kerr oscillator: tunnel splittings, Kummer connection, and a perturbative/non-perturbative relation*, [hexagon:2611.01004](https://hexagonmath.org/2611.01004) (2026).

## Usage

```bash
pip install -e .              # mpmath, numpy, python-flint, and sympy; use ".[figures]" for matplotlib
python reproduce/curie_weiss.py
python reproduce/kerr_cat.py
```

`reproduce/curie_weiss.py` recomputes every numerical value in the Curie–Weiss note and checks it against the paper,
reporting `PASS` or `FAIL` with labels keyed to the paper's numbering (Sections 2–5, Result 3.1, Remark 5.1,
equations (7), (8), (13), and (14), Tables 1–2, and Figures 1–2); cited or assumed statements that are not recomputed
are listed as `NOTE` lines. A full run takes a few minutes on one core and checkpoints intermediate results to
`reproduce/.cache_*.json` so it can be resumed. A saved output log is provided in `reproduce/curie_weiss.log`, and
`reproduce/curie_weiss_figures.py` regenerates both figures.

`reproduce/kerr_cat.py` does the same for the Kerr-cat note, reporting `PASS` or `FAIL` with labels keyed to the paper's
numbering (Sections 2–6, Results 3.1–6.1, Tables 1–3, and the accuracy claims of the abstract). A full run takes about an
hour on one core (exact splittings at 60–120 digits) and checkpoints intermediate results so it can be resumed: splittings
are saved in `reproduce/.cache_kerr_cat.json` and perturbative series in `reproduce/.series_kerr_cat/`. A saved output
log is provided in `reproduce/kerr_cat.log`, and `reproduce/kerr_cat_figures.py` regenerates both figures.

## Files

- `resurgence/series.py`: truncated power-series arithmetic (products, reversion, exp, log, and square root)
- `resurgence/laplace.py`: all-orders Laplace method with *N*-dependent amplitudes (the series *A*± of Sections 2–3)
- `resurgence/largeorder.py`: ratio tests, Richardson extrapolation, and action estimates (Section 3)
- `resurgence/borel.py`: Borel transforms, Borel–Padé poles, and lateral and median resummation (Sections 3 and 5)
- `resurgence/checkpoint.py`: resumable key-value checkpoints for long computations
- `resurgence/examples.py`: validation series (Euler, Stirling, and the quartic anharmonic oscillator)
- `resurgence/models/curie_weiss.py`: exact finite sums, mean-field free-energy functional, critical points, 1/*N*
  phase series, and exact Hubbard–Stratonovich metastable sector (equations (7), (8), and (14); Result 3.1)
- `resurgence/models/curie_weiss_uniform.py`: two-point Chester–Friedman–Ursell expansion (equation (13), Table 1,
  and Figure 1)
- `resurgence/models/curie_weiss_zeros.py`: 1/*N* series of the Lee–Yang zeros at fixed filling (Section 5, Table 2,
  and Figure 2)
- `resurgence/models/kerr_cat.py`: squeezed Kerr oscillator: exact Rayleigh–Schrödinger series in the displaced frame
  (exact rationals via `python-flint`) and exact spectrum by Sturm bisection in each parity sector (Sections 2–6)
- `reproduce/curie_weiss.py`: verification suite for the Curie–Weiss note
- `reproduce/curie_weiss.log`: saved verification log for the Curie–Weiss note
- `reproduce/curie_weiss_figures.py`: figure generator for Figures 1 and 2 of the Curie–Weiss note
- `reproduce/kerr_cat.py`: verification suite for the Kerr-cat note
- `reproduce/kerr_cat.log`: saved verification log for the Kerr-cat note
- `reproduce/kerr_cat_figures.py`: figure generator for Figures 1 and 2 of the Kerr-cat note
