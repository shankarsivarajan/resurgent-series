# Resurgent Series

Companion code for a series of notes applying resurgence and transseries analysis to tractable model systems.

> Shankar Sivarajan, *Resurgence of the finite-N Curie–Weiss model: metastability, the spinodal, and Lee–Yang zeros* (2026).

## Usage

```bash
pip install -e .              # mpmath and numpy; use ".[figures]" for matplotlib
python reproduce/curie_weiss.py
```

`reproduce/curie_weiss.py` recomputes every numerical value in the Curie–Weiss note and checks it against the paper,
reporting `PASS` or `FAIL` with labels keyed to the paper's numbering (Sections 2–5, Result 3.1, Remark 5.1,
equations (7), (8), (13), and (14), Tables 1–2, and Figures 1–2); cited or assumed statements that are not recomputed
are listed as `NOTE` lines. A full run takes a few minutes on a single core and checkpoints intermediate results to
`reproduce/.cache_*.json` so it can be resumed. A saved output log is provided in `reproduce/curie_weiss.log`, and
`reproduce/curie_weiss_figures.py` regenerates both figures.

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
- `reproduce/curie_weiss.py`: verification suite for the Curie–Weiss note
- `reproduce/curie_weiss.log`: saved verification log
- `reproduce/curie_weiss_figures.py`: figure generator for Figures 1 and 2
