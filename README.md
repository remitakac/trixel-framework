# TRIXEL Framework
Reference implementation of the TRIXEL calibrator system (SD, VD, VS) — minimum for maximum.
TRIXEL Framework
https://doi.org/10.5281/zenodo.20721811

T = (V, D, S) — a mathematical framework describing any system through three dimensions:

V (Existence) — what the system is

D (Dynamics) — how it changes

S (Structure) — the scale or form in which it exists

From these, three calibrators measure their mutual relationships.

What this is
TRIXEL is an independent research project by Milan Takáč (Košice, Slovakia), developed collaboratively with AI assistance.
The framework is not affiliated with any academic institution.

Latest preprint (v3.0):
https://doi.org/10.5281/zenodo.20721811

Previous version (v1.0):
https://doi.org/10.5281/zenodo.20610880

Installation
bash
pip install numpy scipy

Quick start
python

import numpy as np
from calibrators import compute_all, dominant_calibrator, check_algebraic_identity

S = np.linspace(0, 10, 500)
V = np.exp(-0.5 * ((S-5)/1.5)**2)
D = -np.gradient(V, S)

c = compute_all(V, D, S)
print(f"VS range: {c['VS'].min():.4f} to {c['VS'].max():.4f}")

err, holds = check_algebraic_identity(V, D, S)
print(f"Identity VD/VS = SD holds: {holds} (error: {err:.2e})")

dom = dominant_calibrator(c['SD'], c['n'])
Example notebook: examples/basic_usage.ipynb

The three calibrators
Calibrator	Formula	Meaning
SD	\	dD/dS\		How fast dynamics change with structure
VS	1/\	dV/dS\		Sensitivity of existence to structural change
VD	SD/n	Bridge between dynamics and existence


Exact identity: VD / VS = SD

Dominance map
In (log SD, log n) space, one calibrator always dominates:

VS dominant — existence highly sensitive to structure

SD dominant — dynamics change rapidly with structure

VD dominant — intermediate regime

Verified at 99.99% accuracy on a 600×600 grid.

Verified results
Algebraic identity VD/VS = SD — exact

Dominance partition theorem — 99.99% accuracy

VS early warning — Burgers turbulence (90/90 runs, FP=0%, FN=0%)

Cross-domain validation — Lotka–Volterra ecology

Real tokamak data — VS identifies stable plasma phases in GOLEM

Not yet verified
VS as disruption precursor on real tokamak data

VS early warning in 2D Navier–Stokes

VS on EEG seizure data

Rogowski coil calibration for GOLEM

Physical interpretation
VS = 1/|dV/dS| drops when the system’s existence becomes rapidly sensitive to structural changes.
This happens before global energy or amplitude changes become visible.

The mapping (choice of V, D, S) must be physically motivated.

How to apply TRIXEL
Choose S (time, wavenumber, radius, position…)

Choose V(S) (energy spectrum, current, population…)

Choose D(S) (dE/dt, dI/dt, dN/dt…)

Run compute_all(V, D, S) and examine VS

Repository structure
Kód
trixel/
├── calibrators.py
├── README.md
├── requirements.txt
├── tests/
│   └── test_identity.py
└── examples/
    └── basic_usage.ipynb
Citation
bibtex
@misc{takac2026trixel,
  author = {Takáč, Milan},
  title  = {TRIXEL 3.0 — Scale-Adaptive Metric Framework},
  year   = {2026},
  doi    = {10.5281/zenodo.20721811},
  url    = {https://doi.org/10.5281/zenodo.20721811}
}
License
MIT License.

Contact
Milan Takáč, Košice, Slovakia
https://doi.org/10.5281/zenodo.20721811
