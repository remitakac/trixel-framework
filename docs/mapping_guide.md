# TRIXEL Mapping Guide: How to Choose V, D, S

## The core question

Before applying TRIXEL, you must answer three questions about your system:

1. **What exists?** → This is V
2. **How does it change?** → This is D  
3. **At what scale or position?** → This is S

The mapping must be **physically motivated** — not chosen to make the numbers look good.

---

## Rules for a valid mapping

**Rule 1: V and D should be independent measurements**

If D = dV/dS (you compute D from V), the calibrators reduce to comparisons of derivatives of V. This is mathematically valid but limits the framework to signal analysis only.

A stronger mapping uses V and D from separate physical measurements:

| System | S | V | D (independent) |
|--------|---|---|-----------------|
| Tokamak | time | plasma current Ip(t) | loop voltage Uloop(t) |
| Ecology | carrying capacity K | population N(K) | growth rate dN/dt from field data |
| EEG | time | signal amplitude | reference signal from separate electrode |

**Rule 2: S must be the natural structural parameter of your system**

S is the parameter along which the system is organized:
- Time series → S = time
- Spectral analysis → S = wavenumber or frequency
- Spatial profile → S = radius, position, depth

**Rule 3: Choose the smallest meaningful scale for S**

Too fine: VS explodes due to noise (use physical smoothing first)  
Too coarse: VS loses sensitivity to real transitions

Physical smoothing window should match the characteristic timescale of the signal you want to detect — not chosen to produce a clean result.

---

## Common mistakes

**Mistake 1: D = dV/dS always**

This works for mathematical demonstrations but misses the point for real systems. Look for a second independent measurement that captures dynamics.

**Mistake 2: Choosing S to make VS look good**

S must have physical meaning before you run the analysis. If you are choosing S after seeing the result, that is tuning.

**Mistake 3: Applying TRIXEL without knowing what VS should do**

State your prediction before computing. VS should drop before a known transition (disruption, phase change, instability). If you don't know what to expect, you can't validate the result.

---

## Step-by-step guide

1. **Identify the structural parameter S** — what organizes your system (time, space, scale)?

2. **Identify existence V(S)** — what is the system at each S? Energy, current, population, amplitude?

3. **Identify dynamics D(S)** — how does the system change? Ideally from an independent measurement, not computed from V.

4. **State your prediction** — where do you expect VS to drop? Before a disruption, at a phase transition, at the onset of instability?

5. **Apply physical smoothing** — based on the characteristic timescale of your system, not the result.

6. **Compute and compare** — run `compute_all(V, D, S)` and check VS against your prediction.

7. **Report honestly** — if VS does not drop before the transition, that is a result too.

---

## Examples with independent V and D

### Tokamak plasma (GOLEM)
```python
S = t_ms                    # time [ms]
V = Ip_smoothed             # plasma current [A] — Rogowski coil
D = Uloop_smoothed          # loop voltage [V] — separate measurement
```
VS drops during rapid current changes (instability onset).  
SD rises when driving voltage changes rapidly.

### 1D Burgers turbulence (spectral)
```python
S = wavenumber_k            # spectral scale
V = E(k, t)                 # energy spectrum at time t
D = dE(k)/dt                # rate of spectral change
```
VS drops at the dissipation scale before global energy increases.  
Verified: 90/90 runs, FP=0%, FN=0% (Addendum 39).

---

## What TRIXEL cannot do (honest limits)

- It does not choose V, D, S for you
- It does not work without physical motivation for the mapping
- VS is sensitive to noise where dV/dS ≈ 0 — physical smoothing required
- Results on 5 tokamak shots are indicative, not statistically robust
- Not yet tested on EEG, 2D Navier-Stokes, or disruption precursor prediction

---

## Reference

Preprint: https://doi.org/10.5281/zenodo.20721811  
Addenda 1–44: available on request or in the preprint supplementary material.
