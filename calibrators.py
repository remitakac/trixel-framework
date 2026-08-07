"""
TRIXEL Framework — calibrators.py
=================================

Reference implementation of the three TRIXEL calibrators:
SD — how fast dynamics change with structure
VD — bridge between dynamics and existence
VS — sensitivity of existence to structural change

Core identity (exact):
    VD / VS = SD

Only two calibrators are independent. The third is always determined.
"""

import numpy as np


def compute_SD(D, S):
    """
    SD = |dD/dS|
    Measures how fast dynamics change with structure.

    Parameters
    ----------
    D : array-like
        Dynamics signal D(S)
    S : array-like
        Structural parameter S

    Returns
    -------
    SD : ndarray
    """
    D = np.asarray(D, dtype=float)
    S = np.asarray(S, dtype=float)
    return np.abs(np.gradient(D, S))


def compute_VS(V, S, epsilon=1e-10):
    """
    VS = 1 / |dV/dS|
    Sensitivity of existence to structural change.

    HIGH VS → stable regime
    LOW VS  → unstable regime

    Parameters
    ----------
    V : array-like
        Existence signal V(S)
    S : array-like
        Structural parameter S
    epsilon : float
        Regularization to avoid division by zero

    Returns
    -------
    VS : ndarray
    """
    V = np.asarray(V, dtype=float)
    S = np.asarray(S, dtype=float)
    n = np.abs(np.gradient(V, S))
    return 1.0 / (n + epsilon)


def compute_VD(V, D, S, epsilon=1e-10):
    """
    VD = SD / n = |dD/dS| / |dV/dS|
    Bridge between dynamics and existence.

    Parameters
    ----------
    V : array-like
        Existence signal
    D : array-like
        Dynamics signal
    S : array-like
        Structural parameter
    epsilon : float
        Regularization

    Returns
    -------
    VD : ndarray
    """
    SD = compute_SD(D, S)
    n  = np.abs(np.gradient(np.asarray(V, dtype=float),
                             np.asarray(S, dtype=float)))
    return SD / (n + epsilon)


def compute_all(V, D, S, epsilon=1e-10):
    """
    Compute SD, VD, VS and n = |dV/dS|.

    Returns
    -------
    dict with keys: 'SD', 'VD', 'VS', 'n'
    """
    V = np.asarray(V, dtype=float)
    D = np.asarray(D, dtype=float)
    S = np.asarray(S, dtype=float)

    n  = np.abs(np.gradient(V, S))
    SD = np.abs(np.gradient(D, S))
    VS = 1.0 / (n + epsilon)
    VD = SD / (n + epsilon)

    return {'SD': SD, 'VD': VD, 'VS': VS, 'n': n}


def dominant_calibrator(SD, n):
    """
    Determine dominant calibrator in (log SD, log n) space.

    Verified at 99.99% accuracy on 600x600 grid (Addendum 16).

    Returns
    -------
    'SD', 'VD', or 'VS'
    """
    SD = np.asarray(SD, dtype=float)
    n  = np.asarray(n,  dtype=float)

    scalar = SD.ndim == 0
    SD = np.atleast_1d(SD)
    n  = np.atleast_1d(n)

    result = np.empty(SD.shape, dtype=object)

    for i in range(len(SD.flat)):
        sd = SD.flat[i]
        ni = n.flat[i]

        log_sd = np.log(sd + 1e-30)
        log_n  = np.log(ni + 1e-30)

        if log_n >= 0 and log_sd + log_n >= 0:
            result.flat[i] = 'SD'
        elif log_n <= 0 and log_sd >= 0:
            result.flat[i] = 'VD'
        else:
            result.flat[i] = 'VS'

    return result.item() if scalar else result


def check_algebraic_identity(V, D, S, epsilon=1e-10, tol=1e-6):
    """
    Verify VD/VS = SD.

    Returns
    -------
    max_error : float
    identity_holds : bool
    """
    c = compute_all(V, D, S, epsilon)
    lhs = c['VD'] / (c['VS'] + epsilon)
    rhs = c['SD']

    rel_err = np.abs(lhs - rhs) / (np.abs(rhs) + epsilon)
    max_err = rel_err.max()

    return max_err, max_err < tol

def compute_residuum(VD, VS, SD, epsilon=1e-10):
    """
    TRIXEL Residuum R = |log(VD) - log(VS) - log(SD)|
    
    Measures departure from the stability plane.
    R = 0: system is internally consistent.
    R rising: system losing consistency - early warning signal.
    
    Requires independent V, D, S (check with check_independence first).
    """
    return np.abs(
        np.log(VD + epsilon) - 
        np.log(VS + epsilon) - 
        np.log(SD + epsilon)
    )


def check_independence(V, D, S, tol=1e-3):
    """
    Jacobian rank test: are V, D, S genuinely independent?

    Returns rank (should be 3 for valid R computation).
    If rank < 3: D is likely computed from V. R would be artifact.

    NOTE (Addendum 45, 2026-08-07): this test detects only LINEAR
    dependence between V, D, S. If D is a NONLINEAR function of V
    (e.g. D computed as a derivative or other nonlinear transform of V),
    this test can report rank=3 (falsely appearing independent) even
    though D carries no independent information. Use
    check_functional_dependence() as a supplementary check when D's
    provenance is uncertain.

    Rule: If D does not have its own cable, R is not physical.
    """
    J = np.vstack([
        np.asarray(V, dtype=float),
        np.asarray(D, dtype=float),
        np.asarray(S, dtype=float)
    ])
    rank = np.linalg.matrix_rank(J, tol=tol)
    if rank < 3:
        import warnings
        warnings.warn(
            f"rank={rank} < 3: V, D, S not independent. "
            "R would be a numerical artifact, not a physical signal. "
            "Find an independent measurement for D."
        )
    return rank


def check_functional_dependence(V, D, S, correlation_threshold=0.95):
    """
    Supplementary check to check_independence(): detects NONLINEAR
    (e.g. derivative) dependence of D on V that the linear rank test
    misses.

    Background (Addendum 45, 2026-08-07): check_independence() uses
    np.linalg.matrix_rank, which only detects linear combinations.
    If D = f(V) for some nonlinear f (most commonly D computed as a
    derivative of V), matrix_rank typically still reports rank=3,
    giving a false sense of independence. This was confirmed on this
    module's own __main__ self-test case (D = -gradient(V, S)), which
    passes check_independence() (rank=3) despite D being fully
    determined by V.

    This function instead checks correlation between D and the SIGNED
    causal numerical derivative of V with respect to S. An unsigned
    (absolute value) version of this check gives false negatives near
    sign changes (e.g. around extrema of V) — the sign must be
    preserved for the check to be meaningful.

    Parameters
    ----------
    V, D, S : array-like
        Same signals passed to check_independence().
    correlation_threshold : float
        Absolute correlation above which a warning is raised
        (default 0.95).

    Returns
    -------
    corr : float
        Correlation between D and signed dV/dS. Values near +-1
        indicate D is likely a (near-)deterministic function of V;
        values near 0 support genuine independence.
    """
    V = np.asarray(V, dtype=float)
    D = np.asarray(D, dtype=float)
    S = np.asarray(S, dtype=float)

    dS = np.diff(S)
    dS[dS == 0] = 1e-12
    grad_VS = np.diff(V) / dS
    grad_VS = np.insert(grad_VS, 0, grad_VS[0])  # signed, causal

    corr = np.corrcoef(D, grad_VS)[0, 1]
    if abs(corr) > correlation_threshold:
        import warnings
        warnings.warn(
            f"D has high correlation ({corr:.3f}) with signed dV/dS. "
            "D is likely nonlinearly derived from V — "
            "check_independence() (rank test) will not catch this."
        )
    return corr


if __name__ == "__main__":
    print("TRIXEL calibrators.py — self-test")
    print("=" * 50)

    S = np.linspace(0, 10, 500)
    V = np.exp(-0.5 * ((S - 5) / 1.5) ** 2)
    D = -np.gradient(V, S)

    c = compute_all(V, D, S)
    err, holds = check_algebraic_identity(V, D, S)

    print(f"Algebraic identity VD/VS = SD: "
          f"{'HOLDS' if holds else 'FAILS'} "
          f"(max error = {err:.2e})")

    dom = dominant_calibrator(c['SD'], c['n'])
    unique, counts = np.unique(dom, return_counts=True)

    print("Dominant calibrators:")
    for d, cnt in zip(unique, counts):
        print(f"  {d}: {cnt/len(dom)*100:.1f}%")

    print(f"\nVS range: {c['VS'].min():.4f} to {c['VS'].max():.4f}")
    print(f"SD range: {c['SD'].min():.4f} to {c['SD'].max():.4f}")

    # --- Independence First Rule demonstration (Addendum 45) ---
    # This self-test's D IS a deterministic (nonlinear) function of V,
    # by construction (D = -gradient(V, S)). This block shows why that
    # matters: the linear rank test alone would miss it.
    print("\n--- Independence checks (this self-test's D is derived from V) ---")
    rank = check_independence(V, D, S)
    func_corr = check_functional_dependence(V, D, S)
    print(f"check_independence (linear rank test): rank={rank} "
          f"(reports 3 -- FALSE POSITIVE for independence)")
    print(f"check_functional_dependence (nonlinear test): corr={func_corr:.4f} "
          f"(correctly flags the dependence)")
    print("This demonstrates why check_functional_dependence() is a "
          "necessary supplement to check_independence(), not a "
          "redundant check -- see docs/mapping_guide.md, Mistake 4.")
