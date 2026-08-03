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
    print("\nSelf-test complete.")
