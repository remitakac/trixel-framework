import numpy as np
from calibrators import (
    compute_all,
    check_algebraic_identity,
    check_independence,
    check_functional_dependence,
)


def test_identity():
    """
    Verify that the algebraic identity VD/VS = SD holds for a synthetic Gaussian profile.
    """
    S = np.linspace(0, 10, 500)
    V = np.exp(-0.5 * ((S - 5) / 1.5) ** 2)
    D = -np.gradient(V, S)
    err, holds = check_algebraic_identity(V, D, S)
    assert holds and err < 1e-6


def test_rank_test_misses_nonlinear_dependence():
    """
    Regression test: check_independence() (linear rank test) reports
    rank=3 for this Gaussian profile even though D = -gradient(V, S)
    is fully determined by V. This is the expected (known) limitation
    of the linear rank test, not a bug -- it documents why
    check_functional_dependence() exists as a supplementary check.
    """
    S = np.linspace(0, 10, 500)
    V = np.exp(-0.5 * ((S - 5) / 1.5) ** 2)
    D = -np.gradient(V, S)

    rank = check_independence(V, D, S)
    assert rank == 3, (
        "Expected the linear rank test to report rank=3 here "
        "(false positive for independence) -- if this changes, "
        "the documented limitation may no longer apply and this "
        "test (and the related docstrings) should be revisited."
    )


def test_functional_dependence_catches_what_rank_test_misses():
    """
    check_functional_dependence() should correctly flag D as
    (nonlinearly) dependent on V in the same case where
    check_independence() gives a false positive.
    """
    S = np.linspace(0, 10, 500)
    V = np.exp(-0.5 * ((S - 5) / 1.5) ** 2)
    D = -np.gradient(V, S)

    corr = check_functional_dependence(V, D, S)
    assert abs(corr) > 0.95, (
        f"Expected high correlation (>0.95) between D and signed dV/dS "
        f"since D is deterministically derived from V, got {corr:.3f}"
    )


def test_functional_dependence_low_for_independent_signals():
    """
    Sanity check: check_functional_dependence() should NOT flag
    genuinely independent V and D (avoids false positives on real,
    independently-measured data).
    """
    rng = np.random.default_rng(0)
    S = np.linspace(0, 10, 500)
    V = np.exp(-0.5 * ((S - 5) / 1.5) ** 2)
    D = rng.normal(size=len(S))  # independent of V by construction

    corr = check_functional_dependence(V, D, S)
    assert abs(corr) < 0.3, (
        f"Expected low correlation for genuinely independent D, "
        f"got {corr:.3f} -- possible false positive."
    )
