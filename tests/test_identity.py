import numpy as np
from calibrators import compute_all, check_algebraic_identity

def test_identity():
  """
  Verify that the algebraic identity VD/VS = SD holds for a synthetic Gaussian profile.
  """
    S = np.linspace(0, 10, 500)
    V = np.exp(-0.5 * ((S - 5) / 1.5) ** 2)
    D = -np.gradient(V, S)
    err, holds = check_algebraic_identity(V, D, S)
    assert holds and err < 1e-6
