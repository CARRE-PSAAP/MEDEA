import warnings

import numpy as np

from medea import mfem

VectorLike = mfem.Vector | list[float] | np.ndarray[np.floating]

class QuadratureSet:
    def __init__(
        self,
        angles: list[VectorLike] | list[float],
        weights: list[float] | np.ndarray[float],
        name: str = "custom",
    ):
        self.weights = np.array([])
        self.angles = []
        if len(angles) != len(weights):
            raise ValueError(
                f"Number of angles and weights must be the same ({len(angles)}!={len(weights)})"
            )

        if all(isinstance(v, float) for v in angles):
            self.angles = np.array([mfem.Vector([v]) for v in angles])
        elif not all(isinstance(v, mfem.Vector) for v in angles):
            self.angles = np.array([mfem.Vector(v) for v in angles])
        else:
            self.angles[:] = np.array(angles)

        if not all((v.Size() == self.angles[0].Size()) for v in self.angles):
            raise ValueError("All angles must be of same dimension")
        else:
            self.dim = self.angles[0].Size()

        if isinstance(weights, list):
            self.weights = np.array(weights)
        else:
            self.weights = weights

        def _normalize(values: np.ndarray, expected, warn_digits, err_digits=None):
            diff = np.abs(values.sum() - expected) / expected
            if err_digits is not None:
                err_tol = 10 ** (-err_digits)
                if diff > err_tol:
                    raise ValueError(
                        f"Invalid quadrature weights for dim {self.dim}, should sum to {expected:.{err_digits+1}f}, got {values.sum():.{err_digits+1}f}"
                    )

            warn_tol = 10 ** (-warn_digits)
            if diff > warn_tol:
                warnings.warn(
                    f"Normalizing quadrature to {expected:.4f}", category=UserWarning
                )
            return values * (expected / values.sum())

        match self.dim:
            case 1:
                self.weights = _normalize(self.weights, 2.0, 8, 5)
            case 2 | 3:
                self.weights = _normalize(self.weights, 4 * np.pi, 8, 5)
            case _:
                raise ValueError(f"Invalid angle dimension {self.dim}")

        self.name = name

def gauss_legendre(N: int) -> QuadratureSet:
    if N < 2:
        raise ValueError("N must be at least 2")
    if N % 2 != 0:
        warnings.warn("N is odd in Gauss-Legendre quadrature set", category=UserWarning)
    mu, w = np.polynomial.legendre.leggauss(N)
    return QuadratureSet([mfem.Vector([m]) for m in mu], w, f"S{N} Gauss-Legendre")

def double_gauss(N: int) -> QuadratureSet:
    if N < 1: 
        raise ValueError("N must be at least 1")
    mu, w = np.polynomial.legendre.leggauss(N)

    return QuadratureSet(np.append((mu-1)/2, (mu+1)/2), np.append(w/2, w/2), f"Double S{N} Gauss-Legendre")

def gauss_lobatto(N: int) -> QuadratureSet:
    if N < 2:
        raise ValueError("N must be at least 2")
    if N % 2 != 0:
        warnings.warn("N is odd in Gauss-Lobatto quadrature set", category=UserWarning)

    # Construct Legendre polynomial P_{n-1}(x)
    c = np.zeros(N)
    c[-1] = 1.0
    P = np.polynomial.legendre.Legendre(c)

    # Interior nodes are the roots of P'_{n-1}(x)
    interior_nodes = P.deriv().roots()
    nodes = np.concatenate(([-1.0], interior_nodes, [1.0]))

    # Compute weights
    weights = 2.0 / (N * (N - 1) * (P(nodes) ** 2))
    return QuadratureSet(nodes.real, weights.real, f"{N}-point Gauss-Lobatto")