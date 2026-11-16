import warnings
from itertools import product
from math import gamma, pi, sqrt

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

def level_symmetric(N: int, dim: int = 3) -> QuadratureSet:
    """Level-symmetric (LQn) quadrature normalized to 4*pi (Ref. Lewis and Miller
    Computational Method of Neutron Transport)"
    """
    if dim not in (2, 3):
        raise ValueError(f"'dim' must be 2 or 3 (got {dim})")
    if N < 2 or N % 2:
        raise ValueError(f"N must be an even integer >= 2 (got {N})")

    # Principal-octant points (0-based level indices i+j+k = n-1); points that
    # are permutations of each other share a weight (class id in `cls`)
    n = N // 2
    i, j = np.nonzero(np.add.outer(np.arange(n), np.arange(n)) < n)
    k = n - 1 - i - j
    cls = np.unique(np.sort([i, j, k], axis=0), axis=1, return_inverse=True)[1].ravel()
    onehot = np.eye(cls.max() + 1)[cls]

    def fit(mu1_sq, order):
        """Cosines (Eq. 8, 9), class weights (octant sum 1) and residual of the
        even-moment equations (Eq. 13, 18) for k >= l, k + l <= order."""
        mu = np.sqrt(np.linspace(mu1_sq, 1 - 2 * mu1_sq, n))
        kk, ll = np.array(
            [(a, b) for a in range(0, order + 1, 2) for b in range(0, a + 1, 2) if a + b <= order]
        ).T
        A = (mu[i][:, None] ** kk * mu[j][:, None] ** ll).T @ onehot
        B = [0.5 * gamma((a + 1) / 2) * gamma((b + 1) / 2) / (sqrt(pi) * gamma((a + b + 3) / 2))
             for a, b in zip(kk, ll)]
        P = np.linalg.lstsq(A, B, rcond=None)[0]
        return mu, P, np.linalg.norm(A @ P - B)

    if N == 2:
        mu1_sq = 1 / 3
    else:
        # Pick mu_1 so the moments through k+l = N hold too: find the first
        # local minimum of the residual on a grid, then zoom in on it.
        xs = np.geomspace(1e-4, 1 / 3 - 1e-6, 400)
        r = np.array([fit(x, N)[2] for x in xs])
        m = np.flatnonzero((r[1:-1] <= r[:-2]) & (r[1:-1] <= r[2:]))[0] + 1
        for _ in range(10):
            xs = np.linspace(xs[m - 1], xs[m + 1], 21)
            m = np.clip(np.argmin([fit(x, N)[2] for x in xs]), 1, 19)
        mu1_sq = xs[m]

    mu, P, resid = fit(mu1_sq, N - 2)
    if resid > 1e-8:
        warnings.warn(f"Level-symmetric S{N} moment equations are inconsistent (residual {resid:.2e})")
    if np.any(P < 0):
        warnings.warn(f"Level-symmetric S{N} has negative weights")

    # Reflect the octant into all 2^dim sign combinations (dim=2 keeps xi > 0
    # and folds the mirrored xi < 0 directions into the weights)
    signs = np.array(list(product((-1, 1), repeat=dim)))
    octant = mu[np.array([i, j, k]).T[:, :dim]]
    angles = (signs[:, None, :] * octant[None]).reshape(-1, dim)
    weights = np.tile(4 * pi * P[cls] / len(signs), len(signs))
    return QuadratureSet(list(angles), weights, f"S{N} level-symmetric")
