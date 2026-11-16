"""Tests for medea.quadrature: QuadratureSet (1D/2D/3D) and the 1D set builders."""

import warnings

import numpy as np
import pytest

from medea import mfem
from medea.quadrature import (
    QuadratureSet,
    double_gauss,
    gauss_legendre,
    gauss_lobatto,
    level_symmetric,
)


def unit_vectors_2d(n):
    th = 2 * np.pi * (np.arange(n) + 0.5) / n
    return [[np.cos(t), np.sin(t)] for t in th]


def unit_vectors_3d():
    s = 1 / np.sqrt(3)
    return [
        [sx * s, sy * s, sz * s] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ]


def values(qs):
    """Angles as an (n, dim) ndarray."""
    return np.array([[v[i] for i in range(qs.dim)] for v in qs.angles])


@pytest.fixture(autouse=True)
def _no_stray_warnings():
    """Unexpected UserWarnings are failures; tests expecting one use pytest.warns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        yield


class TestQuadratureSet1D:
    def test_from_floats(self):
        qs = QuadratureSet([-0.5, 0.5], [1.0, 1.0], "pair")
        assert qs.dim == 1
        assert qs.name == "pair"
        assert len(qs.angles) == 2
        assert all(isinstance(v, mfem.Vector) for v in qs.angles)
        np.testing.assert_allclose(values(qs)[:, 0], [-0.5, 0.5])
        np.testing.assert_allclose(qs.weights, [1.0, 1.0])

    def test_from_lists(self):
        qs = QuadratureSet([[-0.5], [0.5]], [1.0, 1.0])
        assert qs.dim == 1
        np.testing.assert_allclose(values(qs)[:, 0], [-0.5, 0.5])

    def test_from_mfem_vectors(self):
        qs = QuadratureSet([mfem.Vector([-0.5]), mfem.Vector([0.5])], [1.0, 1.0])
        assert qs.dim == 1
        np.testing.assert_allclose(values(qs)[:, 0], [-0.5, 0.5])

    def test_default_name(self):
        assert QuadratureSet([-0.5, 0.5], [1.0, 1.0]).name == "custom"

    def test_weights_sum_to_two(self):
        qs = QuadratureSet([-0.5, 0.5], [1.0, 1.0])
        assert qs.weights.sum() == pytest.approx(2.0)

    def test_numpy_weights(self):
        qs = QuadratureSet([-0.5, 0.5], np.array([1.0, 1.0]))
        assert qs.weights.sum() == pytest.approx(2.0)


class TestQuadratureSet2D:
    def test_construction(self):
        qs = QuadratureSet(unit_vectors_2d(8), [4 * np.pi / 8] * 8, "ring")
        assert qs.dim == 2
        assert len(qs.angles) == 8
        assert all(v.Size() == 2 for v in qs.angles)
        assert qs.weights.sum() == pytest.approx(4 * np.pi)

    def test_mfem_vectors(self):
        angles = [mfem.Vector(a) for a in unit_vectors_2d(4)]
        qs = QuadratureSet(angles, [np.pi] * 4)
        assert qs.dim == 2
        np.testing.assert_allclose(values(qs), unit_vectors_2d(4))

    def test_zero_first_moment(self):
        qs = QuadratureSet(unit_vectors_2d(8), [4 * np.pi / 8] * 8)
        np.testing.assert_allclose(qs.weights @ values(qs), 0.0, atol=1e-12)


class TestQuadratureSet3D:
    def test_construction(self):
        qs = QuadratureSet(unit_vectors_3d(), [4 * np.pi / 8] * 8)
        assert qs.dim == 3
        assert len(qs.angles) == 8
        assert all(v.Size() == 3 for v in qs.angles)
        assert qs.weights.sum() == pytest.approx(4 * np.pi)

    def test_second_moment_is_isotropic(self):
        # int Omega_i Omega_j dOmega = (4 pi / 3) delta_ij
        qs = QuadratureSet(unit_vectors_3d(), [4 * np.pi / 8] * 8)
        a = values(qs)
        np.testing.assert_allclose(
            a.T @ (qs.weights[:, None] * a), 4 * np.pi / 3 * np.eye(3), atol=1e-12
        )


class TestQuadratureSetValidation:
    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same"):
            QuadratureSet([-0.5, 0.5], [2.0])

    def test_mixed_dimensions(self):
        with pytest.raises(ValueError, match="same dimension"):
            QuadratureSet([[1.0, 0.0], [0.0, 0.0, 1.0]], [2 * np.pi, 2 * np.pi])

    def test_dim_too_large(self):
        with pytest.raises(ValueError, match="Invalid angle dimension"):
            QuadratureSet([[1.0, 0, 0, 0], [-1.0, 0, 0, 0]], [2 * np.pi] * 2)

    @pytest.mark.parametrize("scale", [0.5, 2.0])
    def test_grossly_wrong_weights_raise(self, scale):
        with pytest.raises(ValueError, match="Invalid quadrature weights"):
            QuadratureSet([-0.5, 0.5], [scale, scale])

    def test_error_above_tolerance_raises(self):
        # hard-error tolerance is 1e-5 relative
        with pytest.raises(ValueError, match="Invalid quadrature weights"):
            QuadratureSet([-0.5, 0.5], [1.0 + 1e-4] * 2)

    def test_small_error_renormalized_with_warning(self):
        # between warn (1e-8) and error (1e-5) tolerances
        with pytest.warns(UserWarning, match="Normalizing"):
            qs = QuadratureSet([-0.5, 0.5], [1.0 + 1e-6] * 2)
        assert qs.weights.sum() == pytest.approx(2.0, rel=1e-14)

    def test_tiny_error_renormalized_silently(self):
        # below the warn tolerance: renormalized exactly, no warning
        # (the autouse fixture turns any UserWarning into a failure)
        qs = QuadratureSet([-0.5, 0.5], [1.0 + 1e-10] * 2)
        assert qs.weights.sum() == pytest.approx(2.0, rel=1e-14)

    def test_normalization_preserves_ratios(self):
        with pytest.warns(UserWarning):
            qs = QuadratureSet([-0.5, 0.5], [0.5 * (1 + 1e-6), 1.5 * (1 + 1e-6)])
        assert qs.weights[1] / qs.weights[0] == pytest.approx(3.0)
        assert qs.weights.sum() == pytest.approx(2.0, rel=1e-14)

    def test_3d_wrong_total_raises(self):
        with pytest.raises(ValueError, match="Invalid quadrature weights"):
            QuadratureSet(unit_vectors_3d(), [1.0] * 8)


def polynomial_exact(qs, degree):
    """Check sum w_i mu_i^k against int_{-1}^{1} mu^k dmu for k <= degree."""
    mu = values(qs)[:, 0]
    for k in range(degree + 1):
        exact = 0.0 if k % 2 else 2.0 / (k + 1)
        assert (qs.weights * mu**k).sum() == pytest.approx(exact, abs=1e-12), k


class TestGaussLegendre:
    @pytest.mark.parametrize("N", [2, 4, 8, 16])
    def test_structure(self, N):
        qs = gauss_legendre(N)
        assert qs.dim == 1
        assert len(qs.angles) == N
        assert qs.weights.shape == (N,)
        assert qs.name == f"S{N} Gauss-Legendre"
        assert qs.weights.sum() == pytest.approx(2.0)
        assert np.all(qs.weights > 0)

    @pytest.mark.parametrize("N", [2, 4, 8, 16])
    def test_exactness(self, N):
        # N-point Gauss-Legendre integrates degree 2N-1 exactly
        polynomial_exact(gauss_legendre(N), 2 * N - 1)

    @pytest.mark.parametrize("N", [2, 4, 8])
    def test_symmetry(self, N):
        qs = gauss_legendre(N)
        mu = values(qs)[:, 0]
        np.testing.assert_allclose(np.sort(mu), np.sort(-mu), atol=1e-14)
        assert np.all(np.abs(mu) < 1)

    def test_known_s2(self):
        qs = gauss_legendre(2)
        np.testing.assert_allclose(
            np.sort(values(qs)[:, 0]), [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        )
        np.testing.assert_allclose(qs.weights, [1.0, 1.0])

    @pytest.mark.parametrize("N", [-1, 0, 1])
    def test_too_small(self, N):
        with pytest.raises(ValueError, match="at least 2"):
            gauss_legendre(N)

    @pytest.mark.parametrize("N", [3, 5])
    def test_odd_warns(self, N):
        with pytest.warns(UserWarning, match="odd"):
            qs = gauss_legendre(N)
        assert len(qs.angles) == N


class TestDoubleGauss:
    @pytest.mark.parametrize("N", [1, 2, 4, 8])
    def test_structure(self, N):
        qs = double_gauss(N)
        assert qs.dim == 1
        assert len(qs.angles) == 2 * N
        assert qs.name == f"Double S{N} Gauss-Legendre"
        assert qs.weights.sum() == pytest.approx(2.0)
        assert np.all(qs.weights > 0)

    @pytest.mark.parametrize("N", [1, 2, 4, 8])
    def test_half_ranges(self, N):
        mu = values(double_gauss(N))[:, 0]
        assert np.all(mu[:N] < 0) and np.all(mu[:N] > -1)
        assert np.all(mu[N:] > 0) and np.all(mu[N:] < 1)

    @pytest.mark.parametrize("N", [1, 2, 4, 8])
    def test_half_range_weights(self, N):
        # each hemisphere integrates the half-range measure exactly
        qs = double_gauss(N)
        assert qs.weights[:N].sum() == pytest.approx(1.0)
        assert qs.weights[N:].sum() == pytest.approx(1.0)

    @pytest.mark.parametrize("N", [2, 4, 8])
    def test_exactness_on_each_half(self, N):
        qs = double_gauss(N)
        mu = values(qs)[:, 0]
        for k in range(2 * N):
            neg = (qs.weights[:N] * mu[:N] ** k).sum()
            pos = (qs.weights[N:] * mu[N:] ** k).sum()
            assert pos == pytest.approx(1.0 / (k + 1), abs=1e-12)
            assert neg == pytest.approx((-1) ** k / (k + 1), abs=1e-12)

    def test_symmetry(self):
        mu = values(double_gauss(4))[:, 0]
        np.testing.assert_allclose(np.sort(mu), np.sort(-mu), atol=1e-14)

    @pytest.mark.parametrize("N", [-1, 0])
    def test_too_small(self, N):
        with pytest.raises(ValueError, match="at least 1"):
            double_gauss(N)


class TestGaussLobatto:
    @pytest.mark.parametrize("N", [2, 4, 6, 8])
    def test_structure(self, N):
        qs = gauss_lobatto(N)
        assert qs.dim == 1
        assert len(qs.angles) == N
        assert qs.name == f"{N}-point Gauss-Lobatto"
        assert qs.weights.sum() == pytest.approx(2.0)
        assert np.all(qs.weights > 0)

    @pytest.mark.parametrize("N", [2, 4, 6, 8])
    def test_endpoints(self, N):
        mu = np.sort(values(gauss_lobatto(N))[:, 0])
        assert mu[0] == pytest.approx(-1.0)
        assert mu[-1] == pytest.approx(1.0)

    @pytest.mark.parametrize("N", [2, 4, 6, 8])
    def test_exactness(self, N):
        # N-point Gauss-Lobatto integrates degree 2N-3 exactly
        polynomial_exact(gauss_lobatto(N), 2 * N - 3)

    @pytest.mark.parametrize("N", [4, 6])
    def test_symmetry(self, N):
        qs = gauss_lobatto(N)
        mu = values(qs)[:, 0]
        order = np.argsort(mu)
        np.testing.assert_allclose(mu[order], -mu[order][::-1], atol=1e-12)
        np.testing.assert_allclose(
            qs.weights[order], qs.weights[order][::-1], atol=1e-12
        )

    def test_known_s4(self):
        # nodes +-1, +-1/sqrt(5); weights 1/6, 5/6
        qs = gauss_lobatto(4)
        order = np.argsort(values(qs)[:, 0])
        np.testing.assert_allclose(
            values(qs)[order, 0], [-1, -1 / np.sqrt(5), 1 / np.sqrt(5), 1], atol=1e-12
        )
        np.testing.assert_allclose(
            qs.weights[order], [1 / 6, 5 / 6, 5 / 6, 1 / 6], atol=1e-12
        )

    @pytest.mark.parametrize("N", [-1, 0, 1])
    def test_too_small(self, N):
        with pytest.raises(ValueError, match="at least 2"):
            gauss_lobatto(N)

    def test_odd_warns(self):
        with pytest.warns(UserWarning, match="Gauss-Lobatto"):
            qs = gauss_lobatto(5)
        assert len(qs.angles) == 5


class TestLevelSymmetric:
    # The autouse fixture turns any UserWarning into a failure, so every test
    # below that doesn't use pytest.warns also checks that no warning is emitted.
    ORDERS = [2, 4, 6, 8, 10, 12, 16]

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("N", ORDERS)
    def test_structure(self, N, dim):
        qs = level_symmetric(N, dim)
        assert qs.dim == dim
        assert qs.name == f"S{N} level-symmetric"
        n_octant = N * (N + 2) // 8
        assert len(qs.angles) == len(qs.weights) == n_octant * 2**dim
        assert qs.weights.sum() == pytest.approx(4 * np.pi)
        assert np.all(qs.weights > 0)

    @pytest.mark.parametrize("N", ORDERS)
    def test_unit_vectors(self, N):
        np.testing.assert_allclose(
            np.linalg.norm(values(level_symmetric(N, 3)), axis=1), 1.0
        )

    @pytest.mark.parametrize("N", ORDERS)
    def test_level_symmetric_cosines(self, N):
        # One set of N/2 cosines is shared by all axes, equally spaced in mu^2
        mu = values(level_symmetric(N, 3))
        levels = np.unique(np.round(np.abs(mu), 10))
        assert len(levels) == N // 2
        if N > 2:
            np.testing.assert_allclose(
                np.diff(levels**2), np.diff(levels**2)[0], rtol=1e-6
            )

    @pytest.mark.parametrize("N", ORDERS)
    def test_moments_3d(self, N):
        qs = level_symmetric(N, 3)
        w, mu = qs.weights, values(qs)
        np.testing.assert_allclose(w @ mu, 0.0, atol=1e-12)  # flow condition
        np.testing.assert_allclose(w @ mu**2, 4 * np.pi / 3)  # diffusion condition
        # Even moments up to order N-2 match the sphere integral of mu^2 eta^2 = 4pi/15
        if N >= 6:
            assert w @ (mu[:, 0] ** 2 * mu[:, 1] ** 2) == pytest.approx(4 * np.pi / 15)

    @pytest.mark.parametrize("N", ORDERS)
    def test_reflection_symmetry(self, N):
        qs = level_symmetric(N, 3)
        mu = values(qs)
        for axis in range(3):
            flip = mu.copy()
            flip[:, axis] *= -1
            # every direction's mirror image exists with the same weight
            lookup = {tuple(np.round(m, 9)): w for m, w in zip(mu, qs.weights)}
            for m, w in zip(flip, qs.weights):
                assert lookup[tuple(np.round(m, 9))] == pytest.approx(w)

    @pytest.mark.parametrize("N", ORDERS)
    def test_axis_permutation_symmetry(self, N):
        qs = level_symmetric(N, 3)
        mu = values(qs)
        lookup = {tuple(np.round(m, 9)): w for m, w in zip(mu, qs.weights)}
        for perm in ([1, 0, 2], [2, 1, 0], [0, 2, 1]):
            for m, w in zip(mu[:, perm], qs.weights):
                assert lookup[tuple(np.round(m, 9))] == pytest.approx(w)

    @pytest.mark.parametrize("N", [4, 8])
    def test_2d_matches_3d_projection(self, N):
        q2, q3 = level_symmetric(N, 2), level_symmetric(N, 3)
        # Folding xi=+-: the 2D set is the 3D set projected onto (mu, eta)
        proj = {}
        for m, w in zip(values(q3), q3.weights):
            key = tuple(np.round(m[:2], 9))
            proj[key] = proj.get(key, 0.0) + w
        assert len(proj) == len(q2.angles)
        for m, w in zip(values(q2), q2.weights):
            assert proj[tuple(np.round(m, 9))] == pytest.approx(w)

    def test_default_dim_is_3(self):
        assert level_symmetric(4).dim == 3

    def test_high_order_negative_weights_warn(self):
        with pytest.warns(UserWarning, match="negative weights"):
            qs = level_symmetric(20, 3)
        assert qs.weights.sum() == pytest.approx(4 * np.pi)
        assert np.any(qs.weights < 0)

    @pytest.mark.parametrize("dim", [1, 4, 0])
    def test_invalid_dim(self, dim):
        with pytest.raises(ValueError, match="'dim' must be 2 or 3"):
            level_symmetric(4, dim)

    @pytest.mark.parametrize("N", [-2, 0, 1, 3, 5])
    def test_invalid_N(self, N):
        with pytest.raises(ValueError, match="even integer"):
            level_symmetric(N, 3)
