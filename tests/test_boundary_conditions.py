"""Tests for medea.bc.BoundaryCondition: validation against mesh and quadrature."""

import numpy as np
import pytest

from medea._backend import mfem
from medea.boundary_conditions import BoundaryCondition
from medea.quadrature import gauss_legendre

N_ANGLES = 4


@pytest.fixture
def mesh():
    # 1D mesh has two boundary regions (attributes 1 and 2)
    return mfem.Mesh(4, 1.0)


@pytest.fixture
def quad():
    return gauss_legendre(N_ANGLES)


def eval_bdr(coef, mesh, attr):
    """Value of a boundary PWConstCoefficient on the boundary element with ``attr``."""
    for i in range(mesh.GetNBE()):
        if mesh.GetBdrAttribute(i) == attr:
            return coef.Eval(
                mesh.GetBdrElementTransformation(i), mfem.IntegrationPoint()
            )
    raise AssertionError(f"no boundary element with attribute {attr}")


def test_validate_accepts_matching_shape(mesh, quad):
    bc = BoundaryCondition(np.ones((2, N_ANGLES)))
    bc.validate(mesh, quad)


def test_wrong_attribute_count_raises(mesh, quad):
    bc = BoundaryCondition(np.ones((3, N_ANGLES)))
    with pytest.raises(ValueError, match="attributes"):
        bc.validate(mesh, quad)


def test_attributes_not_matching_mesh_raise(mesh, quad):
    # right count, but attribute 3 does not exist on the mesh
    bc = BoundaryCondition(np.ones((2, N_ANGLES)), attributes=[1, 3])
    with pytest.raises(ValueError, match="attributes"):
        bc.validate(mesh, quad)


def test_explicit_attributes_map_rows(mesh, quad):
    data = np.arange(2 * N_ANGLES, dtype=float).reshape(2, N_ANGLES)
    bc = BoundaryCondition(data, attributes=[2, 1])
    bc.validate(mesh, quad)
    for m in range(N_ANGLES):
        assert eval_bdr(bc[m], mesh, 2) == data[0, m]
        assert eval_bdr(bc[m], mesh, 1) == data[1, m]


def test_attributes_length_must_match_values():
    with pytest.raises(ValueError, match="Length of attributes"):
        BoundaryCondition(np.ones((2, N_ANGLES)), attributes=[1])


def test_wrong_angle_count_raises(mesh, quad):
    bc = BoundaryCondition(np.ones((2, N_ANGLES + 1)))
    with pytest.raises(ValueError, match="angles"):
        bc.validate(mesh, quad)


def test_indexing_gives_one_coefficient_per_angle(mesh, quad):
    # data is [attribute][angle]; bc[m] is the coefficient for angle m
    data = np.arange(2 * N_ANGLES, dtype=float).reshape(2, N_ANGLES)
    bc = BoundaryCondition(data)
    bc.validate(mesh, quad)
    for m in range(N_ANGLES):
        for attr in (1, 2):
            assert eval_bdr(bc[m], mesh, attr) == data[attr - 1, m]


def test_indexing_before_validate_fails():
    # coefficients only exist once validate() has run
    bc = BoundaryCondition(np.ones((2, N_ANGLES)))
    with pytest.raises(RuntimeError, match="validated"):
        bc[0]


def test_values_must_be_2d():
    with pytest.raises(ValueError, match="2D"):
        BoundaryCondition(np.ones(N_ANGLES))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0])
def test_values_must_be_finite_and_non_negative(bad):
    data = np.ones((2, N_ANGLES))
    data[0, 0] = bad
    with pytest.raises(ValueError, match="finite and non-negative"):
        BoundaryCondition(data)


def test_values_are_converted_to_float(mesh, quad):
    bc = BoundaryCondition([[1] * N_ANGLES] * 2)
    bc.validate(mesh, quad)
    assert eval_bdr(bc[0], mesh, 1) == 1.0
