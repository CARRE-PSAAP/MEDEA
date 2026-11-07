import pytest

import medea


def test_mode_is_valid():
    assert medea.MFEM_MODE in ("ser", "par")


def test_hello_mfem_counts():
    r = medea.hello_mfem(n=4, order=2)
    assert r.num_elements == 16
    assert r.num_vertices == 25
    assert r.num_dofs == 81  # (n*order + 1)^2


def test_hello_mfem_integral():
    assert medea.hello_mfem(n=3, order=1).integral == pytest.approx(1.0)


def test_same_mfem_as_pymfem():
    # Guards against linking a different MFEM than the one PyMFEM imports.
    import mfem

    assert mfem.__version__.rsplit(".", 1)[0] in medea.hello_mfem().mfem_version


def test_bad_args():
    with pytest.raises(ValueError):
        medea.hello_mfem(n=0)
