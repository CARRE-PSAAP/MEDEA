"""Backend selection. PyMFEM allows one mode per process, so each case runs in
a fresh subprocess.

Skipped under mpiexec: these test selection logic, not ranks, and a child
process that inherits the MPI launcher's environment can join (and hang) the
parent's MPI job.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from medea import _backend

pytestmark = pytest.mark.skipif(
    _backend.size() > 1, reason="selection logic; run on a single rank"
)

HAS_PAR = _backend._parallel_installed()


def run(code, backend=None, tmp_path=None):
    env = {k: v for k, v in os.environ.items() if k != "MEDEA_BACKEND"}
    if backend is not None:
        env["MEDEA_BACKEND"] = backend
    # cwd away from the repo root so the source tree cannot shadow the install
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_auto_matches_what_is_installed(tmp_path):
    r = run("import medea; print(medea.MFEM_MODE)", tmp_path=tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[-1] == ("par" if HAS_PAR else "ser")


def test_force_serial(tmp_path):
    r = run("import medea; print(medea.MFEM_MODE)", backend="ser", tmp_path=tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[-1] == "ser"


@pytest.mark.skipif(not HAS_PAR, reason="needs a parallel PyMFEM")
def test_force_parallel(tmp_path):
    r = run("import medea; print(medea.MFEM_MODE)", backend="par", tmp_path=tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[-1] == "par"


@pytest.mark.skipif(HAS_PAR, reason="serial-only PyMFEM")
def test_force_parallel_without_parallel_build_fails(tmp_path):
    r = run("import medea", backend="par", tmp_path=tmp_path)
    assert r.returncode != 0
    assert "no parallel build" in r.stderr


def test_invalid_value_fails(tmp_path):
    r = run("import medea", backend="bogus", tmp_path=tmp_path)
    assert r.returncode != 0
    assert "MEDEA_BACKEND" in r.stderr


def test_conflict_with_already_imported_mode(tmp_path):
    # mfem.ser is loaded first; asking for par must fail loudly, not misbehave.
    r = run("import mfem.ser, medea", backend="par", tmp_path=tmp_path)
    assert r.returncode != 0
    assert "already" in r.stderr


def test_already_imported_serial_is_respected_in_auto(tmp_path):
    r = run("import mfem.ser, medea; print(medea.MFEM_MODE)", tmp_path=tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[-1] == "ser"


def test_helpers_in_serial_mode(tmp_path):
    code = (
        "import medea; from medea import _backend as b; "
        "print(b.rank(), b.size(), b.is_parallel)"
    )
    r = run(code, backend="ser", tmp_path=tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[-3:] == ["0", "1", "False"]
