# MEDEA
Deterministic solver for charged particle transport problems, built on
[MFEM](https://mfem.org) (via PyMFEM) and intended for integration with MC/DC.

## Install (development)

The compiled extension links against the MFEM that ships inside PyMFEM, so
PyMFEM must be installed *before* MEDEA and the build must not be isolated.

```bash
python -m venv .venv && source .venv/bin/activate   # Python >= 3.11
pip install mfem scikit-build-core pybind11 ninja   # serial PyMFEM (prebuilt wheel)
pip install -e .[test] --no-build-isolation
pytest
python examples/hello_mfem.py
```

Tip: to avoid typing `--no-build-isolation`, set `PIP_NO_BUILD_ISOLATION=0`
in your environment or `no-build-isolation = false` in `pip.conf`.

After editing C++ in `src/`, re-run `pip install -e . --no-build-isolation`.

### Serial vs. parallel

`MEDEA_MFEM_MODE` selects which PyMFEM build to link: `ser` (default) or `par`.

```bash
MEDEA_MFEM_MODE=par pip install -e . --no-build-isolation
```

The prebuilt PyMFEM wheels are serial only. Parallel PyMFEM is a source build
that needs MPI, `swig` and `mpi4py` (see the
[PyMFEM docs](https://github.com/mfem/PyMFEM)); install it first, e.g.
`pip install mfem --no-binary mfem --no-build-isolation -C with-parallel=1`
(flag syntax varies between PyMFEM releases; check their install notes).
MEDEA fails at configure time with a clear message if the requested flavour
is not present. `medea.MFEM_MODE` reports what was linked.
Use `-Ccmake.define.MEDEA_MFEM_ROOT=<dir>` to point at a different MFEM.

> The parallel path is configured but has not yet been tested.
