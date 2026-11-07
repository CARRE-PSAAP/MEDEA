# MEDEA
Deterministic solver for charged particle transport problems, built on
[MFEM](https://mfem.org) (via PyMFEM) and intended for integration with MC/DC.

## Install

Requires Python >= 3.11. MEDEA links against the MFEM that ships inside
[PyMFEM](https://github.com/mfem/PyMFEM), so PyMFEM is installed first and the
build is not isolated.

### Serial (default)
```bash
python -m venv .venv && source .venv/bin/activate
pip install mfem scikit-build-core pybind11 ninja
pip install -e .[test] --no-build-isolation      # drop -e for a regular install
pytest && python examples/hello_mfem.py
```

### Parallel (MPI)
PyPI's PyMFEM wheels are serial only, so parallel PyMFEM is built from source
(slow: compiles MFEM, hypre, METIS). Needs an MPI install (`mpicxx`) and cmake;
the script pulls in `swig`, `mpi4py` and the rest.
```bash
source .venv/bin/activate
scripts/install_parallel.sh --dev                # --dev = editable install
```
It is equivalent to PyMFEM's `pip install ./ -C"with-parallel=Yes"`, followed by
`MEDEA_MFEM_MODE=par pip install -e . --no-build-isolation`. On a cluster, load
your MPI module first. Check the result with `medea.MFEM_MODE` (`"ser"`/`"par"`).

To use an existing MFEM, pass `-Ccmake.define.MEDEA_MFEM_ROOT=<dir>` (a directory
with `include/` and `lib/`). After editing C++ in `src/`, re-run the install command.
