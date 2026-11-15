# MEDEA

**Me**thods for **D**eterministic **E**lectronics transport **A**nalysis: Deterministic solver for charged particle transport problems, intended for integration with MC/DC.


## Install

Requires Python >= 3.11, plus a C++17 compiler and cmake: `pip` builds MEDEA's
small compiled extension (`medea._kernels`). That extension works on plain NumPy
arrays and does not link MFEM or MPI, so it needs no PyMFEM at build time and
the same build serves serial and parallel runs.

### Serial (default)

PyMFEM provides wheels on PyPI for serial MFEM, so a serial build can simply be done with `pip install`:

```bash
python -m venv .venv && source .venv/bin/activate  # or use a conda environment
pip install -e ".[dev]"                            # -e for editable install
python -m pytest && python examples/hello_mfem.py
```

### Parallel (MPI)

Parallel PyMFEM must be manually built on your machine, which requires MPI and Cmake.

Build parallel PyMFEM and install MEDEA all at once using the parallel install script:

```bash
source .venv/bin/activate
scripts/install_parallel.sh --dev                # --dev = editable install
```

It is equivalent to PyMFEM's `pip install ./ -C"with-parallel=Yes"`, followed by
`pip install -e .[parallel]`.

Test and try the parallel build (needs `pytest` and `pytest-mpi`):

```bash
scripts/test_parallel.sh                           # suite on 1 rank, then on 2 (NP=2)
```

Note that parallel execution may fail if `PyMFEM` and `MEDEA` were configured to use different MPI implementations (i.e. conda vs. system MPI).

## Build documentation

To satisfy the documentation build requirements, install MEDEA with the optional "docs" dependencies:

```bash
pip install -e ".[docs]"
```

To build and view the docs,

```bash
cd docs
make html
open build/html/index.html
```

## Contributing

See [CONTRIBUTING](/CONTRIBUTING.md) for guidelines on contributing to the project.

