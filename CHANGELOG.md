# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/), and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html) as a guide.

## [Unreleased]

### Added

- Add CONTRIBUTING and CHANGELOG, from [@hansenkyle]
- Add PR and Issue templates, from [@nglaser3]
- Add serial/parallel backend selection (`medea._backend`): parallel MFEM is used automatically when a parallel PyMFEM is installed, with a `MEDEA_BACKEND=ser|par` override, from [@hansenkyle]
- Add `medea.hello_mfem`, which runs serial or parallel with rank-count-independent results, from [@hansenkyle]
- Add the MFEM-free compiled extension `medea._kernels` with a placeholder `sweep_toy` kernel and a Python reference implementation, from [@hansenkyle]
- Add parallel tests (`tests/par`, pytest-mpi), `scripts/test_parallel.sh`, and a parallel CI job, from [@hansenkyle]

### Changed

- Compiled code no longer links MFEM or MPI, so `pip install -e .` no longer needs PyMFEM at build time, from [@hansenkyle]
- Serial CI now tests Python 3.11 and 3.14, from [@hansenkyle]

### Deprecated

### Removed

- Remove the build-time `MEDEA_MFEM_MODE` switch and the MFEM-linked C++ `hello_mfem`, from [@hansenkyle]

### Fixed

### Security

[Unreleased]: https://github.com/CARRE-PSAAP/MEDEA
[@hansenkyle]: https://github.com/hansenkyle
[@nglaser3]: https://github.com/nglaser3