# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/), and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html) as a guide.

## [Unreleased]

### Added

- Add Sphinx documentation, from [@hansenkyle]
- Add C++ and MFEM scaffolding, from [@hansenkyle]
- Add CONTRIBUTING and CHANGELOG, from [@hansenkyle]
- Add PR and Issue templates, from [@nglaser3]


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