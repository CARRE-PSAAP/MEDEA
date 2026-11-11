# Contributing

MEDEA uses the **GitHub flow** repository workflow:

- All features and changes are developed on their own branch (created from `dev`).
  This way all development can be viewed by other collaborators.
- When your feature is ready for review, create a Pull Request in GitHub to merge your changes into `dev`.

For more information on the GitFlow strategy, see [its documentation](https://docs.github.com/en/get-started/using-github/github-flow) on GitHub.

Prepare a development checkout with the following steps:

```
git clone git@github.com:CARRE-PSAAP/MEDEA.git
git switch dev
git switch -c <your-branch-name>

...

git push
```

Make sure to regularly pull the latest changes from dev:

```
git switch dev
git pull origin dev
git switch <your-branch-name>
git merge dev
```

Make sure to `git push` on occasion so your work can be seen by the other developers.
When your feature is ready to review, create a pull request on the GitHub web interface, or

```
gh pr create
```

then follow the prompts.

## Code Format and Style

Python code uses the `black` style. Run `black .` from project root to format your code before commiting.

C++ code uses the `Google` style, and uses `clang-format` to perform the formatting. Run `clang-format -i <file.h> <file.cpp>` to format your code before commiting.

## Serial/Parallel Code and Compiled Kernels

MEDEA code runs unchanged in serial and under `mpirun`. A few rules keep that true:

- **Never import `mfem.ser` or `mfem.par` directly.** Use `from medea._backend import mfem`; it picks the right module once per process (see `medea/_backend.py`). PyMFEM cannot load both in one process.
- **Compiled kernels (C++ in `src/`) take plain NumPy arrays, never MFEM or MPI types.** Do not include `mfem.hpp` or `mpi.h` there. Pull the data you need (connectivity, face normals, per-cell matrices) out of MFEM in Python once at setup, and pass arrays in. This keeps one build valid for serial and parallel, and independent of the PyMFEM version. CI fails if `medea._kernels` links `libmfem` or `libmpi`.
- **Every compiled kernel has a pure-Python reference** in `medea/_reference.py` and a parity test in `tests/test_kernels.py`.
- **Reductions must be global.** A raw `mfem.Vector` dot product or a rank-local linear form is local in parallel and double-counts dofs on partition boundaries. Use true-dof vectors (`ParallelAssemble`) and an `mpi4py` reduction; see `medea/hello.py`.
- **Tests:** put anything that should pass in both modes in `tests/`, asserting global quantities that do not depend on the rank count. Put parallel-only checks in `tests/par/` (collected only on a parallel install), and mark tests that need several ranks with `@pytest.mark.mpi(min_size=2)`. Run `scripts/test_parallel.sh` locally to test on 1 and 2 ranks.

## Versioning

MEDEA uses [semantic versioning](https://semver.org/#semantic-versioning-200). When changes are merged into `main` from `dev`, this increments the code version, which is numbered MAJOR.MINOR.PATCH:

- MAJOR version: "breaking" API changes
- MINOR version: backward-compatible additions to functionality
- PATCH version: backward-compatible bug fixes

MEDEA will be on major version zero (0.x.y) until a stable API is developed, ready for interfacing with MC/DC.

