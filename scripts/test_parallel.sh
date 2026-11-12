#!/usr/bin/env bash
# Run the test suite against a parallel (MPI) install of MEDEA, twice:
#   1. on a single rank (mpi-marked tests are skipped), then
#   2. under mpiexec with NP ranks (default 2) and --with-mpi (everything runs).
# Usage: source .venv/bin/activate && scripts/test_parallel.sh [pytest args]
# Needs pytest and pytest-mpi (pip install pytest pytest-mpi).
set -euo pipefail

cd "$(dirname "$0")/.."
NP="${NP:-2}"

# Fail, rather than skip, if this is not a parallel install.
export MEDEA_REQUIRE_PARALLEL=1

# Run the `pytest` executable, not `python -m pytest`: the latter puts the
# repo root on sys.path, so a regular (non-editable) install would import the
# source tree's medea/ (which has no compiled _kernels) instead of the installed one.
echo "== single rank =="
pytest tests "$@"

echo "== $NP ranks =="
mpiexec -n "$NP" pytest --with-mpi tests "$@"
