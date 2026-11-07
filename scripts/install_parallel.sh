#!/usr/bin/env bash
# Install MEDEA against a *parallel* (MPI) build of PyMFEM.
# Usage: source .venv/bin/activate && scripts/install_parallel.sh [--dev]
#   --dev  editable install (pip install -e); default is a regular install.
set -euo pipefail

cd "$(dirname "$0")/.."
EDITABLE=""
[[ "${1:-}" == "--dev" ]] && EDITABLE="-e"

die() { echo "error: $*" >&2; exit 1; }

# 1. Prerequisites ------------------------------------------------------------
[[ -n "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]] || die "activate a virtualenv or conda env first"
command -v mpicxx >/dev/null || die "mpicxx not found. Install MPI (Debian/Ubuntu: sudo apt install libopenmpi-dev openmpi-bin; macOS: brew install open-mpi) or, on a cluster, load your MPI module."
command -v cmake  >/dev/null || die "cmake not found (pip install cmake, or use your package manager)"
echo "Using MPI compiler: $(command -v mpicxx)"

# 2. Build tools (swig is available from PyPI, no sudo needed) --------------------
python -m pip install --upgrade pip
python -m pip install numpy swig mpi4py scikit-build-core pybind11 ninja setuptools wheel

# 3. Parallel PyMFEM (source build; takes a while: MFEM + hypre + METIS) ----------
python -m pip install mfem --no-binary mfem --no-build-isolation \
    -C"with-parallel=Yes" --force-reinstall --no-deps --verbose

# 4. MEDEA, linked against the parallel MFEM ----------------------------------
MEDEA_MFEM_MODE=par python -m pip install $EDITABLE . --no-build-isolation

# 5. Smoke test ------------------------------------------------------------------
python - <<'PY'
import medea, mfem.par  # noqa: F401
assert medea.MFEM_MODE == "par", medea.MFEM_MODE
print("OK: MEDEA linked to parallel MFEM:", medea.hello_mfem().mfem_version)
PY
