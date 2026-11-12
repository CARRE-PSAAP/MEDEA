#!/usr/bin/env bash
# Install MEDEA with a parallel (MPI) build of PyMFEM.
# A parallel PyMFEM install also contains the serial module, so MEDEA picks
# parallel automatically (MEDEA_BACKEND=ser forces serial).

# Usage: source .venv/bin/activate && scripts/install_parallel.sh [--dev]
#   --dev  editable install (pip install -e); default is a regular install.

set -euo pipefail

cd "$(dirname "$0")/.."
EDITABLE=""
[[ "${1:-}" == "--dev" ]] && EDITABLE="-e"

die() { echo "error: $*" >&2; exit 1; }

# 1. Prerequisites 
[[ -n "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]] || die "activate a virtualenv or conda env first"
command -v mpicxx >/dev/null || die "mpicxx not found. Install MPI (Debian/Ubuntu: sudo apt install libopenmpi-dev openmpi-bin; macOS: brew install open-mpi) or, on a cluster, load your MPI module."
command -v cmake  >/dev/null || die "cmake not found (pip install cmake, or use your package manager)"
echo "Using MPI compiler: $(command -v mpicxx)"

# 2. Build tools
python -m pip install --upgrade pip
python -m pip install numpy swig mpi4py ninja setuptools wheel

# 3. Parallel PyMFEM (this will take a long time)
python -m pip install mfem --no-binary mfem --no-build-isolation \
    -C"with-parallel=Yes" --force-reinstall --no-deps --verbose

# 4. MEDEA
python -m pip install $EDITABLE ".[parallel]"

# 5. Smoke test
cd "$(mktemp -d)"
python - <<'PY'
import medea
assert medea.MFEM_MODE == "par", medea.MFEM_MODE
print("OK: MEDEA is using parallel MFEM:", medea.hello_mfem().mfem_version)
PY
