"""Hello world: use MFEM (through PyMFEM) from MEDEA, serial or parallel.

    python examples/hello_mfem.py                  # serial (or one rank)
    mpiexec -n 2 python examples/hello_mfem.py     # parallel install only

The code is identical in both cases; MEDEA picks the serial or parallel PyMFEM
(override with MEDEA_BACKEND=ser). Run from outside the repository root if you
installed without -e.
"""

import medea

r = medea.hello_mfem(n=4, order=2)

# Print rank by rank so parallel output is not interleaved.
for rank in range(r.num_ranks):
    if medea.MFEM_MODE == "par":
        medea._backend.comm.Barrier()
    if rank == r.rank:
        print(
            f"rank {r.rank}/{r.num_ranks}: "
            f"{r.local_elements} of {r.num_elements} elements, "
            f"{r.local_dofs} of {r.num_dofs} dofs"
        )

if r.rank == 0:
    print(f"MFEM mode      : {medea.MFEM_MODE}")
    print(f"MFEM version   : {r.mfem_version}")
    print(f"integral of 1  : {r.integral:.12f}  (expected 1)")
