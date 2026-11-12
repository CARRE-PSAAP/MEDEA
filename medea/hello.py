"""Hello world: build a mesh and space with MFEM and integrate f = 1.

The same code runs serial or parallel; ``MFEM_MODE`` decides. In parallel the
mesh is partitioned across ranks and the quantities named ``global_*`` or
``num_*`` are totals over all ranks, so they do not depend on the rank count.
"""

from dataclasses import dataclass

import numpy as np

from . import _backend as backend
from ._backend import mfem


@dataclass(frozen=True)
class HelloResult:
    """Result of :func:`hello_mfem`."""

    mfem_version: str
    num_elements: int  # global
    num_dofs: int  # global true dofs
    integral: float  # integral of f=1 over the unit square (should be 1)
    rank: int
    num_ranks: int
    local_elements: int  # this rank's share of num_elements
    local_dofs: int  # true dofs owned by this rank


def hello_mfem(n=4, order=2, comm=None):
    """Integrate f = 1 over an ``n`` x ``n`` quad mesh of the unit square.

    Uses a degree-``order`` H1 space. Collective in parallel: every rank must
    call it with the same arguments. ``comm`` is an ``mpi4py`` communicator
    (default ``COMM_WORLD``); serial mode accepts only ``None`` or a
    one-rank communicator.
    """
    if n < 1 or order < 1:
        raise ValueError("n and order must be >= 1")

    mesh = mfem.Mesh(n, n, "QUADRILATERAL")
    fec = mfem.H1_FECollection(order, mesh.Dimension())
    one = mfem.ConstantCoefficient(1.0)

    if backend.is_parallel:
        comm = backend.comm if comm is None else comm
        pmesh = mfem.ParMesh(comm, mesh)
        fes = mfem.ParFiniteElementSpace(pmesh, fec)
        b = mfem.ParLinearForm(fes)
        b.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
        b.Assemble()
        # The true-dof vector has one owner per dof with shared contributions
        # already summed, so its entries add up to the integral over all ranks.
        # (Summing the local, unassembled form over ranks would count dofs on
        # partition boundaries once per rank.)
        integral = comm.allreduce(float(np.sum(b.ParallelAssemble().GetDataArray())))
        return HelloResult(
            mfem_version=_mfem_version(),
            num_elements=pmesh.GetGlobalNE(),
            num_dofs=fes.GlobalTrueVSize(),
            integral=integral,
            rank=comm.Get_rank(),
            num_ranks=comm.Get_size(),
            local_elements=pmesh.GetNE(),
            local_dofs=fes.GetTrueVSize(),
        )

    if comm is not None and comm.Get_size() > 1:
        raise ValueError("serial mode cannot use a communicator with >1 rank")
    fes = mfem.FiniteElementSpace(mesh, fec)
    b = mfem.LinearForm(fes)
    b.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
    b.Assemble()
    ones = mfem.GridFunction(fes)
    ones.Assign(1.0)
    return HelloResult(
        mfem_version=_mfem_version(),
        num_elements=mesh.GetNE(),
        num_dofs=fes.GetTrueVSize(),
        integral=float(b(ones)),
        rank=0,
        num_ranks=1,
        local_elements=mesh.GetNE(),
        local_dofs=fes.GetTrueVSize(),
    )


def _mfem_version():
    import mfem as pkg

    return pkg.__version__
