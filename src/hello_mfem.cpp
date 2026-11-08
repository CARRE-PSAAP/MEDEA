#include "hello_mfem.hpp"

#include <stdexcept>

#include "mfem.hpp"

namespace medea {

HelloResult hello_mfem(int n, int order) {
  if (n < 1 || order < 1)
    throw std::invalid_argument("n and order must be >= 1");

  mfem::Mesh mesh =
      mfem::Mesh::MakeCartesian2D(n, n, mfem::Element::QUADRILATERAL);
  mfem::H1_FECollection fec(order, mesh.Dimension());
  mfem::FiniteElementSpace fes(&mesh, &fec);

  mfem::ConstantCoefficient one(1.0);
  mfem::LinearForm b(&fes);
  b.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
  b.Assemble();

  mfem::GridFunction ones(&fes);
  ones = 1.0;

  HelloResult r;
  r.mfem_version = mfem::GetVersionStr();
  r.num_elements = mesh.GetNE();
  r.num_vertices = mesh.GetNV();
  r.num_dofs = fes.GetNDofs();
  r.integral = b(ones);
  return r;
}

}  // namespace medea
