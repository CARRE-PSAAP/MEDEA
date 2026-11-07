#pragma once
#include <string>

namespace medea {

struct HelloResult {
  std::string mfem_version;
  int num_elements;
  int num_vertices;
  int num_dofs;
  double integral;  // integral of f=1 over the unit square (should be 1)
};

// Build a n-by-n quad mesh of the unit square, a degree-`order` H1 space, and
// integrate f = 1 over it. Exercises MFEM linking end to end.
HelloResult hello_mfem(int n, int order);

}  // namespace medea
