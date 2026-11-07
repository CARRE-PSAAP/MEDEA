"""Hello world: call a C++ function (linked to PyMFEM's MFEM) from Python."""
import medea

r = medea.hello_mfem(n=4, order=2)
print(f"MFEM mode      : {medea.MFEM_MODE}")
print(f"MFEM version   : {r.mfem_version}")
print(f"elements/verts : {r.num_elements} / {r.num_vertices}")
print(f"H1 dofs        : {r.num_dofs}")
print(f"integral of 1  : {r.integral:.12f}  (expected 1)")
