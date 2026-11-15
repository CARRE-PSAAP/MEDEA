"""1D / 2D / 3D CubeSat electronics-bay mesh generator (gmsh).

A 1U (10 cm) CubeSat with two stacked PCBs carrying bare silicon dies, a
battery pack, a reaction wheel and thick corner rails.  The dies are the
*sensitive volumes*: the quantity of interest for a shielding study is the
dose integrated over material SILICON.  One die has a tantalum spot shield.

The model is defined once, as 3D boxes.  The 2D mesh is the z = ZC slice of
it and the 1D mesh is the vertical line x = XC, z = ZC through it.

Materials (mesh attribute / physical group id):
    1 vacuum
    2 aluminum (6061 walls, 2 mm; solid 8 mm corner rails)
    3 silicon  (500 um die = SENSITIVE VOLUME)
    4 FR-4     (1.6 mm printed circuit boards)
    5 tantalum (1 mm spot-shield plate over one die)
    6 battery  (homogenized Li-ion pack)
    7 steel    (reaction-wheel flywheel, modeled as a solid puck)

Two mesh types:
    --mesh tet  (default) conforming triangles/tets/segments via gmsh -> .msh
    --mesh hex  axis-aligned quads/hexes via MFEM, locally refined with
                NON-CONFORMING (hanging-node) refinement -> MFEM .mesh file.
                Needs MFEM (medea) but not gmsh.  Geometry is exact because
                every region boundary is a grid line.

Usage:
    python cubesat.py [--dim 1|2|3] [--mesh tet|hex] [--out FILE] [--scale S] [--gui]

All lengths are in cm.  Simplifications: dies sit directly on the boards (no
package or bond wires), the Ta shield is a plate (its supports are out of
plane), rails are solid, the wheel is a solid block.
"""

import argparse
import math
from pathlib import Path

import numpy as np

try:
    import gmsh  # only needed for --mesh tet
except ImportError:
    gmsh = None

# ============================================================== PARAMETERS ==
VACUUM, ALUMINUM, SILICON, FR4, TANTALUM, BATTERY, STEEL = range(1, 8)
NAMES = ["vacuum", "aluminum", "silicon", "fr4", "tantalum", "battery", "steel"]

# --- geometry (cm)
BOX = 10.0  # 1U outer envelope
WALL_T = 0.20  # aluminum wall / end plate thickness
RAIL = 0.80  # corner rail cross-section (square)
PCB_T = 0.16  # 1.6 mm FR-4 board
DIE_T = 0.05  # 500 um silicon die
PCB_Y = (2.5, 6.5)  # y of the bottom face of the (lower, upper) board

# dies: name -> (x0, x1, z0, z1, board index, "top" | "bottom" face of board)
DIES = {
    "fpga": (2.0, 3.2, 4.4, 5.6, 0, "top"),  # bare
    "memory": (5.5, 6.3, 4.6, 5.4, 0, "top"),  # under the Ta plate
    "sensor": (4.0, 5.0, 4.5, 5.5, 1, "top"),
    "adc": (6.0, 7.0, 4.5, 5.5, 1, "bottom"),  # hangs below the upper board
}
CAP_DIE, CAP_GAP, CAP_T, CAP_OVER = "memory", 0.10, 0.10, 0.20  # Ta plate
BATTERY_BOX = (1.5, 8.5, 1.5, 8.5, 1.8)  # x0, x1, z0, z1, thickness; on bottom wall
WHEEL_BOX = (6.0, 9.0, 3.5, 6.5, 0.8)  # x0, x1, z0, z1, height; on upper board

# --- where the 2D / 1D meshes cut the model
XC, ZC = 6.2, 5.0  # 1D line passes through memory die, Ta plate, adc die, wheel

# default output directory: examples/data next to this script (any cwd)
DATA_DIR = Path(__file__).resolve().parent / "data"

# --- element size (cm) per material (order of NAMES): fine for 1D/2D,
#     feature-sized (coarse) for 3D tets
H_FINE = [0.5, 0.10, 0.025, 0.08, 0.05, 0.15, 0.10]
H_3D = [1.0, 0.20, 0.05, 0.16, 0.10, 0.50, 0.40]

# --- hex/quad mesh (--mesh hex): target max element edge (cm) per material, in
#     NAMES order.  Elements are split only along axes longer than the target.
H_HEX = [0.5, 0.10, 0.025, 0.08, 0.05, 0.15, 0.10]
H_HEX_3D = [1.0, 0.20, 0.05, 0.16, 0.10, 0.50, 0.40]

# --- tet mesh grading (3D only for INTERIOR and FAR_*)
GRADE = 1.0  # distance over which sizes grow from a solid's boundary size
INTERIOR = 3.0  # 3D: element size at the center of a solid, relative to its boundary size
FAR_SIZE = 0.8  # 3D: max element size far from the silicon dies (None: off)
FAR_DIST = (1.0, 5.0)  # 3D: distance from silicon over which sizes grow to FAR_SIZE
# ============================================================================


def build_boxes():
    """All solid regions as (material, (x0,y0,z0), (x1,y1,z1)); disjoint."""
    B, T, R = BOX, WALL_T, RAIL
    bx = []

    def add(m, x0, y0, z0, x1, y1, z1):
        bx.append((m, (x0, y0, z0), (x1, y1, z1)))

    add(ALUMINUM, 0, 0, 0, B, B, T)  # end plates (z faces)
    add(ALUMINUM, 0, 0, B - T, B, B, B)
    for x in (0, B - R):  # corner rails, along z
        for y in (0, B - R):
            add(ALUMINUM, x, y, T, x + R, y + R, B - T)
    add(ALUMINUM, 0, R, T, T, B - R, B - T)  # side walls between the rails
    add(ALUMINUM, B - T, R, T, B, B - R, B - T)
    add(ALUMINUM, R, 0, T, B - R, T, B - T)
    add(ALUMINUM, R, B - T, T, B - R, B, B - T)

    for y in PCB_Y:
        add(FR4, T, y, T, B - T, y + PCB_T, B - T)

    sv = {}
    for name, (x0, x1, z0, z1, ib, side) in DIES.items():
        y = PCB_Y[ib] + PCB_T if side == "top" else PCB_Y[ib] - DIE_T
        add(SILICON, x0, y, z0, x1, y + DIE_T, z1)
        sv[name] = bx[-1]
    _, (x0, y0, z0), (x1, y1, z1) = sv[CAP_DIE]
    o, g = CAP_OVER, CAP_GAP
    add(TANTALUM, x0 - o, y1 + g, z0 - o, x1 + o, y1 + g + CAP_T, z1 + o)

    x0, x1, z0, z1, t = BATTERY_BOX
    add(BATTERY, x0, T, z0, x1, T + t, z1)
    x0, x1, z0, z1, h = WHEEL_BOX
    y = PCB_Y[1] + PCB_T
    add(STEEL, x0, y, z0, x1, y + h, z1)
    return bx, sv


def section(boxes, dim):
    """Reduce the 3D boxes to `dim` dimensions by slicing."""
    axes = {3: (0, 1, 2), 2: (0, 1), 1: (1,)}[dim]
    fixed = {3: {}, 2: {2: ZC}, 1: {0: XC, 2: ZC}}[dim]
    return [
        (m, tuple(lo[a] for a in axes), tuple(hi[a] for a in axes))
        for m, lo, hi in boxes
        if all(lo[a] < c < hi[a] for a, c in fixed.items())
    ]


def check_geometry(regs):
    eps = 1e-12
    for m, lo, hi in regs:
        assert all(a < b for a, b in zip(lo, hi)), f"degenerate region {lo} {hi}"
        assert all(-eps <= a and b <= BOX + eps for a, b in zip(lo, hi)), "outside box"
    for i, (_, l1, h1) in enumerate(regs):
        for _, l2, h2 in regs[i + 1 :]:
            assert not all(
                min(b1, b2) - max(a1, a2) > eps for a1, b1, a2, b2 in zip(l1, h1, l2, h2)
            ), f"regions overlap: {l1}-{h1} and {l2}-{h2}"


# ====================================================================== #
#  TET / TRIANGLE / SEGMENT MESH  (gmsh, conforming)                      #
# ====================================================================== #
# ------------------------------------------------------------------ 1D mesh --
def mesh_1d(regs, H):
    """Contiguous line segments between every region boundary along y."""
    geo = gmsh.model.geo
    ys = sorted({0.0, BOX, *[v for _, lo, hi in regs for v in (lo[0], hi[0])]})
    pts = [geo.addPoint(y, 0, 0) for y in ys]
    groups = {}
    for i in range(len(ys) - 1):
        mid = 0.5 * (ys[i] + ys[i + 1])
        m = next((m for m, lo, hi in regs if lo[0] < mid < hi[0]), VACUUM)
        c = geo.addLine(pts[i], pts[i + 1])
        n = math.ceil((ys[i + 1] - ys[i]) / H[m - 1] - 1e-9)
        geo.mesh.setTransfiniteCurve(c, n + 1)
        groups.setdefault(m, []).append(c)
    geo.synchronize()
    return groups


# -------------------------------------------------------------- 2D / 3D mesh --
def mesh_nd(regs, dim, H):
    """Fragment the domain box by all regions; size elements per material."""
    occ, F = gmsh.model.occ, gmsh.model.mesh.field

    def add_box(lo, hi):
        size = [b - a for a, b in zip(lo, hi)]
        if dim == 2:
            return occ.addRectangle(*lo, 0, *size)
        return occ.addBox(*lo, *size)

    whole = add_box((0,) * dim, (BOX,) * dim)
    occ.fragment([(dim, whole)], [(dim, add_box(lo, hi)) for _, lo, hi in regs])
    occ.synchronize()

    # material of each fragment: solids are boxes, so match the bounding box
    # (a center of mass is unreliable for non-convex vacuum regions)
    def material(tag):
        bb = gmsh.model.getBoundingBox(dim, tag)
        bb = bb[:dim] + bb[3 : 3 + dim]
        for m, lo, hi in regs:
            if all(abs(u - v) < 1e-5 for u, v in zip(bb, (*lo, *hi))):
                return m
        return VACUUM

    mat = {t: material(t) for _, t in gmsh.model.getEntities(dim)}

    def closure(tags):  # entities plus everything on their boundaries
        out, cur = set(), [(dim, t) for t in tags]
        while cur:
            out |= set(cur)
            cur = gmsh.model.getBoundary(cur, combined=False, oriented=False)
        return out

    def restrict(field, ents):
        r = F.add("Restrict")
        F.setNumber(r, "InField", field)
        keys = ("PointsList", "CurvesList", "SurfacesList", "VolumesList")
        for d, key in enumerate(keys):
            tags = [t for dd, t in ents if dd == d]
            if tags:
                F.setNumbers(r, key, tags)
        return r

    def const(h):
        f = F.add("MathEval")
        F.setString(f, "F", str(h))
        return f

    def threshold(infield, smin, smax, dmin, dmax):
        th = F.add("Threshold")
        F.setNumber(th, "InField", infield)
        for k, v in zip(("SizeMin", "SizeMax", "DistMin", "DistMax"), (smin, smax, dmin, dmax)):
            F.setNumber(th, k, v)
        return th

    def distance(ents):  # distance to the boundary (curves in 2D, surfaces in 3D)
        d = F.add("Distance")
        key = "CurvesList" if dim == 2 else "SurfacesList"
        F.setNumbers(d, key, sorted({t for dd, t in ents if dd == dim - 1}))
        F.setNumber(d, "Sampling", 50)
        return d

    def combine(kind, fields):
        f = F.add(kind)
        F.setNumbers(f, "FieldsList", fields)
        return f

    def ents_of(m):
        return closure([t for t, mm in mat.items() if mm == m])

    # 3D only: let elements grow away from the sensitive silicon, and toward the
    # interior of each solid (distance from its own boundary)
    far = None
    if dim == 3 and FAR_SIZE and SILICON in mat.values():
        far = threshold(distance(ents_of(SILICON)), 0.0, FAR_SIZE, FAR_DIST[0], FAR_DIST[1])

    def grow(f):  # coarsen field f by the far-from-silicon growth, if any
        return combine("Max", [f, far]) if far is not None else f

    vacuum = closure([t for t, m in mat.items() if m == VACUUM])
    fields, vac_fields = [], [const(H[VACUUM - 1])]
    for m in sorted(set(mat.values()) - {VACUUM}):
        ents = ents_of(m)
        dist = distance(ents)
        inner = [const(H[m - 1])]
        if dim == 3 and m != SILICON:
            inner.append(threshold(dist, H[m - 1], INTERIOR * H[m - 1], 0, GRADE))
        fields.append(restrict(grow(combine("Max", inner)), ents))
        # in vacuum, grow from this solid's size toward the vacuum size
        vac_fields.append(threshold(dist, H[m - 1], H[VACUUM - 1], 0, GRADE))
    fields.append(restrict(grow(combine("Min", vac_fields)), vacuum))
    mn = F.add("Min")
    F.setNumbers(mn, "FieldsList", fields)
    F.setAsBackgroundMesh(mn)
    for opt in ("FromPoints", "FromCurvature", "ExtendFromBoundary"):
        gmsh.option.setNumber("Mesh.MeshSize" + opt, 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay (2D)

    groups = {}
    for t, m in mat.items():
        groups.setdefault(m, []).append(t)
    return groups


# ====================================================================== #
#  HEX / QUAD MESH  (MFEM, axis-aligned, non-conforming refinement)       #
#  Independent of the gmsh code above.                                    #
# ====================================================================== #
def mesh_hex(regs, dim, H):
    """Return (mfem.Mesh, sizes): quads/hexes with attributes = material ids.

    1. Base tensor grid with a grid line at every region boundary, so each base
       cell lies in exactly one material and the geometry is exact.
    2. Repeatedly split cells (hanging nodes allowed) along only those axes
       whose edge is longer than the target size H of the cell's material.
    """
    from medea._backend import mfem

    # base grid coordinates per axis (merge coordinates closer than 1e-9)
    axes = []
    for a in range(dim):
        vals = sorted({0.0, BOX, *[v for _, lo, hi in regs for v in (lo[a], hi[a])]})
        merged = [vals[0]]
        for v in vals[1:]:
            if v - merged[-1] > 1e-9:
                merged.append(v)
        axes.append(merged)
    n = [len(c) - 1 for c in axes]

    # unit-cell Cartesian mesh (integer vertex coordinates), then move the
    # vertices onto the real grid lines
    if dim == 1:
        mesh = mfem.Mesh.MakeCartesian1D(n[0], float(n[0]))
    elif dim == 2:
        mesh = mfem.Mesh.MakeCartesian2D(
            n[0], n[1], mfem.Element.QUADRILATERAL, True, float(n[0]), float(n[1])
        )
    else:
        mesh = mfem.Mesh.MakeCartesian3D(
            n[0], n[1], n[2], mfem.Element.HEXAHEDRON, float(n[0]), float(n[1]), float(n[2])
        )
    for k in range(mesh.GetNV()):
        v = mesh.GetVertexArray(k)
        for a in range(dim):
            v[a] = axes[a][int(round(v[a]))]

    def bbox(e):
        pts = np.array([mesh.GetVertexArray(int(i))[:dim] for i in mesh.GetElementVertices(e)])
        return pts.min(axis=0), pts.max(axis=0)

    def material(lo, hi):
        mid = 0.5 * (lo + hi)
        return next(
            (m for m, rlo, rhi in regs if all(r0 < c < r1 for r0, c, r1 in zip(rlo, mid, rhi))),
            VACUUM,
        )

    for e in range(mesh.GetNE()):
        mesh.SetAttribute(e, material(*bbox(e)))

    mesh.EnsureNCMesh()
    for _ in range(40):
        refs = mfem.RefinementArray()
        for e in range(mesh.GetNE()):
            lo, hi = bbox(e)
            h = H[mesh.GetAttribute(e) - 1]
            bits = sum(1 << a for a in range(dim) if hi[a] - lo[a] > h * (1 + 1e-9))
            if bits:
                refs.Append(mfem.Refinement(e, bits))
        if refs.Size() == 0:
            break
        mesh.GeneralRefinement(refs, 1)  # 1 = non-conforming
    return mesh


def show_hex(mesh, dim, run=True):
    """View an MFEM hex/quad mesh in the gmsh GUI, one colored entity per material.

    Display only (nodes are duplicated per material); the mesh file written is
    the MFEM one.  Hanging nodes show as T-junctions.
    """
    gmsh.initialize()
    gmsh.model.add("cubesat_hex")
    gtype = {1: 1, 2: 3, 3: 5}[dim]  # gmsh line / quad / hex8; vertex order matches MFEM
    elems = {}
    for e in range(mesh.GetNE()):
        elems.setdefault(mesh.GetAttribute(e), []).append(e)
    next_node = 1
    for m, es in elems.items():
        gmsh.model.addDiscreteEntity(dim, m)
        verts = sorted({int(v) for e in es for v in mesh.GetElementVertices(e)})
        tag = {v: next_node + i for i, v in enumerate(verts)}
        coords = np.concatenate([np.pad(mesh.GetVertexArray(v)[:dim], (0, 3 - dim)) for v in verts])
        gmsh.model.mesh.addNodes(dim, m, list(tag.values()), coords)
        conn = [tag[int(v)] for e in es for v in mesh.GetElementVertices(e)]
        gmsh.model.mesh.addElementsByType(m, gtype, [], conn)
        gmsh.model.addPhysicalGroup(dim, [m], m, NAMES[m - 1])
        next_node += len(verts)
    if run:
        gmsh.fltk.run()
    return sum(len(gmsh.model.mesh.getElements(dim, m)[1][0]) for m in elems)


# ------------------------------------------------------------------- main --
def sensitive_report(sv):
    print("sensitive volumes (material 3, cm):")
    for name, (_, lo, hi) in sv.items():
        print(f"  {name:7s} x=[{lo[0]},{hi[0]}] y=[{lo[1]:.3f},{hi[1]:.3f}] z=[{lo[2]},{hi[2]}]")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dim", type=int, choices=[1, 2, 3], default=2)
    p.add_argument("--mesh", choices=["tet", "hex"], default="tet")
    p.add_argument("--out", help="output file (default examples/data/cubesat_<dim>d[_hex].msh/.mesh)")
    p.add_argument("--scale", type=float, default=1.0, help="scale all mesh sizes")
    p.add_argument("--gui", action="store_true", help="open the mesh in the gmsh GUI")
    a = p.parse_args()

    boxes, sv = build_boxes()
    regs = section(boxes, a.dim)
    check_geometry(regs)

    if a.mesh == "hex":
        out = a.out or str(DATA_DIR / f"cubesat_{a.dim}d_hex.mesh")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        H = [h * a.scale for h in (H_HEX_3D if a.dim == 3 else H_HEX)]
        mesh = mesh_hex(regs, a.dim, H)
        mesh.Print(out)
        counts = {NAMES[m - 1]: 0 for m in range(1, 8)}
        for e in range(mesh.GetNE()):
            counts[NAMES[mesh.GetAttribute(e) - 1]] += 1
        print(f"wrote {out}: {a.dim}D hex/quad, {mesh.GetNE()} elements, "
              f"nonconforming={mesh.Nonconforming()}, per material {counts}")
        sensitive_report(sv)
        if a.gui:
            assert gmsh is not None, "gmsh is required for --gui"
            show_hex(mesh, a.dim)
            gmsh.finalize()
        return

    assert gmsh is not None, "gmsh is required for --mesh tet (pip install gmsh)"
    out = a.out or str(DATA_DIR / f"cubesat_{a.dim}d.msh")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    H = [h * a.scale for h in (H_3D if a.dim == 3 else H_FINE)]

    gmsh.initialize()
    gmsh.model.add("cubesat")
    groups = mesh_1d(regs, H) if a.dim == 1 else mesh_nd(regs, a.dim, H)
    for m, tags in groups.items():
        gmsh.model.addPhysicalGroup(a.dim, tags, m, NAMES[m - 1])
    gmsh.model.mesh.generate(a.dim)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(out)

    counts = {}
    for m, tags in groups.items():
        counts[NAMES[m - 1]] = sum(
            len(e) for t in tags for e in gmsh.model.mesh.getElements(a.dim, t)[1]
        )
    print(f"wrote {out}: {a.dim}D, elements per material {counts}")
    sensitive_report(sv)
    if a.gui:
        gmsh.fltk.run()
    gmsh.finalize()


if __name__ == "__main__":
    main()
