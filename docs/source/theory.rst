MEDEA Theory Guide
==================

Sweep Algorithm
~~~~~~~~~~~~~~~

When the Boltzmann Transport Equation is discretized with the Discontinuous Galerkin (DG) method and discrete ordinates, the upwind condition for
angular flux results in a coupling between a cell and its "upstream" neighbors. The mesh can be "swept" to invert the transport operator, where each
cell can be solved individually if the angular flux of its upstream neighbors is known. This is equivalent to forward substitution to solve a block
lower triangular system.

In order to determine an ordering for cells, a graph is formed, where each cell in the mesh is a vertex, and each face is a directed edge. A topological
ordering alorithm is then used to determine a sweep order. MEDEA uses Kahn's algorithm. Once the graph is built, the "in-degree" (number of incoming edges) of each
graph vertex is calculated. All vertices with degree 0 are added to a queue (these cells are independent of any other cells). Then, while the queue is not empty:

- Remove the first vertex in the queue and append it to the result vector
- Decrease the in-degree of each child of this vertex. If any of them now have a degree of zero, add it to the queue.

If an ordering exists for this mesh, every cell will be added to the result vector. If any cells are leftover, no ordering exists for this mesh.

:class:`medea.sweep.TransportSweep` implements Kahn's algorithm and the block lower triangular solver.

.. note::

    When the graph is built in :class:`medea.sweep.TransportSweep`, the unit normal vector is evaluated only at a single point on each face; the sweep
    only supports meshes with flat faces for now.


