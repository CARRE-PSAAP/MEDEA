MEDEA Theory Guide
==================

Sweep Algorithm
~~~~~~~~~~~~~~~

When the Boltzmann Transport Equation is discretized with the Discontinuous Galerkin (DG) method and discrete ordinates, the upwind condition for
angular flux results in a coupling between a cell and its "upstream" neighbors. The mesh can be "swept" to invert the transport operator, where each
cell can be solved individually if the angular flux of its upstream neighbors is known. This is equivalent to forward substitution to solve a block
lower triangular system.

In order to determine an ordering for cells, a graph is formed, where each cell in the mesh is a vertex, and each face is a directed edge. A topological
ordering alorithm is then used to determine a sweep order. MEDEA uses Kahn's algorithm (:prf:ref:`kahn-ordering`).


.. prf:algorithm:: Kahn
    :label: kahn-ordering

    **Inputs** Given a directed graph $G(V, E)$ of vertices $V = \{v_i\}_{i=1}^{\vert V\vert}$, edges $E =\{(u, v)\}$

    **Output** Compute a topological ordering $\pi: \{1, 2,\dots, \lvert V\rvert\} \rightarrow V$ such that $\pi^{-1}(u) \lt \pi^{-1}(v) \forall (u, v) \in E$

    1. Compute the in-degree $D_i$ of all $v_i$
    2. Add all $v_i \vert D_i = 0$ to :code:`queue`
    3. while :code:`queue` is not empty:

        1. Remove a vertex $v_i$ from :code:`queue`
        2. Append $i$ to :code:`output`
        3. For each immediate child $v_j$ of $v_i$:

            1. Decrease the in-degree $D_j$
            2. If $D_j = 0$, add $v_j$ to the queue

    If $\vert \pi\vert \neq \vert V \vert$, the graph is cyclic and no such topological ordering exists. Else return :code:`output`

:class:`medea.sweep.TransportSweep` implements Kahn's algorithm and the block lower triangular solver.

.. note::

    When the graph is built in :class:`medea.sweep.TransportSweep`, the unit normal vector is evaluated only at a single point on each face; the sweep
    only supports meshes with flat faces for now.


