MEDEA Theory Guide
==================

Boltzmann Transport Equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The linear Boltzmann Transport Equation (BTE) describes the advection (transport) of particles and collisions that redistribute a particle population in phase space.
In operator notation, the steady-state BTE is

.. math::

    \mathcal{L}\psi(\vec{r}, \vec{\Omega}, E) = \mathcal{S}\psi(\vec{r}, \vec{\Omega}, E) + Q_{ext}(\vec{r}, \vec{\Omega}, E)

where :math:`\vec{r}` is the spatial position vector, :math:`\vec{\Omega}` is the direction of flight (unit vector), :math:`E` is particle energy, and :math:`\psi` is the angular flux, which describes the distribution of particles.

:math:`\mathcal{L}` is the transport operator, which describes the straight path a particle follows between collisions:

.. math::

        \mathcal{L}\psi \stackrel{\triangle}{=} \vec{\Omega} \cdot \vec{\nabla} \psi(\vec{r}, \vec{\Omega}, E)

:math:`\mathcal{S}` is the collision operator, which describes the redistribution of particles when they interact with matter:

.. math::

    \mathcal{S}\psi \stackrel{\triangle}{=} \int\limits_0^\infty \int\limits_{4\pi} \Sigma_{s}(\vec{r}, \vec{\Omega}\cdot\vec{\Omega}^\prime, E^\prime \rightarrow E)\psi(\vec{r}, \vec{\Omega}^\prime, E^\prime) d\Omega^\prime dE^\prime -  \Sigma_t(\vec{r}, E)\psi(\vec{r}, \vec{\Omega}, E)

The integral term in the scattering operator represents a particle interacting with matter at position :math:`\vec{r}`, with initial direction and energy :math:`\vec{\Omega}^\prime` and :math:`E^\prime`, "scattering" off the material and exiting with angle and energy :math:`\vec{\Omega}` and :math:`E`.
The :math:`-\Sigma_t\psi` term represents the loss of particles with angle and energy :math:`\vec{\Omega}` and :math:`E`, either by scattering to another angle and energy, or by being absorbed by the material.

:math:`Q_{ext}(\vec{r}, \vec{\Omega}, E)` is the independent source of particles within the medium, which does not depend on the flux :math:`\psi`.
This may represent spontaneous fission in the case of electrons, beta decay in the case of electrons, etc.

The BTE requires the following conditions to be valid:

- Particles travel in a straight line between interactions
- Scattering interactions happen instantaneously, and at a single point in space
- Particle collisions with the same species (i.e. particle-particle instead of particle-material) are far more infrequent than collisions with the material and can be neglected

In order to form a well-posed problem, the transport equation must be accompanied by boundary conditions, often incoming flux boundary boundary conditions (e.g. perscribe no incoming neutrons to a reactor area, or a known spectrum of electron radiation in a sattelite orbit).
The steady-state transport problem in 3D geometry with incoming flux boundary conditions is

.. math::
    \begin{gathered}
        \mathcal{L}\psi(\vec{r}, \vec{\Omega}, E) = \mathcal{S}\psi(\vec{r}, \vec{\Omega}, E) + Q_{ext}(\vec{r}, \vec{\Omega}, E), \vec{\Omega}, E), \quad \vec{r} \in \Gamma, \vec{\Omega} \in  \mathbb{S}^2, E \in [0, \infty)\\
        \psi(\vec{r}, \vec{\Omega}, E) = \psi_{in}(\vec{r}_\gamma, \vec{\Omega}, E), \quad \vec{\Omega} \cdot \vec{n}_e < 0, \vec{r} \in \partial\Gamma
    \end{gathered}

where :math:`\Gamma` is the spatial domain, :math:`\partial\Gamma` is the domain boundary, :math:`\vec{n}_e` is the outward normal unit vector at the domain boundary, and :math:`\psi_{in}` is the known incoming flux, which may be zero.

Electron Interaction Physics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Second Moment Method
~~~~~~~~~~~~~~~~~~~~
