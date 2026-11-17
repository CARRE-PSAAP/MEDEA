MEDEA Theory Guide
==================

Boltzmann Transport Equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The linear Boltzmann Transport Equation (BTE) describes the advection (transport) of particles and collisions that redistribute a particle population in phase space. The steady-state BTE is

.. math::
    :nowrap:

    \begin{multline}
        \vec{\Omega} \cdot \vec{\nabla} \psi(\vec{r}, \vec{\Omega}, E) + \Sigma_{t}(\vec{r}, E)\psi(\vec{r}, \vec{\Omega}, E) \\
        = \int\limits_{4\pi}d\Omega^\prime \int\limits_0^\infty dE^\prime \Sigma_s(\vec{r}, \vec{\Omega}^\prime \cdot \vec{\Omega}, E^\prime \rightarrow E) \psi(\vec{r}, \vec{\Omega}^\prime, E^\prime) \\
        + Q_{ext}(\vec{r}, \vec{\Omega}, E), \qquad \vec{r} \in \Gamma, \vec{\Omega} \in \mathbb{S}^2, E \in [0, \infty)
    \end{multline}

where 

- :math:`\vec{r}`: spatial position vector
- :math:`\vec{\Omega}`: direction of flight (unit vector)
- :math:`E`: particle energy
- :math:`\psi`: angular flux, interpreted as a path length density of particles
- :math:`\Sigma_t` is the total cross section (the sum of all interaction cross sections)
- :math:`\Sigma_s` is the double-differential scattering cross section.

In operator notation, this is

.. math::

    \mathcal{L}\psi = \mathcal{S}\psi + Q_{ext}(\vec{r},\vec{\Omega}, E),

where :math:`\mathcal{L}` is the transport operator, which describes the straight path a particle follows between collisions,

.. math::

        \mathcal{L}\psi \stackrel{\triangle}{=} \vec{\Omega} \cdot \vec{\nabla} \psi(\vec{r}, \vec{\Omega}, E),

and :math:`\mathcal{S}` is the collision operator, which describes the redistribution of particles when they interact with matter,

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

In order to define a well-posed transport problem, the transport equation must be accompanied by boundary conditions, often incoming flux boundary boundary conditions (e.g. perscribe no incoming neutrons to a reactor area, or a known spectrum of electron radiation in a sattelite orbit).
The steady-state transport problem in 3D geometry with incoming flux boundary conditions is

.. math::
    :nowrap:

    \begin{gathered}
        \mathcal{L}\psi = \mathcal{S}\psi + Q_{ext}(\vec{r}, \vec{\Omega}, E), \vec{\Omega}, E), \quad \vec{r} \in \Gamma, \vec{\Omega} \in  \mathbb{S}^2, E \in [0, \infty)\\
        \psi(\vec{r}, \vec{\Omega}, E) = \psi_{in}(\vec{r}_\gamma, \vec{\Omega}, E), \quad \vec{\Omega} \cdot \vec{n}_e < 0, \vec{r} \in \partial\Gamma
    \end{gathered}

where :math:`\partial\Gamma` is the domain boundary, :math:`\vec{n}_e` is the outward normal unit vector at the domain boundary, and :math:`\psi_{in}` is the known incoming flux, which may be zero. 

For more information on the derivation of the transport equation see :footcite:t:`prinja2010general` and :footcite:t:`lewis1984computational`

Electron Interaction Physics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Electron interactions with matter include:

- Elastic scattering (Coulomb interaction with nuclei)
- Inelastic scattering with atomic electrons (Coulomb)
- Radiative Bremsstrahlung
- Ionization
- Excitation

Each of these interactions are characterized by very small angular deflection and/or very small energy loss, and a small mean free path (large cross section). 
Each of these properties requires special treatement in both Monte Carlo and deterministic methods.

- The scattering cross section outgoing angular distribution is often represented in deterministic methods using Legendre polynomials, where in neutron transport only a few Legendre moments are needed in order to sufficiently represent the distribution and obtain an accurate result. 
  Electron scattering is highly anisotropic and highly forward-peaked, so the scattering angle must be specially treated to account for this.
- Energy loss through Coulomb collisions is very small, so a multigroup treatment of the energy variable in the BTE would require millions of groups in order to capture energy loss.
  This can be done with the Boltzmann Fokker-Planck equation and Generalized Fokker-Planck equation for example.
- Nearly all electron interactions are scattering (rather than absorption). A high scattering ratio leads to slow convergence of iterative methods, so acceleration techniques must be used.

.. footbibliography::