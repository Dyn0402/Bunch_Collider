"""
bunch_collider: Particle bunch collision simulation library.

Provides tools for simulating the collision of relativistic particle bunches,
including beam-beam effects such as the hourglass effect (beta* broadening),
crossing angles, transverse offsets, and arbitrary longitudinal bunch profiles.

The C++ extension (compiled via pybind11) is used internally for fast density
calculations over 3D grids.

Main classes
------------
BunchDensity
    Represents a single particle bunch with a 3D density distribution.
BunchCollider
    Manages two bunches and runs the full collision simulation.

Example
-------
>>> import numpy as np
>>> from bunch_collider import BunchCollider
>>>
>>> sim = BunchCollider()
>>> sim.set_bunch_rs(np.array([0., 0., -6e6]), np.array([0., 0., 6e6]))
>>> sim.set_bunch_beta_stars(85, 85)       # cm
>>> sim.set_bunch_sigmas(np.array([170., 170.]), np.array([170., 170.]))  # um
>>> sim.run_sim_parallel()
>>> zs, z_dist = sim.get_z_density_dist()
"""

from .bunch_density import BunchDensity
from .bunch_collider import BunchCollider
from .measure import Measure

__version__ = "0.1.0"
__all__ = ["BunchDensity", "BunchCollider", "Measure"]
