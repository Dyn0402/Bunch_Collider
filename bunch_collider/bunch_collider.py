"""
Two-bunch collision simulator.

``BunchCollider`` wraps two ``BunchDensity`` instances and orchestrates the
time-stepping loop that integrates the density product over the collision
region to produce the z-vertex distribution and luminosity.

All positions are in **micrometres (um)**, angles in **radians**, time in
**nanoseconds (ns)**, and beta* values in **centimetres (cm)**.
"""

import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor as Pool
from scipy.ndimage import gaussian_filter1d

from .bunch_density import BunchDensity


class BunchCollider:
    """
    Simulate the head-on (or crossing-angle) collision of two particle bunches.

    The simulation discretises time into ``n_points_t`` steps spanning the
    approach of the two bunches.  At each step the 3-D density product is
    evaluated on an ``(n_points_x, n_points_y, n_points_z)`` Cartesian grid,
    then integrated over x and y to give the z-vertex distribution.

    All physical parameters have sensible defaults representative of a
    typical collider experiment, but should be overridden via the setter
    methods before running a simulation.

    Parameters are set via setters (``set_bunch_*``); run ``run_sim()`` or
    ``run_sim_parallel()`` to execute, then retrieve results with
    ``get_z_density_dist()``.

    Attributes
    ----------
    bunch1, bunch2 : BunchDensity
        The two colliding bunches.
    z_shift : float
        Shift applied to the z axis when returning results (um).
    amplitude : float
        Scale factor applied to the z distribution when returning results.
    bkg : float
        Background density added to the density product (arb.).
    z_bounds : tuple of float or None
        (z_min, z_max) in um for the z-axis grid.  ``None`` uses the bunch
        positions plus ``z_lim_sigma`` sigma.
    gaus_smearing_sigma : float or None
        If set, apply a 1-D Gaussian smear to the z distribution (cm).
    gaus_z_efficiency_width : float or None
        If set, weight the z distribution by a Gaussian efficiency (cm).
    n_points_x, n_points_y, n_points_z, n_points_t : int
        Grid dimensions in space and time.
    parallel_threads : int
        Number of worker processes used by ``run_sim_parallel()``.
    """

    def __init__(self):
        self.bunch1 = BunchDensity()
        self.bunch2 = BunchDensity()

        self.bunch1_beta_original = np.array([0., 0., +1.])
        self.bunch2_beta_original = np.array([0., 0., -1.])
        self.bunch1.set_beta(*self.bunch1_beta_original)
        self.bunch2.set_beta(*self.bunch2_beta_original)

        self.bunch1.set_sigma(150., 150., 1.1e6)
        self.bunch2.set_sigma(150., 150., 1.1e6)  # um

        self.bunch1_r_original = np.array([0., 0., -6.e6])
        self.bunch2_r_original = np.array([0., 0., +6.e6])
        self.set_bunch_rs(self.bunch1_r_original, self.bunch2_r_original)

        self.z_shift = 0.      # um
        self.amplitude = 1.    # arb.

        self.bkg = 0.          # background density

        self.z_bounds = (-265. * 1e4, 265. * 1e4)   # um

        self.gaus_smearing_sigma = None
        self.gaus_z_efficiency_width = None

        self.x_lim_sigma = 10
        self.y_lim_sigma = 10
        self.z_lim_sigma = 5

        self.n_points_x = 61
        self.n_points_y = 61
        self.n_points_z = 151
        self.n_points_t = 61

        self.bunch1_longitudinal_fit_parameter_path = None
        self.bunch2_longitudinal_fit_parameter_path = None
        self.bunch1_longitudinal_fit_scaling = 1.
        self.bunch2_longitudinal_fit_scaling = 1.
        self.bunch1_longitudinal_profile_file_path = None
        self.bunch2_longitudinal_profile_file_path = None

        self.x, self.y, self.z = None, None, None
        self.average_density_product_xyz = None
        self.z_dist = None

        self.parallel_threads = os.cpu_count()

    # ------------------------------------------------------------------
    # Bunch parameter setters
    # ------------------------------------------------------------------

    def set_bunch_sigmas(self, sigma1, sigma2):
        """
        Set the transverse beam widths (um).

        Parameters
        ----------
        sigma1, sigma2 : array-like, shape (2,) or (3,)
            (sigma_x, sigma_y[, sigma_z]) for bunch 1 and 2.
        """
        self.bunch1.set_sigma(*sigma1)
        self.bunch2.set_sigma(*sigma2)

    def set_bunch_betas(self, beta1, beta2):
        """Set the dimensionless velocity vectors for both bunches."""
        self.bunch1_beta_original = np.asarray(beta1, dtype=float)
        self.bunch2_beta_original = np.asarray(beta2, dtype=float)
        self.bunch1.set_beta(*self.bunch1_beta_original)
        self.bunch2.set_beta(*self.bunch2_beta_original)

    def set_bunch_rs(self, r1, r2):
        """
        Set the initial positions of both bunch centres (um).

        Parameters
        ----------
        r1, r2 : array-like, shape (3,)
            (x, y, z) initial position of bunch 1 and 2.
        """
        self.bunch1_r_original = np.asarray(r1, dtype=float)
        self.bunch2_r_original = np.asarray(r2, dtype=float)
        self.bunch1.set_initial_z(self.bunch1_r_original[2])
        self.bunch2.set_initial_z(self.bunch2_r_original[2])
        self.bunch1.set_offsets(*self.bunch1_r_original[:2])
        self.bunch2.set_offsets(*self.bunch2_r_original[:2])

    def set_bunch_offsets(self, offset1, offset2):
        """
        Set the transverse (x, y) offsets of both bunches (um).

        Parameters
        ----------
        offset1, offset2 : array-like, shape (2,)
        """
        self.bunch1_r_original[:2] = offset1
        self.bunch2_r_original[:2] = offset2
        self.bunch1.set_offsets(*self.bunch1_r_original[:2])
        self.bunch2.set_offsets(*self.bunch2_r_original[:2])

    def set_bunch_beta_stars(self, beta_star1_x, beta_star2_x,
                             beta_star1_y=None, beta_star2_y=None):
        """
        Set the beta* (hourglass) values for both bunches (cm).

        Parameters
        ----------
        beta_star1_x, beta_star2_x : float
            Beta* in the x plane for bunch 1 and 2.
        beta_star1_y, beta_star2_y : float, optional
            Beta* in the y plane. Defaults to the x value.
        """
        self.bunch1.set_beta_star(beta_star1_x, beta_star1_y)
        self.bunch2.set_beta_star(beta_star2_x, beta_star2_y)

    def set_bunch_beta_star_shifts(self, beta_star1_shift_x, beta_star2_shift_x,
                                   beta_star1_shift_y=None, beta_star2_shift_y=None):
        """Set the longitudinal beta* shift for both bunches (cm)."""
        self.bunch1.set_beta_star_shift(beta_star1_shift_x, beta_star1_shift_y)
        self.bunch2.set_beta_star_shift(beta_star2_shift_x, beta_star2_shift_y)

    def set_bunch_crossing(self, crossing_angle1_x, crossing_angle1_y,
                           crossing_angle2_x, crossing_angle2_y):
        """
        Set the crossing half-angles for both bunches (rad).

        Parameters
        ----------
        crossing_angle1_x : float
            Bunch-1 half-angle in the x-z plane.
        crossing_angle1_y : float
            Bunch-1 half-angle in the y-z plane.
        crossing_angle2_x, crossing_angle2_y : float
            Corresponding angles for bunch 2.
        """
        self.bunch1.set_angles(crossing_angle1_x, crossing_angle1_y)
        self.bunch2.set_angles(crossing_angle2_x, crossing_angle2_y)

    def set_bunch_lengths(self, length1, length2):
        """
        Set the single-Gaussian longitudinal bunch lengths (um).

        Parameters
        ----------
        length1, length2 : float
            RMS bunch lengths (um).
        """
        self.bunch1.set_bunch_length(length1)
        self.bunch2.set_bunch_length(length2)

    def set_bunch_delays(self, delay1, delay2):
        """Set timing delays for both bunches (ns)."""
        self.bunch1.set_delay(delay1)
        self.bunch2.set_delay(delay2)

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------

    def set_z_shift(self, z_shift):
        """Set an offset applied to the z axis in output distributions (um)."""
        self.z_shift = z_shift

    def set_amplitude(self, amp):
        """Set a scale factor applied to the z distribution in output."""
        self.amplitude = amp

    def set_z_bounds(self, z_bounds):
        """
        Set the z-axis range for the simulation grid (um).

        Parameters
        ----------
        z_bounds : tuple of float
            ``(z_min, z_max)`` in um.
        """
        self.z_bounds = z_bounds

    def set_gaus_smearing_sigma(self, sigma):
        """Apply a Gaussian smear to the returned z distribution (cm)."""
        self.gaus_smearing_sigma = sigma

    def set_gaus_z_efficiency_width(self, width):
        """Weight the returned z distribution by a Gaussian efficiency (cm)."""
        self.gaus_z_efficiency_width = width

    def set_bkg(self, bkg):
        """Set the background density added to the density product."""
        self.bkg = bkg

    def set_grid_size(self, n_points_x=None, n_points_y=None,
                      n_points_z=None, n_points_t=None):
        """
        Override the number of grid points.

        Parameters
        ----------
        n_points_x, n_points_y, n_points_z, n_points_t : int, optional
            New values; ``None`` leaves the current value unchanged.
        """
        if n_points_x is not None:
            self.n_points_x = n_points_x
        if n_points_y is not None:
            self.n_points_y = n_points_y
        if n_points_z is not None:
            self.n_points_z = n_points_z
        if n_points_t is not None:
            self.n_points_t = n_points_t

    # ------------------------------------------------------------------
    # Longitudinal profile helpers
    # ------------------------------------------------------------------

    def set_longitudinal_fit_parameters_from_file(self, bunch1_path, bunch2_path):
        """Load quad-Gaussian fit parameters for both bunches from files."""
        self.bunch1_longitudinal_fit_parameter_path = bunch1_path
        self.bunch2_longitudinal_fit_parameter_path = bunch2_path
        self.bunch1.read_longitudinal_beam_profile_fit_parameters_from_file(bunch1_path)
        self.bunch2.read_longitudinal_beam_profile_fit_parameters_from_file(bunch2_path)

    def set_longitudinal_fit_scaling(self, scale1, scale2):
        """Apply a uniform scale to all longitudinal length parameters."""
        self.bunch1_longitudinal_fit_scaling = scale1
        self.bunch2_longitudinal_fit_scaling = scale2
        self.bunch1.set_longitudinal_beam_profile_scaling(scale1)
        self.bunch2.set_longitudinal_beam_profile_scaling(scale2)

    def set_longitudinal_profiles_from_file(self, bunch1_path, bunch2_path):
        """Load tabulated longitudinal profiles for both bunches from files."""
        self.bunch1_longitudinal_profile_file_path = bunch1_path
        self.bunch2_longitudinal_profile_file_path = bunch2_path
        self.bunch1.read_longitudinal_beam_profile_from_file(bunch1_path)
        self.bunch2.read_longitudinal_beam_profile_from_file(bunch2_path)

    def check_profile_normalizations(self):
        """Check that both longitudinal profiles integrate to unity."""
        total1 = self.bunch1.check_profile_normalization()
        total2 = self.bunch2.check_profile_normalization()
        print(f'Bunch 1 total density: {total1}, Bunch 2 total density: {total2}')
        if total1 is not None and not np.isclose(total1, 1.0, atol=1e-6):
            print(f'WARNING: Bunch 1 profile is not normalised: {total1}')
        if total2 is not None and not np.isclose(total2, 1.0, atol=1e-6):
            print(f'WARNING: Bunch 2 profile is not normalised: {total2}')

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def generate_grid(self):
        """Construct the spatial and temporal grids used in the simulation."""
        dt = ((self.bunch2.initial_z - self.bunch1.initial_z)
              / self.bunch1.c / self.n_points_t)
        self.bunch1.dt = self.bunch2.dt = dt

        self.x = np.linspace(
            -self.x_lim_sigma * self.bunch1.transverse_sigma[0],
             self.x_lim_sigma * self.bunch1.transverse_sigma[0],
            self.n_points_x,
        )
        self.y = np.linspace(
            -self.y_lim_sigma * self.bunch1.transverse_sigma[1],
             self.y_lim_sigma * self.bunch1.transverse_sigma[1],
            self.n_points_y,
        )
        if self.z_bounds is not None:
            self.z = np.linspace(self.z_bounds[0], self.z_bounds[1],
                                 self.n_points_z)
        else:
            z_min = min(self.bunch1_r_original[2], self.bunch2_r_original[2])
            z_max = max(self.bunch1_r_original[2], self.bunch2_r_original[2])
            sigma_z = self.bunch1.longitudinal_params['sigma1']
            self.z = np.linspace(
                z_min - self.z_lim_sigma * sigma_z,
                z_max + self.z_lim_sigma * sigma_z,
                self.n_points_z,
            )

    def run_sim(self, print_params=False):
        """
        Run the collision simulation (single-threaded, summation over time).

        Results are stored in ``average_density_product_xyz`` and ``z_dist``.

        Parameters
        ----------
        print_params : bool
            Print a parameter summary before running.
        """
        self.set_bunch_rs(self.bunch1_r_original, self.bunch2_r_original)
        self.bunch1.set_beta(*self.bunch1_beta_original)
        self.bunch2.set_beta(*self.bunch2_beta_original)
        self.bunch1.set_angles(self.bunch1.angle_x, self.bunch1.angle_y)
        self.average_density_product_xyz = None
        self.z_dist = None

        self.generate_grid()
        x_3d, y_3d, z_3d = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.bunch1.calculate_r_and_beta()
        self.bunch2.calculate_r_and_beta()
        if print_params:
            print(self)

        for _ in range(self.n_points_t):
            dp = self.bunch1.density(x_3d, y_3d, z_3d) * self.bunch2.density(x_3d, y_3d, z_3d)
            dp += self.bkg * (self.bunch1.density(x_3d, y_3d, z_3d)
                              + self.bunch2.density(x_3d, y_3d, z_3d))
            if self.average_density_product_xyz is None:
                self.average_density_product_xyz = dp
            else:
                self.average_density_product_xyz += dp
            self.bunch1.propagate()
            self.bunch2.propagate()

        self.average_density_product_xyz /= self.n_points_t
        self.z_dist = np.sum(self.average_density_product_xyz, axis=(0, 1))

    def compute_time_step(self, time_step_index):
        """
        Compute the density product for a single time step.

        Used internally by ``run_sim_parallel()``.

        Parameters
        ----------
        time_step_index : int

        Returns
        -------
        np.ndarray, shape (n_points_x, n_points_y, n_points_z)
        """
        bunch1_copy = self.bunch1.copy()
        bunch2_copy = self.bunch2.copy()
        bunch1_copy.propagate_n_steps(time_step_index)
        bunch2_copy.propagate_n_steps(time_step_index)

        x_3d, y_3d, z_3d = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        d1 = bunch1_copy.density(x_3d, y_3d, z_3d)
        d2 = bunch2_copy.density(x_3d, y_3d, z_3d)
        dp = d1 * d2 + self.bkg * (d1 + d2)
        return dp

    def run_sim_parallel(self, print_params=False):
        """
        Run the collision simulation using parallel worker processes.

        Time steps are distributed across ``parallel_threads`` workers.
        The result is integrated over time and transverse coordinates using
        the trapezoidal rule.  Results are stored in
        ``average_density_product_xyz`` and ``z_dist``.

        Parameters
        ----------
        print_params : bool
            Print a parameter summary before running.
        """
        self.set_bunch_rs(self.bunch1_r_original, self.bunch2_r_original)
        self.bunch1.set_beta(*self.bunch1_beta_original)
        self.bunch2.set_beta(*self.bunch2_beta_original)
        self.bunch1.set_angles(self.bunch1.angle_x, self.bunch1.angle_y)
        self.average_density_product_xyz = None
        self.z_dist = None

        self.generate_grid()
        self.bunch1.calculate_r_and_beta()
        self.bunch2.calculate_r_and_beta()
        if print_params:
            print(self)

        with Pool(max_workers=self.parallel_threads) as pool:
            density_products = list(pool.map(self.compute_time_step,
                                             range(self.n_points_t)))

        density_products = np.array(density_products)  # (n_t, nx, ny, nz)
        integrated_over_t = np.trapezoid(density_products, dx=self.bunch1.dt, axis=0)
        integrated_over_tx = np.trapezoid(integrated_over_t, x=self.x, axis=0)
        self.average_density_product_xyz = integrated_over_t
        self.z_dist = np.trapezoid(integrated_over_tx, x=self.y, axis=0)

    # ------------------------------------------------------------------
    # Results / diagnostics
    # ------------------------------------------------------------------

    def get_grid_info(self):
        """
        Return a summary dict of grid spacing and extent.

        Returns
        -------
        dict
        """
        return {
            'dx': self.x[1] - self.x[0],
            'dy': self.y[1] - self.y[0],
            'dz': self.z[1] - self.z[0],
            'dt': self.bunch1.dt,
            'n_points_t': self.n_points_t,
            'x_range': (self.x[0], self.x[-1]),
            'y_range': (self.y[0], self.y[-1]),
            'z_range': (self.z[0], self.z[-1]),
            'n_points_x': self.n_points_x,
            'n_points_y': self.n_points_y,
            'n_points_z': self.n_points_z,
        }

    def get_beam_sigmas(self):
        """Return the transverse sigma arrays for both bunches."""
        return self.bunch1.transverse_sigma, self.bunch2.transverse_sigma

    def get_bunch_crossing_angles(self):
        """Return the crossing angles (angle_x, angle_y) for both bunches."""
        return (self.bunch1.angle_x, self.bunch1.angle_y,
                self.bunch2.angle_x, self.bunch2.angle_y)

    def get_z_density_dist(self):
        """
        Return the z-vertex distribution after any smearing / efficiency weighting.

        Returns
        -------
        z_vals : np.ndarray
            z positions in cm.
        z_dist : np.ndarray
            Collision density at each z (arb. units).
        """
        z_vals = (self.z - self.z_shift) / 1e4   # um → cm
        z_dist = self.amplitude * self.z_dist
        if self.gaus_z_efficiency_width is not None:
            z_dist = z_dist * _gaus(z_vals, 1, 0, self.gaus_z_efficiency_width)
        if self.gaus_smearing_sigma is not None:
            z_spacing = z_vals[1] - z_vals[0]
            z_dist = gaussian_filter1d(z_dist, self.gaus_smearing_sigma / z_spacing)
        return z_vals, z_dist

    def get_x_density_dist(self):
        """Return the x-projected density distribution (sum over y and z)."""
        return self.x, np.sum(self.average_density_product_xyz, axis=(1, 2))

    def get_y_density_dist(self):
        """Return the y-projected density distribution (sum over x and z)."""
        return self.y, np.sum(self.average_density_product_xyz, axis=(0, 2))

    def get_relativistic_moller_factor(self):
        """
        Compute the relativistic Møller flux factor for the two bunches.

        Returns
        -------
        float
            Møller factor in um/ns.
        """
        v1 = self.bunch1.beta * self.bunch1.c
        v2 = self.bunch2.beta * self.bunch2.c
        return np.sqrt(
            np.linalg.norm(v1 - v2) ** 2
            - np.linalg.norm(np.cross(v1, v2)) ** 2 / self.bunch1.c ** 2
        )

    def get_naked_luminosity(self, observed=False):
        """
        Compute the per-particle luminosity (integrated density product × Møller factor).

        Parameters
        ----------
        observed : bool
            If ``True``, integrate the observable z distribution (after
            smearing / efficiency) rather than the raw one.

        Returns
        -------
        float
            Luminosity in um^-2 · ns^-1 (per particle pair per crossing).
        """
        if observed:
            zs, z_dist = self.get_z_density_dist()
            zs = zs * 1e4   # cm → um
        else:
            zs, z_dist = self.z, self.z_dist
        return np.trapezoid(z_dist, zs) * self.get_relativistic_moller_factor()

    def get_param_string(self):
        """Return a compact human-readable parameter summary."""
        return (
            f'Beta*: ({self.bunch1.beta_star_x}, {self.bunch1.beta_star_y}), '
            f'({self.bunch2.beta_star_x}, {self.bunch2.beta_star_y}) cm\n'
            f'Beam widths: {self.bunch1.transverse_sigma[0]:.1f} (x), '
            f'{self.bunch1.transverse_sigma[1]:.1f} (y) um\n'
            f'Beam lengths: {self.bunch1.get_beam_length() / 1e4:.1f}, '
            f'{self.bunch2.get_beam_length() / 1e4:.1f} cm\n'
            f'Crossing angles y: {self.bunch1.angle_y * 1e3:.2f}, '
            f'{self.bunch2.angle_y * 1e3:.2f} mrad\n'
            f'Crossing angles x: {self.bunch1.angle_x * 1e3:.2f}, '
            f'{self.bunch2.angle_x * 1e3:.2f} mrad\n'
            f'Beam offsets (x): {self.bunch1_r_original[0]:.0f}, '
            f'{self.bunch2_r_original[0]:.0f} um'
        )

    def __str__(self):
        return (
            f'BunchCollider:\n'
            f'  z_shift={self.z_shift}, amplitude={self.amplitude}\n'
            f'  grid: x={self.n_points_x}, y={self.n_points_y}, '
            f'z={self.n_points_z}, t={self.n_points_t}\n'
            f'  x_lim={self.x_lim_sigma}σ, y_lim={self.y_lim_sigma}σ, '
            f'z_lim={self.z_lim_sigma}σ\n'
            f'  bunch1 r0={self.bunch1_r_original}\n'
            f'  bunch2 r0={self.bunch2_r_original}\n'
            f'  Gaussian smearing σ: {self.gaus_smearing_sigma}\n'
            f'  Gaussian z efficiency width: {self.gaus_z_efficiency_width}\n'
            f'  Background: {self.bkg}\n'
            f'\nBunch 1:\n{self.bunch1}\n'
            f'\nBunch 2:\n{self.bunch2}\n'
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _gaus(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))
