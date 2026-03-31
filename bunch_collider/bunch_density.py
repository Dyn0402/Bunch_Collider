"""
Single-bunch density model for relativistic particle bunches.

The bunch is represented as a 3D distribution: a transverse (x, y) Gaussian
whose width can grow with distance from the interaction point (hourglass /
beta-function broadening), multiplied by an arbitrary longitudinal (z) profile
described either by a sum of up to four Gaussians or by a tabulated PDF.

All positions are in **micrometres (um)**, angles in **radians**, time in
**nanoseconds (ns)**, and beta* values in **centimetres (cm)**.
"""

import numpy as np
from scipy.optimize import curve_fit as cf
import copy

from . import _bunch_density_cpp as bdcpp


class BunchDensity:
    """
    3D density distribution for a single relativistic particle bunch.

    The bunch travels at velocity ``beta * c`` and is initially placed at
    ``initial_z`` along the beam axis with optional transverse offsets.
    Crossing angles rotate the bunch direction in the x-z and y-z planes.
    Beta-function (hourglass) broadening widens the transverse profile away
    from the interaction point.

    The longitudinal profile is a normalised mixture of up to four Gaussians
    (parameters stored in ``longitudinal_params``), or a tabulated PDF loaded
    from file.

    Parameters are set via setter methods; call ``density()`` to evaluate the
    3D density on a grid.

    Attributes
    ----------
    c : float
        Speed of light in um/ns.
    transverse_sigma : np.ndarray, shape (2,)
        RMS beam width at the interaction point in x and y (um).
    beta : np.ndarray, shape (3,)
        Dimensionless velocity vector (v/c).
    r : np.ndarray, shape (3,)
        Current bunch centre position (um).
    t : float
        Current time (ns).
    dt : float
        Propagation timestep (ns).
    angle_x : float
        Crossing half-angle in the x-z plane (rad).
    angle_y : float
        Crossing half-angle in the y-z plane (rad).
    beta_star_x, beta_star_y : float or None
        Beta-function value at the IP in x and y (cm). ``None`` disables
        hourglass broadening.
    beta_star_shift_x, beta_star_shift_y : float
        Longitudinal shift of the beta* minimum from z = 0 (cm).
    delay : float
        Timing delay applied to the bunch's initial position (ns).
    longitudinal_params : dict
        Parameters for the quad-Gaussian longitudinal profile.
    longitudinal_width_scaling : float
        Uniform scale factor applied to all longitudinal length parameters.
    """

    c = 299792458. * 1e6 / 1e9  # um/ns

    def __init__(self):
        self.transverse_sigma = np.array([0., 0.], dtype=np.float64)  # um
        self.beta = np.array([0., 0., 1.], dtype=np.float64)          # v/c
        self.r = np.array([0., 0., 0.], dtype=np.float64)             # um
        self.t = 0.    # ns
        self.dt = 0.   # ns

        self.angle_x = 0.   # rad (rotation in x-z plane)
        self.angle_y = 0.   # rad (rotation in y-z plane)

        self.beta_star_x = None   # cm
        self.beta_star_y = None   # cm
        self.beta_star_shift_x = 0.   # cm
        self.beta_star_shift_y = 0.   # cm

        self.delay = 0.   # ns

        self.longitudinal_params = {
            'mu1': 0., 'sigma1': 1.,
            'a2': 0., 'mu2': 0., 'sigma2': 1.,
            'a3': 0., 'mu3': 0., 'sigma3': 1.,
            'a4': 0., 'mu4': 0., 'sigma4': 1.,
        }
        self.effective_longitudinal_params = self.longitudinal_params.copy()

        self.longitudinal_profile_zs = None         # tabulated z positions (um)
        self.longitudinal_profile_densities = None  # tabulated densities (um^-1)
        self.longitudinal_width_scaling = 1.

        self.initial_z = 0.   # um
        self.offset_x = 0.   # um
        self.offset_y = 0.   # um

        self.reset = True   # recalculate r and beta before next density call

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_initial_z(self, z):
        """Set the initial z position of the bunch centre (um)."""
        self.initial_z = z
        self.reset = True

    def set_offsets(self, x_offset, y_offset):
        """Set the transverse offsets of the bunch centre (um)."""
        self.offset_x = x_offset
        self.offset_y = y_offset
        self.reset = True

    def set_beta(self, x, y, z):
        """Set the dimensionless velocity vector (v/c)."""
        self.beta = np.array([x, y, z], dtype=np.float64)
        self.reset = True

    def set_beta_star(self, beta_star_x, beta_star_y=None):
        """
        Set the beta* (hourglass) parameter (cm).

        Parameters
        ----------
        beta_star_x : float
            Beta* in the x plane (cm).
        beta_star_y : float, optional
            Beta* in the y plane (cm). Defaults to ``beta_star_x``.
        """
        self.beta_star_x = beta_star_x
        self.beta_star_y = beta_star_x if beta_star_y is None else beta_star_y
        self.reset = True

    def set_beta_star_shift(self, beta_star_shift_x, beta_star_shift_y=None):
        """
        Set the longitudinal shift of the beta* minimum from z = 0 (cm).

        Parameters
        ----------
        beta_star_shift_x : float
        beta_star_shift_y : float, optional
            Defaults to ``beta_star_shift_x``.
        """
        self.beta_star_shift_x = beta_star_shift_x
        self.beta_star_shift_y = (beta_star_shift_x if beta_star_shift_y is None
                                  else beta_star_shift_y)
        self.reset = True

    def set_sigma(self, x, y, z=None):
        """
        Set the transverse (and optionally longitudinal) beam width (um).

        Parameters
        ----------
        x, y : float
            RMS transverse widths (um).
        z : float, optional
            RMS longitudinal width (um). Sets the ``sigma1`` Gaussian parameter.
        """
        self.transverse_sigma = np.array([x, y], dtype=np.float64)
        if z is not None:
            self.longitudinal_params['sigma1'] = z
            self.effective_longitudinal_params['sigma1'] = z
        self.reset = True

    def set_bunch_length(self, length):
        """
        Set the single-Gaussian longitudinal bunch length (um).

        Parameters
        ----------
        length : float
            RMS bunch length (um).
        """
        self.longitudinal_params['sigma1'] = length
        self.effective_longitudinal_params['sigma1'] = length
        self.reset = True

    def set_angles(self, angle_x, angle_y):
        """
        Set the bunch crossing half-angles (rad).

        Parameters
        ----------
        angle_x : float
            Half-angle in the x-z plane (rad).
        angle_y : float
            Half-angle in the y-z plane (rad).
        """
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.reset = True

    def set_delay(self, delay):
        """Set the timing delay for the bunch (ns)."""
        self.delay = delay
        self.reset = True

    # ------------------------------------------------------------------
    # Longitudinal profile
    # ------------------------------------------------------------------

    def read_longitudinal_beam_profile_fit_parameters_from_file(self, fit_out_path):
        """
        Read quad-Gaussian fit parameters from a text file and apply them.

        Parameters
        ----------
        fit_out_path : str
            Path to the fit parameter file.
        """
        fit_params = read_longitudinal_beam_profile_fit_parameters(fit_out_path)
        self.longitudinal_params = fit_params
        self.effective_longitudinal_params = fit_params.copy()
        self.reset = True

    def read_longitudinal_beam_profile_from_file(self, profile_path):
        """
        Load a tabulated longitudinal density profile (z, density) from file.

        The file must have a single header row followed by two columns:
        z positions (um) and density values (um^-1, normalised to unit integral).

        Parameters
        ----------
        profile_path : str or None
            Path to the profile file. Pass ``None`` to clear any loaded profile.
        """
        if profile_path is None:
            print("No longitudinal profile file provided. Using default parameters.")
            self.longitudinal_profile_zs = None
            self.longitudinal_profile_densities = None
        else:
            data = np.loadtxt(profile_path, skiprows=1)
            self.longitudinal_profile_zs = data[:, 0]
            self.longitudinal_profile_densities = data[:, 1]
        self.reset = True

    def set_longitudinal_beam_profile_scaling(self, scaling):
        """
        Scale all longitudinal length parameters by a uniform factor.

        Parameters
        ----------
        scaling : float
            Scale factor applied to all mu and sigma Gaussian parameters.
        """
        self.longitudinal_width_scaling = scaling
        for key in self.longitudinal_params:
            if 'sigma' in key or 'mu' in key:
                self.effective_longitudinal_params[key] = (
                    self.longitudinal_params[key] * scaling
                )
        self.reset = True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_beam_length(self):
        """
        Estimate the effective single-Gaussian RMS length of the bunch (um).

        Fits a single Gaussian to the ``quad_gaus_pdf`` longitudinal profile
        and returns twice the fitted sigma.

        Returns
        -------
        float
            Effective RMS bunch length (um).
        """
        p = self.effective_longitudinal_params
        x = np.linspace(-abs(self.initial_z), abs(self.initial_z), 1000)
        y = quad_gaus_pdf(x,
                          p['mu1'], p['sigma1'],
                          p['a2'],  p['mu2'],  p['sigma2'],
                          p['a3'],  p['mu3'],  p['sigma3'],
                          p['a4'],  p['mu4'],  p['sigma4'])
        popt, _ = cf(gaus_pdf, x, y, p0=[p['mu1'], p['sigma1']])
        return 2 * popt[1]

    def check_profile_normalization(self):
        """
        Verify that the tabulated longitudinal profile integrates to unity.

        Returns
        -------
        float or None
            Integral of the profile, or ``None`` if no profile is loaded.
        """
        if (self.longitudinal_profile_densities is not None
                and self.longitudinal_profile_zs is not None):
            total = np.trapezoid(self.longitudinal_profile_densities,
                                 self.longitudinal_profile_zs)
            print(f'Total density from profile: {total:.6f}, '
                  f'is normalised: {np.isclose(total, 1.0, atol=1e-6)}')
            return total
        else:
            print("Longitudinal profile not set. Cannot check normalisation.")
            return None

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------

    def calculate_r_and_beta(self):
        """
        Compute ``r`` and ``beta`` from the initial-z, offsets, angles, and delay.

        Called automatically before the first density evaluation (or whenever
        ``reset`` is True).
        """
        delayed_z = self.initial_z + np.sign(self.initial_z) * self.delay * self.c
        r_rotated = delayed_z * np.array(
            [np.sin(self.angle_x), np.sin(self.angle_y), 1.], dtype=np.float64
        )
        self.r = r_rotated + np.array([self.offset_x, self.offset_y, 0.],
                                       dtype=np.float64)
        self.beta = -r_rotated / np.linalg.norm(r_rotated)
        self.t = 0.
        self.reset = False

    def propagate(self):
        """Advance the bunch by one timestep ``dt``."""
        self.r += self.beta * self.c * self.dt
        self.t += self.dt

    def propagate_n_steps(self, n):
        """
        Advance the bunch by ``n`` timesteps.

        Parameters
        ----------
        n : int
            Number of steps to advance.
        """
        self.r += self.beta * self.c * self.dt * n
        self.t += self.dt * n

    def copy(self):
        """Return a deep copy of this bunch."""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Density evaluation
    # ------------------------------------------------------------------

    def density(self, x, y, z):
        """
        Evaluate the 3D density on a grid.

        Uses the tabulated profile (if loaded) or the Gaussian mixture model.
        Delegates to the C++ extension for performance.

        Parameters
        ----------
        x, y, z : np.ndarray, shape (nx, ny, nz)
            Grid coordinates from ``np.meshgrid(..., indexing='ij')``.

        Returns
        -------
        np.ndarray, shape (nx, ny, nz)
            Density values (um^-3).
        """
        if self.reset:
            self.calculate_r_and_beta()

        beta_star_x = self.beta_star_x if self.beta_star_x is not None else 0
        beta_star_y = self.beta_star_y if self.beta_star_y is not None else 0

        if (self.longitudinal_profile_densities is not None
                and self.longitudinal_profile_zs is not None):
            return bdcpp.density_interpolated_pdf(
                x, y, z,
                self.r[0], self.r[1], self.r[2],
                self.transverse_sigma[0], self.transverse_sigma[1],
                self.angle_x, self.angle_y,
                beta_star_x, beta_star_y,
                self.beta_star_shift_x, self.beta_star_shift_y,
                self.longitudinal_profile_zs, self.longitudinal_profile_densities,
            )

        gaussians = extract_gaussian_list(self.effective_longitudinal_params)
        return bdcpp.density_n_gaussians(
            x, y, z,
            self.r[0], self.r[1], self.r[2],
            self.transverse_sigma[0], self.transverse_sigma[1],
            self.angle_x, self.angle_y,
            beta_star_x, beta_star_y,
            self.beta_star_shift_x, self.beta_star_shift_y,
            gaussians,
        )

    def density_arbitrary(self, x, y, z):
        """
        Evaluate the density using the Gaussian mixture longitudinal model.

        Unlike ``density()``, this always uses the Gaussian model even when a
        tabulated profile is loaded.

        Parameters
        ----------
        x, y, z : np.ndarray
            Grid coordinates.

        Returns
        -------
        np.ndarray
            Density values (um^-3).
        """
        if self.reset:
            self.calculate_r_and_beta()

        beta_star_x = self.beta_star_x if self.beta_star_x is not None else 0
        beta_star_y = self.beta_star_y if self.beta_star_y is not None else 0
        gaussians = extract_gaussian_list(self.effective_longitudinal_params)
        return bdcpp.density_n_gaussians(
            x, y, z,
            self.r[0], self.r[1], self.r[2],
            self.transverse_sigma[0], self.transverse_sigma[1],
            self.angle_x, self.angle_y,
            beta_star_x, beta_star_y,
            self.beta_star_shift_x, self.beta_star_shift_y,
            gaussians,
        )

    def density_interpolate(self, x, y, z):
        """
        Evaluate the density using the tabulated longitudinal profile.

        Requires that ``read_longitudinal_beam_profile_from_file`` has been
        called beforehand.

        Parameters
        ----------
        x, y, z : np.ndarray
            Grid coordinates.

        Returns
        -------
        np.ndarray
            Density values (um^-3).
        """
        if self.reset:
            self.calculate_r_and_beta()

        beta_star_x = self.beta_star_x if self.beta_star_x is not None else 0
        beta_star_y = self.beta_star_y if self.beta_star_y is not None else 0
        return bdcpp.density_interpolated_pdf(
            x, y, z,
            self.r[0], self.r[1], self.r[2],
            self.transverse_sigma[0], self.transverse_sigma[1],
            self.angle_x, self.angle_y,
            beta_star_x, beta_star_y,
            self.beta_star_shift_x, self.beta_star_shift_y,
            self.longitudinal_profile_zs, self.longitudinal_profile_densities,
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __str__(self):
        return (
            f'Bunch Parameters:\n'
            f'  Initial Z:            {self.initial_z:.0f} um\n'
            f'  Offsets (x, y):       {self.offset_x:.4f}, {self.offset_y:.4f} um\n'
            f'  Beta (v/c):           {self.beta}\n'
            f'  Transverse sigma:     {self.transverse_sigma} um\n'
            f'  Crossing angles:      ({self.angle_x * 1e3:.4f}, {self.angle_y * 1e3:.4f}) mrad\n'
            f'  Position:             {self.r} um\n'
            f'  Time:                 {self.t} ns\n'
            f'  Timestep:             {self.dt} ns\n'
            f'  Delay:                {self.delay} ns\n'
            f'  Beta*:                ({self.beta_star_x}, {self.beta_star_y}) cm\n'
            f'  Beta* shift:          ({self.beta_star_shift_x}, {self.beta_star_shift_y}) cm\n'
            f'  Long. params:         {self.longitudinal_params}\n'
            f'  Effective long.:      {self.effective_longitudinal_params}\n'
            f'  Long. width scaling:  {self.longitudinal_width_scaling}\n'
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def read_longitudinal_beam_profile_fit_parameters(fit_out_path):
    """
    Parse a quad-Gaussian fit parameter file.

    Parameters
    ----------
    fit_out_path : str
        Path to the parameter file.  Lines 0-2 are skipped (header); each
        subsequent line has the form ``param_name: value``.

    Returns
    -------
    dict
        Parameter dictionary compatible with ``BunchDensity.longitudinal_params``.
    """
    with open(fit_out_path, 'r') as f:
        lines = f.readlines()
    return {
        param: float(val)
        for line in lines[3:]
        for param, val in [line.strip().split(': ')]
    }


def gaus_pdf(x, b, c):
    """Normalised 1-D Gaussian PDF with mean ``b`` and std ``c``."""
    return np.exp(-(x - b) ** 2 / (2 * c ** 2)) / (c * np.sqrt(2 * np.pi))


def quad_gaus_pdf(x, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4):
    """
    Normalised mixture of up to four Gaussians.

    The first component has weight 1; subsequent components have relative
    weights ``a2``, ``a3``, ``a4``.
    """
    return (
        gaus_pdf(x, b1, c1)
        + a2 * gaus_pdf(x, b2, c2)
        + a3 * gaus_pdf(x, b3, c3)
        + a4 * gaus_pdf(x, b4, c4)
    ) / (1 + a2 + a3 + a4)


def extract_gaussian_list(longitudinal_params):
    """
    Convert a flat longitudinal-parameter dict into a list of ``[a, mu, sigma]``
    triples expected by the C++ density functions.

    The first Gaussian always has weight ``a = 1.0``; additional Gaussians are
    included if their ``a{i}`` key is present in the dict.

    Parameters
    ----------
    longitudinal_params : dict

    Returns
    -------
    list of [float, float, float]
    """
    gaussians = [[1.0, longitudinal_params['mu1'], longitudinal_params['sigma1']]]
    i = 2
    while f'a{i}' in longitudinal_params:
        gaussians.append([
            longitudinal_params[f'a{i}'],
            longitudinal_params[f'mu{i}'],
            longitudinal_params[f'sigma{i}'],
        ])
        i += 1
    return gaussians
