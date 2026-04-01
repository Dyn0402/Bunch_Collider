"""
Vernier scan simulation.

Sweeps the transverse separation between the two bunches and records the
integrated collision rate at each step, mimicking the real Vernier scan
technique used to measure absolute luminosity at colliders.

The integrated rate as a function of offset follows a Gaussian whose width
is related to the transverse beam size.

Run::

    python examples/vernier_scan.py
"""

import numpy as np
import matplotlib
import importlib

def _pick_backend():
    for name in ('TkAgg', 'Qt5Agg', 'Qt6Agg', 'WxAgg'):
        try:
            matplotlib.use(name)
            importlib.import_module(f'matplotlib.backends.backend_{name.lower()}')
            return name
        except Exception:
            continue
    matplotlib.use('Agg')
    return 'Agg'

_BACKEND = _pick_backend()
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from bunch_collider import BunchCollider


def vernier_fit(x, a, sigma, x0):
    """Expected Vernier scan shape (Gaussian in the beam separation)."""
    return a * np.exp(-(x - x0) ** 2 / (4 * sigma ** 2))


def main():
    # --- Scan parameters ---
    x_offsets = np.linspace(-1000., 1000., 9)   # um
    beam_width = 135.    # um  (transverse RMS)
    beam_length = 1.2e6  # um  (longitudinal RMS)
    beta_star = 85       # cm
    z_initial = 6.0e6   # um

    integrated_rates = []

    for i, x_offset in enumerate(x_offsets):
        print(f"[{i + 1}/{len(x_offsets)}] x offset = {x_offset:.0f} μm")

        sim = BunchCollider()
        sim.set_bunch_rs(
            np.array([x_offset, 0., -z_initial]),
            np.array([0.,       0.,  z_initial]),
        )
        sim.set_bunch_sigmas(
            np.array([beam_width, beam_width]),
            np.array([beam_width, beam_width]),
        )
        sim.set_bunch_lengths(beam_length, beam_length)
        sim.set_bunch_beta_stars(beta_star, beta_star)
        sim.run_sim_parallel()

        zs, z_dist = sim.get_z_density_dist()
        integrated_rates.append(np.trapezoid(z_dist, zs))

    integrated_rates = np.array(integrated_rates)

    # --- Fit ---
    p0 = [integrated_rates.max(), beam_width, 0.]
    popt, pcov = curve_fit(vernier_fit, x_offsets, integrated_rates, p0=p0)
    popt[1] = abs(popt[1])   # sigma is squared in fit; ensure positive
    perr = np.sqrt(np.diag(pcov))

    print(f"\nFit results:")
    print(f"  A     = {popt[0]:.3e} ± {perr[0]:.1e}")
    print(f"  σ     = {popt[1]:.1f} ± {perr[1]:.1f} μm")
    print(f"  x₀    = {popt[2]:.1f} ± {perr[2]:.1f} μm")

    # --- Plot ---
    x_fine = np.linspace(x_offsets.min(), x_offsets.max(), 300)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x_offsets, integrated_rates, color='steelblue', zorder=5,
               label='Simulation')
    ax.plot(x_fine, vernier_fit(x_fine, *popt), 'r-', lw=1.5,
            label='Gaussian fit')
    ax.annotate(
        f'A = {popt[0]:.2e} ± {perr[0]:.1e}\n'
        f'σ = {popt[1]:.1f} ± {perr[1]:.1f} μm\n'
        f'x₀ = {popt[2]:.1f} ± {perr[2]:.1f} μm',
        xy=(0.03, 0.82), xycoords='axes fraction',
        bbox=dict(facecolor='wheat', alpha=0.5),
    )
    ax.set_xlabel('x Offset (μm)')
    ax.set_ylabel('Integrated Collision Rate (arb.)')
    ax.set_title('Vernier Scan Simulation')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if _BACKEND == 'Agg':
        out = "vernier_scan.png"
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"No interactive display found — figure saved to {out}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
