"""
Basic head-on collision: z-vertex distribution.

Demonstrates the minimal workflow to simulate a head-on bunch collision and
plot the resulting z-vertex distribution fitted with a Gaussian.

Produces a single figure with three panels:

* **Top-left** — x-z beam envelopes: both beams are head-on and centred at
  x = 0, so they fully overlap.  The hourglass (beta-function) broadening
  away from the IP is clearly visible.
* **Top-right** — y-z beam envelopes: identical to x-z for a round beam with
  no crossing angle.
* **Bottom** — z-vertex distribution with Gaussian fit, highlighting the
  non-Gaussian tail introduced by the hourglass effect.

Run::

    python examples/basic_collision.py
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
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit

from bunch_collider import BunchCollider


def gaus(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))


def _draw_envelope(ax, z_cm, sigma_um, beta_star_cm,
                   x_offset_um=0., color='steelblue', label=None):
    """Draw the ±1σ hourglass beam envelope on *ax* (y in μm, z in cm)."""
    env = sigma_um * np.sqrt(1. + (z_cm / beta_star_cm) ** 2)
    ax.plot(z_cm,  env + x_offset_um, color=color, lw=1.2, label=label)
    ax.plot(z_cm, -env + x_offset_um, color=color, lw=1.2)
    ax.fill_between(z_cm, -env + x_offset_um, env + x_offset_um,
                    color=color, alpha=0.15)


def main():
    # --- Collision parameters ---
    beam_width  = 170     # um  (transverse RMS)
    beam_length = 1.1e6   # um  (longitudinal RMS)
    beta_star   = 85      # cm  (hourglass parameter)
    z_initial   = 6.0e6   # um  (initial |z| of each bunch)

    # --- Set up the collider ---
    sim = BunchCollider()
    sim.set_bunch_rs(np.array([0., 0., -z_initial]),
                     np.array([0., 0.,  z_initial]))
    sim.set_bunch_sigmas(np.array([beam_width, beam_width]),
                         np.array([beam_width, beam_width]))
    sim.set_bunch_lengths(beam_length, beam_length)
    sim.set_bunch_beta_stars(beta_star, beta_star)

    # --- Run ---
    print("Running simulation...")
    sim.run_sim_parallel()

    zs, z_dist = sim.get_z_density_dist()   # z in cm

    # --- Gaussian fit ---
    fit_mask = np.abs(zs) < 80
    p0 = [np.max(z_dist[fit_mask]), 0., 20.]
    popt, pcov = curve_fit(gaus, zs[fit_mask], z_dist[fit_mask], p0=p0)
    perr = np.sqrt(np.diag(pcov))

    # --- Quantify non-Gaussianity ---
    gaus_vals = gaus(zs, *popt)
    peak = np.max(z_dist)
    max_dev_pct = np.max(np.abs(z_dist - gaus_vals)) / peak * 100

    # curve_fit uncertainties are meaningless for a noiseless simulation curve;
    # format them as "N/A" if covariance estimation failed
    def fmt_err(v, e, fmt):
        return f'{v:{fmt}} ± {e:{fmt}}' if np.isfinite(e) else f'{v:{fmt}}'

    print(f"Fit:  A = {fmt_err(popt[0], perr[0], '.3e')}")
    print(f"      μ = {fmt_err(popt[1], perr[1], '.2f')} cm")
    print(f"      σ = {fmt_err(popt[2], perr[2], '.2f')} cm")
    print(f"Max deviation from Gaussian fit: {max_dev_pct:.1f}% of peak")

    # --- Figure ---
    z_env = np.linspace(zs[0], zs[-1], 400)   # cm

    fig = plt.figure(figsize=(11, 7))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)
    ax_xz   = fig.add_subplot(gs[0, 0])
    ax_yz   = fig.add_subplot(gs[0, 1])
    ax      = fig.add_subplot(gs[1, :])

    # --- Beam envelope panels ---
    for ax_env, transverse_label, title in [
        (ax_xz, 'x (μm)', 'Beam Envelopes — x-z'),
        (ax_yz, 'y (μm)', 'Beam Envelopes — y-z'),
    ]:
        _draw_envelope(ax_env, z_env, beam_width, beta_star,
                       color='steelblue', label='Beam 1 & 2 (head-on)')
        ax_env.axhline(0, color='gray', lw=0.6, alpha=0.4)
        ax_env.axvline(0, color='gray', lw=0.6, alpha=0.4)
        ax_env.set_xlabel('z (cm)')
        ax_env.set_ylabel(transverse_label)
        ax_env.set_title(title)
        ax_env.legend(fontsize=8)
        ax_env.grid(alpha=0.25)

    # sigma annotation on x-z panel
    sigma0 = beam_width
    sigma_beta = beam_width * np.sqrt(2)
    ax_xz.annotate('', xy=(beta_star, sigma_beta), xytext=(0, sigma0),
                   arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
    ax_xz.text(beta_star * 0.5, (sigma0 + sigma_beta) / 2,
               r'$\beta^*$', ha='center', va='bottom', fontsize=8)

    # --- z-distribution panel ---
    ax.plot(zs, z_dist, 'b-', lw=1.5, label='Simulation')
    ax.plot(zs, gaus_vals, 'r--', lw=1.5,
            label=rf'Gaussian fit  σ = {popt[2]:.1f} cm')

    # Beam parameters box (left)
    ax.annotate(
        'Beam parameters\n'
        f'  Transverse σ = {beam_width} μm\n'
        f'  Bunch length = {beam_length / 1e4:.0f} cm\n'
        f'  β* = {beta_star} cm\n'
        '  Head-on, no crossing angle',
        xy=(0.02, 0.97), xycoords='axes fraction',
        ha='left', va='top', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4),
    )

    # Fit results box (right)
    ax.annotate(
        'Gaussian fit\n'
        f'  A = {fmt_err(popt[0], perr[0], ".2e")}\n'
        f'  μ = {fmt_err(popt[1], perr[1], ".2f")} cm\n'
        f'  σ = {fmt_err(popt[2], perr[2], ".2f")} cm\n'
        f'  Max deviation: {max_dev_pct:.1f}% of peak\n',
        xy=(0.98, 0.97), xycoords='axes fraction',
        ha='right', va='top', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )
    ax.annotate('Distribution is not exactly Gaussian\n'
                '  due to the hourglass effect!',
                xy=(0.02, 0.7), xycoords='axes fraction',
                ha='left', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=1.0),
    )

    ax.set_xlabel('z (cm)')
    ax.set_ylabel('Collision density (arb.)')
    ax.set_title('Z-Vertex Distribution — Head-On Collision')
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if _BACKEND == 'Agg':
        out = "basic_collision.png"
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"No interactive display found — figure saved to {out}")
        print("Install python3.12-tk (sudo apt install python3.12-tk) for interactive plots.")
    else:
        plt.show()


if __name__ == '__main__':
    main()
