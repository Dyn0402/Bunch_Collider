"""
Beam envelope visualisation for colliding bunches.

Shows the ±1σ transverse profiles of two colliding bunches in the x-z plane
(the "hourglass" shape from beta-function broadening) alongside the resulting
z-vertex distribution.  Three configurations are compared side-by-side:

* Offset + crossing angle
* Offset only (no crossing angle)
* Offset + opposite crossing angle

This illustrates how both the transverse separation and the beam crossing angle
affect the shape of the luminosity-weighted z-vertex distribution.

Run::

    python examples/beam_envelope.py
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

from bunch_collider import BunchCollider


def _draw_envelope(ax, z_cm, sigma_um, beta_star_cm,
                   x_offset_um=0., angle_rad=0., color='steelblue', label=None):
    """Draw the ±1σ hourglass beam envelope on *ax* (y in mm, z in cm)."""
    env = sigma_um * np.sqrt(1. + (z_cm / beta_star_cm) ** 2)
    z_um = z_cm * 1e4
    y_top =  env * np.cos(angle_rad) - z_um * np.sin(angle_rad) + x_offset_um
    y_bot = -env * np.cos(angle_rad) - z_um * np.sin(angle_rad) + x_offset_um
    # Display in mm for readability at these scales
    ax.plot(z_cm, y_top * 1e-3, color=color, lw=1.2, label=label)
    ax.plot(z_cm, y_bot * 1e-3, color=color, lw=1.2)
    ax.fill_between(z_cm, y_bot * 1e-3, y_top * 1e-3, color=color, alpha=0.15)


def main(
    beta_star=85.,         # cm
    beam_width=150.,       # um
    x_offset=900.,         # um  transverse offset of beam 1
    angle_x=0.1e-3,        # rad crossing half-angle (cols 0 and 2)
    z_range=(-250., 250.), # cm
    n_z=500,
):
    """
    Plot ±1σ beam envelopes and z-vertex distributions for three
    offset/angle configurations.
    """
    z_cm = np.linspace(*z_range, n_z)

    configs = [
        dict(x_off= x_offset, angle= angle_x,
             title='Offset + crossing angle',
             label=rf'$\theta={angle_x*1e3:.1f}$ mrad, offset $={x_offset:.0f}$ μm'),
        dict(x_off= x_offset, angle=0.,
             title='Offset only',
             label=rf'$\theta=0$, offset $={x_offset:.0f}$ μm'),
        dict(x_off=-x_offset, angle=-angle_x,
             title='Offset + opposite angle',
             label=rf'$\theta={-angle_x*1e3:.1f}$ mrad, offset $={-x_offset:.0f}$ μm'),
    ]

    fig, axs = plt.subplots(2, 3, figsize=(11, 5), sharex='all', sharey='row')

    for col, cfg in enumerate(configs):
        ax_env   = axs[0, col]
        ax_zdist = axs[1, col]

        # Beam 1: offset + angle; Beam 2: centred, no angle
        _draw_envelope(ax_env, z_cm, beam_width, beta_star,
                       x_offset_um=cfg['x_off'], angle_rad=cfg['angle'],
                       color='steelblue', label='Beam 1')
        _draw_envelope(ax_env, z_cm, beam_width, beta_star,
                       x_offset_um=0., angle_rad=0.,
                       color='goldenrod', label='Beam 2')

        ax_env.axhline(0, color='gray', lw=0.6, alpha=0.5)
        ax_env.axvline(0, color='gray', lw=0.6, alpha=0.5)
        ax_env.set_title(cfg['title'])
        ax_env.annotate(cfg['label'],
                        xy=(0.5, 0.97), xycoords='axes fraction',
                        ha='center', va='top', fontsize=7.5,
                        bbox=dict(facecolor='white', edgecolor='gray',
                                  boxstyle='round,pad=0.3', alpha=0.8))
        if col == 0:
            ax_env.set_ylabel('x (mm)')
            ax_env.legend(loc='lower left', fontsize=7)

        # Simulation
        sim = BunchCollider()
        sim.set_bunch_beta_stars(beta_star, beta_star)
        sim.set_bunch_offsets(np.array([cfg['x_off'], 0.]),
                              np.array([0.,           0.]))
        sim.set_bunch_sigmas(np.array([beam_width, beam_width]),
                             np.array([beam_width, beam_width]))
        sim.set_bunch_crossing(cfg['angle'], 0., 0., 0.)
        print(f"  Running simulation {col + 1}/3 ...")
        sim.run_sim_parallel()
        z_vals, z_dist = sim.get_z_density_dist()

        ax_zdist.plot(z_vals, z_dist, color='black')
        ax_zdist.set_ylim(bottom=0)
        ax_zdist.set_xlim(*z_range)
        ax_zdist.axhline(0, color='gray', lw=0.6, alpha=0.5)
        ax_zdist.set_xlabel('z (cm)')
        if col == 0:
            ax_zdist.set_ylabel('Luminosity density (arb.)')

    # Suppress redundant y-tick labels on shared columns
    for col in (1, 2):
        axs[0, col].tick_params(labelleft=False)
        axs[1, col].tick_params(labelleft=False)

    fig.suptitle('Beam Envelopes and Z-Vertex Distributions', y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.06, hspace=0.1)
    if _BACKEND == 'Agg':
        out = "beam_envelope.png"
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"No interactive display found — figure saved to {out}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
