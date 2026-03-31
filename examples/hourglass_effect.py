"""
Hourglass (beta*) effect on the z-vertex distribution.

Produces a single figure with three panels:

* **Top-left** — x-z beam envelopes: shows how the beta-function broadens
  each bunch away from the IP, and how the two beams partially overlap with
  a transverse offset.
* **Top-right** — y-z beam envelopes: beams are centred at y = 0 so they
  fully overlap in this projection, but the hourglass broadening is still
  visible.
* **Bottom** — z-vertex distribution: compares the collision density with and
  without hourglass broadening.  A transverse x-offset is used to make the
  hourglass effect clearly visible.

Run::

    python examples/hourglass_effect.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from bunch_collider import BunchCollider


def _draw_envelope(ax, z_cm, sigma_um, beta_star_cm,
                   x_offset_um=0., color='steelblue', label=None,
                   linestyle='-'):
    """Draw the ±1σ hourglass beam envelope on *ax* (y in μm, z in cm)."""
    env = sigma_um * np.sqrt(1. + (z_cm / beta_star_cm) ** 2)
    ax.plot(z_cm,  env + x_offset_um, color=color, lw=1.2, ls=linestyle, label=label)
    ax.plot(z_cm, -env + x_offset_um, color=color, lw=1.2, ls=linestyle)
    ax.fill_between(z_cm, -env + x_offset_um, env + x_offset_um,
                    color=color, alpha=0.12)


def main(
    beam_width=170.,     # um
    beam_length=1.1e6,   # um
    beta_star=85.,       # cm
    x_offset=700.,       # um  (peripheral collision makes the effect clearer)
    z_initial=6.0e6,     # um
):
    # -----------------------------------------------------------------------
    # Simulations
    # -----------------------------------------------------------------------
    def make_sim(with_hourglass):
        sim = BunchCollider()
        sim.set_bunch_rs(np.array([x_offset, 0., -z_initial]),
                         np.array([0.,        0.,  z_initial]))
        sim.set_bunch_sigmas(np.array([beam_width, beam_width]),
                             np.array([beam_width, beam_width]))
        sim.set_bunch_lengths(beam_length, beam_length)
        if with_hourglass:
            sim.set_bunch_beta_stars(beta_star, beta_star)
        return sim

    print("Running simulation without hourglass...")
    sim_off = make_sim(False)
    sim_off.run_sim_parallel()
    zs_off, z_dist_off = sim_off.get_z_density_dist()

    print("Running simulation with hourglass...")
    sim_on = make_sim(True)
    sim_on.run_sim_parallel()
    zs_on, z_dist_on = sim_on.get_z_density_dist()

    integral_off = np.trapezoid(z_dist_off, zs_off)
    integral_on  = np.trapezoid(z_dist_on,  zs_on)
    effect_pct = (1. - integral_on / integral_off) * 100.
    direction = "fewer" if effect_pct > 0 else "more"
    print(f"Hourglass gives {abs(effect_pct):.1f}% {direction} collisions")

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    z_env = np.linspace(zs_on[0], zs_on[-1], 400)   # cm, same range as simulation

    fig = plt.figure(figsize=(11, 7))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)
    ax_xz   = fig.add_subplot(gs[0, 0])
    ax_yz   = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[1, :])

    # --- x-z envelope panel ---
    # Beam 1 (offset in x)
    _draw_envelope(ax_xz, z_env, beam_width, beta_star,
                   x_offset_um=x_offset, color='steelblue', label=f'Beam 1 (x = {x_offset:.0f} μm)')
    # Beam 2 (centred)
    _draw_envelope(ax_xz, z_env, beam_width, beta_star,
                   x_offset_um=0., color='goldenrod', label='Beam 2 (x = 0)')
    ax_xz.axhline(0, color='gray', lw=0.6, alpha=0.5)
    ax_xz.axvline(0, color='gray', lw=0.6, alpha=0.5)
    ax_xz.set_xlabel('z (cm)')
    ax_xz.set_ylabel('x (μm)')
    ax_xz.set_title('Beam Envelopes — x-z')
    ax_xz.legend(fontsize=8)
    ax_xz.grid(alpha=0.25)

    # --- y-z envelope panel ---
    # Both beams centred at y = 0
    _draw_envelope(ax_yz, z_env, beam_width, beta_star,
                   x_offset_um=0., color='steelblue', label='Beam 1')
    _draw_envelope(ax_yz, z_env, beam_width, beta_star,
                   x_offset_um=0., color='goldenrod', label='Beam 2')
    ax_yz.axhline(0, color='gray', lw=0.6, alpha=0.5)
    ax_yz.axvline(0, color='gray', lw=0.6, alpha=0.5)
    ax_yz.set_xlabel('z (cm)')
    ax_yz.set_ylabel('y (μm)')
    ax_yz.set_title('Beam Envelopes — y-z')
    ax_yz.annotate('Both beams centred\nat y = 0 (full overlap)',
                   xy=(0.97, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax_yz.grid(alpha=0.25)

    # --- z-distribution comparison panel ---
    ax_dist.plot(zs_off, z_dist_off, 'k-',  lw=1.5, label='No hourglass')
    ax_dist.plot(zs_on,  z_dist_on,  'r-',  lw=1.5,
                 label=rf'Hourglass  ($\beta^*={beta_star}$ cm)')
    ax_dist.annotate(
        f'Transverse x-offset: {x_offset:.0f} μm\n'
        f'Hourglass gives {abs(effect_pct):.1f}% {direction} collisions',
        xy=(0.03, 0.95), xycoords='axes fraction', va='top',
        bbox=dict(facecolor='wheat', alpha=0.4),
    )
    ax_dist.set_xlabel('z (cm)')
    ax_dist.set_ylabel('Collision density (arb.)')
    ax_dist.set_title('Z-Vertex Distribution — Hourglass Effect')
    ax_dist.set_ylim(bottom=0)
    ax_dist.grid(alpha=0.3)
    ax_dist.legend()

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
