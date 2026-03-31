"""
Animated bunch collision visualisation.

Animates the density product and density sum of two colliding bunches in the
x-z and y-z planes as they pass through each other, and optionally saves a GIF.

A companion static figure shows the ±1σ beam envelopes (hourglass profiles) in
both the x-z and y-z planes, providing the optical context for the collision.

Two entry points are provided:

* ``animate_collision()``  — two bunches colliding, shows density product and sum.
* ``animate_single_bunch()`` — single bunch propagating through the grid.

Run::

    python examples/animation.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from bunch_collider import BunchDensity


def _draw_envelope(ax, z_cm, sigma_um, beta_star_cm,
                   x_offset_um=0., color='steelblue', label=None):
    """Draw the ±1σ hourglass beam envelope on *ax* (y in μm, z in cm)."""
    env = sigma_um * np.sqrt(1. + (z_cm / beta_star_cm) ** 2)
    ax.plot(z_cm,  env + x_offset_um, color=color, lw=1.2, label=label)
    ax.plot(z_cm, -env + x_offset_um, color=color, lw=1.2)
    ax.fill_between(z_cm, -env + x_offset_um, env + x_offset_um,
                    color=color, alpha=0.15)


# -----------------------------------------------------------------------
# Two-bunch collision
# -----------------------------------------------------------------------

def animate_collision(
    beam_width=150.,     # um  transverse RMS
    beam_length=1.3e6,   # um  longitudinal RMS
    beta_star=85.,       # cm
    x_offset=900.,       # um  transverse separation
    z_initial=6.0e6,     # um
    n_steps=50,
    n_grid=101,
    save_path=None,      # e.g. "collision.gif"
):
    """
    Animate the density product and sum of two colliding bunches.

    Parameters
    ----------
    beam_width : float
        Transverse RMS beam width (um).
    beam_length : float
        Longitudinal RMS bunch length (um).
    beta_star : float
        Beta* hourglass parameter (cm).
    x_offset : float
        Transverse offset of bunch 1 in x (um).
    z_initial : float
        Initial |z| position of each bunch (um).
    n_steps : int
        Number of time steps.
    n_grid : int
        Number of grid points per transverse axis.
    save_path : str or None
        If given, save the animation to this file (requires Pillow).
    """
    bunch1 = BunchDensity()
    bunch1.set_initial_z(-z_initial)
    bunch1.set_offsets(x_offset, 0.)
    bunch1.set_beta(0., 0., 1.)
    bunch1.set_sigma(beam_width, beam_width, beam_length)
    bunch1.set_angles(0., 0.)
    bunch1.set_beta_star(beta_star)

    bunch2 = BunchDensity()
    bunch2.set_initial_z(z_initial)
    bunch2.set_offsets(0., 0.)
    bunch2.set_beta(0., 0., -1.)
    bunch2.set_sigma(beam_width, beam_width, beam_length)
    bunch2.set_angles(0., 0.)
    bunch2.set_beta_star(beta_star)

    dt = 2 * z_initial / bunch1.c / n_steps
    bunch1.dt = bunch2.dt = dt
    bunch1.calculate_r_and_beta()
    bunch2.calculate_r_and_beta()

    x = np.linspace(-10 * beam_width, 10 * beam_width, n_grid)
    y = np.linspace(-10 * beam_width, 10 * beam_width, n_grid)
    z = np.linspace(-7 * beam_length, 7 * beam_length, n_grid + 5)
    z_cm = z / 1e4

    X3, Y3, Z3 = np.meshgrid(x, y, z, indexing='ij')

    # --- Static beam envelope figure ---
    fig_env, axs_env = plt.subplots(1, 2, figsize=(11, 3.5),
                                    sharey=True, sharex=True)
    z_env = np.linspace(z_cm.min(), z_cm.max(), 500)
    # x-z plane: bunch 1 offset in x, bunch 2 centred
    _draw_envelope(axs_env[0], z_env, beam_width, beta_star,
                   x_offset_um=x_offset, color='steelblue', label=f'Bunch 1 (x = {x_offset:.0f} μm)')
    _draw_envelope(axs_env[0], z_env, beam_width, beta_star,
                   x_offset_um=0., color='goldenrod', label='Bunch 2 (x = 0)')
    axs_env[0].set_xlim(z_cm.min(), z_cm.max())
    axs_env[0].set_ylim(x.min(), x.max())
    axs_env[0].axhline(0, color='gray', lw=0.6, alpha=0.5)
    axs_env[0].axvline(0, color='gray', lw=0.6, alpha=0.5)
    axs_env[0].set_xlabel('z (cm)')
    axs_env[0].set_ylabel('x (μm)')
    axs_env[0].set_title('Beam Envelopes — x-z')
    axs_env[0].legend(fontsize=8)
    axs_env[0].grid(alpha=0.25)
    # y-z plane: both bunches centred at y = 0
    _draw_envelope(axs_env[1], z_env, beam_width, beta_star,
                   x_offset_um=0., color='steelblue', label='Bunch 1')
    _draw_envelope(axs_env[1], z_env, beam_width, beta_star,
                   x_offset_um=0., color='goldenrod', label='Bunch 2')
    axs_env[1].axhline(0, color='gray', lw=0.6, alpha=0.5)
    axs_env[1].axvline(0, color='gray', lw=0.6, alpha=0.5)
    axs_env[1].set_xlabel('z (cm)')
    axs_env[1].set_ylabel('y (μm)')
    axs_env[1].set_title('Beam Envelopes — y-z')
    axs_env[1].annotate('Both bunches centred at y = 0',
                        xy=(0.97, 0.95), xycoords='axes fraction',
                        ha='right', va='top', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    axs_env[1].grid(alpha=0.25)
    fig_env.suptitle('Static Beam Envelopes (±1σ hourglass profiles)', y=1.01)
    fig_env.tight_layout()

    # Pre-compute frames
    frames_dp_xz, frames_dp_yz = [], []
    frames_ds_xz, frames_ds_yz = [], []

    for i in range(n_steps):
        d1 = bunch1.density(X3, Y3, Z3)
        d2 = bunch2.density(X3, Y3, Z3)
        dp = d1 * d2
        ds = d1 + d2
        frames_dp_xz.append(np.sum(dp, axis=1))
        frames_dp_yz.append(np.sum(dp, axis=0))
        frames_ds_xz.append(np.sum(ds, axis=1))
        frames_ds_yz.append(np.sum(ds, axis=0))
        bunch1.propagate()
        bunch2.propagate()

    # Derive vmax from the actual projected data so the colorscale is correct.
    # (The 3D peak density is not the right scale for a y-summed projection.)
    vmax_dp = max(f.max() for f in frames_dp_xz + frames_dp_yz)
    vmax_ds = max(f.max() for f in frames_ds_xz + frames_ds_yz)

    # Set up figure — sharey='row' links the transverse axis within each row
    fig, ax = plt.subplots(2, 2, figsize=(12, 6), sharey='row')
    ext = [z_cm.min(), z_cm.max(), x.min(), x.max()]

    im = {
        'dp_xz': ax[0, 0].imshow(frames_dp_xz[0], extent=ext, origin='lower',
                                  cmap='hot', vmin=0, vmax=vmax_dp, aspect='auto'),
        'dp_yz': ax[1, 0].imshow(frames_dp_yz[0],
                                  extent=[z_cm.min(), z_cm.max(), y.min(), y.max()],
                                  origin='lower', cmap='hot', vmin=0,
                                  vmax=vmax_dp, aspect='auto'),
        'ds_xz': ax[0, 1].imshow(frames_ds_xz[0], extent=ext, origin='lower',
                                  cmap='viridis', vmin=0, vmax=vmax_ds, aspect='auto'),
        'ds_yz': ax[1, 1].imshow(frames_ds_yz[0],
                                  extent=[z_cm.min(), z_cm.max(), y.min(), y.max()],
                                  origin='lower', cmap='viridis', vmin=0,
                                  vmax=vmax_ds, aspect='auto'),
    }
    ax[0, 0].set_title('Density Product (x-z)')
    ax[1, 0].set_title('Density Product (y-z)')
    ax[0, 1].set_title('Density Sum (x-z)')
    ax[1, 1].set_title('Density Sum (y-z)')
    for row in ax:
        for a in row:
            a.set_xlabel('z (cm)')
    ax[0, 0].set_ylabel('x (μm)')
    ax[1, 0].set_ylabel('y (μm)')
    # Right column shares y-axis with the left; suppress redundant tick labels
    ax[0, 1].tick_params(labelleft=False)
    ax[1, 1].tick_params(labelleft=False)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.3)

    def update(frame):
        idx = min(frame, n_steps - 1)
        im['dp_xz'].set_data(frames_dp_xz[idx])
        im['dp_yz'].set_data(frames_dp_yz[idx])
        im['ds_xz'].set_data(frames_ds_xz[idx])
        im['ds_yz'].set_data(frames_ds_yz[idx])
        return list(im.values())

    anim = FuncAnimation(fig, update, frames=int(n_steps * 1.5),
                         interval=80, blit=True)
    if save_path:
        anim.save(save_path, writer='pillow', fps=10)
        print(f"Saved animation to {save_path}")

    plt.show()


# -----------------------------------------------------------------------
# Single bunch propagation
# -----------------------------------------------------------------------

def animate_single_bunch(
    beam_width=100.,   # um
    beam_length=1.0e6, # um
    angle_x=1.e-4,     # rad crossing angle in x-z plane
    z_initial=9.0e6,   # um
    n_steps=120,
    n_grid_x=80,
    n_grid_z=120,
):
    """
    Animate a single bunch propagating through the interaction region.

    Parameters
    ----------
    beam_width : float
        Transverse RMS beam width (um).
    beam_length : float
        Longitudinal RMS bunch length (um).
    angle_x : float
        Bunch angle in the x-z plane (rad).
    z_initial : float
        Initial z position of the bunch (um).
    n_steps : int
        Number of propagation steps.
    n_grid_x : int
        Grid points in the transverse direction.
    n_grid_z : int
        Grid points along z.
    """
    bunch = BunchDensity()
    bunch.set_initial_z(-z_initial)
    bunch.set_offsets(0., 0.)
    bunch.set_angles(angle_x, 0.)
    bunch.set_beta(0., 0., 1.)
    bunch.set_sigma(beam_width, beam_width, beam_length)
    bunch.dt = 2 * z_initial / bunch.c / n_steps
    bunch.calculate_r_and_beta()

    x = np.linspace(-15 * beam_width, 15 * beam_width, n_grid_x)
    y = np.linspace(-15 * beam_width, 15 * beam_width, n_grid_x)
    z = np.linspace(-15 * beam_length, 15 * beam_length, n_grid_z)
    X3, Y3, Z3 = np.meshgrid(x, y, z, indexing='ij')
    max_density = 1. / ((2 * np.pi) ** 1.5 * beam_width ** 2 * beam_length)

    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    kw = dict(origin='lower', cmap='jet', vmin=0, vmax=max_density, aspect='auto')
    im_xz = ax[0].imshow(np.zeros((n_grid_x, n_grid_z)),
                          extent=[z.min(), z.max(), x.min(), x.max()], **kw)
    im_yz = ax[1].imshow(np.zeros((n_grid_x, n_grid_z)),
                          extent=[z.min(), z.max(), y.min(), y.max()], **kw)
    ax[0].set(title='Bunch density in x-z', xlabel='z (μm)', ylabel='x (μm)')
    ax[1].set(title='Bunch density in y-z', xlabel='z (μm)', ylabel='y (μm)')
    fig.tight_layout()

    # Pre-compute frames
    frames_xz, frames_yz = [], []
    for _ in range(n_steps):
        d = bunch.density(X3, Y3, Z3)
        frames_xz.append(np.sum(d, axis=1))
        frames_yz.append(np.sum(d, axis=0))
        bunch.propagate()

    def update(frame):
        idx = min(frame, n_steps - 1)
        im_xz.set_data(frames_xz[idx])
        im_yz.set_data(frames_yz[idx])
        return im_xz, im_yz

    FuncAnimation(fig, update, frames=int(n_steps * 1.3),
                  interval=60, blit=True)
    plt.show()


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main():
    # Uncomment the example you want to run:
    animate_collision()
    # animate_single_bunch()


if __name__ == '__main__':
    main()
