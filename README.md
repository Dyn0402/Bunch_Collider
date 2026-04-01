# bunch-collider

A Python library for simulating the collision of relativistic particle bunches, with a fast C++ backend via [pybind11](https://github.com/pybind/pybind11).

Designed for accelerator physics applications such as luminosity calculations, z-vertex distribution modelling, Vernier scan analysis, and hourglass (beta*) effect studies.

---

## Features

- **3D Gaussian bunch density** with transverse (x, y) and arbitrary longitudinal (z) profiles
- **Hourglass (beta-function) broadening** — transverse beam size grows away from the IP
- **Crossing angles** in both x-z and y-z planes
- **Arbitrary longitudinal profiles** — up to four Gaussian components, or a tabulated (interpolated) PDF loaded from file
- **Parallel simulation** — time steps distributed across CPU cores via `ProcessPoolExecutor`
- **Relativistic Møller factor** for luminosity calculations
- **Fast C++ core** — density evaluated entirely in C++ via pybind11

---

## Requirements

### Python dependencies (installed automatically by pip)
- Python ≥ 3.8
- numpy, scipy, matplotlib
- pybind11 ≥ 2.10 *(build-time only)*

### C++ compiler (must be installed separately)

The density calculation is implemented in C++ and **must be compiled on your
machine** — the compiled binary is not distributed with the package.
`pip install .` triggers this compilation automatically, but you need a
working C++ compiler first:

| Platform | Recommended toolchain |
|----------|-----------------------|
| **Linux** | `sudo apt install build-essential python3-dev` (Debian/Ubuntu) |
| **macOS** | `xcode-select --install` (installs Apple Clang) |
| **Windows** | [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) — select "Desktop development with C++" |

> **Why is this needed?**  The 3D density integral is evaluated on a grid of
> millions of points at each time step.  The C++ extension makes this ~100×
> faster than pure Python.  Without it the package will not import at all.

---

## Installation

### 1. Install system dependencies (Linux/Debian/Ubuntu)

```bash
sudo apt install build-essential python3-dev python3-tk
```

> `python3-tk` is needed for interactive matplotlib plot windows. If you skip
> this, examples will save plots to PNG files instead of opening a window.

### 2. Clone the repository

```bash
git clone https://github.com/Dyn0402/Bunch_Collider.git
cd Bunch_Collider
```

### 3. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 4. Install the package

```bash
pip install ".[examples]"
```

This fetches all Python dependencies (numpy, scipy, matplotlib), downloads
pybind11, and compiles the C++ extension.  You only need to do this once.

### 5. Verify the installation

```bash
python -c "from bunch_collider import BunchCollider, BunchDensity; print('Installation successful!')"
```

### 6. Run an example

```bash
python examples/basic_collision.py
```

A figure window should open showing the z-vertex distribution. If no display
is available the plot is saved to `basic_collision.png` instead.

---

### Editable install (for development)

```bash
pip install -e ".[examples]"
```

Changes to `.py` files take effect immediately. If you edit
`bunch_collider/_bunch_density_cpp.cpp` rerun `pip install -e ".[examples]"`
(or `python setup.py build_ext --inplace`) to recompile.

---

## Quick Start

```python
import numpy as np
from bunch_collider import BunchCollider

# Create a collider with default settings
sim = BunchCollider()

# Set the initial positions of the two bunches (um)
sim.set_bunch_rs(np.array([0., 0., -6e6]), np.array([0., 0., 6e6]))

# Set transverse beam widths (um)
sim.set_bunch_sigmas(np.array([170., 170.]), np.array([170., 170.]))

# Set longitudinal bunch lengths (um)
sim.set_bunch_lengths(1.1e6, 1.1e6)

# Enable the hourglass (beta*) effect
sim.set_bunch_beta_stars(85, 85)  # cm

# Run the parallel simulation
sim.run_sim_parallel()

# Get the z-vertex distribution
zs, z_dist = sim.get_z_density_dist()  # zs in cm
```

---

## API Overview

### `BunchDensity`

Represents a single relativistic particle bunch.

| Method | Description |
|--------|-------------|
| `set_initial_z(z)` | Initial z position (um) |
| `set_offsets(x, y)` | Transverse offsets (um) |
| `set_sigma(x, y[, z])` | Transverse (and longitudinal) RMS widths (um) |
| `set_bunch_length(l)` | Longitudinal RMS width (um) |
| `set_beta(x, y, z)` | Velocity vector (v/c) |
| `set_angles(angle_x, angle_y)` | Crossing half-angles (rad) |
| `set_beta_star(bx[, by])` | Beta* hourglass parameter (cm) |
| `set_beta_star_shift(sx[, sy])` | Longitudinal shift of beta* minimum (cm) |
| `set_delay(delay)` | Timing delay (ns) |
| `read_longitudinal_beam_profile_from_file(path)` | Load tabulated z profile |
| `read_longitudinal_beam_profile_fit_parameters_from_file(path)` | Load Gaussian fit params |
| `density(x, y, z)` | Evaluate density on a 3D grid (C++ accelerated) |
| `propagate()` | Advance the bunch by one timestep |
| `calculate_r_and_beta()` | Recompute position and velocity from initial params |

### `BunchCollider`

Manages two bunches and runs the collision simulation.

| Method | Description |
|--------|-------------|
| `set_bunch_rs(r1, r2)` | Initial positions (um) |
| `set_bunch_sigmas(s1, s2)` | Transverse widths (um) |
| `set_bunch_lengths(l1, l2)` | Longitudinal lengths (um) |
| `set_bunch_beta_stars(bx1, bx2[, by1, by2])` | Beta* values (cm) |
| `set_bunch_crossing(ax1, ay1, ax2, ay2)` | Crossing angles (rad) |
| `set_bunch_offsets(o1, o2)` | Transverse offsets (um) |
| `set_grid_size(nx, ny, nz, nt)` | Override grid dimensions |
| `set_z_bounds(bounds)` | z-axis range (um) |
| `run_sim()` | Serial simulation |
| `run_sim_parallel()` | Parallel simulation (recommended) |
| `get_z_density_dist()` | Returns `(z_cm, z_dist)` |
| `get_naked_luminosity()` | Integrated luminosity (per particle pair) |
| `get_relativistic_moller_factor()` | Møller flux factor |

---

## Examples

The `examples/` directory contains ready-to-run scripts:

| Script | Description |
|--------|-------------|
| `basic_collision.py` | Head-on collision, Gaussian fit to z-vertex distribution |
| `hourglass_effect.py` | Compare with/without beta* hourglass broadening |
| `vernier_scan.py` | Scan transverse offset to measure effective beam size |
| `animation.py` | Animate the density product as bunches pass through each other |

Run any example from the repository root after installing:

```bash
python examples/basic_collision.py
```

---

## Units

| Quantity | Unit |
|----------|------|
| Position / width / length | micrometres (μm) |
| Angle | radians |
| Time / timestep | nanoseconds (ns) |
| Beta* | centimetres (cm) |
| Velocity | dimensionless (β = v/c) |

---

## Rebuilding the C++ Extension (development)

If you are editing `bunch_collider/_bunch_density_cpp.cpp` and want to
recompile without a full reinstall:

```bash
python setup.py build_ext --inplace
```

This recompiles and places the `.so` / `.pyd` file directly inside
`bunch_collider/`, so the package can be imported from the repo root without
installing.  For normal use, `pip install .` is preferred.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
