# scripts/

Experiment-specific analysis scripts for sPHENIX RHIC Vernier scan data.

These scripts are **not** standalone — they require external data files
(ROOT vertex distributions, CAD measurement CSVs, longitudinal bunch profile
fits) that are not included in this repository.

They are preserved here as reference implementations showing how
``BunchCollider`` can be used in a full analysis pipeline.

| Script | Description |
|--------|-------------|
| `vernier_z_vertex_fitting.py` | Full fitting pipeline: reads ROOT z-vertex histograms and CAD data, fits crossing angles and beam widths to each scan step |
| `vernier_z_vertex_fitting_clean.py` | Cleaner orchestration layer over `vernier_z_vertex_fitting.py`; reads optimised parameters from file and runs final fits |
| `Measure.py` | Original copy of the value-with-uncertainty class (now also available as `bunch_collider.measure.Measure`) |

For a self-contained demonstration of the fitting workflow using synthetic
data, see `examples/fit_simulation_to_data.py`.
