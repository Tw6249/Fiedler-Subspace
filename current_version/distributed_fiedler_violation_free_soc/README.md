# Fiedler-subspace experiments

This directory contains only the code and results used by the current manuscript,
plus the MuJoCo preflight for the planned four-UAV experiment. The canonical
manuscript is `../paper/main.tex`.

## Structure

- `violation_free_soc_core.py`: spectral, conic-control, and allocation routines.
- `run_paper_experiments.py`: crossing, exact-subspace topology, and repeated-root experiments.
- `run_topology_experiment.py`: strong topology-disconnection stress test used in the paper.
- `run_end_to_end_experiment.py`: estimated-subspace controller experiment.
- `plot_paper_figures.py`: overview, crossing, all-iterate safety, and recovery figures.
- `plot_end_to_end_figure.py`: end-to-end comparison figure.
- `plot_physical_experiment.py`: four-robot physical-run figures and metrics.
- `paper_data/`: the numerical records supporting the current manuscript.
- `paper_figures/`: the PDF/PNG figures included by the current manuscript.
- `mujoco_preflight/`: MuJoCo dynamics, network emulation, outputs, and migration-facing checks.

## Reproduce the paper experiments

Run from this directory:

```powershell
python -m pip install -r requirements.txt
python run_paper_experiments.py
python run_topology_experiment.py
python run_end_to_end_experiment.py
python run_end_to_end_experiment.py --outdir paper_data/end_to_end_tightened --beta-total 0.003
python plot_paper_figures.py
python plot_end_to_end_figure.py
python plot_physical_experiment.py
python -m pytest -q
```

The fixed `beta_total=0.003` case is an empirically selected tightening, not an
online-certified error bound. It is retained as empirical robustness evidence.

## Physical-robot experiment

The physical-run plotter reads `../../四车实验/实验结果/timeseries.csv`. It
evaluates all six unordered robot pairs; the finite-range channel cutoff
determines the active graph. Centralized eigendecomposition is used only to
reconstruct offline ground-truth metrics. The manuscript includes the motion
composite at
`../../object_motion_compositor/results/VID_20260809_003148/composite.png`
directly from LaTeX.

```powershell
python plot_physical_experiment.py
```

The command writes the standalone architecture, trajectory, and connectivity
panels to `paper_figures/`, together with a four-panel diagnostic figure that
is not used in the manuscript. LaTeX assembles the standalone assets with
`\subfloat`; the pair-distance panel is omitted from the paper. The command
writes the hash-traceable numerical summary to
`paper_data/physical_experiment/metrics.json`.
See `paper_data/physical_experiment/README.md` for provenance and the boundary
between recorded data and the revised distributed deployment description.

## MuJoCo preflight

The current migration candidate uses the original estimator/controller structure
with two rounds per finite-consensus call, a 5 Hz high-level controller, and a
fixed 10 ms delay for one complete neighbor communication round. This gives 12
synchronous rounds per control cycle, or 60 rounds/s. It is a MuJoCo timing and
integration candidate, not evidence that the target boards and Wi-Fi link have
already achieved this rate.

```powershell
python -m mujoco_preflight.run_current_onboard_preflight `
  --outdir mujoco_preflight/outputs_current_reproduction
python -m pytest -q test_mujoco_preflight.py
```

The frozen current result is in
`mujoco_preflight/outputs_current_onboard_k02_100hz_v1/`. Its minimum sampled
algebraic connectivity is 0.588418, maximum projected cycle time is about
122.1 ms, and no deadline is missed in the fixed-delay model. The maximum local
Ritz-value error is 0.308677, so the scalar Ritz output remains a coarse logging
diagnostic and is not a safety certificate.

See `mujoco_preflight/README.md` for the only supported configuration,
communication accounting, figures, result interpretation, and hardware gates.
Legacy MuJoCo profiles, sweeps, and outputs have been removed so that this entry
cannot accidentally launch a stale experiment.
