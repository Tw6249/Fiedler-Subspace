# Four-robot physical experiment

This directory contains the generated metric record used by
`../../../paper/sections/10_physical_experiment.tex`. The raw 200-sample record is
retained at `../../../../四车实验/实验结果/timeseries.csv`; it is not duplicated
here.

Reproduce the figures and metrics from
`current_version/distributed_fiedler_violation_free_soc/`:

```powershell
python plot_physical_experiment.py
```

The script evaluates the complete six-pair candidate graph. Pair weights are
set by the finite-range channel model, with transition distance 1.60 m and
cutoff distance 2.12 m. The centralized eigendecomposition in the script is an
offline ground-truth calculation and is not part of the applied control path.

`metrics.json` records the source CSV path and SHA-256 digest so that the paper
numbers remain traceable to the retained run. The physical implementation
description in the paper follows the revised deployment specified by the
authors: each robot rejects out-of-range information and solves its own local
SOCP onboard. The older run metadata under `四车实验/实验结果/` still describes
the superseded centralized ROS 2 port and should not be used as evidence for
the revised deployment architecture.
