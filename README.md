# Fiedler-Subspace

> Distributed, violation-free coordination through the Fiedler subspace.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](current_version/distributed_fiedler_violation_free_soc/requirements.txt)
[![License](https://img.shields.io/badge/license-see%20notice-lightgrey)](LICENSE)
[![Project page](https://img.shields.io/badge/project%20page-live-7c3aed)](https://tw6249.github.io/Fiedler-Subspace/)

Fiedler-Subspace is a research codebase for distributed multi-robot coordination under connectivity and safety constraints. It combines algebraic-connectivity estimation, subspace-aware optimization, finite-range communication, and safety-preserving control with simulation and physical-experiment tooling.

## What is here?

- **Core method** — spectral estimation, violation-free conic control, allocation, and diagnostics.
- **Reproducible experiments** — crossing, topology-disconnection, repeated-root, and end-to-end studies used by the manuscript.
- **Physical validation** — four-robot experiment utilities, motion-capture adapters, ROS 2 control packages, and recorded metrics.
- **MuJoCo preflight** — a fixed, reviewable four-UAV integration candidate with network-delay emulation and regression tests.

## Start here

```powershell
cd current_version/distributed_fiedler_violation_free_soc
python -m pip install -r requirements.txt
python -m pytest -q
python run_paper_experiments.py
python run_topology_experiment.py
python run_end_to_end_experiment.py
python plot_paper_figures.py
```

The supported MuJoCo entry point is documented in [`mujoco_preflight/README.md`](current_version/distributed_fiedler_violation_free_soc/mujoco_preflight/README.md). It is a timing and integration candidate; hardware and Wi-Fi validation still require the stated experimental gates.

## Repository map

| Path | Purpose |
| --- | --- |
| [`current_version/distributed_fiedler_violation_free_soc`](current_version/distributed_fiedler_violation_free_soc) | Canonical implementation and manuscript experiments |
| [`最新实物实验/动捕实验/实验`](最新实物实验/动捕实验/实验) | Motion-capture, ROS 2, and MuJoCo deployment package |
| [`四车实验`](四车实验) | Four-robot experiment workspace and recorded results |
| [`docs/index.html`](https://tw6249.github.io/Fiedler-Subspace/) | Visual project homepage for GitHub Pages |

## Reproducibility notes

The canonical experiment scripts write outputs to explicit directories and refuse accidental overwrites. Numerical records and figures should be treated as provenance-bearing artifacts: do not edit a result in place; create a new output directory and record the command used.

Raw videos, ROS bags, local caches, LaTeX build products, and temporary audit renders are intentionally excluded from the public Git history by [`.gitignore`](.gitignore). Publish large release artifacts through GitHub Releases or an external archival service.

## Citation

The citation block will be updated when the manuscript receives its final bibliographic information. Until then, please cite the accompanying manuscript and link to this repository.

## Status

Research software under active development. Interfaces and experiment paths may change before the archival release.

The initial public release includes the canonical numerical implementation, retained figures, and the four-robot CSV. Manuscript working drafts, ROS workspaces, and video-processing tools remain in the local research workspace. No open-source license has been granted yet; see LICENSE.
