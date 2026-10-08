# SIRDwell

Research code for relative observed-state dwell quantification using mutation-state occupancy and a transition backbone. The Python package retains the import name `relobstq_mhn`; the distribution and repository are named **SIRDwell**.

The method computes state inflow `F_hat`, the occupancy-to-inflow ratio `R_raw`, and the median-normalized score `R_star`. These scores describe relative accumulation in observed mutation states. They are dimensionless and do not estimate calendar time.

## Installation

Use Python **3.11 or 3.12**. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[mhn,test]'
```

The core dependencies are NumPy, pandas, SciPy, and PyYAML. The `mhn` extra installs the official `mhn==1.2.3` backend for cMHN fitting. Preprocessing, supplied-backbone scoring, simulations, and the selected longitudinal fallback calculation can be used without that extra:

```bash
python -m pip install -e '.[test]'
```

Computations produce tables and configuration metadata. The repository contains no plotting code. Core calculations do not require plotting packages; the optional MHN backend retains its upstream dependency set.

## Repository layout

```text
configs/                      Experiment configurations
data/                         Longitudinal input snapshots, provenance, and checksums
docs/configuration.md         Input schemas, parameters, and workflow dependencies
experiments/                  Thin command-line entry points
src/relobstq_mhn/
    core/                     State representation, inflow, scoring, bootstrap, cMHN
    data/                     Mutation and cohort table preparation
    evaluation/               Ranking, classification, and interval metrics
    simulation/               Mutation trajectories and implanted dwell signals
    workflows/                Experiment computations and result tables
    io/                       YAML configuration and result serialization
tests/                        Synthetic-data tests for core methods and workflows
```

The three longitudinal study input snapshots are bundled under `data/raw/`, with source notices and SHA-256 checksums. Restore them with `python -m experiments.install_data`; see [the data guide](data/README.md). GENIE raw records and patient-level derived inputs require provider access and are not bundled. Manuscript figures and experiment result archives are not included.

## Running the workflows

Run commands from the repository root so that the relative paths in `configs/` resolve correctly. Supply the input files described in [the configuration guide](docs/configuration.md) before running cohort analyses.

```bash
# Prepare the fixed cross-sectional event panels.
python -m experiments.prepare_cross_sectional

# Fit cMHN and calculate state scores.
python -m experiments.run_cross_sectional

# Calculate controls, information-gain summaries, and topology routes.
python -m experiments.run_secondary

# Run simulations using their known generating backbone.
python -m experiments.run_simulation
python -m experiments.run_topology_robustness

# Evaluate the configured longitudinal cohorts.
python -m experiments.run_longitudinal
```

Each entry point accepts `--config PATH`. The preparation and cross-sectional entry points also accept a repeatable `--cohort` argument:

```bash
python -m experiments.prepare_cross_sectional --cohort AACR_LUAD
python -m experiments.run_cross_sectional --cohort AACR_LUAD
```

Once all cohort inputs are available, the aggregate entry point runs the configured workflows in dependency order:

```bash
python -m experiments.run_all
```

The simulation commands accept `--repeats N`; this changes the configured experiment. Use separate output paths when changing parameters.

## Using the method API

The computation layer accepts pandas tables and NumPy arrays. For a known transition matrix, the state-scoring API returns scores, edges, and an optional bootstrap summary:

```python
from relobstq_mhn import ScoreThresholds
from relobstq_mhn.core.pipeline import score_states_from_mhn

scores, edges, bootstrap_summary = score_states_from_mhn(
    occupancy,
    theta,
    events,
    thresholds=ScoreThresholds(
        minimum_state_count=5,
        minimum_inflow=1e-8,
        high_confidence_state_count=10,
    ),
)
```

`occupancy` contains `state`, `stage`, `genotype`, `event_count`, `N_v`, and `L_v`. `theta` is a finite square log-hazard matrix whose row and column order matches `events`. See [the configuration guide](docs/configuration.md) for the full contracts.

## Scientific scope

The two simulation workflows use their known generating transition backbone; their recovery analyses do not include cMHN refitting error. The configured longitudinal workflow retains the selected frequency/cooccurrence backbone built from the full cohort. Its patient-fold occupancy scoring shares that backbone and is not a strictly out-of-fold prediction analysis. Separate patient-grouped cross-fitting helpers remain available in `workflows/longitudinal_preparation.py`.

Selected GLASS and CRC pair-level provenance remains unresolved. The longitudinal runner exposes the calculation and configuration; it does not establish an exact raw-data reconstruction of the selected manuscript pairs. Details are stated with the longitudinal configuration in the guide.

## Tests and licensing

```bash
python -m pytest
```

Tests exercise the method and workflows with synthetic inputs. Dataset-specific numerical reproduction requires the corresponding local data and configuration.

The project has not yet designated an open-source license; see [LICENSE_NOTICE.md](LICENSE_NOTICE.md). External datasets and the official MHN dependency retain their own licenses.
