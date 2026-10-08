# Configuration and usage

The files in `configs/` define the command-line experiments. Paths are resolved relative to the current working directory; run the entry points from the repository root. Numerical parameters, event-panel order, seeds, and inclusion rules are part of the experiment definition. An edited configuration defines a different run and should use its own output directory.

The YAML defaults are authoritative for the entry points. Some reusable workflow dataclasses have smaller generic defaults, so a direct Python API call without an explicit configuration can differ from a configured command-line run.

## Workflow dependencies and paths

| Entry point | Configuration | Inputs | Default output |
| --- | --- | --- | --- |
| `experiments.prepare_cross_sectional` | `cross_sectional_preparation.yaml` | `data/cross_sectional_harmonized/<cohort>/` | `data/prepared/cross_sectional/<cohort>/` |
| `experiments.run_cross_sectional` | `cross_sectional.yaml` | `data/prepared/cross_sectional/<cohort>/` | `outputs/cross_sectional/<cohort>/` |
| `experiments.run_secondary` | `secondary.yaml` | Prepared matrices and cross-sectional results | `outputs/secondary/<cohort>/` |
| `experiments.run_simulation` | `simulation.yaml` | Generated internally | `outputs/simulation/dwell_gradient/` |
| `experiments.run_topology_robustness` | `topology_robustness.yaml` | Generated internally | `outputs/simulation/topology_robustness/` |
| `experiments.run_longitudinal` | `longitudinal.yaml` | `data/longitudinal/cbioportal/<study>/` | `outputs/longitudinal/` |

Cross-sectional preparation precedes fitting and scoring. Secondary analyses require both the prepared matrix and fitted cross-sectional tables. Simulations and longitudinal analysis are independent of that chain. `python -m experiments.run_all` runs the configured workflows in dependency order.

Every command accepts `--config PATH`. `prepare_cross_sectional` and `run_cross_sectional` accept repeatable `--cohort` filters. `run_simulation` and `run_topology_robustness` accept `--repeats N`, which overrides only the repeat count. Use `python -m experiments.<entry_point> --help` for command-specific options.

## Cross-sectional input tables

The preparation configuration uses three cohort identifiers: `AACR_LUAD`, `AACR_COAD`, and `AACR_IDC`. Its dataset label is `AACR Project GENIE v18.0-public`. Preparation starts from already harmonized, cohort-filtered tables available under the applicable provider terms; it does not perform raw-release cohort selection.

Place two CSV files in each `data/cross_sectional_harmonized/<cohort>/` directory:

| File | Columns | Contract |
| --- | --- | --- |
| `analysis_metadata.csv` | `analysis_id`, `stage_group`; optional `patient_id`, `sample_id` and other metadata | One row per analysis unit; unique `analysis_id`. |
| `mutations_long.csv` | `analysis_id`, `gene`; optional `consequence` | One row per mutation record; IDs refer to analysis units in the metadata. |

Keep identifiers consistently represented as strings across files. Genes are normalized to uppercase, duplicate analysis-unit/gene records are removed, and nonfunctional records are excluded when `consequence` is supplied. If the column is absent, the table should already contain the intended mutation records. A zero in the binary matrix means that no selected mutation record was found for that analysis unit and event; assay callability requires information beyond this schema.

Stage labels are standardized to `early`, `local_advanced`, `primary`, `metastatic`, or `unknown`. State identifiers combine the standardized stage and canonical genotype, for example `primary::KRAS+TP53`; `WT` denotes an empty selected-event genotype. `stage_group` should be populated with the intended stage information before preparation.

The cohort-specific 15-event panels and their order are supplied in `cross_sectional_preparation.yaml` and `cross_sectional.yaml`. Preparation uses these fixed panels, with `minimum_state_count: 5`. Each configured event must have a mutation record in the cohort input. The two configurations must agree on the model panel; preserve the matrix column order when reusing a fitted backbone.

Preparation writes the following files:

| File | Contents |
| --- | --- |
| `mhn_training_matrix.csv` | Binary event columns only, in configured order; rows sorted by `analysis_id`. |
| `mhn_row_index_map.csv` | The corresponding `analysis_id` and row index, with available state/sample metadata. |
| `state_table.csv` | One row per analysis unit, containing `analysis_id`, `stage_group`, genotype and state assignments. |
| `tables/state_occupancy.tsv` | State counts and observed occupancy fractions. |
| `tables/event_panel.tsv` | Ordered panel and selection-rule label. |
| `tables/preparation_qc.tsv` | Matrix and identifier consistency summaries. |

The first three files are the cross-sectional scoring input contract. If supplying prepared files directly, the matrix must contain only binary events, its row count must match the row map, IDs must be unique, and every row-map ID must have a stage assignment in `state_table.csv`. Stored genotype signatures are checked against the matrix when the complete event panel is reused.

## Cross-sectional model and scoring defaults

`cross_sectional.yaml` fits the official cMHN backend with CPU execution and an L1 penalty. The regularization candidates are scaled by the cohort size: `lambda = lambda_multiplier / n_samples`.

| Setting | Configured default |
| --- | --- |
| Lambda multipliers | `[0.1, 0.3, 1.0, 3.0, 10.0]` |
| Cross-validation folds | `5` |
| One-standard-error selection | `pick_1se: true` |
| Maximum iterations | `5000` |
| Relative tolerance | `1e-7` |
| Random seed | `20260630` |
| Minimum state count | `5` |
| Minimum inflow | `1e-8` |
| High-confidence state count | `10` |
| Score epsilon | `1e-12` |
| Bootstrap replicates / top-k | `500` / `10` |
| Number of reported top states | `25` |

The method constructs same-stage one-event-addition edges. It computes inflow from observed predecessor occupancy and transition probabilities, then uses

```text
R_raw(v)  = L_v / (F_hat_v + epsilon)
R_star(v) = R_raw(v) / median(R_raw among count- and inflow-eligible states)
```

`N_v` is the observed count and `L_v` is its occupancy fraction. Eligibility requires both the count and inflow thresholds; the high-confidence flag additionally requires the larger count threshold. Interpret numerical scores together with these flags. Bootstrap uncertainty resamples state counts while holding the transition backbone fixed.

Cross-sectional output tables include `state_occupancy`, `state_edges`, `state_scores`, `top_relative_dwell_states`, `theta`, `cv_scores`, `bootstrap_summary`, and `quality_control`. `theta.tsv` records the target-event rows and named source-event columns. Its orientation and event order must agree with the binary matrix.

The Python workflow `run_cross_sectional_cohort` also accepts a supplied `theta` argument. This bypasses model fitting while retaining matrix validation and scoring; the command-line entry point uses the configured cMHN fit.

## Secondary analyses

`secondary.yaml` reads, for each configured cohort, the prepared `mhn_training_matrix.csv` and cross-sectional `tables/{state_scores,state_edges,state_occupancy,theta}.tsv`. It verifies theta shape and event order before calculating the additional tables.

| Setting | Configured default |
| --- | --- |
| Top-k | `10` |
| Inflow-shuffle replicates | `400` |
| Random seed | `20260630`, plus the zero-based cohort index |
| Topology route targets | `6` |
| Minimum state count / inflow | `5` / `1e-8` |

Outputs cover inflow computability, the `R_star` landscape, information gain, denominator ablation, inflow-pairing falsification, and dominant-predecessor topology routes. They use the fitted backbone and observed cohorts from the cross-sectional workflow.

## Known-backbone simulations

Both simulation workflows generate mutation trajectories from a known log-hazard matrix and use that same matrix for state inflow scoring. Implanted dwell levels are relative multiplicative factors. These simulations assess score recovery conditional on the generating backbone; cMHN fitting and its estimation error are outside their computation.

| Setting | Dwell gradient | Topology robustness |
| --- | --- | --- |
| Configuration | `simulation.yaml` | `topology_robustness.yaml` |
| Event count | `15` | `12` |
| Pilot samples | `30000` | `12000` |
| Samples per repeat | `5000` | `3000` |
| Repeats | `60` | `10` per condition |
| Dwell levels | `[0.25, 0.5, 1.0, 2.0, 4.0]` | `[0.25, 0.5, 1.0, 2.0, 4.0]` |
| States per level | `5` | `2` |
| Maximum simulated time | `10.0` | `10.0` |
| Maximum events | `8` | `7` |
| Minimum pilot count | `25` | `8` |
| Base random seed | `20260901` | `20260901` |
| Minimum state count / inflow | `5` / `1e-8` | `5` / `1e-8` |

The dwell-gradient configuration uses `theta_sparsity: 0.10`. Truth-state selection includes `8` one-event, `12` two-event, and `5` three/four-event candidates, with count thresholds of `80`, `45`, and the minimum pilot count, respectively. Repeat seeds are derived from the base seed and repeat number.

Topology robustness spans `linear`, `branching`, `mutual_exclusivity`, and `mixed` topology templates, sparsities `[0.05, 0.10, 0.20]`, and dwell placements `early`, `middle`, and `late`. Placements correspond to one-event, two-event, and three/four-event states. The workflow derives deterministic seeds for conditions and repeats from the configured base seed.

Outputs contain generating-backbone, implanted-truth, repeat-level score, recovery-metric, and evaluation-coverage tables. Inspect coverage alongside recovery metrics: a truth state may not be observed or may fail eligibility thresholds in a sampled cohort. The tests use smaller synthetic configurations; the YAML files define the full experiment sizes.

## Selected longitudinal calculation

`longitudinal.yaml` configures the three selected studies:

| Study ID | Label | Ordering information | Study-specific overrides |
| --- | --- | --- | --- |
| `difg_glass` | GLASS | Primary/recurrence sample types; surgery timeline when available | `max_events: 8` |
| `coadread_mskcc` | CRC-triplets | Primary samples before metastases | Shared defaults |
| `mnm_washu_2016` | MNM-WashU | Numeric sample-ID suffix, with `-1` as baseline | `max_events: 10`, minimum state count `1`, high-confidence count `2` |

Install the bundled unchanged cBioPortal inputs with `python -m experiments.install_data`. Source versions, data licenses and checksums are described in [the data guide](../data/README.md). The installer restores standard cBioPortal tab-separated exports in `data/longitudinal/cbioportal/<study>/`:

| File | Required fields | Use |
| --- | --- | --- |
| `data_clinical_sample.txt` | `SAMPLE_ID`, `PATIENT_ID`; `SAMPLE_TYPE` for GLASS and CRC | Patient/sample mapping and study-specific temporal ordering. |
| `data_mutations*.txt` | `Hugo_Symbol`, `Tumor_Sample_Barcode` | Driver mutation records; `Variant_Classification` and/or `Consequence` are used for functional filtering when present. |
| `data_timeline_surgery.txt` | `SAMPLE_ID`, `START_DATE` | Optional GLASS surgery times, overriding sample-type ranks when available. |

The reader skips comment lines beginning with `#`. MNM sample IDs require a terminal numeric suffix such as `-1` or `-2`. Ordering can represent recurrence rank or primary/metastatic role rather than elapsed calendar time; interval outputs must be interpreted with the study-specific ordering scale.

The configured driver-selection threshold combines support and frequency. Genes are ranked by support, study priority, and gene name; if too few meet the threshold, the candidate list falls back to the available driver records before the event-count cap is applied.

| Setting | Configured default |
| --- | --- |
| Maximum selected events | `12`, with study overrides above |
| Minimum events for MHN evaluation gate | `5` |
| Minimum event frequency / support | `0.005` / `2` |
| Minimum state count / inflow | `2` / `1e-8` |
| High-confidence state count | `4` |
| Top reported states per study | `8` |
| Bootstrap replicates | `1000` |
| Base random seed | `20260827` |
| Validation seeds | GLASS `20260827`, CRC `20260827`, MNM `20260828` |
| Event-loss pair exclusion | `true` |
| Transition backend | `frequency_cooccurrence_backbone` |
| cMHN enabled | `false` |

The selected calculation builds its frequency/cooccurrence backbone from the full cohort. The retained evaluation derives occupancy scores and thresholds using patient folds, but the transition backbone remains shared across those folds. This is a full-cohort-backbone consistency analysis, not a strictly out-of-fold model prediction. Installing MHN does not change the selected backend while `mhn.enabled` remains `false`. The retained MHN settings (`fixed_lambda_multiplier: 1.0`, `max_iterations: 1500`, `relative_tolerance: 1e-6`) do not fit a cMHN model under that configuration.

Pairs combine adjacent ordered patient timepoints and exclude apparent loss of selected events under the monotone-progression rule. Output predictions include score-source and pair-QC fields so that missing scores and excluded pairs remain visible. GLASS and CRC selected pair-level generating provenance remains unresolved; the configured implementation cannot by itself establish exact reconstruction of the manuscript-selected pairs or their aggregate values.

The selected implementation is in `workflows/selected_longitudinal.py`. Per-study outputs under `<result_root>/<study>/tables/` contain sample metadata, driver mutations, event support, event matrices, occupancy, theta, scores, edges, ordered timepoints, pair predictions, persistence summaries, and dwell contrasts. Aggregated tables are written under `<result_root>/tables/`.

For a separate patient-grouped cross-fitting analysis, `workflows/longitudinal_preparation.py` exposes `prepare_longitudinal_pairs`. Its metadata input requires `analysis_id`, `patient_id`, `collection_time`, and `stage_group`; its event matrix requires a unique `analysis_id` plus binary event columns. It fits a backbone within each training-patient fold, unless fold-specific theta matrices are supplied. This helper defines a separate analysis and does not replace the selected longitudinal calculation.

## Result serialization and licensing

Workflows return tables to Python callers and write them when an output directory is configured. Result files include TSV tables, resolved settings, model/backend metadata, runtime input/environment metadata, and checksums as appropriate to the entry point. These are generated scientific outputs and are not distributed in the source repository. The three longitudinal input snapshots are distributed separately under `data/raw/`; GENIE inputs require provider access as explained in the data guide.

The project license has not yet been designated. See [LICENSE_NOTICE.md](../LICENSE_NOTICE.md); dataset and dependency licenses remain applicable independently.
