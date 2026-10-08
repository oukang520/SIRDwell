# Experiment input data

This directory distributes the original cBioPortal tab-separated input files used by the configured longitudinal calculation. These are curated mutation and clinical exports, rather than raw sequencing reads. All retained input files are unchanged at the byte level after installation. GLASS mutation tables use gzip compression and 8 MiB parts for repository storage; compression does not filter records or change fields.

## Install and verify

Run from the repository root with Python 3.11 or 3.12. Installation uses only the Python standard library.

```bash
# Verify every distributed file part against the manifest.
python -m experiments.install_data --verify-only

# Restore the input files at the configured paths.
python -m experiments.install_data

# Run the configured longitudinal calculation.
python -m experiments.run_longitudinal
```

To install one study, add `--study difg_glass`, `--study coadread_mskcc`, or `--study mnm_washu_2016`. Repeat `--study` to select several. `--data-root PATH` changes the installation root; a corresponding custom `data_root` is then needed in the experiment configuration. Existing matching files are retained. Existing files with different contents cause an error and are not overwritten.

```text
data/
    manifest.json             Original-file and storage-part SHA-256 checksums
    sources.csv               Providers, versions, acquisition dates, access and citations
    raw/longitudinal/cbioportal/<study>/
                              Original files, source notices and compressed mutation parts
    longitudinal/cbioportal/<study>/
                              Locally restored inputs (ignored by Git)
```

`manifest.json` records the stored-part size and checksum, original-file size and checksum, compression mode, study, and target filename. All selected parts are checked before installation; every restored file is checked before it is moved into its final location. No external download is required for the three bundled studies.

## Sources and data licenses

| Dataset | Study ID | Documented version | Source publication |
| --- | --- | --- | --- |
| GLASS | `difg_glass` | `2022-05-31` release; GRCh37 | Varn et al., *Cell* (2022), [doi:10.1016/j.cell.2022.04.038](https://doi.org/10.1016/j.cell.2022.04.038) |
| CRC-triplets | `coadread_mskcc` | Retained cBioPortal snapshot; no independent numbered release | Brannon et al., *Genome Biology* (2014), [doi:10.1186/s13059-014-0454-7](https://doi.org/10.1186/s13059-014-0454-7) |
| MNM-WashU | `mnm_washu_2016` | Retained cBioPortal snapshot; no independent numbered release | Welch et al., *New England Journal of Medicine* (2016), [doi:10.1056/NEJMoa1605949](https://doi.org/10.1056/NEJMoa1605949) |

Contains information from the GLASS, MSK colorectal, and Washington University AML/MDS study databases distributed through [cBioPortal DataHub](https://github.com/cBioPortal/datahub). The bundled study data are available under the [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/). Preserve attribution and source notices, keep redistributed or adapted datasets open, and offer them under the same ODbL license. Each original study `LICENSE`, metadata and available readme are retained alongside the data. This dataset license is independent of the research-code license notice.

GLASS includes all three original `data_mutations*.txt` tables, clinical sample and patient tables, surgery times, and their available metadata. Expression, copy-number and treatment tables are outside the released calculation's inputs and are not bundled. CRC and MNM include the files in their retained study directories. The three studies keep separate source directories.

The recorded GLASS and CRC archive acquisition date is 2026-08-27. A reliable local MNM acquisition date is not recorded. Publication years are not data-version labels. Original archive URLs and retained archive SHA-256 values are in `sources.csv`; those URLs can change upstream. The checksums in `manifest.json` identify the actual files distributed here. A historical DataHub Git commit is not asserted.

## AACR Project GENIE inputs

The cross-sectional workflows use AACR Project GENIE **18.0-public**, available through the [fixed Synapse release folder](https://www.synapse.org/Synapse:syn68719152). Register at [GENIE Synapse](https://genie.synapse.org/), obtain access under the provider's current process, and accept the applicable data terms. The [v18 data guide, page 3](https://www.aacr.org/wp-content/uploads/2025/08/18.0-data_guide.pdf#page=3) requires express written permission from the AACR Project GENIE Coordinating Center for redistribution. Permission enquiries: `genieinfo@aacr.org`.

GENIE raw records and patient-level derived cohort inputs are not bundled. The configured preparation starts from cohort-filtered, harmonized `analysis_metadata.csv` and `mutations_long.csv` under `data/cross_sectional_harmonized/{AACR_LUAD,AACR_COAD,AACR_IDC}/`. See [the configuration guide](../docs/configuration.md#cross-sectional-input-tables) for the schema. It does not implement the complete provider-raw-to-harmonized cohort selection. The original study citation is the AACR Project GENIE Consortium, *Cancer Discovery* (2017), [doi:10.1158/2159-8290.CD-17-0151](https://doi.org/10.1158/2159-8290.CD-17-0151); cite data version 18.0-public separately.

## Calculation scope

The simulation configurations generate their own inputs and require no external dataset. Secondary analyses use results generated by the cross-sectional calculation.

The longitudinal configuration remains the selected full-cohort frequency/cooccurrence-backbone calculation. Bundling these source inputs does not resolve the historical GLASS/CRC selected pair-level generating provenance or guarantee reconstruction of selected manuscript pairs. See [the longitudinal configuration](../docs/configuration.md#selected-longitudinal-calculation) for the calculation and its scope.
