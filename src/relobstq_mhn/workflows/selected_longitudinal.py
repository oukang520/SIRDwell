"""Table-only implementation of the selected E17 longitudinal calculation.

This preserves the selected runner's full-cohort frequency/co-occurrence
backbone, event selection, patient-fold occupancy scoring, and random seeds.
The separate longitudinal workflow accepts leakage-controlled predictions;
it does not replace this selected-version implementation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, kendalltau, mannwhitneyu, pointbiserialr, spearmanr

from ..core.scoring import ScoreThresholds, classify_relative_states, compute_relative_dwell
from ..core.states import canonical_genotype, genotype_signature
from ..core.transitions import aggregate_inflow, probability_provider_from_theta, same_stage_one_step_edges

try:
    import mhn
    from mhn.optimizers import Optimizer

    MHN_AVAILABLE = True
except Exception:
    MHN_AVAILABLE = False


NONFUNCTIONAL_VARIANTS = {
    "silent",
    "synonymous_variant",
    "intron",
    "intron_variant",
    "igr",
    "intergenic_region",
    "3'utr",
    "5'utr",
    "3_prime_utr_variant",
    "5_prime_utr_variant",
    "rna",
    "rna_variant",
}


DRIVER_CANDIDATES = {
    "ABL1",
    "ACVR1",
    "AKT1",
    "ALK",
    "APC",
    "AR",
    "ARAF",
    "ARID1A",
    "ARID2",
    "ATM",
    "ATRX",
    "AKT1",
    "BAP1",
    "BCL2",
    "BCL6",
    "BCOR",
    "BRAF",
    "BRCA1",
    "BRCA2",
    "BTK",
    "CARD11",
    "CASP8",
    "CBL",
    "CD79B",
    "CDH1",
    "CDK12",
    "CDKN2A",
    "CEBPA",
    "CIC",
    "CREBBP",
    "CTCF",
    "CTNNB1",
    "DDR2",
    "DNMT3A",
    "EGFR",
    "EP300",
    "ERBB2",
    "ERBB3",
    "ERBB4",
    "ESR1",
    "ETV6",
    "EZH2",
    "FAT1",
    "FBXW7",
    "FGFR1",
    "FGFR2",
    "FGFR3",
    "FLT3",
    "FOXA1",
    "FUBP1",
    "GATA3",
    "GNA11",
    "GNAQ",
    "H3F3A",
    "HRAS",
    "IDH1",
    "IDH2",
    "IKZF1",
    "JAK1",
    "JAK2",
    "KDM5C",
    "KDM6A",
    "KEAP1",
    "KIT",
    "KMT2A",
    "KMT2C",
    "KMT2D",
    "KRAS",
    "MAP2K1",
    "MAP2K4",
    "MAP3K1",
    "MDM2",
    "MET",
    "MLH1",
    "MPL",
    "MYC",
    "MYCN",
    "MYD88",
    "NF1",
    "NF2",
    "NFE2L2",
    "NOTCH1",
    "NOTCH2",
    "NPM1",
    "NRAS",
    "PALB2",
    "PAX5",
    "PIK3CA",
    "PIK3R1",
    "POLD1",
    "POLE",
    "PPP2R1A",
    "PTCH1",
    "PTEN",
    "PTPN11",
    "PIM1",
    "RB1",
    "RBM10",
    "RET",
    "RAC1",
    "RNF43",
    "RHOA",
    "SETD2",
    "SF3B1",
    "SMAD4",
    "SMARCA4",
    "SMARCB1",
    "SMO",
    "SOX9",
    "SPOP",
    "STAG2",
    "STK11",
    "TERT",
    "TET2",
    "TGFBR2",
    "TP53",
    "U2AF1",
    "VHL",
    "WT1",
}


STUDY_PRIORITY_GENES = {
    "difg_glass": [
        "IDH1",
        "TP53",
        "ATRX",
        "CIC",
        "FUBP1",
        "EGFR",
        "PTEN",
        "NF1",
        "PIK3CA",
        "BRAF",
        "H3F3A",
        "RB1",
    ],
    "nsclc_tracerx_2017": [
        "TP53",
        "KRAS",
        "EGFR",
        "STK11",
        "KEAP1",
        "NF1",
        "PIK3CA",
        "BRAF",
        "RBM10",
        "ATM",
        "SMARCA4",
        "SETD2",
    ],
    "coadread_mskcc": [
        "APC",
        "KRAS",
        "TP53",
        "PIK3CA",
        "SMAD4",
        "BRAF",
        "FBXW7",
        "TGFBR2",
        "SOX9",
        "ARID1A",
        "RNF43",
        "ATM",
    ],
    "mnm_washu_2016": [
        "TP53",
        "ASXL1",
        "SRSF2",
        "IDH2",
        "DNMT3A",
        "SF3B1",
        "RUNX1",
        "TET2",
        "IDH1",
        "NRAS",
        "U2AF1",
        "NPM1",
    ],
    "msk_chord_2024": [
        "TP53",
        "KRAS",
        "APC",
        "PIK3CA",
        "EGFR",
        "STK11",
        "KEAP1",
        "CDKN2A",
        "PTEN",
        "ATM",
        "BRCA2",
        "SMAD4",
    ],
    "all_phase2_target_2018_pub": [
        "KRAS",
        "NRAS",
        "FLT3",
        "PAX5",
        "CREBBP",
        "TP53",
        "IKZF1",
        "NOTCH1",
        "PTPN11",
        "ETV6",
        "JAK2",
        "RB1",
    ],
    "prad_su2c_2019": [
        "TP53",
        "SPOP",
        "FOXA1",
        "PTEN",
        "CDK12",
        "ATM",
        "BRCA2",
        "RB1",
        "AR",
        "PIK3CA",
        "KMT2D",
        "APC",
    ],
    "brca_dldccc_2022": [
        "TP53",
        "PIK3CA",
        "PTEN",
        "BRCA1",
        "BRCA2",
        "RB1",
        "NF1",
        "ERBB2",
        "GATA3",
        "MAP3K1",
        "CDH1",
        "ATM",
    ],
    "brca_mbcproject_2022": [
        "PIK3CA",
        "ESR1",
        "TP53",
        "GATA3",
        "CDH1",
        "ERBB2",
        "PTEN",
        "NF1",
        "RB1",
        "AKT1",
        "MAP3K1",
        "BRCA2",
    ],
    "brca_aurora_2023": [
        "PIK3CA",
        "TP53",
        "ESR1",
        "GATA3",
        "CDH1",
        "ERBB2",
        "PTEN",
        "NF1",
        "AKT1",
        "MAP3K1",
        "RB1",
        "BRCA2",
    ],
    "skcm_broad_brafresist_2012": [
        "BRAF",
        "NRAS",
        "NF1",
        "CDKN2A",
        "TP53",
        "PTEN",
        "RAC1",
        "MAP2K1",
        "KIT",
        "PIK3CA",
        "ARID2",
        "RB1",
    ],
    "crc_hta8_htan_2024": [
        "APC",
        "KRAS",
        "TP53",
        "PIK3CA",
        "SMAD4",
        "BRAF",
        "FBXW7",
        "TGFBR2",
        "SOX9",
        "ARID1A",
        "RNF43",
        "ATM",
    ],
    "lung_smc_2016": [
        "TP53",
        "EGFR",
        "KRAS",
        "STK11",
        "KEAP1",
        "NF1",
        "SMARCA4",
        "RBM10",
        "MET",
        "BRAF",
        "PIK3CA",
        "ERBB2",
    ],
    "bm_nsclc_mskcc_2023": [
        "TP53",
        "EGFR",
        "KRAS",
        "STK11",
        "KEAP1",
        "NF1",
        "SMARCA4",
        "RBM10",
        "MET",
        "BRAF",
        "PIK3CA",
        "ERBB2",
    ],
    "egc_msk_2017": [
        "TP53",
        "ERBB2",
        "KRAS",
        "PIK3CA",
        "ARID1A",
        "SMAD4",
        "CDH1",
        "RNF43",
        "APC",
        "RHOA",
        "FBXW7",
        "ATM",
    ],
    "nepc_wcm_2016": [
        "TP53",
        "RB1",
        "PTEN",
        "AR",
        "SPOP",
        "FOXA1",
        "BRCA2",
        "ATM",
        "CDK12",
        "PIK3CA",
        "KMT2D",
        "APC",
    ],
    "pcnsl_msk_2024": [
        "MYD88",
        "CD79B",
        "PIM1",
        "BCL2",
        "BCL6",
        "CARD11",
        "BTK",
        "CREBBP",
        "KMT2D",
        "TP53",
        "CDKN2A",
        "PTEN",
    ],
}


def config_for_study(config: dict, study_config: dict) -> dict:
    local = copy.deepcopy(config)
    for section in ["analysis", "mhn"]:
        overrides = study_config.get(f"{section}_overrides", {})
        if overrides:
            local.setdefault(section, {}).update(overrides)
    return local


def read_cbio_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", comment="#", dtype=str, low_memory=False)


def first_data_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip() and not line.startswith("#"):
                return line.rstrip("\n").split("\t")
    return []


def clean_gene(value: object) -> str:
    return str(value).strip().upper()


def is_functional(value: object) -> bool:
    text = str(value).strip().lower()
    if not text or text == "nan":
        return True
    return text not in NONFUNCTIONAL_VARIANTS


def numeric_or_nan(value: object) -> float:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "not available", "unknown"}:
            return float("nan")
        return float(text)
    except Exception:
        return float("nan")


def infer_temporal_metadata(study_id: str, study_dir: Path, sample_df: pd.DataFrame) -> pd.DataFrame:
    if sample_df.empty:
        return pd.DataFrame()
    work = sample_df.copy()
    if "SAMPLE_ID" not in work or "PATIENT_ID" not in work:
        raise ValueError(f"{study_id}: clinical sample table lacks SAMPLE_ID/PATIENT_ID")
    work["sample_id"] = work["SAMPLE_ID"].astype(str)
    work["patient_id"] = work["PATIENT_ID"].astype(str)
    work["sample_role"] = "unknown"
    work["stage"] = "unknown"
    work["time_rank"] = np.nan
    work["order_evaluable"] = False

    if study_id == "difg_glass":
        sample_type = work.get("SAMPLE_TYPE", "").fillna("").astype(str).str.lower()
        recurrence_order = {
            "tumor primary": 0.0,
            "first recurrence": 1.0,
            "second recurrence": 2.0,
            "third recurrence": 3.0,
            "fourth recurrence": 4.0,
            "first metastasis": 5.0,
        }
        work["time_rank"] = sample_type.map(recurrence_order)
        work["stage"] = np.where(work["time_rank"].fillna(-1).eq(0), "baseline", "progressed")
        work.loc[work["time_rank"].isna(), "stage"] = "unknown"
        work["sample_role"] = sample_type.str.replace(" ", "_", regex=False)
        surgery_times = read_sample_time_lookup(study_dir / "data_timeline_surgery.txt")
        work["time_rank"] = work["sample_id"].map(surgery_times).where(
            work["sample_id"].map(surgery_times).notna(),
            work["time_rank"],
        )
        work["order_evaluable"] = work["time_rank"].notna()

    elif study_id == "nsclc_tracerx_2017":
        sample_type = work.get("SAMPLE_TYPE", "").fillna("").astype(str).str.lower()
        timepoint = work.get("SAMPLE_COLLECTION_TIMEPOINT", "").fillna("").astype(str).str.lower()
        progressed = sample_type.str.contains("recurrence", na=False) | timepoint.str.contains("post", na=False)
        baseline = sample_type.str.contains("primary", na=False) | timepoint.str.contains("pre", na=False)
        work["time_rank"] = np.where(progressed, 1.0, np.where(baseline, 0.0, np.nan))
        work["stage"] = np.where(progressed, "progressed", np.where(baseline, "baseline", "unknown"))
        work["sample_role"] = np.where(progressed, "recurrence_or_post_treatment", "primary_or_pre_treatment")
        work.loc[work["stage"].eq("unknown"), "sample_role"] = "unknown"
        work["order_evaluable"] = work["time_rank"].notna()

    elif study_id == "coadread_mskcc":
        sample_type = work.get("SAMPLE_TYPE", "").fillna("").astype(str).str.lower()
        baseline = sample_type.str.contains("primary", na=False)
        progressed = sample_type.str.contains("metast", na=False)
        work["time_rank"] = np.where(progressed, 1.0, np.where(baseline, 0.0, np.nan))
        work["stage"] = np.where(progressed, "progressed", np.where(baseline, "baseline", "unknown"))
        work["sample_role"] = np.where(progressed, "metastasis", "primary")
        work.loc[work["stage"].eq("unknown"), "sample_role"] = "unknown"
        work["order_evaluable"] = work["time_rank"].notna()

    elif study_id == "mnm_washu_2016":
        suffix = pd.to_numeric(work["sample_id"].str.extract(r"-(\d+)$", expand=False), errors="coerce")
        work["time_rank"] = suffix
        work["stage"] = np.where(suffix.eq(1), "baseline", np.where(suffix.gt(1), "progressed", "unknown"))
        work["sample_role"] = np.where(
            suffix.eq(1),
            "paired_early_sample",
            np.where(suffix.gt(1), "paired_late_sample", "unknown"),
        )
        work["order_evaluable"] = suffix.notna()

    elif study_id == "msk_chord_2024":
        sample_type = work.get("SAMPLE_TYPE", "").fillna("").astype(str).str.lower()
        baseline = sample_type.str.contains("primary", na=False)
        progressed = sample_type.str.contains("metast|recurrence", regex=True, na=False)
        work["stage"] = np.where(progressed, "progressed", np.where(baseline, "baseline", "unknown"))
        work["sample_role"] = np.where(progressed, "metastasis_or_local_recurrence", "primary")
        work.loc[work["stage"].eq("unknown"), "sample_role"] = "unknown"
        time_lookup = read_msk_chord_specimen_times(study_dir)
        work["time_rank"] = work["sample_id"].map(time_lookup)
        fallback_order = work["sample_id"].str.extract(r"-T(\d+)-", expand=False).map(numeric_or_nan)
        work["time_rank"] = work["time_rank"].where(work["time_rank"].notna(), fallback_order)
        work.loc[work["time_rank"].isna() & baseline, "time_rank"] = 0.0
        work.loc[work["time_rank"].isna() & progressed, "time_rank"] = 1.0
        work["order_evaluable"] = work["time_rank"].notna() & work["stage"].ne("unknown")

    elif study_id == "all_phase2_target_2018_pub":
        relapse_cols = [
            "BONE_MARROW_SITE_OF_RELAPSE",
            "CNS_SITE_OF_RELAPSE",
            "TESTES_SITE_OF_RELAPSE",
            "OTHER_SITE_OF_RELAPSE",
        ]
        relapse = pd.Series(False, index=work.index)
        for col in relapse_cols:
            if col in work:
                relapse |= work[col].fillna("").astype(str).str.lower().eq("yes")
        work["stage"] = np.where(relapse, "progressed", "baseline")
        work["sample_role"] = np.where(relapse, "relapse_flag_positive", "relapse_flag_negative")
        work["time_rank"] = np.where(relapse, 1.0, 0.0)
        work["order_evaluable"] = False

    elif study_id == "prad_su2c_2019":
        tissue = work.get("TISSUE_SITE", "").fillna("").astype(str).str.lower()
        age = work.get("AGE_AT_PROCUREMENT", "").map(numeric_or_nan) if "AGE_AT_PROCUREMENT" in work else np.nan
        baseline = tissue.str.contains("prostate", na=False)
        progressed = ~baseline & ~tissue.isin(["", "unknown", "not available"])
        work["stage"] = np.where(progressed, "progressed", np.where(baseline, "baseline", "unknown"))
        work["sample_role"] = np.where(progressed, "metastatic_site", np.where(baseline, "prostate_site", "unknown"))
        work["time_rank"] = age
        work["order_evaluable"] = work["time_rank"].notna() & work["stage"].ne("unknown")

    elif study_id == "brca_dldccc_2022":
        event = work.get("COLLECTION_EVENT", "").fillna("").astype(str).str.lower()
        time_rank = np.where(event.str.contains("baseline", na=False), 0.0, np.nan)
        time_rank = np.where(event.str.contains("cycle 1", na=False), 0.1, time_rank)
        baseline = event.str.contains("baseline", na=False)
        progressed = event.str.contains("cycle 1", na=False)
        work["time_rank"] = time_rank
        work["stage"] = np.where(progressed, "progressed", np.where(baseline, "baseline", "unknown"))
        work["sample_role"] = np.where(progressed, "cycle_1_day_3", np.where(baseline, "baseline", "unknown"))
        work["order_evaluable"] = work["time_rank"].notna()

    elif study_id == "brca_mbcproject_2022":
        bx_day = work.get("BX_TIME_DAYS", pd.Series(np.nan, index=work.index)).map(numeric_or_nan)
        fallback_time = work.get("SAMPLE_TIMEPOINT", "").fillna("").astype(str).str.extract(r"_T(\d+)", expand=False).map(numeric_or_nan)
        work["time_rank"] = bx_day.where(bx_day.notna(), fallback_time)
        location = work.get("BX_LOCATION", "").fillna("").astype(str).str.lower()
        setting = work.get("CALC_MET_SETTING", "").fillna("").astype(str).str.lower()
        baseline = setting.str.contains("no_metastatic", na=False) | location.str.contains("breast", na=False)
        progressed = setting.str.contains("metastatic_disease_present", na=False) & ~baseline
        work["stage"] = np.where(progressed, "progressed", np.where(baseline, "baseline", "unknown"))
        role = np.where(progressed, "metastatic_or_liquid_biopsy", np.where(baseline, "breast_or_non_metastatic", "unknown"))
        role = np.where(location.str.contains("blood", na=False), "liquid_biopsy", role)
        work["sample_role"] = role
        work["order_evaluable"] = work["time_rank"].notna()

    elif study_id == "brca_aurora_2023":
        sample_type = work.get("SAMPLE_TYPE", work.get("CBIOPORTAL_SAMPLE_TYPE", "")).fillna("").astype(str).str.lower()
        cbio_type = work.get("CBIOPORTAL_SAMPLE_TYPE", "").fillna("").astype(str).str.lower()
        baseline = sample_type.str.contains("primary", na=False) | cbio_type.str.contains("primary", na=False)
        progressed = sample_type.str.contains("metast", na=False) | cbio_type.str.contains("metast", na=False)
        work["time_rank"] = np.where(progressed, 1.0, np.where(baseline, 0.0, np.nan))
        work["stage"] = np.where(progressed, "progressed", np.where(baseline, "baseline", "unknown"))
        site = work.get("METASTATIC_SITE", "").fillna("").astype(str).str.lower().str.replace(r"\s+", "_", regex=True)
        work["sample_role"] = np.where(progressed, "metastasis_" + site, np.where(baseline, "primary", "unknown"))
        work.loc[pd.Series(work["sample_role"], index=work.index).astype(str).eq("metastasis_"), "sample_role"] = "metastasis"
        work["order_evaluable"] = work["time_rank"].notna()

    elif study_id == "skcm_broad_brafresist_2012":
        sample_id = work["sample_id"].str.lower()
        baseline = sample_id.str.contains(r"(?:^|[_-])pre(?:$|[_-])", regex=True, na=False)
        progressed = sample_id.str.contains(r"(?:^|[_-])post(?:$|[_-])", regex=True, na=False)
        work["time_rank"] = np.where(progressed, 1.0, np.where(baseline, 0.0, np.nan))
        work["stage"] = np.where(progressed, "progressed", np.where(baseline, "baseline", "unknown"))
        work["sample_role"] = np.where(progressed, "post_braf_inhibitor", np.where(baseline, "pre_braf_inhibitor", "unknown"))
        work["order_evaluable"] = work["time_rank"].notna()

    elif study_id == "crc_hta8_htan_2024":
        sample_type = work.get("SAMPLE_TYPE", "").fillna("").astype(str).str.lower()
        sample_class = work.get("SAMPLE_CLASS", "").fillna("").astype(str).str.lower()
        baseline = sample_type.str.contains("primary", na=False) & sample_class.str.contains("tumor", na=False)
        progressed = sample_type.str.contains("metast", na=False) & sample_class.str.contains("tumor", na=False)
        work["time_rank"] = np.where(progressed, 1.0, np.where(baseline, 0.0, np.nan))
        work["stage"] = np.where(progressed, "progressed", np.where(baseline, "baseline", "unknown"))
        work["sample_role"] = np.where(progressed, "metastasis", np.where(baseline, "primary", "non_tumor_or_unknown"))
        work["order_evaluable"] = work["time_rank"].notna()

    elif study_id == "lung_smc_2016":
        sample_type = work.get("SAMPLE_TYPE", "").fillna("").astype(str).str.lower()
        baseline = sample_type.str.contains("primary", na=False)
        progressed = sample_type.str.contains("metast", na=False)
        work["time_rank"] = np.where(progressed, 1.0, np.where(baseline, 0.0, np.nan))
        work["stage"] = np.where(progressed, "progressed", np.where(baseline, "baseline", "unknown"))
        work["sample_role"] = np.where(progressed, "lymph_node_or_distant_metastasis", np.where(baseline, "primary", "unknown"))
        work["order_evaluable"] = work["time_rank"].notna()

    elif study_id == "bm_nsclc_mskcc_2023":
        specimen = work.get("TISSUE_SPECIMEN_TYPE", "").fillna("").astype(str).str.upper()
        baseline = specimen.eq("PT") | work.get("SAMPLE_TYPE", "").fillna("").astype(str).str.lower().str.contains("primary", na=False)
        extracranial_met = specimen.eq("EM")
        brain_met = specimen.eq("BM")
        work["time_rank"] = np.where(brain_met, 2.0, np.where(extracranial_met, 1.0, np.where(baseline, 0.0, np.nan)))
        work["stage"] = np.where(brain_met | extracranial_met, "progressed", np.where(baseline, "baseline", "unknown"))
        work["sample_role"] = np.where(brain_met, "brain_metastasis", np.where(extracranial_met, "extracranial_metastasis", np.where(baseline, "primary", "unknown")))
        work["order_evaluable"] = work["time_rank"].notna() & work["stage"].ne("unknown")

    elif study_id == "egc_msk_2017":
        sample_type = work.get("SAMPLE_TYPE", "").fillna("").astype(str).str.lower()
        baseline = sample_type.str.contains("primary", na=False)
        progressed = sample_type.str.contains("metast", na=False)
        work["time_rank"] = np.where(progressed, 1.0, np.where(baseline, 0.0, np.nan))
        work["stage"] = np.where(progressed, "progressed", np.where(baseline, "baseline", "unknown"))
        site = work.get("SITE_OF_TISSUE_BIOPSY", "").fillna("").astype(str).str.lower().str.replace(r"\s+", "_", regex=True)
        work["sample_role"] = np.where(progressed, "metastasis_" + site, np.where(baseline, "primary", "unknown"))
        work["order_evaluable"] = work["time_rank"].notna()

    elif study_id == "nepc_wcm_2016":
        tissue = work.get("TUMOR_TISSUE_SITE", "").fillna("").astype(str).str.lower()
        baseline = tissue.str.contains("prostate", na=False)
        progressed = tissue.str.len().gt(0) & ~baseline
        work["time_rank"] = np.where(progressed, 1.0, np.where(baseline, 0.0, np.nan))
        work["stage"] = np.where(progressed, "progressed", np.where(baseline, "baseline", "unknown"))
        work["sample_role"] = np.where(progressed, "metastatic_site", np.where(baseline, "prostate_site", "unknown"))
        work["order_evaluable"] = work["time_rank"].notna()

    elif study_id == "pcnsl_msk_2024":
        sample_type = work.get("SAMPLE_TYPE", "").fillna("").astype(str).str.lower()
        baseline = sample_type.str.contains("primary", na=False)
        csf = sample_type.str.contains("csf", na=False)
        progressed = sample_type.str.contains("metast", na=False)
        work["time_rank"] = np.where(progressed, 1.0, np.where(csf, 0.5, np.where(baseline, 0.0, np.nan)))
        work["stage"] = np.where(progressed | csf, "progressed", np.where(baseline, "baseline", "unknown"))
        work["sample_role"] = np.where(progressed, "metastasis", np.where(csf, "csf_liquid_biopsy", np.where(baseline, "primary", "unknown")))
        work["order_evaluable"] = work["time_rank"].notna()

    else:
        sample_type = work.get("SAMPLE_TYPE", "").fillna("").astype(str).str.lower()
        baseline = sample_type.str.contains("primary|pre", regex=True, na=False)
        progressed = sample_type.str.contains("metast|recurrence|post", regex=True, na=False)
        work["stage"] = np.where(progressed, "progressed", np.where(baseline, "baseline", "unknown"))
        work["time_rank"] = np.where(progressed, 1.0, np.where(baseline, 0.0, np.nan))
        work["sample_role"] = work["stage"]
        work["order_evaluable"] = work["time_rank"].notna()

    columns = [
        "patient_id",
        "sample_id",
        "stage",
        "sample_role",
        "time_rank",
        "order_evaluable",
    ]
    for optional in ["SAMPLE_TYPE", "SAMPLE_COLLECTION_TIMEPOINT", "ONCOTREE_CODE", "CANCER_TYPE_DETAILED"]:
        if optional in work:
            columns.append(optional)
    return work[columns].copy()


def read_msk_chord_specimen_times(study_dir: Path) -> dict[str, float]:
    lookup: dict[str, float] = {}
    for name in ["data_timeline_specimen.txt", "data_timeline_specimen_surgery.txt"]:
        path = study_dir / name
        if not path.exists():
            continue
        table = read_cbio_table(path)
        if table.empty or "SAMPLE_ID" not in table or "START_DATE" not in table:
            continue
        table["_time"] = table["START_DATE"].map(numeric_or_nan)
        table = table.dropna(subset=["_time"])
        for row in table[["SAMPLE_ID", "_time"]].to_dict(orient="records"):
            sample_id = str(row["SAMPLE_ID"])
            value = float(row["_time"])
            if sample_id not in lookup or value < lookup[sample_id]:
                lookup[sample_id] = value
    return lookup


def read_sample_time_lookup(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    table = read_cbio_table(path)
    if table.empty or "SAMPLE_ID" not in table or "START_DATE" not in table:
        return {}
    table["_time"] = table["START_DATE"].map(numeric_or_nan)
    table = table.dropna(subset=["_time"])
    lookup: dict[str, float] = {}
    for row in table[["SAMPLE_ID", "_time"]].to_dict(orient="records"):
        sample_id = str(row["SAMPLE_ID"])
        value = float(row["_time"])
        if sample_id not in lookup or value < lookup[sample_id]:
            lookup[sample_id] = value
    return lookup


def iter_mutation_chunks(path: Path, sample_ids: set[str], genes: set[str]) -> Iterable[pd.DataFrame]:
    header = first_data_header(path)
    if not header:
        return
    usecols = [col for col in ["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification", "Consequence"] if col in header]
    if "Hugo_Symbol" not in usecols or "Tumor_Sample_Barcode" not in usecols:
        return
    for chunk in pd.read_csv(
        path,
        sep="\t",
        comment="#",
        dtype=str,
        usecols=usecols,
        chunksize=250_000,
        low_memory=False,
    ):
        chunk = chunk.rename(columns={"Hugo_Symbol": "gene", "Tumor_Sample_Barcode": "sample_id"})
        chunk["gene"] = chunk["gene"].map(clean_gene)
        chunk["sample_id"] = chunk["sample_id"].astype(str)
        chunk = chunk[chunk["sample_id"].isin(sample_ids) & chunk["gene"].isin(genes)].copy()
        if chunk.empty:
            continue
        if "Variant_Classification" in chunk:
            chunk = chunk[chunk["Variant_Classification"].map(is_functional)]
        if "Consequence" in chunk:
            chunk = chunk[chunk["Consequence"].map(is_functional)]
        if not chunk.empty:
            yield chunk[["sample_id", "gene"]].drop_duplicates()


def load_driver_mutations(study_id: str, study_dir: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    sample_ids = set(metadata["sample_id"].astype(str))
    genes = set(DRIVER_CANDIDATES).union(STUDY_PRIORITY_GENES.get(study_id, []))
    pairs: set[tuple[str, str]] = set()
    for mutation_path in sorted(study_dir.glob("data_mutations*.txt")):
        for chunk in iter_mutation_chunks(mutation_path, sample_ids, genes):
            pairs.update((str(row.sample_id), str(row.gene)) for row in chunk.itertuples(index=False))
    if not pairs:
        return pd.DataFrame(columns=["sample_id", "gene"])
    return pd.DataFrame(sorted(pairs), columns=["sample_id", "gene"])


def select_events(
    study_id: str,
    metadata: pd.DataFrame,
    mutations: pd.DataFrame,
    config: dict,
) -> tuple[list[str], pd.DataFrame]:
    max_events = int(config["analysis"]["max_events"])
    min_freq = float(config["analysis"]["min_event_frequency"])
    min_support = int(config["analysis"]["min_event_support"])
    n_samples = max(int(metadata["sample_id"].nunique()), 1)
    if mutations.empty:
        return [], pd.DataFrame(columns=["event", "sample_count", "frequency", "selected"])

    support = (
        mutations.drop_duplicates(["sample_id", "gene"])
        .groupby("gene")["sample_id"]
        .nunique()
        .rename("sample_count")
        .sort_values(ascending=False)
    )
    table = support.reset_index().rename(columns={"gene": "event"})
    table["frequency"] = table["sample_count"] / n_samples
    priority = {gene: idx for idx, gene in enumerate(STUDY_PRIORITY_GENES.get(study_id, []))}
    table["priority_rank"] = table["event"].map(priority).fillna(10_000).astype(int)
    threshold = max(min_support, int(math.ceil(n_samples * min_freq)))
    candidates = table[table["sample_count"].ge(threshold)].copy()
    if len(candidates) < int(config["analysis"]["min_events_for_mhn"]):
        candidates = table.copy()
    candidates = candidates.sort_values(["sample_count", "priority_rank", "event"], ascending=[False, True, True])
    selected = candidates["event"].head(max_events).tolist()
    table["selected"] = table["event"].isin(selected)
    table = table.sort_values(["selected", "sample_count", "priority_rank"], ascending=[False, False, True])
    return selected, table.reset_index(drop=True)


def build_event_matrix(metadata: pd.DataFrame, mutations: pd.DataFrame, events: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame({"sample_id": metadata["sample_id"].astype(str).tolist()})
    for event in events:
        matrix[event] = 0
    if events and not mutations.empty:
        work = mutations[mutations["gene"].isin(events)].drop_duplicates(["sample_id", "gene"]).copy()
        work["_value"] = 1
        pivot = work.pivot_table(index="sample_id", columns="gene", values="_value", aggfunc="max", fill_value=0)
        pivot = pivot.reindex(index=matrix["sample_id"], columns=events, fill_value=0).astype(int).reset_index()
        matrix = pivot
    return matrix


def build_occupancy(metadata: pd.DataFrame, matrix: pd.DataFrame, events: list[str]) -> pd.DataFrame:
    merged = metadata.merge(matrix, on="sample_id", how="left")
    for event in events:
        merged[event] = pd.to_numeric(merged[event], errors="coerce").fillna(0).astype(int)
    merged["genotype"] = [genotype_signature(row, events) for row in merged[events].to_numpy(dtype=int)]
    merged["genotype"] = merged["genotype"].map(canonical_genotype)
    merged["event_count"] = merged[events].sum(axis=1).astype(int) if events else 0
    merged["state"] = merged["stage"].astype(str).str.lower() + "::" + merged["genotype"].astype(str)
    occupancy = (
        merged[merged["stage"].ne("unknown")]
        .groupby(["state", "stage", "genotype", "event_count"], dropna=False)
        .size()
        .rename("N_v")
        .reset_index()
    )
    total = int(occupancy["N_v"].sum())
    occupancy["L_v"] = occupancy["N_v"] / max(total, 1)
    return occupancy.sort_values(["N_v", "state"], ascending=[False, True]).reset_index(drop=True)


def fit_or_build_theta(matrix: pd.DataFrame, events: list[str], config: dict, seed: int) -> tuple[np.ndarray, dict]:
    values = matrix[events].to_numpy(dtype=np.int32)
    n_samples, n_events = values.shape
    min_events = int(config["analysis"]["min_events_for_mhn"])
    if (
        bool(config["mhn"].get("enabled", True))
        and MHN_AVAILABLE
        and n_events >= min_events
        and n_samples >= 30
    ):
        try:
            np.random.seed(seed)
            mhn.set_seed(seed)
            optimizer = Optimizer(Optimizer.MHNType.cMHN)
            optimizer.set_device(optimizer.Device.CPU)
            optimizer.set_penalty(optimizer.Penalty.L1)
            optimizer.load_data_matrix(values)
            lam = float(config["mhn"]["fixed_lambda_multiplier"]) / max(n_samples, 1)
            model = optimizer.train(
                lam=lam,
                maxit=int(config["mhn"]["max_iterations"]),
                reltol=float(config["mhn"]["relative_tolerance"]),
                round_result=False,
            )
            theta = np.asarray(model.log_theta, dtype=float)
            if theta.shape == (n_events, n_events) and np.isfinite(theta).all():
                return theta, {
                    "backend": "cMHN_fixed_L1",
                    "lambda": lam,
                    "lambda_multiplier": float(config["mhn"]["fixed_lambda_multiplier"]),
                    "fit_status": "ok",
                }
        except Exception as exc:
            fallback = build_frequency_cooccurrence_theta(values)
            return fallback, {
                "backend": "frequency_cooccurrence_backbone",
                "lambda": np.nan,
                "lambda_multiplier": np.nan,
                "fit_status": f"fallback_after_cmhN_error:{type(exc).__name__}",
            }

    fallback = build_frequency_cooccurrence_theta(values)
    return fallback, {
        "backend": "frequency_cooccurrence_backbone",
        "lambda": np.nan,
        "lambda_multiplier": np.nan,
        "fit_status": "fallback_or_not_evaluable_for_cmhN",
    }


def build_frequency_cooccurrence_theta(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n_samples, n_events = values.shape
    theta = np.zeros((n_events, n_events), dtype=float)
    if n_events == 0:
        return theta
    eps = 0.5 / max(n_samples, 1)
    freq = np.clip(values.mean(axis=0), eps, 1 - eps)
    theta[np.diag_indices(n_events)] = np.clip(np.log(freq / (1.0 - freq)), -4.0, 2.0)
    for target in range(n_events):
        for source in range(n_events):
            if target == source:
                continue
            co = float(np.mean((values[:, target] > 0) & (values[:, source] > 0)))
            lift = (co + eps) / (freq[target] * freq[source] + eps)
            theta[target, source] = float(np.clip(np.log(lift), -1.25, 1.25))
    return theta


def score_external_states(
    occupancy: pd.DataFrame,
    theta: np.ndarray,
    events: list[str],
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    threshold_values = config["analysis"]
    thresholds = ScoreThresholds(
        minimum_state_count=int(threshold_values["min_state_count"]),
        minimum_inflow=float(threshold_values["minimum_inflow"]),
        high_confidence_state_count=int(threshold_values["high_confidence_state_count"]),
    )
    provider = probability_provider_from_theta(theta, events)
    edges = same_stage_one_step_edges(
        occupancy,
        events,
        provider,
        observed_sources_only=True,
        rule="external_longitudinal_one_step",
    )
    inflow = aggregate_inflow(
        occupancy,
        edges,
        rule="external_longitudinal_one_step",
        minimum_state_count=thresholds.minimum_state_count,
        minimum_inflow=thresholds.minimum_inflow,
    )
    try:
        scores, normalizer = compute_relative_dwell(inflow, thresholds)
    except ValueError:
        relaxed = ScoreThresholds(
            minimum_state_count=max(1, thresholds.minimum_state_count - 1),
            minimum_inflow=thresholds.minimum_inflow,
            high_confidence_state_count=max(2, thresholds.high_confidence_state_count - 1),
        )
        scores, normalizer = compute_relative_dwell(inflow, relaxed)
    scores = classify_relative_states(scores)
    return scores.sort_values(["R_star", "N_v"], ascending=[False, False]).reset_index(drop=True), edges, normalizer


def aggregate_timepoint_events(metadata: pd.DataFrame, matrix: pd.DataFrame, events: list[str]) -> pd.DataFrame:
    merged = metadata.merge(matrix, on="sample_id", how="left")
    for event in events:
        merged[event] = pd.to_numeric(merged[event], errors="coerce").fillna(0).astype(int)
    usable = merged[merged["order_evaluable"].astype(bool) & merged["time_rank"].notna()].copy()
    if usable.empty:
        return pd.DataFrame()
    rows = []
    for (patient_id, time_rank), group in usable.groupby(["patient_id", "time_rank"], dropna=True):
        stage = "progressed" if group["stage"].eq("progressed").any() else "baseline"
        event_values = group[events].max(axis=0).astype(int).to_dict()
        genotype = genotype_signature(np.array([event_values[event] for event in events], dtype=int), events)
        rows.append(
            {
                "patient_id": patient_id,
                "time_rank": float(time_rank),
                "stage": stage,
                "sample_count_at_timepoint": int(len(group)),
                "genotype": genotype,
                "state": f"{stage}::{genotype}",
                **event_values,
            }
        )
    return pd.DataFrame(rows).sort_values(["patient_id", "time_rank"]).reset_index(drop=True)


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for value in pos:
        wins += float(np.sum(value > neg)) + 0.5 * float(np.sum(value == neg))
    return wins / (len(pos) * len(neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    positives = int(labels.sum())
    if positives == 0 or len(labels) == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    cumulative_positives = np.cumsum(ordered_labels)
    ranks = np.arange(1, len(ordered_labels) + 1)
    precision_at_rank = cumulative_positives / ranks
    return float(np.sum(precision_at_rank * ordered_labels) / positives)


def safe_rank_correlation(x: np.ndarray, y: np.ndarray, method: str = "spearman") -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or np.unique(x).size <= 1 or np.unique(y).size <= 1:
        return np.nan, np.nan
    try:
        if method == "kendall":
            value, p_value = kendalltau(x, y, nan_policy="omit")
        else:
            value, p_value = spearmanr(x, y, nan_policy="omit")
    except Exception:
        return np.nan, np.nan
    return float(value) if np.isfinite(value) else np.nan, float(p_value) if np.isfinite(p_value) else np.nan


def safe_point_biserial(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if len(labels) < 3 or np.unique(labels).size <= 1 or np.unique(scores).size <= 1:
        return np.nan, np.nan
    try:
        value, p_value = pointbiserialr(labels, scores)
    except Exception:
        return np.nan, np.nan
    return float(value) if np.isfinite(value) else np.nan, float(p_value) if np.isfinite(p_value) else np.nan


def bootstrap_ci(
    evaluable: pd.DataFrame,
    metric_fn,
    seed: int,
    n_bootstrap: int,
) -> tuple[float, float]:
    if evaluable.empty or n_bootstrap <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    patient_ids = evaluable["patient_id"].astype(str).drop_duplicates().to_numpy()
    values = []
    if len(patient_ids) >= 2:
        patient_groups = {patient: group for patient, group in evaluable.groupby(evaluable["patient_id"].astype(str))}
        for _ in range(n_bootstrap):
            sampled_patients = rng.choice(patient_ids, size=len(patient_ids), replace=True)
            sample = pd.concat([patient_groups[patient] for patient in sampled_patients], ignore_index=True)
            value = metric_fn(sample)
            if np.isfinite(value):
                values.append(float(value))
    else:
        row_count = len(evaluable)
        for _ in range(n_bootstrap):
            sample = evaluable.iloc[rng.integers(0, row_count, size=row_count)].copy()
            value = metric_fn(sample)
            if np.isfinite(value):
                values.append(float(value))
    if len(values) < max(20, n_bootstrap // 20):
        return np.nan, np.nan
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


def dwell_contrast(
    study_id: str,
    metadata: pd.DataFrame,
    matrix: pd.DataFrame,
    scores: pd.DataFrame,
    timepoints: pd.DataFrame,
    events: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_lookup = scores.set_index("state")[["R_star", "log2_R_star", "eligible_relobstq"]].to_dict(orient="index")
    merged = metadata.merge(matrix, on="sample_id", how="left")
    for event in events:
        merged[event] = pd.to_numeric(merged[event], errors="coerce").fillna(0).astype(int)
    merged["genotype"] = [genotype_signature(row, events) for row in merged[events].to_numpy(dtype=int)]
    merged["genotype"] = merged["genotype"].map(canonical_genotype)
    merged["state"] = merged["stage"].astype(str).str.lower() + "::" + merged["genotype"]
    merged["R_star"] = merged["state"].map(lambda state: score_lookup.get(state, {}).get("R_star", np.nan))
    merged["log2_R_star"] = merged["state"].map(lambda state: score_lookup.get(state, {}).get("log2_R_star", np.nan))
    merged["eligible_relobstq"] = merged["state"].map(lambda state: score_lookup.get(state, {}).get("eligible_relobstq", False))
    eligible = merged[merged["eligible_relobstq"].astype(bool) & merged["stage"].isin(["baseline", "progressed"])].copy()
    rows = []
    baseline = eligible.loc[eligible["stage"].eq("baseline"), "log2_R_star"].dropna()
    progressed = eligible.loc[eligible["stage"].eq("progressed"), "log2_R_star"].dropna()
    if len(baseline) and len(progressed):
        try:
            p_value = float(mannwhitneyu(progressed, baseline, alternative="two-sided").pvalue)
        except Exception:
            p_value = np.nan
        rows.append(
            {
                "study_id": study_id,
                "baseline_n": int(len(baseline)),
                "progressed_n": int(len(progressed)),
                "baseline_median_log2_R": float(np.median(baseline)),
                "progressed_median_log2_R": float(np.median(progressed)),
                "progressed_minus_baseline_median_log2_R": float(np.median(progressed) - np.median(baseline)),
                "mannwhitney_p": p_value,
            }
        )
    else:
        rows.append(
            {
                "study_id": study_id,
                "baseline_n": int(len(baseline)),
                "progressed_n": int(len(progressed)),
                "baseline_median_log2_R": np.nan,
                "progressed_median_log2_R": np.nan,
                "progressed_minus_baseline_median_log2_R": np.nan,
                "mannwhitney_p": np.nan,
            }
        )

    paired_rows = []
    if not timepoints.empty:
        eligible_scores = scores[scores["eligible_relobstq"].astype(bool)].copy()
        state_log = eligible_scores.set_index("state")["log2_R_star"].to_dict()
        for patient_id, group in timepoints.groupby("patient_id"):
            group = group.sort_values("time_rank")
            if group["time_rank"].nunique() < 2:
                continue
            records = group.to_dict(orient="records")
            for earlier, later in zip(records[:-1], records[1:]):
                early_r = state_log.get(earlier["state"], np.nan)
                late_r = state_log.get(later["state"], np.nan)
                if np.isfinite(early_r) and np.isfinite(late_r):
                    paired_rows.append(
                        {
                            "study_id": study_id,
                            "patient_id": patient_id,
                            "early_state": earlier["state"],
                            "late_state": later["state"],
                            "early_log2_R": float(early_r),
                            "late_log2_R": float(late_r),
                            "delta_late_minus_early_log2_R": float(late_r - early_r),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(paired_rows)


def build_longitudinal_pair_table(
    study_id: str,
    timepoints: pd.DataFrame,
    events: list[str],
) -> pd.DataFrame:
    if timepoints.empty or not events:
        return pd.DataFrame()
    rows = []
    pair_id = 0
    for patient_id, group in timepoints.groupby("patient_id"):
        group = group.sort_values("time_rank")
        if group["time_rank"].nunique() < 2:
            continue
        records = group.to_dict(orient="records")
        for earlier, later in zip(records[:-1], records[1:]):
            early_events = {event for event in events if int(earlier[event]) == 1}
            late_events = {event for event in events if int(later[event]) == 1}
            union = early_events.union(late_events)
            intersection = early_events.intersection(late_events)
            gained = sorted(late_events.difference(early_events))
            lost = sorted(early_events.difference(late_events))
            pair_id += 1
            rows.append(
                {
                    "study_id": study_id,
                    "pair_id": pair_id,
                    "patient_id": patient_id,
                    "early_time_rank": float(earlier["time_rank"]),
                    "late_time_rank": float(later["time_rank"]),
                    "time_interval_rank": float(later["time_rank"]) - float(earlier["time_rank"]),
                    "early_stage": earlier["stage"],
                    "late_stage": later["stage"],
                    "early_genotype": earlier["genotype"],
                    "late_genotype": later["genotype"],
                    "early_state": earlier["state"],
                    "late_state": later["state"],
                    "early_event_count": len(early_events),
                    "late_event_count": len(late_events),
                    "gained_events": "+".join(gained),
                    "lost_events": "+".join(lost),
                    "n_gained": len(gained),
                    "n_lost": len(lost),
                    "empirical_persistent": int(early_events == late_events),
                    "minimum_observed_dwell_interval": (
                        float(later["time_rank"]) - float(earlier["time_rank"])
                        if early_events == late_events
                        else 0.0
                    ),
                    "jaccard_similarity": len(intersection) / max(len(union), 1),
                }
            )
    return pd.DataFrame(rows)


def apply_pair_qc(pairs: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    if pairs.empty:
        return pairs.copy(), {
            "pair_qc_raw_pairs": 0,
            "pair_qc_retained_pairs": 0,
            "pair_qc_excluded_event_loss_pairs": 0,
        }
    work = pairs.copy()
    qc = config["analysis"].get("pair_qc", {})
    excluded_event_loss = 0
    if bool(qc.get("exclude_event_loss", False)) and "n_lost" in work:
        loss_mask = pd.to_numeric(work["n_lost"], errors="coerce").fillna(0).gt(0)
        excluded_event_loss = int(loss_mask.sum())
        work = work[~loss_mask].copy()
    return work.copy(), {
        "pair_qc_raw_pairs": int(len(pairs)),
        "pair_qc_retained_pairs": int(len(work)),
        "pair_qc_excluded_event_loss_pairs": excluded_event_loss,
    }


def weighted_genotype_log2_r(scores: pd.DataFrame) -> dict[str, float]:
    eligible = scores[scores["eligible_relobstq"].astype(bool)].copy()
    if eligible.empty:
        return {}
    eligible["genotype_key"] = eligible["state"].astype(str).str.split("::", n=1).str[1]
    result: dict[str, float] = {}
    for genotype, group in eligible.groupby("genotype_key"):
        values = pd.to_numeric(group["log2_R_star"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        weights = pd.to_numeric(group["N_v"], errors="coerce").fillna(1.0).clip(lower=1.0)
        valid = values.notna()
        if valid.any():
            result[str(genotype)] = float(np.average(values[valid], weights=weights[valid]))
    return result


def score_from_training_patients(
    metadata: pd.DataFrame,
    matrix: pd.DataFrame,
    events: list[str],
    theta: np.ndarray,
    train_patients: set[str],
    config: dict,
) -> tuple[pd.DataFrame, float]:
    train_metadata = metadata[metadata["patient_id"].astype(str).isin(train_patients)].copy()
    train_sample_ids = set(train_metadata["sample_id"].astype(str))
    train_matrix = matrix[matrix["sample_id"].astype(str).isin(train_sample_ids)].copy()
    if train_metadata.empty or train_matrix.empty:
        return pd.DataFrame(), float("nan")
    occupancy = build_occupancy(train_metadata, train_matrix, events)
    if occupancy.empty:
        return pd.DataFrame(), float("nan")
    try:
        scores, _, _ = score_external_states(occupancy, theta, events, config)
    except Exception:
        return pd.DataFrame(), float("nan")
    eligible = scores[scores["eligible_relobstq"].astype(bool)].copy()
    if eligible.empty:
        return scores, float("nan")
    threshold = float(np.nanmedian(eligible["log2_R_star"].replace([np.inf, -np.inf], np.nan)))
    return scores, threshold


def evaluate_rstar_dwell_persistence(
    study_id: str,
    metadata: pd.DataFrame,
    matrix: pd.DataFrame,
    timepoints: pd.DataFrame,
    theta: np.ndarray,
    events: list[str],
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_pairs = build_longitudinal_pair_table(study_id, timepoints, events)
    _, empty_pair_qc = apply_pair_qc(raw_pairs, config)
    if raw_pairs.empty:
        return raw_pairs, pd.DataFrame(
            [
                {
                    "study_id": study_id,
                    "total_ordered_pairs": int(len(raw_pairs)),
                    **empty_pair_qc,
                    "evaluable_pairs": 0,
                    "persistent_pairs": 0,
                    "changed_pairs": 0,
                    "persistence_rate": np.nan,
                    "auc": np.nan,
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "high_rstar_persistence_rate": np.nan,
                    "low_rstar_persistence_rate": np.nan,
                    "delta_persistence_rate_high_minus_low": np.nan,
                    "high_rstar_pair_count": 0,
                    "low_rstar_pair_count": 0,
                    "rstar_contrast_rule": "not_evaluable",
                    "spearman_r_jaccard": np.nan,
                    "spearman_p_jaccard": np.nan,
                    "spearman_r_interval": np.nan,
                    "spearman_p_interval": np.nan,
                    "spearman_r_minimum_dwell_interval": np.nan,
                    "spearman_p_minimum_dwell_interval": np.nan,
                    "median_interval_high_R": np.nan,
                    "median_interval_low_R": np.nan,
                    "delta_interval_high_minus_low": np.nan,
                    "mean_minimum_dwell_high_R": np.nan,
                    "mean_minimum_dwell_low_R": np.nan,
                    "delta_minimum_dwell_high_minus_low": np.nan,
                    "minimum_dwell_mannwhitney_p": np.nan,
                    "interval_mannwhitney_p": np.nan,
                    "median_log2_R_persistent": np.nan,
                    "median_log2_R_changed": np.nan,
                    "mannwhitney_p": np.nan,
                    "exact_state_score_fraction": np.nan,
                }
            ]
        )

    pairs = raw_pairs.copy()
    pair_patients = pairs["patient_id"].astype(str).drop_duplicates().tolist()
    rng = np.random.default_rng(int(config["random_seed"]) + 17 + len(pair_patients))
    rng.shuffle(pair_patients)
    n_folds = min(5, max(2, len(pair_patients))) if len(pair_patients) >= 2 else 1
    fold_lookup = {patient: index % n_folds for index, patient in enumerate(pair_patients)}
    all_patients = set(metadata["patient_id"].astype(str).unique())
    prediction_rows = []

    for fold in range(n_folds):
        validation_patients = {patient for patient, fold_index in fold_lookup.items() if fold_index == fold}
        train_patients = all_patients.difference(validation_patients)
        train_scores, threshold = score_from_training_patients(
            metadata,
            matrix,
            events,
            theta,
            train_patients,
            config,
        )
        if train_scores.empty or not np.isfinite(threshold):
            continue
        eligible = train_scores[train_scores["eligible_relobstq"].astype(bool)].copy()
        state_score = eligible.set_index("state")["log2_R_star"].to_dict()
        genotype_score = weighted_genotype_log2_r(train_scores)
        validation_pairs = pairs[pairs["patient_id"].astype(str).isin(validation_patients)].copy()
        for pair in validation_pairs.to_dict(orient="records"):
            early_state = str(pair["early_state"])
            early_genotype = str(pair["early_genotype"])
            score_source = "missing"
            predicted_score = np.nan
            if early_state in state_score:
                predicted_score = float(state_score[early_state])
                score_source = "exact_state"
            elif early_genotype in genotype_score:
                predicted_score = float(genotype_score[early_genotype])
                score_source = "genotype_weighted"
            predicted_long = np.nan
            if np.isfinite(predicted_score):
                predicted_long = int(predicted_score >= threshold)
            prediction_rows.append(
                {
                    **pair,
                    "fold": fold + 1,
                    "train_median_log2_R_threshold": threshold,
                    "predicted_log2_R": predicted_score,
                    "predicted_long_dwell": predicted_long,
                    "score_source": score_source,
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        predictions = pairs.copy()
        predictions["predicted_log2_R"] = np.nan
        predictions["predicted_long_dwell"] = np.nan
        predictions["score_source"] = "missing"

    filtered_predictions, pair_qc = apply_pair_qc(predictions, config)
    predictions["pair_qc_pass"] = False
    predictions.loc[filtered_predictions.index, "pair_qc_pass"] = True
    evaluable = filtered_predictions[np.isfinite(pd.to_numeric(filtered_predictions["predicted_log2_R"], errors="coerce"))].copy()
    total_pairs = int(len(raw_pairs))
    if evaluable.empty:
        summary = pd.DataFrame(
            [
                {
                    "study_id": study_id,
                    "total_ordered_pairs": total_pairs,
                    **pair_qc,
                    "evaluable_pairs": 0,
                    "persistent_pairs": 0,
                    "changed_pairs": 0,
                    "persistence_rate": np.nan,
                    "auc": np.nan,
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "high_rstar_persistence_rate": np.nan,
                    "low_rstar_persistence_rate": np.nan,
                    "delta_persistence_rate_high_minus_low": np.nan,
                    "high_rstar_pair_count": 0,
                    "low_rstar_pair_count": 0,
                    "rstar_contrast_rule": "not_evaluable",
                    "spearman_r_jaccard": np.nan,
                    "spearman_p_jaccard": np.nan,
                    "spearman_r_interval": np.nan,
                    "spearman_p_interval": np.nan,
                    "spearman_r_minimum_dwell_interval": np.nan,
                    "spearman_p_minimum_dwell_interval": np.nan,
                    "median_interval_high_R": np.nan,
                    "median_interval_low_R": np.nan,
                    "delta_interval_high_minus_low": np.nan,
                    "mean_minimum_dwell_high_R": np.nan,
                    "mean_minimum_dwell_low_R": np.nan,
                    "delta_minimum_dwell_high_minus_low": np.nan,
                    "minimum_dwell_mannwhitney_p": np.nan,
                    "interval_mannwhitney_p": np.nan,
                    "median_log2_R_persistent": np.nan,
                    "median_log2_R_changed": np.nan,
                    "mannwhitney_p": np.nan,
                    "exact_state_score_fraction": np.nan,
                }
            ]
        )
        return predictions, summary

    labels = evaluable["empirical_persistent"].astype(int).to_numpy()
    score_values = evaluable["predicted_log2_R"].astype(float).to_numpy()
    pred_long = evaluable["predicted_long_dwell"].astype(int).to_numpy()
    persistent_pairs = int(labels.sum())
    changed_pairs = int(len(labels) - labels.sum())
    auc = binary_auc(labels, score_values) if persistent_pairs and changed_pairs else np.nan
    accuracy = float(np.mean(pred_long == labels)) if len(labels) else np.nan
    if persistent_pairs and changed_pairs:
        sensitivity = float(np.mean(pred_long[labels == 1] == 1))
        specificity = float(np.mean(pred_long[labels == 0] == 0))
        balanced_accuracy = (sensitivity + specificity) / 2.0
    else:
        balanced_accuracy = np.nan
    evaluable["rstar_contrast_group"] = "middle"
    score_series = evaluable["predicted_log2_R"].astype(float)
    low_cut = float(score_series.quantile(1.0 / 3.0))
    high_cut = float(score_series.quantile(2.0 / 3.0))
    contrast_rule = "outer_tertile_predicted_log2_R"
    if np.isfinite(low_cut) and np.isfinite(high_cut) and low_cut < high_cut:
        low_mask = score_series.le(low_cut)
        high_mask = score_series.ge(high_cut)
    else:
        median_cut = float(score_series.median())
        contrast_rule = "median_predicted_log2_R"
        low_mask = score_series.lt(median_cut)
        high_mask = score_series.ge(median_cut)
    evaluable.loc[low_mask, "rstar_contrast_group"] = "low"
    evaluable.loc[high_mask, "rstar_contrast_group"] = "high"
    predictions["rstar_contrast_group"] = "not_evaluable"
    predictions.loc[evaluable.index, "rstar_contrast_group"] = evaluable["rstar_contrast_group"]
    high = evaluable[evaluable["rstar_contrast_group"].eq("high")]
    low = evaluable[evaluable["rstar_contrast_group"].eq("low")]
    high_rate = float(high["empirical_persistent"].mean()) if not high.empty else np.nan
    low_rate = float(low["empirical_persistent"].mean()) if not low.empty else np.nan
    delta_rate = high_rate - low_rate if np.isfinite(high_rate) and np.isfinite(low_rate) else np.nan
    base_rate = float(labels.mean()) if len(labels) else np.nan
    average_precision_value = average_precision(labels, score_values)
    average_precision_lift = (
        average_precision_value / base_rate
        if np.isfinite(average_precision_value) and np.isfinite(base_rate) and base_rate > 0
        else np.nan
    )
    rank_biserial = 2.0 * auc - 1.0 if np.isfinite(auc) else np.nan
    point_biserial_r, point_biserial_p = safe_point_biserial(labels, score_values)
    jaccard_values = evaluable["jaccard_similarity"].astype(float).to_numpy()
    rho, rho_p = safe_rank_correlation(score_values, jaccard_values, method="spearman")
    kendall_state_tau, kendall_state_p = safe_rank_correlation(score_values, jaccard_values, method="kendall")
    high_persistent = int(high["empirical_persistent"].astype(int).sum()) if not high.empty else 0
    high_changed = int(len(high) - high_persistent)
    low_persistent = int(low["empirical_persistent"].astype(int).sum()) if not low.empty else 0
    low_changed = int(len(low) - low_persistent)
    if np.isfinite(high_rate) and np.isfinite(low_rate) and low_rate > 0:
        top_bottom_risk_ratio = high_rate / low_rate
    elif np.isfinite(high_rate) and high_rate > 0 and np.isfinite(low_rate) and low_rate == 0:
        top_bottom_risk_ratio = np.inf
    else:
        top_bottom_risk_ratio = np.nan
    if len(high) and len(low):
        top_bottom_odds_ratio = ((high_persistent + 0.5) * (low_changed + 0.5)) / (
            (high_changed + 0.5) * (low_persistent + 0.5)
        )
        try:
            _, top_bottom_fisher_p = fisher_exact(
                [[high_persistent, high_changed], [low_persistent, low_changed]],
                alternative="greater",
            )
        except Exception:
            top_bottom_fisher_p = np.nan
    else:
        top_bottom_odds_ratio = np.nan
        top_bottom_fisher_p = np.nan
    top_rstar_lift_over_base = (
        high_rate / base_rate if np.isfinite(high_rate) and np.isfinite(base_rate) and base_rate > 0 else np.nan
    )
    bottom_rstar_lift_over_base = (
        low_rate / base_rate if np.isfinite(low_rate) and np.isfinite(base_rate) and base_rate > 0 else np.nan
    )
    persistent_scores = evaluable.loc[evaluable["empirical_persistent"].astype(int).eq(1), "predicted_log2_R"]
    changed_scores = evaluable.loc[evaluable["empirical_persistent"].astype(int).eq(0), "predicted_log2_R"]
    if len(persistent_scores) and len(changed_scores):
        mw_p = float(mannwhitneyu(persistent_scores, changed_scores, alternative="greater").pvalue)
    else:
        mw_p = np.nan
    interval_valid = evaluable[
        np.isfinite(pd.to_numeric(evaluable["time_interval_rank"], errors="coerce"))
        & pd.to_numeric(evaluable["time_interval_rank"], errors="coerce").gt(0)
    ].copy()
    interval_scores = interval_valid["predicted_log2_R"].astype(float).to_numpy()
    interval_values = np.log1p(interval_valid["time_interval_rank"].astype(float).to_numpy())
    if len(interval_valid) >= 3 and np.unique(interval_scores).size > 1 and np.unique(interval_values).size > 1:
        try:
            interval_rho, interval_p = spearmanr(interval_scores, interval_values, nan_policy="omit")
        except Exception:
            interval_rho, interval_p = np.nan, np.nan
    else:
        interval_rho, interval_p = np.nan, np.nan
    high_interval = high[
        np.isfinite(pd.to_numeric(high["time_interval_rank"], errors="coerce"))
        & pd.to_numeric(high["time_interval_rank"], errors="coerce").gt(0)
    ]["time_interval_rank"].astype(float)
    low_interval = low[
        np.isfinite(pd.to_numeric(low["time_interval_rank"], errors="coerce"))
        & pd.to_numeric(low["time_interval_rank"], errors="coerce").gt(0)
    ]["time_interval_rank"].astype(float)
    median_high_interval = float(np.median(high_interval)) if len(high_interval) else np.nan
    median_low_interval = float(np.median(low_interval)) if len(low_interval) else np.nan
    delta_interval = (
        median_high_interval - median_low_interval
        if np.isfinite(median_high_interval) and np.isfinite(median_low_interval)
        else np.nan
    )
    if len(high_interval) and len(low_interval):
        interval_mw_p = float(mannwhitneyu(high_interval, low_interval, alternative="greater").pvalue)
    else:
        interval_mw_p = np.nan
    minimum_dwell_valid = evaluable[
        np.isfinite(pd.to_numeric(evaluable["minimum_observed_dwell_interval"], errors="coerce"))
    ].copy()
    minimum_dwell_scores = minimum_dwell_valid["predicted_log2_R"].astype(float).to_numpy()
    minimum_dwell_values = np.log1p(
        minimum_dwell_valid["minimum_observed_dwell_interval"].astype(float).to_numpy()
    )
    if (
        len(minimum_dwell_valid) >= 3
        and np.unique(minimum_dwell_scores).size > 1
        and np.unique(minimum_dwell_values).size > 1
    ):
        try:
            minimum_dwell_rho, minimum_dwell_p = spearmanr(minimum_dwell_scores, minimum_dwell_values, nan_policy="omit")
        except Exception:
            minimum_dwell_rho, minimum_dwell_p = np.nan, np.nan
    else:
        minimum_dwell_rho, minimum_dwell_p = np.nan, np.nan
    minimum_dwell_tau, minimum_dwell_tau_p = safe_rank_correlation(
        minimum_dwell_scores,
        minimum_dwell_values,
        method="kendall",
    )
    high_minimum_dwell = high[
        np.isfinite(pd.to_numeric(high["minimum_observed_dwell_interval"], errors="coerce"))
    ]["minimum_observed_dwell_interval"].astype(float)
    low_minimum_dwell = low[
        np.isfinite(pd.to_numeric(low["minimum_observed_dwell_interval"], errors="coerce"))
    ]["minimum_observed_dwell_interval"].astype(float)
    mean_high_minimum_dwell = float(np.mean(high_minimum_dwell)) if len(high_minimum_dwell) else np.nan
    mean_low_minimum_dwell = float(np.mean(low_minimum_dwell)) if len(low_minimum_dwell) else np.nan
    delta_minimum_dwell = (
        mean_high_minimum_dwell - mean_low_minimum_dwell
        if np.isfinite(mean_high_minimum_dwell) and np.isfinite(mean_low_minimum_dwell)
        else np.nan
    )
    if len(high_minimum_dwell) and len(low_minimum_dwell):
        minimum_dwell_mw_p = float(mannwhitneyu(high_minimum_dwell, low_minimum_dwell, alternative="greater").pvalue)
    else:
        minimum_dwell_mw_p = np.nan
    n_bootstrap = int(config["analysis"].get("bootstrap_replicates", 0))
    bootstrap_seed = int(config["random_seed"]) + int(hashlib.sha256(study_id.encode("utf-8")).hexdigest()[:8], 16) % 10_000

    def _auc_metric(frame: pd.DataFrame) -> float:
        y = frame["empirical_persistent"].astype(int).to_numpy()
        s = frame["predicted_log2_R"].astype(float).to_numpy()
        return binary_auc(y, s) if y.sum() and len(y) > y.sum() else np.nan

    def _ap_metric(frame: pd.DataFrame) -> float:
        return average_precision(
            frame["empirical_persistent"].astype(int).to_numpy(),
            frame["predicted_log2_R"].astype(float).to_numpy(),
        )

    def _delta_metric(frame: pd.DataFrame) -> float:
        local_scores = frame["predicted_log2_R"].astype(float)
        local_low_cut = float(local_scores.quantile(1.0 / 3.0))
        local_high_cut = float(local_scores.quantile(2.0 / 3.0))
        if np.isfinite(local_low_cut) and np.isfinite(local_high_cut) and local_low_cut < local_high_cut:
            local_low = frame[local_scores.le(local_low_cut)]
            local_high = frame[local_scores.ge(local_high_cut)]
        else:
            local_median = float(local_scores.median())
            local_low = frame[local_scores.lt(local_median)]
            local_high = frame[local_scores.ge(local_median)]
        if local_low.empty or local_high.empty:
            return np.nan
        return float(local_high["empirical_persistent"].mean() - local_low["empirical_persistent"].mean())

    def _minimum_dwell_rho_metric(frame: pd.DataFrame) -> float:
        value, _ = safe_rank_correlation(
            frame["predicted_log2_R"].astype(float).to_numpy(),
            np.log1p(frame["minimum_observed_dwell_interval"].astype(float).to_numpy()),
            method="spearman",
        )
        return value

    auc_ci_low, auc_ci_high = bootstrap_ci(evaluable, _auc_metric, bootstrap_seed + 1, n_bootstrap)
    ap_ci_low, ap_ci_high = bootstrap_ci(evaluable, _ap_metric, bootstrap_seed + 2, n_bootstrap)
    delta_ci_low, delta_ci_high = bootstrap_ci(evaluable, _delta_metric, bootstrap_seed + 3, n_bootstrap)
    dwell_rho_ci_low, dwell_rho_ci_high = bootstrap_ci(
        evaluable,
        _minimum_dwell_rho_metric,
        bootstrap_seed + 4,
        n_bootstrap,
    )
    summary = pd.DataFrame(
        [
            {
                "study_id": study_id,
                "total_ordered_pairs": total_pairs,
                **pair_qc,
                "evaluable_pairs": int(len(evaluable)),
                "persistent_pairs": persistent_pairs,
                "changed_pairs": changed_pairs,
                "persistence_rate": float(labels.mean()) if len(labels) else np.nan,
                "auc": auc,
                "auc_ci_low": auc_ci_low,
                "auc_ci_high": auc_ci_high,
                "rank_biserial_persistence": rank_biserial,
                "average_precision": average_precision_value,
                "average_precision_ci_low": ap_ci_low,
                "average_precision_ci_high": ap_ci_high,
                "average_precision_lift": average_precision_lift,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "high_rstar_persistence_rate": high_rate,
                "low_rstar_persistence_rate": low_rate,
                "delta_persistence_rate_high_minus_low": delta_rate,
                "delta_persistence_ci_low": delta_ci_low,
                "delta_persistence_ci_high": delta_ci_high,
                "high_rstar_pair_count": int(len(high)),
                "low_rstar_pair_count": int(len(low)),
                "rstar_contrast_rule": contrast_rule,
                "top_bottom_risk_ratio": top_bottom_risk_ratio,
                "top_bottom_odds_ratio": top_bottom_odds_ratio,
                "top_bottom_fisher_p": top_bottom_fisher_p,
                "top_rstar_lift_over_base": top_rstar_lift_over_base,
                "bottom_rstar_lift_over_base": bottom_rstar_lift_over_base,
                "point_biserial_r_persistence": point_biserial_r,
                "point_biserial_p_persistence": point_biserial_p,
                "spearman_r_jaccard": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p_jaccard": float(rho_p) if np.isfinite(rho_p) else np.nan,
                "kendall_tau_jaccard": kendall_state_tau,
                "kendall_p_jaccard": kendall_state_p,
                "spearman_r_interval": float(interval_rho) if np.isfinite(interval_rho) else np.nan,
                "spearman_p_interval": float(interval_p) if np.isfinite(interval_p) else np.nan,
                "spearman_r_minimum_dwell_interval": (
                    float(minimum_dwell_rho) if np.isfinite(minimum_dwell_rho) else np.nan
                ),
                "spearman_p_minimum_dwell_interval": (
                    float(minimum_dwell_p) if np.isfinite(minimum_dwell_p) else np.nan
                ),
                "spearman_r_minimum_dwell_ci_low": dwell_rho_ci_low,
                "spearman_r_minimum_dwell_ci_high": dwell_rho_ci_high,
                "kendall_tau_minimum_dwell_interval": minimum_dwell_tau,
                "kendall_p_minimum_dwell_interval": minimum_dwell_tau_p,
                "median_interval_high_R": median_high_interval,
                "median_interval_low_R": median_low_interval,
                "delta_interval_high_minus_low": delta_interval,
                "mean_minimum_dwell_high_R": mean_high_minimum_dwell,
                "mean_minimum_dwell_low_R": mean_low_minimum_dwell,
                "delta_minimum_dwell_high_minus_low": delta_minimum_dwell,
                "minimum_dwell_mannwhitney_p": minimum_dwell_mw_p,
                "interval_mannwhitney_p": interval_mw_p,
                "median_log2_R_persistent": float(np.median(persistent_scores)) if len(persistent_scores) else np.nan,
                "median_log2_R_changed": float(np.median(changed_scores)) if len(changed_scores) else np.nan,
                "mannwhitney_p": mw_p,
                "exact_state_score_fraction": float(evaluable["score_source"].eq("exact_state").mean()),
            }
        ]
    )
    return predictions, summary


def make_metric_audit(dwell_persistence: pd.DataFrame) -> pd.DataFrame:
    definitions = [
        {
            "metric": "auc",
            "label": "AUC",
            "role": "primary",
            "direction": "higher_than_0.5",
            "rationale": "Threshold-free discrimination of persistent versus changed adjacent states.",
            "minimum_evidence_rule": "Primary retained cohorts >0.5.",
        },
        {
            "metric": "delta_persistence_rate_high_minus_low",
            "label": "Top-bottom persistent difference",
            "role": "primary",
            "direction": "positive",
            "rationale": "Directly compares empirical persistence in the highest versus lowest predicted R* terciles.",
            "minimum_evidence_rule": "Primary retained cohorts positive.",
        },
        {
            "metric": "spearman_r_minimum_dwell_interval",
            "label": "Spearman rho minimum dwell",
            "role": "primary",
            "direction": "positive",
            "rationale": "Ranks R* against a conservative lower bound of observed dwell interval.",
            "minimum_evidence_rule": "Primary retained cohorts positive.",
        },
        {
            "metric": "average_precision_lift",
            "label": "Average precision lift",
            "role": "supportive",
            "direction": "higher_than_1",
            "rationale": "Evaluates enrichment of persistent states near the top of the R* ranking relative to prevalence.",
            "minimum_evidence_rule": "At least one retained cohort >1 and no coherent contradiction.",
        },
        {
            "metric": "spearman_r_jaccard",
            "label": "Spearman rho state similarity",
            "role": "supportive",
            "direction": "positive",
            "rationale": "Tests whether high R* aligns with broader genotype similarity, not only exact persistence.",
            "minimum_evidence_rule": "Primary retained cohorts positive.",
        },
        {
            "metric": "rank_biserial_persistence",
            "label": "Rank-biserial effect",
            "role": "supportive",
            "direction": "positive",
            "rationale": "Effect-size transform of AUC, easier to interpret as a rank advantage.",
            "minimum_evidence_rule": "Primary retained cohorts positive.",
        },
        {
            "metric": "top_bottom_risk_ratio",
            "label": "Top-bottom persistence risk ratio",
            "role": "supportive",
            "direction": "higher_than_1",
            "rationale": "Fold-change version of the top-versus-bottom R* persistence comparison.",
            "minimum_evidence_rule": "Primary retained cohorts >1.",
        },
        {
            "metric": "minimum_dwell_mannwhitney_p",
            "label": "Minimum dwell one-sided Mann-Whitney P",
            "role": "supportive",
            "direction": "lower_is_better",
            "rationale": "Tests whether top R* pairs have larger conservative minimum dwell intervals than bottom R* pairs.",
            "minimum_evidence_rule": "Useful as uncertainty annotation, not as a standalone main claim.",
        },
        {
            "metric": "balanced_accuracy",
            "label": "Balanced accuracy",
            "role": "secondary",
            "direction": "higher_than_0.5",
            "rationale": "Depends on an arbitrary median threshold; useful QC but less aligned with relative ranking.",
            "minimum_evidence_rule": "Do not use as a main positive claim if class balance is poor.",
        },
        {
            "metric": "accuracy",
            "label": "Accuracy",
            "role": "not_recommended_main",
            "direction": "higher_than_base_rate",
            "rationale": "Strongly distorted by persistent-class prevalence in small longitudinal cohorts.",
            "minimum_evidence_rule": "Report only descriptively.",
        },
        {
            "metric": "spearman_r_interval",
            "label": "Spearman rho raw interval",
            "role": "not_recommended_main",
            "direction": "positive",
            "rationale": "Raw follow-up interval does not require the state to persist and is therefore less specific to dwell time.",
            "minimum_evidence_rule": "Do not use as primary evidence.",
        },
    ]
    rows = []
    for item in definitions:
        metric = item["metric"]
        values = []
        if metric in dwell_persistence.columns:
            values = pd.to_numeric(dwell_persistence[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().tolist()
        if item["direction"] == "higher_than_0.5":
            supportive_count = sum(value > 0.5 for value in values)
        elif item["direction"] == "higher_than_1":
            supportive_count = sum(value > 1.0 for value in values)
        elif item["direction"] == "lower_is_better":
            supportive_count = sum(value < 0.05 for value in values)
        elif item["direction"] == "higher_than_base_rate":
            supportive_count = 0
            if metric in dwell_persistence.columns and "persistence_rate" in dwell_persistence.columns:
                for _, row in dwell_persistence.iterrows():
                    value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
                    base = pd.to_numeric(pd.Series([row.get("persistence_rate")]), errors="coerce").iloc[0]
                    if np.isfinite(value) and np.isfinite(base) and value > base:
                        supportive_count += 1
        else:
            supportive_count = sum(value > 0 for value in values)
        evaluable_count = len(values)
        if evaluable_count:
            median_value = float(np.nanmedian(values))
            min_value = float(np.nanmin(values))
            max_value = float(np.nanmax(values))
        else:
            median_value = min_value = max_value = np.nan
        if item["role"] == "primary":
            recommendation = "main"
        elif item["role"] == "supportive":
            recommendation = "supplement"
        elif item["role"] == "secondary":
            recommendation = "qc_only"
        else:
            recommendation = "descriptive_only"
        rows.append(
            {
                **item,
                "evaluable_cohorts": evaluable_count,
                "supportive_cohorts": supportive_count,
                "median_value": median_value,
                "min_value": min_value,
                "max_value": max_value,
                "recommendation": recommendation,
            }
        )
    return pd.DataFrame(rows)


def process_study(study_id: str, study_config: dict, config: dict, result_root: Path) -> dict:
    local_config = config_for_study(config, study_config)
    seed_overrides = config.get("analysis", {}).get("validation_seed_overrides", {})
    validation_seed = int(seed_overrides.get(study_id, config["random_seed"]))
    local_config["random_seed"] = validation_seed
    study_dir = Path(config["data_root"]) / study_id
    tables_dir = result_root / study_id / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    sample_df = read_cbio_table(study_dir / "data_clinical_sample.txt")
    metadata = infer_temporal_metadata(study_id, study_dir, sample_df)
    mutations = load_driver_mutations(study_id, study_dir, metadata)
    events, event_support = select_events(study_id, metadata, mutations, local_config)
    selected_mutations = mutations[mutations["gene"].isin(events)].copy()
    matrix = build_event_matrix(metadata, selected_mutations, events)
    occupancy = build_occupancy(metadata, matrix, events)
    study_seed = int(hashlib.sha256(study_id.encode("utf-8")).hexdigest()[:8], 16) % 10_000
    theta, fit_meta = fit_or_build_theta(matrix, events, local_config, validation_seed + study_seed)
    scores, edges, normalizer = score_external_states(occupancy, theta, events, local_config)
    timepoints = aggregate_timepoint_events(metadata, matrix, events)
    dwell_predictions, dwell_persistence = evaluate_rstar_dwell_persistence(
        study_id,
        metadata,
        matrix,
        timepoints,
        theta,
        events,
        local_config,
    )
    dwell, paired_dwell = dwell_contrast(study_id, metadata, matrix, scores, timepoints, events)

    metadata.to_csv(tables_dir / "sample_metadata.tsv", sep="\t", index=False)
    mutations.to_csv(tables_dir / "driver_mutations_long.tsv", sep="\t", index=False)
    event_support.to_csv(tables_dir / "event_support.tsv", sep="\t", index=False)
    matrix.to_csv(tables_dir / "event_matrix.tsv", sep="\t", index=False)
    occupancy.to_csv(tables_dir / "state_occupancy.tsv", sep="\t", index=False)
    pd.DataFrame(theta, index=events, columns=events).rename_axis("target_event").to_csv(tables_dir / "theta.tsv", sep="\t")
    scores.to_csv(tables_dir / "state_scores.tsv", sep="\t", index=False)
    edges.to_csv(tables_dir / "state_edges.tsv", sep="\t", index=False)
    timepoints.to_csv(tables_dir / "ordered_timepoint_states.tsv", sep="\t", index=False)
    dwell_predictions.to_csv(tables_dir / "dwell_persistence_predictions.tsv", sep="\t", index=False)
    dwell_persistence.to_csv(tables_dir / "dwell_persistence_summary.tsv", sep="\t", index=False)
    dwell.to_csv(tables_dir / "dwell_stage_contrast.tsv", sep="\t", index=False)
    paired_dwell.to_csv(tables_dir / "paired_dwell_delta.tsv", sep="\t", index=False)
    (tables_dir / "fit_metadata.json").write_text(json.dumps(fit_meta, indent=2), encoding="utf-8")

    multi_sample = metadata.groupby("patient_id")["sample_id"].nunique()
    ordered_pairs = int(dwell_persistence["evaluable_pairs"].iloc[0]) if not dwell_persistence.empty else 0
    raw_ordered_pairs = int(dwell_persistence["total_ordered_pairs"].iloc[0]) if not dwell_persistence.empty else 0
    retained_pair_qc = int(dwell_persistence["pair_qc_retained_pairs"].iloc[0]) if not dwell_persistence.empty else 0
    excluded_loss_pairs = (
        int(dwell_persistence["pair_qc_excluded_event_loss_pairs"].iloc[0]) if not dwell_persistence.empty else 0
    )
    eligible_scores = scores[scores["eligible_relobstq"].astype(bool)].copy()
    qc = {
        "study_id": study_id,
        "short_name": study_config["short_name"],
        "display_name": study_config["display_name"],
        "samples": int(metadata["sample_id"].nunique()),
        "patients": int(metadata["patient_id"].nunique()),
        "multi_sample_patients": int((multi_sample > 1).sum()),
        "raw_ordered_pair_count": int(raw_ordered_pairs),
        "pair_qc_retained_count": int(retained_pair_qc),
        "pair_qc_excluded_loss_count": int(excluded_loss_pairs),
        "ordered_pair_count": int(ordered_pairs),
        "selected_events": int(len(events)),
        "validation_random_seed": validation_seed,
        "analysis_max_events": int(local_config["analysis"]["max_events"]),
        "analysis_min_state_count": int(local_config["analysis"]["min_state_count"]),
        "observed_states": int(len(occupancy)),
        "eligible_states": int(scores["eligible_relobstq"].astype(bool).sum()) if "eligible_relobstq" in scores else 0,
        "top_R_star": float(eligible_scores["R_star"].replace([np.inf, -np.inf], np.nan).max()) if not eligible_scores.empty else np.nan,
        "median_R_star": float(eligible_scores["R_star"].median()) if not eligible_scores.empty else np.nan,
        "rstar_normalizer": float(normalizer),
        "backend": fit_meta["backend"],
        "fit_status": fit_meta["fit_status"],
    }
    return {
        "qc": qc,
        "scores": scores.assign(study_id=study_id, short_name=study_config["short_name"]),
        "dwell": dwell,
        "paired_dwell": paired_dwell,
        "dwell_predictions": dwell_predictions,
        "dwell_persistence": dwell_persistence,
    }


def run_selected_longitudinal(config: dict) -> dict[str, pd.DataFrame]:
    """Write selected-version E17 numerical tables and return aggregate tables.

    Relative paths resolve from the current working directory. This function
    does not regenerate or certify the provenance of frozen manuscript outputs.
    """
    result_root = Path(config["result_root"])
    (result_root / "tables").mkdir(parents=True, exist_ok=True)

    outputs = {}
    for study_id, study_config in config["studies"].items():
        outputs[study_id] = process_study(study_id, study_config, config, result_root)

    cohort_qc = pd.DataFrame([value["qc"] for value in outputs.values()])
    cohort_qc.to_csv(result_root / "tables" / "cohort_qc.tsv", sep="\t", index=False)
    scores_all = pd.concat([value["scores"] for value in outputs.values()], ignore_index=True)
    scores_all.to_csv(result_root / "tables" / "state_scores_all.tsv", sep="\t", index=False)
    top_states = (
        scores_all[scores_all["eligible_relobstq"].astype(bool)]
        .sort_values(["study_id", "R_star", "N_v"], ascending=[True, False, False])
        .groupby("study_id", group_keys=False)
        .head(int(config["analysis"]["top_states_per_study"]))
        .reset_index(drop=True)
    )
    top_states.to_csv(result_root / "tables" / "top_external_rstar_states.tsv", sep="\t", index=False)
    dwell = pd.concat([value["dwell"] for value in outputs.values()], ignore_index=True)
    paired_dwell_frames = [value["paired_dwell"] for value in outputs.values() if not value["paired_dwell"].empty]
    paired_dwell = pd.concat(paired_dwell_frames, ignore_index=True) if paired_dwell_frames else pd.DataFrame()
    dwell_prediction_frames = [
        value["dwell_predictions"] for value in outputs.values() if not value["dwell_predictions"].empty
    ]
    dwell_predictions = pd.concat(dwell_prediction_frames, ignore_index=True) if dwell_prediction_frames else pd.DataFrame()
    dwell_persistence = pd.concat([value["dwell_persistence"] for value in outputs.values()], ignore_index=True)
    dwell.to_csv(result_root / "tables" / "dwell_stage_contrast_all.tsv", sep="\t", index=False)
    paired_dwell.to_csv(result_root / "tables" / "paired_dwell_delta_all.tsv", sep="\t", index=False)
    dwell_predictions.to_csv(result_root / "tables" / "dwell_persistence_predictions_all.tsv", sep="\t", index=False)
    dwell_persistence.to_csv(result_root / "tables" / "dwell_persistence_summary_all.tsv", sep="\t", index=False)
    metric_audit_all = make_metric_audit(dwell_persistence)
    metric_audit_all.to_csv(result_root / "tables" / "metric_audit_all_integrated.tsv", sep="\t", index=False)
    primary_studies = set(config["analysis"].get("primary_validation_studies", config["studies"].keys()))
    primary_dwell_persistence = dwell_persistence[dwell_persistence["study_id"].isin(primary_studies)].copy()
    metric_audit = make_metric_audit(primary_dwell_persistence)
    metric_audit.to_csv(result_root / "tables" / "metric_audit.tsv", sep="\t", index=False)

    return {
        "cohort_qc": cohort_qc,
        "state_scores_all": scores_all,
        "top_external_rstar_states": top_states,
        "dwell_stage_contrast_all": dwell,
        "paired_dwell_delta_all": paired_dwell,
        "dwell_persistence_predictions_all": dwell_predictions,
        "dwell_persistence_summary_all": dwell_persistence,
        "metric_audit_all_integrated": metric_audit_all,
        "metric_audit": metric_audit,
    }
