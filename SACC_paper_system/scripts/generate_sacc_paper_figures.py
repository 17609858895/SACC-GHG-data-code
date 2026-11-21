"""Reproduce the SACC analysis and the ordered Fig. 1--9 evidence system.

SACC = Source-aware Accounting-Constrained Compositional Consensus learning.
The script uses only the files in this project. It does not treat accounting
closure as external validation and it does not interpret attribution
fragility as mitigation potential.
"""

from __future__ import annotations

import json
import math
import os
import re
import string
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("OMP_NUM_THREADS", "3")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.text import Text
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SYSTEM_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "DATA"
# Lineage unit used for perturbations and source concentration.
#   "series"   : time-consistent source series (default). The year-specific GHGI
#                Annex tables A-5 to A-9 (2023 to 2019) and the renamed Table
#                3-25/3-25b are merged so that one source series is not split
#                into year fragments.
#   "recorded" : the 103 MetaSources entries exactly as recorded in the release
#                (original submission; reported as a sensitivity analysis).
LINEAGE_LEVEL = os.environ.get("SACC_LINEAGE_LEVEL", "series").strip().lower()
if LINEAGE_LEVEL not in {"series", "recorded"}:
    raise ValueError("SACC_LINEAGE_LEVEL must be 'series' or 'recorded'")
_OUTPUT_SUFFIX = "" if LINEAGE_LEVEL == "series" else "sensitivity_recorded_lineages"
FIG_ROOT = SYSTEM_ROOT / "figures"
TABLE_ROOT = SYSTEM_ROOT / "tables" / _OUTPUT_SUFFIX if _OUTPUT_SUFFIX else SYSTEM_ROOT / "tables"
FIGURE_OUTPUT_ROOT = FIG_ROOT / _OUTPUT_SUFFIX if _OUTPUT_SUFFIX else FIG_ROOT
NOTEBOOK_ROOT = SYSTEM_ROOT / "notebooks"
FIG_ROOT.mkdir(parents=True, exist_ok=True)
TABLE_ROOT.mkdir(parents=True, exist_ok=True)
NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(FIG_ROOT))

from ml_publication_style import (  # noqa: E402
    align_x_ticklabels,
    configure_publication_style,
    numeric_ticks,
    save_figure_with_subfigures,
    style_axis,
)
from scientific_palettes import STYLE_COLORS, get_palette  # noqa: E402


configure_publication_style()
COLORS = get_palette("nature_new", 8)
INK = STYLE_COLORS["ink"]
MUTED = STYLE_COLORS["muted"]
LIGHT = STYLE_COLORS["light"]
EDGE = "#4B5563"
YEARS = [2019, 2020, 2021, 2022, 2023]
SOURCE_ATTENUATION = 0.10
RETENTION_LEVELS = [0.00, 0.10, 0.25, 0.50, 0.75, 1.00]
RANDOM_STATE = 42
PANEL_WIDTH_IN = 4.4
PANEL_HEIGHT_IN = 3.1
MIN_ANNOTATION_FONT = 8.5
MAX_FIGURE_FONT = 13.0
THREE_CATEGORY_BAR_WIDTH = 0.60


# 100-year AR5 global warming potentials. The unspecified aggregate is already
# reported in kg CO2e in the source workbook and therefore receives factor 1.
GWP_AR5 = {
    "Carbon dioxide": 1,
    "Methane": 28,
    "Nitrous oxide": 265,
    "HFC-23": 12400,
    "HFC-32": 677,
    "HFC-125": 3170,
    "HFC-134a": 1300,
    "HFC-143a": 4800,
    "HFC-227ea": 3350,
    "HFC-236fa": 8060,
    "Carbon tetrafluoride": 6630,
    "Hexafluoroethane": 11100,
    "Perfluoropropane": 8900,
    "Perfluorocyclobutane": 9540,
    "Sulfur hexafluoride": 23500,
    "Nitrogen trifluoride": 16100,
    "HFCs and PFCs, unspecified": 1,
}


def gas_group(gas: str) -> str:
    if gas == "Carbon dioxide":
        return "CO$_2$"
    if gas == "Methane":
        return "CH$_4$"
    if gas == "Nitrous oxide":
        return "N$_2$O"
    return "F-gases"


def lineage_series(meta_source: str) -> str:
    """Map a recorded MetaSources entry to its time-consistent source series."""
    name = re.sub(r"^EPA_GHGI_T_A_[5-9]\.", "EPA_GHGI_T_A_5-9.", str(meta_source))
    return name.replace("EPA_GHGI_T_3_25b.", "EPA_GHGI_T_3_25.")


def read_project_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    m2_files = sorted(DATA_ROOT.glob("GHG_national_*_m2_*.parquet"))
    m1_files = sorted(DATA_ROOT.glob("GHG_national_*_m1_*.parquet"))
    if len(m2_files) != 5 or len(m1_files) != 5:
        raise FileNotFoundError("Expected five annual m1 and five annual m2 parquet files")
    m2 = pd.concat([pd.read_parquet(p) for p in m2_files], ignore_index=True)
    m1 = pd.concat([pd.read_parquet(p) for p in m1_files], ignore_index=True)
    for frame in (m1, m2):
        frame["SectorProducedBy"] = frame["SectorProducedBy"].astype(str)
        frame["FlowAmount"] = pd.to_numeric(frame["FlowAmount"], errors="coerce").fillna(0.0)
        missing = sorted(set(frame["Flowable"]) - set(GWP_AR5))
        if missing:
            raise ValueError(f"Unmapped gases: {missing}")
        frame["CO2e_kg"] = frame["FlowAmount"] * frame["Flowable"].map(GWP_AR5)
        frame["GasGroup"] = frame["Flowable"].map(gas_group)
        if "MetaSources" in frame:
            frame["Lineage"] = (
                frame["MetaSources"].map(lineage_series)
                if LINEAGE_LEVEL == "series"
                else frame["MetaSources"]
            )
    sector_file = DATA_ROOT / "GHGs_by_Detailed_Sector_US_2019-2023_v2025.1.0.xlsx"
    sectors = pd.read_excel(sector_file, sheet_name="Sectors", dtype=str)
    sector_names = dict(zip(sectors["Sector"].astype(str), sectors["SectorName"].astype(str)))
    return m2, m1, sector_names


def matrix_from_long(
    frame: pd.DataFrame,
    sectors: list[str],
    gases: list[str],
    value: str = "CO2e_kg",
) -> np.ndarray:
    return (
        frame.pivot_table(
            index="SectorProducedBy",
            columns="Flowable",
            values=value,
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=sectors, columns=gases, fill_value=0.0)
        .to_numpy(float)
    )


def closure_table(m2: pd.DataFrame, m1: pd.DataFrame) -> pd.DataFrame:
    coarse_codes = sorted(m1["SectorProducedBy"].unique(), key=lambda x: (-len(x), x))

    def parent(code: str) -> str | None:
        for candidate in coarse_codes:
            if code.startswith(candidate):
                return candidate
        return None

    detailed = m2[["Year", "SectorProducedBy", "Flowable", "CO2e_kg"]].copy()
    detailed["AggregateSector"] = detailed["SectorProducedBy"].map(parent)
    detailed = (
        detailed.dropna(subset=["AggregateSector"])
        .groupby(["Year", "AggregateSector", "Flowable"], as_index=False)["CO2e_kg"]
        .sum()
        .rename(columns={"CO2e_kg": "m2_reaggregated_kg"})
    )
    coarse = (
        m1.groupby(["Year", "SectorProducedBy", "Flowable"], as_index=False)["CO2e_kg"]
        .sum()
        .rename(columns={"SectorProducedBy": "AggregateSector", "CO2e_kg": "m1_kg"})
    )
    out = coarse.merge(detailed, on=["Year", "AggregateSector", "Flowable"], how="inner")
    denom = np.maximum(np.abs(out["m1_kg"]), 1.0)
    out["relative_error"] = np.abs(out["m2_reaggregated_kg"] - out["m1_kg"]) / denom
    return out


def multiplicative_zero_replacement(composition: np.ndarray) -> np.ndarray:
    comp = np.asarray(composition, dtype=float).copy()
    comp = np.clip(comp, 0.0, None)
    row_sums = comp.sum(axis=1, keepdims=True)
    comp = np.divide(comp, row_sums, out=np.zeros_like(comp), where=row_sums > 0)
    for i in range(comp.shape[0]):
        positive = comp[i] > 0
        if not positive.any():
            comp[i] = 1.0 / comp.shape[1]
            continue
        if positive.all():
            continue
        delta = 0.65 * comp[i, positive].min()
        n_zero = int((~positive).sum())
        replacement_mass = delta * n_zero
        if replacement_mass >= 0.5:
            delta = 0.5 / n_zero
            replacement_mass = 0.5
        comp[i, ~positive] = delta
        comp[i, positive] *= (1.0 - replacement_mass) / comp[i, positive].sum()
    return comp / comp.sum(axis=1, keepdims=True)


def clr_transform(amounts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    composition = multiplicative_zero_replacement(amounts)
    logged = np.log(composition)
    return logged - logged.mean(axis=1, keepdims=True), composition


def cluster_labels(x: np.ndarray, k: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(x)


def constrained_perturb(base: np.ndarray, block: np.ndarray, retention: float) -> np.ndarray:
    """Attenuate one accounting block while exactly restoring every gas total."""
    candidate = base - (1.0 - retention) * block
    candidate = np.clip(candidate, 0.0, None)
    gas_scale = np.divide(
        base.sum(axis=0),
        candidate.sum(axis=0),
        out=np.ones(base.shape[1]),
        where=candidate.sum(axis=0) > 0,
    )
    candidate *= gas_scale[None, :]
    if not np.allclose(candidate.sum(axis=0), base.sum(axis=0), rtol=1e-10, atol=1e-2):
        raise AssertionError("Gas-wise accounting totals were not conserved")
    return candidate


def consensus_labels(label_runs: list[np.ndarray], k: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(label_runs[0])
    consensus = np.zeros((n, n), dtype=float)
    for labels in label_runs:
        consensus += labels[:, None] == labels[None, :]
    consensus /= len(label_runs)
    distance = np.clip(1.0 - consensus, 0.0, 1.0)
    try:
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
    return model.fit_predict(distance), consensus


def align_to_reference(labels: np.ndarray, reference: np.ndarray) -> np.ndarray:
    a = np.unique(labels)
    b = np.unique(reference)
    counts = np.zeros((len(a), len(b)), dtype=int)
    for i, la in enumerate(a):
        for j, lb in enumerate(b):
            counts[i, j] = np.sum((labels == la) & (reference == lb))
    rows, cols = linear_sum_assignment(-counts)
    mapping = {a[i]: b[j] for i, j in zip(rows, cols)}
    return np.array([mapping.get(v, v) for v in labels])


def relabel_by_emissions(labels: np.ndarray, totals: np.ndarray) -> np.ndarray:
    order = sorted(np.unique(labels), key=lambda c: -float(totals[labels == c].mean()))
    mapping = {old: new + 1 for new, old in enumerate(order)}
    return np.array([mapping[x] for x in labels], dtype=int)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    value = spearmanr(x, y, nan_policy="omit").statistic
    return float(value) if np.isfinite(value) else 0.0


def humanize_attribution(name: str) -> str:
    replacements = {
        "BEA Detail Use SUT": "BEA detail use",
        "BEA Detail Supply": "BEA detail supply",
        "BEA Detail GrossOutput IO": "BEA gross output",
        "EIA MECS Energy": "EIA MECS energy",
        "USDA CoA Cropland NAICS": "USDA cropland NAICS",
        "USDA CoA Cropland": "USDA cropland",
        "USGS MYB Lead": "USGS lead",
    }
    clean = name.replace("_", " ")
    return replacements.get(clean, clean)


def compact_attribution(name: str) -> str:
    """Concise family labels for two-column journal artwork."""
    label = humanize_attribution(name)
    replacements = {
        "BEA detail use": "BEA use",
        "BEA detail supply": "BEA supply",
        "BEA gross output": "BEA output",
        "EIA MECS energy": "EIA energy",
        "USDA cropland NAICS": "USDA NAICS",
        "USDA cropland": "USDA land",
    }
    return replacements.get(label, label)


def humanize_meta_source(name: str) -> str:
    clean = name.replace("EPA_GHGI_", "")
    if "." in clean:
        code, description = clean.split(".", 1)
        code = code.replace("T_A_", "TA.").replace("T_", "T").replace("_", ".").replace("5-9", "5–9")
        description = description.replace("_", " ").replace("manufactur", "manufacturing")
        description = description.replace("manufacturinging", "manufacturing")
        description = description.capitalize()
        description = description.replace("Ng manufacturing", "Natural-gas manufacturing")
        description = description.replace("Natural gas manufacturing", "Natural-gas manufacturing")
        return f"{description} ({code})"
    code = clean.replace("T_A_", "TA.").replace("T_", "T").replace("_", ".")
    return f"Direct source ({code})"


def compact_meta_source(name: str) -> str:
    """Short display label for panels that must remain readable at column width."""
    return humanize_meta_source(name).replace("Natural-gas manufacturing", "Natural-gas mfg.")


def compact_sector_name(name: str, width: int = 29) -> str:
    replacements = {
        "State and local government": "State/local government",
        "State/local government electric utilities": "State/local electric utilities",
        "Electric Power Generation, Transmission and Distribution": "Electric power & grid",
        "Water, Sewage and Other Systems": "Water/sewage systems",
        "Natural Gas Distribution": "Natural gas distribution",
        "Air Transportation": "Air transport",
        "Truck Transportation": "Truck transport",
        "Police Protection": "Police protection",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return textwrap.fill(name, width=width, break_long_words=False, break_on_hyphens=False)


@dataclass
class Analysis:
    m2: pd.DataFrame
    m1: pd.DataFrame
    sector_names: dict[str, str]
    sectors: list[str]
    gases: list[str]
    base: np.ndarray
    clr: np.ndarray
    composition: np.ndarray
    source_names: list[str]
    source_matrices: list[np.ndarray]
    perturbed: list[np.ndarray]
    perturbed_clr: list[np.ndarray]
    k_table: pd.DataFrame
    selected_k: int
    labels: np.ndarray
    consensus: np.ndarray
    source_label_runs: list[np.ndarray]
    stability: np.ndarray
    source_hhi: np.ndarray
    fragility: np.ndarray
    priority: np.ndarray
    rank_runs: np.ndarray
    closure: pd.DataFrame
    model_comparison: pd.DataFrame
    retention_sensitivity: pd.DataFrame
    retention_sector_stability: pd.DataFrame
    loo_summary: pd.DataFrame
    loo_assignments: np.ndarray
    loo_cluster_emissions: pd.DataFrame
    family_summary: pd.DataFrame
    family_transition: np.ndarray
    family_origin_change: pd.DataFrame
    most_disruptive_family: str


def compute_retention_sensitivity(
    base: np.ndarray,
    source_matrices: list[np.ndarray],
    selected_k: int,
    reference_labels: np.ndarray,
    source_hhi: np.ndarray,
    reference_priority: np.ndarray,
    sectors: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat the source-lineage stress test across predeclared retention levels."""
    upper = np.triu_indices(len(reference_labels), 1)
    emission_pct = pd.Series(base.sum(axis=1)).rank(pct=True).to_numpy()
    ref_top = set(np.argsort(reference_priority)[-50:])
    summary_rows: list[dict] = []
    sector_rows: list[dict] = []
    for retention in RETENTION_LEVELS:
        runs = []
        infeasible = 0
        for block in source_matrices:
            try:
                candidate = constrained_perturb(base, block, retention)
            except AssertionError:
                infeasible += 1
                continue
            runs.append(cluster_labels(clr_transform(candidate)[0], selected_k))
        if not runs:
            raise RuntimeError(f"No feasible source perturbations at retention={retention}")
        aligned = [align_to_reference(labels, reference_labels) for labels in runs]
        stability = np.mean(np.vstack(aligned) == reference_labels[None, :], axis=0)
        fragility = 1.0 - stability * (1.0 - source_hhi)
        priority = np.sqrt(emission_pct * pd.Series(fragility).rank(pct=True).to_numpy())
        _, consensus = consensus_labels(runs, selected_k)
        aris = np.array([adjusted_rand_score(reference_labels, labels) for labels in runs])
        top = set(np.argsort(priority)[-50:])
        summary_rows.append(
            {
                "source_retention": retention,
                "median_source_ARI": float(np.median(aris)),
                "minimum_source_ARI": float(np.min(aris)),
                "PAC": float(np.mean((consensus[upper] > 0.10) & (consensus[upper] < 0.90))),
                "median_sector_stability": float(np.median(stability)),
                "fraction_stability_ge_0_95": float(np.mean(stability >= 0.95)),
                "priority_rank_spearman": safe_spearman(priority, reference_priority),
                "top50_priority_overlap": len(top & ref_top) / 50.0,
                "evaluated_lineages": len(runs),
                "infeasible_lineages": infeasible,
            }
        )
        sector_rows.extend(
            {
                "source_retention": retention,
                "Sector": sector,
                "PartitionStability": float(value),
            }
            for sector, value in zip(sectors, stability)
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(sector_rows)


def compute_leave_one_year_out(
    m2: pd.DataFrame,
    sectors: list[str],
    gases: list[str],
    source_names: list[str],
    selected_k: int,
    reference_labels: np.ndarray,
    reference_priority: np.ndarray,
    reference_base: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Rebuild the source-consensus solution after omitting each inventory year."""
    upper = np.triu_indices(len(sectors), 1)
    summary_rows: list[dict] = []
    assignments: list[np.ndarray] = []
    cluster_rows: list[dict] = []
    ref_top = set(np.argsort(reference_priority)[-50:])
    full_cluster_gt = {
        c: float(reference_base[reference_labels == c].sum() / 1e12)
        for c in sorted(np.unique(reference_labels))
    }
    for omitted_year in YEARS:
        subset = m2[m2["Year"] != omitted_year]
        mean_long = subset.groupby(["SectorProducedBy", "Flowable"], as_index=False)["CO2e_kg"].sum()
        mean_long["CO2e_kg"] /= len(YEARS) - 1
        base = matrix_from_long(mean_long, sectors, gases)
        clr, _ = clr_transform(base)

        source_long = subset.groupby(
            ["Lineage", "SectorProducedBy", "Flowable"], as_index=False
        )["CO2e_kg"].sum()
        source_long["CO2e_kg"] /= len(YEARS) - 1
        source_matrices = [
            matrix_from_long(source_long[source_long["Lineage"] == source], sectors, gases)
            for source in source_names
        ]
        runs = [
            cluster_labels(clr_transform(constrained_perturb(base, block, SOURCE_ATTENUATION))[0], selected_k)
            for block in source_matrices
        ]
        final0, consensus = consensus_labels(runs, selected_k)
        aligned_final = align_to_reference(final0, reference_labels)
        assignments.append(aligned_final)
        aligned_runs = [align_to_reference(labels, final0) for labels in runs]
        stability = np.mean(np.vstack(aligned_runs) == final0[None, :], axis=0)

        source_sector = np.stack([x.sum(axis=1) for x in source_matrices], axis=1)
        source_share = np.divide(
            source_sector,
            source_sector.sum(axis=1, keepdims=True),
            out=np.zeros_like(source_sector),
            where=source_sector.sum(axis=1, keepdims=True) > 0,
        )
        hhi = np.sum(source_share**2, axis=1)
        fragility = 1.0 - stability * (1.0 - hhi)
        emission_pct = pd.Series(base.sum(axis=1)).rank(pct=True).to_numpy()
        priority = np.sqrt(emission_pct * pd.Series(fragility).rank(pct=True).to_numpy())
        top = set(np.argsort(priority)[-50:])
        summary_rows.append(
            {
                "omitted_year": omitted_year,
                "ARI_vs_full": adjusted_rand_score(reference_labels, final0),
                "assignment_agreement": float(np.mean(aligned_final == reference_labels)),
                "silhouette": silhouette_score(clr, final0),
                "PAC": float(np.mean((consensus[upper] > 0.10) & (consensus[upper] < 0.90))),
                "priority_rank_spearman": safe_spearman(priority, reference_priority),
                "top50_priority_overlap": len(top & ref_top) / 50.0,
            }
        )
        for cluster in sorted(np.unique(reference_labels)):
            gt = float(base[aligned_final == cluster].sum() / 1e12)
            cluster_rows.append(
                {
                    "omitted_year": omitted_year,
                    "Archetype": f"A{cluster}",
                    "AnnualMean_GtCO2e": gt,
                    "change_vs_full_pct": 100.0 * (gt / full_cluster_gt[cluster] - 1.0),
                }
            )
    return pd.DataFrame(summary_rows), np.vstack(assignments), pd.DataFrame(cluster_rows)


def compute_family_stress(
    m2: pd.DataFrame,
    sectors: list[str],
    gases: list[str],
    selected_k: int,
    reference_labels: np.ndarray,
    reference_clr: np.ndarray,
    reference_base: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, str]:
    """Jointly attenuate broad attribution-source families."""
    family_long = m2.groupby(
        ["AttributionSources", "SectorProducedBy", "Flowable"], as_index=False
    )["CO2e_kg"].sum()
    family_long["CO2e_kg"] /= len(YEARS)
    family_names = sorted(family_long["AttributionSources"].unique())
    family_matrices = [
        matrix_from_long(family_long[family_long["AttributionSources"] == family], sectors, gases)
        for family in family_names
    ]
    if not np.allclose(np.sum(family_matrices, axis=0), reference_base, rtol=1e-9, atol=1e-3):
        raise AssertionError("Attribution-source families do not reconstruct the base matrix")

    rows: list[dict] = []
    origin_rows: list[dict] = []
    aligned_results: dict[str, np.ndarray] = {}
    for family, block in zip(family_names, family_matrices):
        candidate = constrained_perturb(reference_base, block, SOURCE_ATTENUATION)
        candidate_clr = clr_transform(candidate)[0]
        labels = cluster_labels(candidate_clr, selected_k)
        aligned = align_to_reference(labels, reference_labels)
        aligned_results[family] = aligned
        shift = np.linalg.norm(candidate_clr - reference_clr, axis=1)
        for cluster in sorted(np.unique(reference_labels)):
            idx = reference_labels == cluster
            changed = aligned[idx] != reference_labels[idx]
            origin_rows.append(
                {
                    "AttributionSource": family,
                    "Archetype": f"A{cluster}",
                    "changed_sectors": int(changed.sum()),
                    "changed_fraction": float(changed.mean()),
                }
            )
        rows.append(
            {
                "AttributionSource": family,
                "CO2e_share": float(block.sum() / reference_base.sum()),
                "ARI_vs_full": adjusted_rand_score(reference_labels, labels),
                "changed_sectors": int(np.sum(aligned != reference_labels)),
                "median_Aitchison_shift": float(np.median(shift)),
                "p95_Aitchison_shift": float(np.percentile(shift, 95)),
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["ARI_vs_full", "changed_sectors"], ascending=[True, False]
    ).reset_index(drop=True)
    disruptive = str(summary.iloc[0]["AttributionSource"])
    transition = np.zeros((selected_k, selected_k), dtype=int)
    aligned = aligned_results[disruptive]
    for original, stressed in zip(reference_labels, aligned):
        transition[int(original) - 1, int(stressed) - 1] += 1
    return summary, transition, pd.DataFrame(origin_rows), disruptive


def run_analysis() -> Analysis:
    m2, m1, sector_names = read_project_data()
    sectors = sorted(m2["SectorProducedBy"].unique())
    gases = sorted(m2["Flowable"].unique())
    n_years = len(YEARS)

    mean_long = (
        m2.groupby(["SectorProducedBy", "Flowable"], as_index=False)["CO2e_kg"].sum()
    )
    mean_long["CO2e_kg"] /= n_years
    base = matrix_from_long(mean_long, sectors, gases)
    clr, composition = clr_transform(base)

    source_long = (
        m2.groupby(["Lineage", "SectorProducedBy", "Flowable"], as_index=False)["CO2e_kg"].sum()
    )
    source_long["CO2e_kg"] /= n_years
    source_names = sorted(source_long["Lineage"].unique())
    source_matrices = [
        matrix_from_long(source_long[source_long["Lineage"] == source], sectors, gases)
        for source in source_names
    ]
    if not np.allclose(np.sum(source_matrices, axis=0), base, rtol=1e-9, atol=1e-3):
        raise AssertionError("Lineage contributions do not reconstruct the five-year mean matrix")

    perturbed: list[np.ndarray] = []
    perturbed_clr: list[np.ndarray] = []
    for source_matrix in source_matrices:
        candidate = constrained_perturb(base, source_matrix, SOURCE_ATTENUATION)
        perturbed.append(candidate)
        perturbed_clr.append(clr_transform(candidate)[0])

    fast_replot = os.environ.get("SACC_FAST_REPLOT", "0") == "1"
    cached_k = TABLE_ROOT / "sacc_k_selection.csv"
    if fast_replot and cached_k.exists():
        k_table = pd.read_csv(cached_k)
        eligible = k_table[k_table["minimum_cluster_size"] >= 20]
        selected_k = int(eligible.sort_values("selection_score", ascending=False).iloc[0]["K"])
        source_label_runs0 = [cluster_labels(x, selected_k) for x in perturbed_clr]
        labels0, consensus = consensus_labels(source_label_runs0, selected_k)
    else:
        rows = []
        k_results: dict[int, tuple[np.ndarray, np.ndarray, list[np.ndarray]]] = {}
        upper = np.triu_indices(len(sectors), 1)
        for k in range(3, 9):
            runs = [cluster_labels(x, k) for x in perturbed_clr]
            final, consensus = consensus_labels(runs, k)
            sizes = np.bincount(final, minlength=k)
            pac = float(np.mean((consensus[upper] > 0.10) & (consensus[upper] < 0.90)))
            aris = [adjusted_rand_score(final, labels) for labels in runs]
            rows.append(
                {
                    "K": k,
                    "silhouette": silhouette_score(clr, final),
                    "davies_bouldin": davies_bouldin_score(clr, final),
                    "PAC": pac,
                    "median_source_ARI": float(np.median(aris)),
                    "minimum_source_ARI": float(np.min(aris)),
                    "minimum_cluster_size": int(sizes.min()),
                    "cluster_sizes": ";".join(map(str, sorted(sizes.tolist(), reverse=True))),
                }
            )
            k_results[k] = (final, consensus, runs)
        k_table = pd.DataFrame(rows)
        k_table["selection_score"] = (
            k_table["silhouette"] - 0.5 * k_table["PAC"] - 0.02 * (k_table["K"] - 3)
        )
        eligible = k_table[k_table["minimum_cluster_size"] >= 20]
        selected_k = int(eligible.sort_values("selection_score", ascending=False).iloc[0]["K"])
        labels0, consensus, source_label_runs0 = k_results[selected_k]
    totals = base.sum(axis=1)
    labels = relabel_by_emissions(labels0, totals)
    source_label_runs = [align_to_reference(r, labels0) for r in source_label_runs0]
    source_label_runs = [
        relabel_by_emissions(r, totals) if set(np.unique(r)) == set(np.unique(labels0)) else r
        for r in source_label_runs
    ]
    # Re-align after relabeling to avoid dependence on arbitrary cluster numbering.
    source_label_runs = [align_to_reference(r, labels) for r in source_label_runs]
    stability = np.mean(np.vstack(source_label_runs) == labels[None, :], axis=0)

    source_sector = np.stack([x.sum(axis=1) for x in source_matrices], axis=1)
    source_share = np.divide(
        source_sector,
        source_sector.sum(axis=1, keepdims=True),
        out=np.zeros_like(source_sector),
        where=source_sector.sum(axis=1, keepdims=True) > 0,
    )
    source_hhi = np.sum(source_share**2, axis=1)
    fragility = 1.0 - stability * (1.0 - source_hhi)
    emission_pct = pd.Series(totals).rank(pct=True).to_numpy()
    fragility_pct = pd.Series(fragility).rank(pct=True).to_numpy()
    priority = np.sqrt(emission_pct * fragility_pct)

    rank_runs = []
    for candidate in perturbed:
        values = candidate.sum(axis=1)
        rank_runs.append(pd.Series(-values).rank(method="average").to_numpy())
    rank_runs = np.vstack(rank_runs)

    cached_comparison = TABLE_ROOT / "sacc_model_comparison.csv"
    if fast_replot and cached_comparison.exists():
        model_comparison = pd.read_csv(cached_comparison)
    else:
        raw_x = np.log1p(base)
        raw_base = cluster_labels(raw_x, selected_k)
        clr_base = cluster_labels(clr, selected_k)
        methods = []
        for method, x, ref, transform in [
            ("Raw log K-means", raw_x, raw_base, lambda z: np.log1p(z)),
            ("CLR K-means", clr, clr_base, lambda z: clr_transform(z)[0]),
        ]:
            aris = [adjusted_rand_score(ref, cluster_labels(transform(z), selected_k)) for z in perturbed]
            methods.append(
                {
                    "method": method,
                    "silhouette": silhouette_score(x, ref),
                    "median_source_ARI": float(np.median(aris)),
                    "minimum_source_ARI": float(np.min(aris)),
                }
            )
        sacc_aris = [adjusted_rand_score(labels, r) for r in source_label_runs]
        methods.append(
            {
                "method": "SACC consensus",
                "silhouette": silhouette_score(clr, labels),
                "median_source_ARI": float(np.median(sacc_aris)),
                "minimum_source_ARI": float(np.min(sacc_aris)),
            }
        )
        model_comparison = pd.DataFrame(methods)

    heavy_cache = {
        "retention": TABLE_ROOT / "source_retention_sensitivity.csv",
        "retention_sector": TABLE_ROOT / "source_retention_sector_stability.csv",
        "loo_summary": TABLE_ROOT / "leave_one_year_out_sensitivity.csv",
        "loo_assignments": TABLE_ROOT / "leave_one_year_out_assignments.csv",
        "loo_emissions": TABLE_ROOT / "leave_one_year_out_archetype_emissions.csv",
        "family": TABLE_ROOT / "attribution_family_stress.csv",
        "family_transition": TABLE_ROOT / "attribution_family_transition.csv",
        "family_origin": TABLE_ROOT / "attribution_family_archetype_change.csv",
    }
    if fast_replot and all(path.exists() for path in heavy_cache.values()):
        retention_sensitivity = pd.read_csv(heavy_cache["retention"])
        retention_sector_stability = pd.read_csv(heavy_cache["retention_sector"])
        loo_summary = pd.read_csv(heavy_cache["loo_summary"])
        loo_table = pd.read_csv(heavy_cache["loo_assignments"], index_col="Sector").reindex(sectors)
        loo_assignments = loo_table[[f"Omit_{year}" for year in YEARS]].to_numpy(int).T
        loo_cluster_emissions = pd.read_csv(heavy_cache["loo_emissions"])
        family_summary = pd.read_csv(heavy_cache["family"])
        family_transition = pd.read_csv(heavy_cache["family_transition"], index_col=0).to_numpy(int)
        family_origin_change = pd.read_csv(heavy_cache["family_origin"])
        most_disruptive_family = str(family_summary.iloc[0]["AttributionSource"])
    else:
        retention_sensitivity, retention_sector_stability = compute_retention_sensitivity(
            base,
            source_matrices,
            selected_k,
            labels,
            source_hhi,
            priority,
            sectors,
        )
        loo_summary, loo_assignments, loo_cluster_emissions = compute_leave_one_year_out(
            m2,
            sectors,
            gases,
            source_names,
            selected_k,
            labels,
            priority,
            base,
        )
        family_summary, family_transition, family_origin_change, most_disruptive_family = compute_family_stress(
            m2,
            sectors,
            gases,
            selected_k,
            labels,
            clr,
            base,
        )

    return Analysis(
        m2=m2,
        m1=m1,
        sector_names=sector_names,
        sectors=sectors,
        gases=gases,
        base=base,
        clr=clr,
        composition=composition,
        source_names=source_names,
        source_matrices=source_matrices,
        perturbed=perturbed,
        perturbed_clr=perturbed_clr,
        k_table=k_table,
        selected_k=selected_k,
        labels=labels,
        consensus=consensus,
        source_label_runs=source_label_runs,
        stability=stability,
        source_hhi=source_hhi,
        fragility=fragility,
        priority=priority,
        rank_runs=rank_runs,
        closure=closure_table(m2, m1),
        model_comparison=model_comparison,
        retention_sensitivity=retention_sensitivity,
        retention_sector_stability=retention_sector_stability,
        loo_summary=loo_summary,
        loo_assignments=loo_assignments,
        loo_cluster_emissions=loo_cluster_emissions,
        family_summary=family_summary,
        family_transition=family_transition,
        family_origin_change=family_origin_change,
        most_disruptive_family=most_disruptive_family,
    )


def style_heatmap(
    ax,
    image,
    xlabels,
    ylabels,
    xlabel=None,
    ylabel=None,
    colorbar=True,
    xrotation=45,
):
    x_ha = "center" if xrotation == 0 else "right"
    ax.set_xticks(
        np.arange(len(xlabels)),
        xlabels,
        rotation=xrotation,
        ha=x_ha,
        rotation_mode="anchor",
    )
    ax.set_yticks(np.arange(len(ylabels)), ylabels)
    ax.tick_params(axis="x", which="major", bottom=True, top=False,
                   direction="out", length=4.2, width=1.1, labelsize=10.5)
    ax.tick_params(axis="y", which="major", left=True, right=False,
                   direction="out", length=4.2, width=1.1, labelsize=10.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("normal")
        label.set_fontstyle("normal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontweight="normal")
    if ylabel:
        ax.set_ylabel(ylabel, fontweight="normal")
    if colorbar:
        cb = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=9.0, width=1.1, length=3.5)
        for label in cb.ax.get_yticklabels():
            label.set_fontweight("normal")


def enforce_readable_text(fig, minimum: float = MIN_ANNOTATION_FONT,
                          maximum: float = MAX_FIGURE_FONT) -> None:
    """Keep figure text compact, readable, aligned, and regular weight.

    Axes labels and tick labels already use the larger shared publication
    style. This pass prevents both illegibly small notes and oversized panel
    annotations after multi-panel assembly, then fixes the final x-tick anchor
    after all panel-specific rotations have been applied.
    """
    for artist in fig.findobj(match=Text):
        artist.set_fontweight("normal")
        if artist.get_text():
            artist.set_fontsize(min(maximum, max(minimum, artist.get_fontsize())))
    for ax in fig.axes:
        align_x_ticklabels(ax)


def build_figure_1(a: Analysis) -> list[Callable]:
    counts = [118, len(a.sectors), len(a.gases), len(a.source_names)]
    count_labels = ["Aggregate\nsectors", "Detailed\nsectors", "Gases", "Source\nlineages"]
    annual = a.m2.groupby(["Year", "GasGroup"], as_index=False)["CO2e_kg"].sum()
    annual["Gt_CO2e"] = annual["CO2e_kg"] / 1e12
    allocation_years = pd.DataFrame(
        {
            2019: [2018, 2017, 2017, 2017, 2017, 2019],
            2020: [2018, 2017, 2017, 2017, 2017, 2020],
            2021: [2018, 2022, 2022, 2017, 2017, 2020],
            2022: [2018, 2022, 2022, 2017, 2017, 2020],
            2023: [2018, 2022, 2022, 2017, 2017, 2020],
        },
        index=["EIA MECS", "USDA cropland", "USDA NAICS", "BEA Make", "BEA Use", "USGS Lead"],
    )
    lags = np.array(allocation_years.columns)[None, :] - allocation_years.to_numpy()
    quality_cols = ["DataReliability", "TemporalCorrelation", "GeographicalCorrelation",
                    "TechnologicalCorrelation", "DataCollection"]
    qcounts = a.m2[quality_cols].round(6).value_counts().sort_values(ascending=False)
    qvalues = np.array(qcounts.index.tolist(), dtype=float)
    active = (
        a.m2.groupby(["Year", "GasGroup", "SectorProducedBy"], as_index=False)["CO2e_kg"]
        .sum()
        .query("CO2e_kg > 0")
        .groupby(["GasGroup", "Year"])["SectorProducedBy"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(index=["CO$_2$", "CH$_4$", "N$_2$O", "F-gases"], columns=YEARS, fill_value=0)
    )
    active_pct = 100 * active / len(a.sectors)
    lineage_totals = a.m2.groupby("Lineage")["CO2e_kg"].sum().sort_values(ascending=False)
    lineage_cumulative = 100 * lineage_totals.cumsum() / lineage_totals.sum()

    def p1(ax):
        bars = ax.bar(range(4), counts, color=COLORS[:4], edgecolor=EDGE, linewidth=1.2)
        ax.set_yscale("log")
        ax.set_xticks(range(4), count_labels)
        for bar, value in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, value * 1.17, f"{value:,}", ha="center",
                    fontsize=12, fontweight="bold", color=INK)
        style_axis(ax, ylabel="Count (log scale)")
        ax.set_ylim(8, 1000)

    def p2(ax):
        for group, color, marker in zip(["CO$_2$", "CH$_4$", "N$_2$O", "F-gases"], COLORS[:4], ["o", "s", "^", "D"]):
            d = annual[annual["GasGroup"] == group]
            ax.plot(d["Year"], d["Gt_CO2e"], marker=marker, lw=2.4, ms=7, color=color,
                    markeredgecolor=EDGE, label=group)
        style_axis(ax, xlabel="Inventory year", ylabel="Emissions (Gt CO$_2$e)")
        ax.set_xticks(YEARS)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4,
                  columnspacing=1.0, handletextpad=0.4)
        numeric_ticks(ax, x=False, y=True, n=5)

    def p3(ax):
        lag_min = min(-1, int(lags.min()))
        lag_max = max(1, int(lags.max()))
        zero_position = (0 - lag_min) / (lag_max - lag_min)
        cmap = LinearSegmentedColormap.from_list(
            "lag",
            [(0.0, "#4E85AC"), (zero_position, "#F7F7F7"), (1.0, "#D26A6A")],
        )
        # A linear normalization keeps each one-year lag interval equally spaced.
        # White is anchored at zero without stretching the negative side.
        norm = Normalize(vmin=lag_min, vmax=lag_max)
        im = ax.imshow(lags, aspect="auto", cmap=cmap, norm=norm)
        for i in range(lags.shape[0]):
            for j in range(lags.shape[1]):
                value = int(lags[i, j])
                text_color = "white" if abs(value) >= 4 else INK
                ax.text(j, i, str(value), ha="center", va="center", fontweight="bold", color=text_color)
        style_heatmap(ax, im, YEARS, allocation_years.index.tolist(), xlabel="Inventory year", colorbar=True)
        colorbar_ax = ax.figure.axes[-1]
        colorbar_ax.set_yticks(np.arange(lag_min, lag_max + 1, 1))
        colorbar_ax.set_ylabel("Inventory year - source year", fontweight="bold")

    def p4(ax):
        cmap = LinearSegmentedColormap.from_list("quality", ["#F7F7F7", COLORS[1], COLORS[4]])
        im = ax.imshow(qvalues, aspect="auto", cmap=cmap, vmin=1, vmax=5)
        for i in range(qvalues.shape[0]):
            for j in range(qvalues.shape[1]):
                ax.text(j, i, f"{qvalues[i,j]:.0f}", ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white" if qvalues[i, j] >= 4 else INK)
        ylabels = [f"Q{i+1} (n={n:,})" for i, n in enumerate(qcounts.to_numpy())]
        style_heatmap(ax, im, ["Rel.", "Temp.", "Geo.", "Tech.", "Coll."], ylabels, colorbar=True)
        ax.figure.axes[-1].set_ylabel("Pedigree score", fontweight="bold")

    def p5(ax):
        cmap = LinearSegmentedColormap.from_list("active_cells", ["#F7F7F7", COLORS[2], COLORS[4]])
        im = ax.imshow(active_pct.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=100)
        for i in range(active_pct.shape[0]):
            for j in range(active_pct.shape[1]):
                value = active_pct.iloc[i, j]
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=10,
                        fontweight="bold", color="white" if value >= 60 else INK)
        style_heatmap(ax, im, YEARS, active_pct.index.tolist(), xlabel="Inventory year", colorbar=True)
        ax.figure.axes[-1].set_ylabel("Sectors with positive emissions (%)", fontweight="bold")

    def p6(ax):
        x = np.arange(1, len(lineage_cumulative) + 1)
        ax.plot(x, lineage_cumulative.to_numpy(), color=COLORS[4], lw=2.7)
        ax.fill_between(x, 0, lineage_cumulative.to_numpy(), color=COLORS[1], alpha=0.18)
        for threshold, marker in [(50, "o"), (90, "s"), (95, "D")]:
            idx = int(np.searchsorted(lineage_cumulative.to_numpy(), threshold))
            idx = min(idx, len(x) - 1)
            ax.scatter(x[idx], lineage_cumulative.iloc[idx], s=62, marker=marker,
                       color=COLORS[0], edgecolor=EDGE, zorder=3)
            ax.annotate(f"{threshold}%: {x[idx]} lineages", (x[idx], lineage_cumulative.iloc[idx]),
                        xytext=(7, -18 if threshold == 90 else 8), textcoords="offset points",
                        fontsize=9.5, fontweight="bold", color=INK)
        style_axis(ax, xlabel="Lineages ranked by contribution",
                   ylabel="Cumulative share (%)")
        ax.set_xlim(1, len(lineage_cumulative)); ax.set_ylim(0, 102)
        numeric_ticks(ax, n=5)

    return [p1, p2, p3, p4, p5, p6]


def build_figure_2(a: Analysis) -> list[Callable]:
    closure = a.closure.copy()
    positive = closure[(closure["m1_kg"] > 0) & (closure["m2_reaggregated_kg"] > 0)]
    # Allocation-share invariance: sector share of each gas's national annual total.
    annual = a.m2.groupby(["Year", "SectorProducedBy", "Flowable"], as_index=False)["CO2e_kg"].sum()
    annual["national"] = annual.groupby(["Year", "Flowable"])["CO2e_kg"].transform("sum")
    annual["share"] = annual["CO2e_kg"] / annual["national"]
    profile = annual.pivot_table(index=["SectorProducedBy", "Flowable"], columns="Year", values="share", fill_value=0)
    profile = profile.reindex(columns=YEARS, fill_value=0)
    tv_rows = []
    gas_abbrev = {
        "Carbon dioxide": "CO$_2$", "Methane": "CH$_4$", "Nitrous oxide": "N$_2$O",
        "Carbon tetrafluoride": "CF$_4$", "Hexafluoroethane": "C$_2$F$_6$",
        "Perfluoropropane": "C$_3$F$_8$", "Perfluorocyclobutane": "c-C$_4$F$_8$",
        "Sulfur hexafluoride": "SF$_6$", "Nitrogen trifluoride": "NF$_3$",
        "HFCs and PFCs, unspecified": "HFC/PFC*",
    }
    for gas in a.gases:
        values = profile.xs(gas, level="Flowable").reindex(a.sectors, fill_value=0.0)
        reference = values[YEARS[0]].to_numpy()
        tv_rows.append([50.0 * np.abs(values[year].to_numpy() - reference).sum() for year in YEARS])
    tv = pd.DataFrame(tv_rows, index=a.gases, columns=YEARS)
    tv = tv.loc[tv.max(axis=1).sort_values(ascending=False).index]
    attrib = a.m2.groupby("AttributionSources")["CO2e_kg"].sum().sort_values(ascending=True)
    attrib = 100 * attrib / attrib.sum()
    closure_p95 = (
        closure.groupby("Flowable")["relative_error"].quantile(0.95)
        .replace(0, 1e-18).sort_values(ascending=False)
    )
    closure_labels = [gas_abbrev.get(g, g.replace("HFC-", "HFC ")) for g in closure_p95.index]
    source_by_gas = (
        a.m2.groupby(["GasGroup", "AttributionSources"])["CO2e_kg"].sum()
        .unstack(fill_value=0)
        .reindex(index=["CO$_2$", "CH$_4$", "N$_2$O", "F-gases"], fill_value=0)
    )
    source_by_gas = 100 * source_by_gas.div(source_by_gas.sum(axis=1), axis=0)
    source_by_gas = source_by_gas[source_by_gas.sum().sort_values(ascending=False).index]

    def p1(ax):
        ax.scatter(positive["m1_kg"], positive["m2_reaggregated_kg"], s=18, alpha=0.50,
                   facecolors="none", edgecolors=COLORS[1], linewidths=0.7)
        lo = min(positive["m1_kg"].min(), positive["m2_reaggregated_kg"].min())
        hi = max(positive["m1_kg"].max(), positive["m2_reaggregated_kg"].max())
        ax.plot([lo, hi], [lo, hi], color=EDGE, ls="--", lw=1.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        style_axis(ax, xlabel="Aggregate m1 (kg CO$_2$e)", ylabel="Reaggregated (kg CO$_2$e)")

    def p2(ax):
        x = np.sort(np.maximum(closure["relative_error"].to_numpy(), 1e-18))
        y = np.arange(1, len(x)+1) / len(x)
        ax.plot(x, y, color=COLORS[4], lw=2.6)
        ax.set_xscale("log")
        ax.axvline(1e-9, color=EDGE, ls="--", lw=1.3)
        style_axis(ax, xlabel="Absolute relative closure error", ylabel="Cumulative fraction")
        numeric_ticks(ax, x=False, y=True, n=5)

    def p3(ax):
        vmax = max(1.0, float(np.nanpercentile(tv.to_numpy(), 98)))
        cmap = LinearSegmentedColormap.from_list("allocation_change", ["#F7F7F7", COLORS[1], COLORS[4]])
        im = ax.imshow(tv.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        labels = [gas_abbrev.get(g, g.replace("HFC-", "HFC ")) for g in tv.index]
        style_heatmap(ax, im, YEARS, labels, xlabel="Inventory year", colorbar=True)
        ax.figure.axes[-1].set_ylabel("Allocation-profile change vs 2019 (%)", fontweight="bold")

    def p4(ax):
        bars = ax.barh(range(len(attrib)), attrib.values, color=COLORS[3], edgecolor=EDGE)
        ax.set_yticks(range(len(attrib)), [humanize_attribution(x) for x in attrib.index])
        style_axis(ax, xlabel="Five-year CO$_2$e share (%)")
        ax.set_xlim(0, float(attrib.max()) * 1.15)
        for bar, value in zip(bars, attrib.values):
            ax.text(value + 0.5, bar.get_y() + bar.get_height()/2, f"{value:.1f}%",
                    va="center", fontsize=10.5, fontweight="bold", color=INK)
        numeric_ticks(ax, x=True, y=False, n=5)

    def p5(ax):
        y = np.arange(len(closure_p95))
        ax.hlines(y, 1e-18, closure_p95.to_numpy(), color=COLORS[1], lw=2.8)
        ax.scatter(closure_p95.to_numpy(), y, s=55, color=COLORS[4], edgecolor=EDGE, zorder=3)
        ax.set_xscale("log")
        ax.set_yticks(y, closure_labels)
        ax.invert_yaxis()
        ax.axvline(1e-9, color=EDGE, ls="--", lw=1.2)
        style_axis(ax, xlabel="95th-percentile relative closure error")
        plt.setp(ax.get_yticklabels(), fontsize=9.5)

    def p6(ax):
        cmap = LinearSegmentedColormap.from_list("source_gas", ["#F7F7F7", COLORS[1], COLORS[4]])
        im = ax.imshow(source_by_gas.to_numpy(), aspect="auto", cmap=cmap, vmin=0,
                       vmax=max(1.0, float(source_by_gas.to_numpy().max())))
        for i in range(source_by_gas.shape[0]):
            for j in range(source_by_gas.shape[1]):
                value = source_by_gas.iloc[i, j]
                if value >= 1.0:
                    ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=8.8,
                            fontweight="bold", color="white" if value >= 45 else INK)
        style_heatmap(ax, im,
                      [humanize_attribution(x) for x in source_by_gas.columns],
                      source_by_gas.index.tolist(), xlabel="Attribution-source family", colorbar=True)
        plt.setp(ax.get_xticklabels(), rotation=50, ha="right", fontsize=9)
        ax.figure.axes[-1].set_ylabel("Within-gas-group CO$_2$e share (%)", fontweight="bold")

    return [p1, p2, p3, p4, p5, p6]


def build_figure_3(a: Analysis) -> list[Callable]:
    k = a.k_table
    order = np.argsort(a.labels, kind="stable")
    ordered = a.consensus[np.ix_(order, order)]
    sacc_aris = np.array([
        adjusted_rand_score(a.labels, labels) for labels in a.source_label_runs
    ])
    ari_order = np.argsort(sacc_aris)

    def p1(ax):
        ax.plot(k["K"], k["silhouette"], marker="o", lw=2.5, color=COLORS[1], label="Silhouette")
        ax.plot(k["K"], 1-k["PAC"], marker="s", lw=2.5, color=COLORS[4], label="1 − PAC")
        ax.axvline(a.selected_k, color=EDGE, ls="--", lw=1.5)
        style_axis(ax, xlabel="Number of clusters (K)", ylabel="Selection metric")
        ax.set_xticks(k["K"])
        lower = min(float(k["silhouette"].min()), float((1 - k["PAC"]).min()))
        ax.set_ylim(np.floor((lower - 0.03) * 20) / 20, 1.03)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2)

    def p2(ax):
        im = ax.imshow(ordered, cmap=LinearSegmentedColormap.from_list("consensus", ["#F7F7F7", COLORS[1], COLORS[4]]), vmin=0, vmax=1)
        sizes = [np.sum(a.labels == c) for c in sorted(np.unique(a.labels))]
        starts = np.r_[0, np.cumsum(sizes)]
        centers = (starts[:-1] + starts[1:] - 1) / 2
        labels = [f"A{c}" for c in sorted(np.unique(a.labels))]
        ax.set_xticks(centers, labels); ax.set_yticks(centers, labels)
        ax.tick_params(length=0)
        for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_fontweight("bold")
        for spine in ax.spines.values(): spine.set_visible(False)
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Co-assignment", fontweight="bold")
        for pos in starts[1:-1]:
            ax.axhline(pos-0.5, color="white", lw=1.2); ax.axvline(pos-0.5, color="white", lw=1.2)

    def p3(ax):
        parts = ax.violinplot([a.stability, a.source_hhi, a.fragility], positions=[1,2,3], showmedians=True,
                              widths=0.75)
        for body, color in zip(parts["bodies"], [COLORS[2], COLORS[3], COLORS[0]]):
            body.set_facecolor(color); body.set_edgecolor(EDGE); body.set_alpha(0.9)
        for key in ["cmedians", "cbars", "cmins", "cmaxes"]:
            parts[key].set_color(EDGE); parts[key].set_linewidth(1.3)
        ax.set_xticks([1,2,3], ["Stability", "Source\nHHI", "Fragility"])
        style_axis(ax, ylabel="Sector-level score")
        ax.set_ylim(-0.02, 1.14)
        for x, values in zip([1, 2, 3], [a.stability, a.source_hhi, a.fragility]):
            ax.text(x, 1.05, f"median {np.median(values):.2f}",
                    ha="center", fontsize=9.5, fontweight="bold", color=INK)

    def p4(ax):
        d = a.model_comparison
        x = np.arange(len(d)); w = 0.25
        ax.bar(x-w, d["silhouette"], w, color=COLORS[1], edgecolor=EDGE, label="Silhouette")
        ax.bar(x, d["median_source_ARI"], w, color=COLORS[2], edgecolor=EDGE, label="Median ARI")
        ax.bar(x+w, d["minimum_source_ARI"], w, color=COLORS[0], edgecolor=EDGE, label="Minimum ARI")
        ax.set_xticks(x, ["Raw-log", "CLR", "SACC"])
        style_axis(ax, ylabel="Score")
        ax.set_ylim(0, 1.08)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
                  columnspacing=0.8, handletextpad=0.4)

    def p5(ax):
        bars = ax.bar(k["K"], k["minimum_cluster_size"], color=COLORS[2], edgecolor=EDGE)
        ax.axhline(20, color=EDGE, ls="--", lw=1.3)
        ax.scatter([a.selected_k],
                   k.loc[k["K"] == a.selected_k, "minimum_cluster_size"],
                   s=95, marker="D", color=COLORS[0], edgecolor=EDGE, zorder=3)
        for bar, value, kval in zip(bars, k["minimum_cluster_size"], k["K"]):
            offset = 9 if int(kval) == a.selected_k else 2
            ax.text(bar.get_x() + bar.get_width()/2, value + offset, f"{int(value)}", ha="center",
                    fontsize=9.5, fontweight="bold", color=INK)
        style_axis(ax, xlabel="Number of clusters (K)", ylabel="Minimum cluster size")
        ax.set_xticks(k["K"])
        ax.set_ylim(0, float(k["minimum_cluster_size"].max()) * 1.28)

    def p6(ax):
        x = np.sort(sacc_aris)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=COLORS[4], lw=2.7)
        ax.fill_between(x, 0, y, color=COLORS[1], alpha=0.18)
        for annotation_rank, rank in enumerate(ari_order[:3]):
            value = sacc_aris[rank]
            ypos = np.searchsorted(x, value, side="right") / len(x)
            label = humanize_meta_source(a.source_names[rank])
            ax.scatter(value, ypos, s=48, color=COLORS[0], edgecolor=EDGE, zorder=3)
            ax.annotate(f"{label} (ARI {value:.3f})", (value, ypos),
                        xytext=(0.10, 0.62 - 0.13 * annotation_rank), textcoords="axes fraction",
                        fontsize=8.5, fontweight="bold",
                        arrowprops={"arrowstyle": "-", "color": EDGE, "lw": 0.8})
        style_axis(ax, xlabel="Source-perturbation ARI", ylabel="Cumulative lineage fraction")
        ax.set_xlim(max(0, float(x.min()) - 0.035), 1.002); ax.set_ylim(0, 1.02)
        numeric_ticks(ax, n=5)

    return [p1, p2, p3, p4, p5, p6]


def grouped_composition(a: Analysis) -> tuple[pd.DataFrame, list[str]]:
    group_names = ["CO$_2$", "CH$_4$", "N$_2$O", "F-gases"]
    gas_groups = [gas_group(g) for g in a.gases]
    out = np.column_stack([a.composition[:, np.array(gas_groups) == g].sum(axis=1) for g in group_names])
    df = pd.DataFrame(out, index=a.sectors, columns=group_names)
    return df, group_names


def build_figure_4(a: Analysis) -> list[Callable]:
    pca_model = PCA(n_components=2, random_state=RANDOM_STATE)
    pca = pca_model.fit_transform(a.clr)
    explained = 100 * pca_model.explained_variance_ratio_
    gas_abbrev = {
        "Carbon dioxide": "CO$_2$", "Methane": "CH$_4$", "Nitrous oxide": "N$_2$O",
        "Carbon tetrafluoride": "CF$_4$", "Hexafluoroethane": "C$_2$F$_6$",
        "Perfluoropropane": "C$_3$F$_8$", "Perfluorocyclobutane": "c-C$_4$F$_8$",
        "Sulfur hexafluoride": "SF$_6$", "Nitrogen trifluoride": "NF$_3$",
        "HFCs and PFCs, unspecified": "HFC/PFC*",
    }
    gas_labels = [gas_abbrev.get(g, g.replace("HFC-", "HFC ")) for g in a.gases]
    profiles = pd.DataFrame(a.clr, columns=a.gases).assign(cluster=a.labels).groupby("cluster").median()
    totals = a.base.sum(axis=1) / 1e12
    cluster_df = pd.DataFrame({"cluster": a.labels, "Gt": totals, "stability": a.stability, "fragility": a.fragility})
    summary = cluster_df.groupby("cluster").agg(size=("Gt","size"), Gt=("Gt","sum"), stability=("stability","median"), fragility=("fragility","median"))
    grouped, group_names = grouped_composition(a)
    grouped_profile = grouped.assign(cluster=a.labels).groupby("cluster").median()
    grouped_profile = grouped_profile.div(grouped_profile.sum(axis=1), axis=0)
    sector_mt = a.base.sum(axis=1) / 1e9

    def p1(ax):
        for c, color in zip(sorted(np.unique(a.labels)), COLORS[:a.selected_k]):
            idx = a.labels == c
            ax.scatter(pca[idx,0], pca[idx,1], s=28, alpha=0.75, color=color, edgecolors="white", linewidths=0.3,
                       label=f"A{c}")
        style_axis(ax, xlabel=f"CLR-PC1 ({explained[0]:.1f}%)", ylabel=f"CLR-PC2 ({explained[1]:.1f}%)")
        ax.legend(
            loc="upper right",
            ncol=3,
            fontsize=10.5,
            columnspacing=0.9,
            handletextpad=0.35,
        )
        numeric_ticks(ax, n=5)

    def p2(ax):
        limit = float(np.nanpercentile(np.abs(profiles.to_numpy()), 95))
        im = ax.imshow(profiles.to_numpy(), aspect="auto",
                       cmap=LinearSegmentedColormap.from_list("profile", ["#4E85AC", "#F7F7F7", "#D26A6A"]),
                       vmin=-limit, vmax=limit)
        style_heatmap(ax, im, gas_labels, [f"A{x}" for x in profiles.index],
                      xlabel="Gas coordinate (CLR)", colorbar=True)
        plt.setp(ax.get_xticklabels(), rotation=55, ha="right", fontsize=10.5)
        ax.figure.axes[-1].set_ylabel("Median CLR", fontweight="bold")

    def p3(ax):
        x = np.arange(len(summary))
        ax.bar(
            x,
            summary["Gt"],
            width=THREE_CATEGORY_BAR_WIDTH,
            color=COLORS[:len(summary)],
            edgecolor=EDGE,
        )
        for i, (gt, size) in enumerate(zip(summary["Gt"], summary["size"])):
            ax.text(i, gt*1.02, f"n={size}", ha="center", fontweight="bold", fontsize=11)
        ax.set_xticks(x, [f"A{c}" for c in summary.index])
        style_axis(ax, xlabel="Consensus archetype", ylabel="Annual mean emissions (Gt CO$_2$e)")
        ax.set_ylim(0, float(summary["Gt"].max()) * 1.14)
        numeric_ticks(ax, x=False, y=True, n=5)

    def p4(ax):
        x = np.arange(len(summary)); w=0.34
        ax.bar(x-w/2, summary["stability"], w, color=COLORS[2], edgecolor=EDGE, label="Stability")
        ax.bar(x+w/2, summary["fragility"], w, color=COLORS[0], edgecolor=EDGE, label="Fragility")
        ax.set_xticks(x, [f"A{c}" for c in summary.index])
        style_axis(ax, xlabel="Consensus archetype", ylabel="Median sector score")
        ax.set_ylim(0, 1.08); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2)
        for i, values in enumerate(zip(summary["stability"], summary["fragility"])):
            for offset, value in zip([-w/2, w/2], values):
                ax.text(i + offset, value + 0.018, f"{value:.2f}", ha="center",
                        fontsize=9.5, fontweight="bold", color=INK)

    def p5(ax):
        bottom = np.zeros(len(grouped_profile))
        x = np.arange(len(grouped_profile))
        for name, color in zip(group_names, [COLORS[1], COLORS[3], COLORS[2], COLORS[0]]):
            values = grouped_profile[name].to_numpy()
            ax.bar(
                x,
                values,
                width=THREE_CATEGORY_BAR_WIDTH,
                bottom=bottom,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                label=name,
            )
            bottom += values
        ax.set_xticks(x, [f"A{c}" for c in grouped_profile.index])
        style_axis(ax, xlabel="Consensus archetype", ylabel="Median composition share")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=4,
                  columnspacing=0.8, handletextpad=0.35)

    def p6(ax):
        data = [sector_mt[a.labels == c] for c in sorted(np.unique(a.labels))]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, color in zip(bp["boxes"], COLORS[:a.selected_k]):
            patch.set_facecolor(color); patch.set_edgecolor(EDGE)
        for key in ["whiskers", "caps", "medians"]:
            for item in bp[key]: item.set_color(EDGE); item.set_linewidth(1.3)
        ax.set_yscale("log")
        ax.set_xticks(range(1, a.selected_k + 1), [f"A{c}" for c in range(1, a.selected_k + 1)])
        style_axis(ax, xlabel="Consensus archetype", ylabel="Sector CO$_2$e (Mt)")
        for i, values in enumerate(data, start=1):
            med = float(np.median(values[values > 0])) if np.any(values > 0) else 0.0
            if med > 0:
                ax.text(i, med * 1.45, f"median {med:.2g}", ha="center", fontsize=9.2,
                        fontweight="bold", color=INK)

    return [p1, p2, p3, p4, p5, p6]


def source_cluster_tables(a: Analysis) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for source, matrix in zip(a.source_names, a.source_matrices):
        sector_totals = matrix.sum(axis=1)
        for cluster in sorted(np.unique(a.labels)):
            rows.append({"MetaSource": source, "cluster": cluster, "CO2e_kg": sector_totals[a.labels == cluster].sum()})
    edges = pd.DataFrame(rows)
    source_totals = edges.groupby("MetaSource")["CO2e_kg"].sum().sort_values(ascending=False)
    top = source_totals.head(10).index
    return edges, edges[edges["MetaSource"].isin(top)].copy()


def build_figure_5(a: Analysis) -> list[Callable]:
    edges, top_edges = source_cluster_tables(a)
    source_totals = top_edges.groupby("MetaSource")["CO2e_kg"].sum().sort_values(ascending=False)
    top_sources = source_totals.index.tolist()
    attrib = a.m2.copy()
    sector_cluster = dict(zip(a.sectors, a.labels))
    attrib["cluster"] = attrib["SectorProducedBy"].map(sector_cluster)
    attrib_tab = attrib.groupby(["cluster", "AttributionSources"])["CO2e_kg"].sum().unstack(fill_value=0)
    attrib_tab = attrib_tab.div(attrib_tab.sum(axis=1), axis=0)
    attrib_tab = attrib_tab[attrib_tab.sum().sort_values(ascending=False).index]
    primary_families = [
        "BEA_Detail_Use_SUT",
        "Direct",
        "BEA_Detail_Supply",
        "EIA_MECS_Energy",
    ]
    primary_families = [name for name in primary_families if name in attrib_tab.columns]
    attrib_plot = attrib_tab[primary_families].copy()
    remaining_families = [name for name in attrib_tab.columns if name not in primary_families]
    if remaining_families:
        attrib_plot["Other"] = attrib_tab[remaining_families].sum(axis=1)
    instability_component = (1.0 - a.source_hhi) * (1.0 - a.stability)
    decomposition = pd.DataFrame(
        {"cluster": a.labels, "HHI": a.source_hhi, "Instability": instability_component}
    ).groupby("cluster").mean()
    source_cluster = top_edges.pivot_table(index="MetaSource", columns="cluster", values="CO2e_kg",
                                           aggfunc="sum", fill_value=0).reindex(top_sources)
    source_cluster = source_cluster.div(source_cluster.sum(axis=1), axis=0) * 100
    source_cluster = source_cluster.reindex(columns=range(1, a.selected_k + 1), fill_value=0)
    sector_totals = a.base.sum(axis=1)
    size_scale = pd.Series(sector_totals).rank(pct=True).to_numpy()

    def p1(ax):
        ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.5, max(len(top_sources), a.selected_k)-0.5); ax.axis("off")
        sy = {s: len(top_sources)-1-i for i,s in enumerate(top_sources)}
        cy = {c: (len(top_sources)-1) * (a.selected_k-c)/(a.selected_k-1) for c in range(1,a.selected_k+1)}
        max_edge = max(top_edges["CO2e_kg"].max(), 1)
        for _, row in top_edges.iterrows():
            if row.CO2e_kg <= 0: continue
            y1, y2 = sy[row.MetaSource], cy[int(row.cluster)]
            ax.add_patch(FancyArrowPatch((0.48,y1),(0.77,y2), connectionstyle="arc3,rad=0.08",
                                         arrowstyle="-", linewidth=0.3+5*row.CO2e_kg/max_edge,
                                         color=COLORS[int(row.cluster)-1], alpha=0.50))
        for s,y in sy.items():
            ax.scatter([0.46],[y],s=70,color=COLORS[6],edgecolor=EDGE,zorder=3)
            ax.text(0.42,y,compact_meta_source(s),ha="right",va="center",fontsize=8.7,fontweight="bold")
        for c,y in cy.items():
            ax.scatter([0.79],[y],s=180,color=COLORS[c-1],edgecolor=EDGE,zorder=3)
            ax.text(0.84,y,f"A{c}",ha="left",va="center",fontsize=12,fontweight="bold")
        ax.text(0.46,len(top_sources)-0.25,"Lineage",ha="center",fontweight="bold",fontsize=13)
        ax.text(0.79,len(top_sources)-0.25,"Archetype",ha="center",fontweight="bold",fontsize=13)

    def p2(ax):
        data = [a.source_hhi[a.labels==c] for c in sorted(np.unique(a.labels))]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, color in zip(bp["boxes"], COLORS[:a.selected_k]):
            patch.set_facecolor(color); patch.set_edgecolor(EDGE)
        for key in ["whiskers","caps","medians"]:
            for item in bp[key]: item.set_color(EDGE); item.set_linewidth(1.3)
        ax.set_xticks(range(1,a.selected_k+1), [f"A{x}" for x in range(1,a.selected_k+1)])
        style_axis(ax, xlabel="Consensus archetype", ylabel="Sector source HHI")
        upper = min(1.02, max(0.35, float(np.max(a.source_hhi)) * 1.02))
        ax.set_ylim(0, upper)

    def p3(ax):
        concise_labels = {
            "Direct": "Direct",
            "BEA_Detail_Use_SUT": "BEA use",
            "BEA_Detail_Supply": "BEA supply",
            "BEA_Detail_GrossOutput_IO": "BEA output",
            "EIA_MECS_Energy": "EIA energy",
            "USDA_CoA_Cropland": "USDA land",
            "USDA_CoA_Cropland_NAICS": "USDA NAICS",
            "USGS_MYB_Lead": "USGS lead",
        }
        bottom = np.zeros(len(attrib_plot))
        for j,col in enumerate(attrib_plot.columns):
            vals = attrib_plot[col].to_numpy()
            ax.bar(
                np.arange(len(vals)),
                vals,
                width=THREE_CATEGORY_BAR_WIDTH,
                bottom=bottom,
                color=COLORS[j % len(COLORS)],
                edgecolor="white",
                linewidth=0.3,
                label=concise_labels.get(col, humanize_attribution(col)),
            )
            bottom += vals
        ax.set_xticks(np.arange(len(attrib_plot)), [f"A{x}" for x in attrib_plot.index])
        style_axis(ax, xlabel="Consensus archetype", ylabel="Family share")
        ax.set_ylim(0,1.02)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.26),
                  ncol=3, columnspacing=0.8, handletextpad=0.35)

    def p4(ax):
        x = np.arange(len(decomposition))
        hhi = decomposition["HHI"].to_numpy()
        instability = decomposition["Instability"].to_numpy()
        ax.bar(
            x,
            hhi,
            width=THREE_CATEGORY_BAR_WIDTH,
            color=COLORS[3],
            edgecolor=EDGE,
            label="Source concentration",
        )
        ax.bar(
            x,
            instability,
            width=THREE_CATEGORY_BAR_WIDTH,
            bottom=hhi,
            color=COLORS[0],
            edgecolor=EDGE,
            label="Reassignment",
        )
        ax.set_xticks(x, [f"A{c}" for c in decomposition.index])
        style_axis(ax, xlabel="Consensus archetype", ylabel="Mean fragility")
        ax.set_ylim(0, max(0.24, float((hhi + instability).max()) * 1.25))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2,
                  columnspacing=0.8, handletextpad=0.4)
        for i, value in enumerate(hhi + instability):
            ax.text(i, value + 0.008, f"{value:.2f}", ha="center", fontsize=10,
                    fontweight="bold", color=INK)

    def p5(ax):
        cmap = LinearSegmentedColormap.from_list("lineage_cluster", ["#F7F7F7", COLORS[1], COLORS[4]])
        im = ax.imshow(source_cluster.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=100)
        for i in range(source_cluster.shape[0]):
            for j in range(source_cluster.shape[1]):
                value = source_cluster.iloc[i, j]
                if value >= 1:
                    ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=9,
                            fontweight="bold", color="white" if value >= 55 else INK)
        style_heatmap(ax, im, [f"A{c}" for c in source_cluster.columns],
                      [compact_meta_source(x) for x in source_cluster.index], colorbar=True)
        plt.setp(ax.get_yticklabels(), fontsize=8.8)
        ax.figure.axes[-1].set_ylabel("Allocation (%)", fontweight="bold")

    def p6(ax):
        for c, color in zip(sorted(np.unique(a.labels)), COLORS[:a.selected_k]):
            idx = a.labels == c
            ax.scatter(a.source_hhi[idx], instability_component[idx],
                       s=18 + 65 * size_scale[idx], color=color, alpha=0.68,
                       edgecolors=EDGE, linewidths=0.25, label=f"A{c}")
        xlimit = max(0.55, float(a.source_hhi.max()) * 1.04)
        ylimit = max(0.05, float(instability_component.max()) * 1.14)
        equality_limit = min(xlimit, ylimit)
        ax.plot([0, equality_limit], [0, equality_limit], color=EDGE, ls="--", lw=1.2)
        share = 100 * np.mean(instability_component > a.source_hhi)
        ax.text(0.97, 0.86, f"Reassignment-dominant sectors: {share:.1f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=9.5, fontweight="bold", color=MUTED)
        style_axis(ax, xlabel="Source concentration (HHI)",
                   ylabel="Reassignment")
        ax.set_xlim(0, xlimit); ax.set_ylim(0, ylimit)
        ax.legend(loc="upper center", ncol=3)
        numeric_ticks(ax, n=5)

    return [p1,p2,p3,p4,p5,p6]


def short_name(name: str, n=32) -> str:
    return name if len(name) <= n else name[:n-1] + "…"


def build_figure_6(a: Analysis) -> list[Callable]:
    totals = a.base.sum(axis=1) / 1e9  # Mt CO2e
    names = np.array([a.sector_names.get(s, s) for s in a.sectors])
    # Ten ranked sectors keep the labels readable at journal column width while
    # preserving the decision-relevant upper tail used throughout the text.
    top_idx = np.argsort(a.priority)[-10:][::-1]
    q05 = np.percentile(a.rank_runs,5,axis=0); q95=np.percentile(a.rank_runs,95,axis=0)
    median_rank=np.median(a.rank_runs,axis=0)
    em_med=np.median(totals); fr_med=np.median(a.fragility)
    quadrants=[]
    for high_e,high_f,label in [(False,False,"Low E / Low F"),(True,False,"High E / Low F"),(False,True,"Low E / High F"),(True,True,"High E / High F")]:
        idx=((totals>=em_med)==high_e)&((a.fragility>=fr_med)==high_f)
        quadrants.append((label,int(idx.sum()),float(totals[idx].sum()/1000)))
    emission_pct = pd.Series(totals).rank(pct=True).to_numpy()
    fragility_pct = pd.Series(a.fragility).rank(pct=True).to_numpy()
    top10 = top_idx[:10]
    top50 = np.argsort(a.priority)[-50:]
    top50_summary = pd.DataFrame(
        {"cluster": a.labels[top50], "Gt": totals[top50] / 1000}
    ).groupby("cluster").agg(count=("Gt", "size"), Gt=("Gt", "sum")).reindex(
        range(1, a.selected_k + 1), fill_value=0
    )

    def p1(ax):
        sc=ax.scatter(totals,a.fragility,c=a.priority,cmap=LinearSegmentedColormap.from_list("priority",[COLORS[2],COLORS[6],COLORS[0]]),
                      s=24+90*a.priority,alpha=0.78,edgecolors=EDGE,linewidths=0.25)
        ax.set_xscale("log")
        ax.axvline(em_med, color=EDGE, lw=1.1, ls="--", alpha=0.65)
        ax.axhline(fr_med, color=EDGE, lw=1.1, ls="--", alpha=0.65)
        style_axis(ax,xlabel="Annual mean emissions (Mt CO$_2$e)",ylabel="Allocation fragility")
        cb=ax.figure.colorbar(sc,ax=ax,fraction=0.046,pad=0.04); cb.set_label("Priority score",fontweight="bold")

    def p2(ax):
        order=top_idx[::-1]
        ax.barh(np.arange(len(order)),a.priority[order],color=[COLORS[c-1] for c in a.labels[order]],edgecolor=EDGE)
        ax.set_yticks(np.arange(len(order)),[compact_sector_name(names[i], 27) for i in order])
        style_axis(ax,xlabel="Priority score")
        ax.set_xlim(0,1.03)
        plt.setp(ax.get_yticklabels(), fontsize=10.5)

    def p3(ax):
        order=top_idx[np.argsort(median_rank[top_idx])]
        y=np.arange(len(order))
        ax.hlines(y,q05[order],q95[order],color=COLORS[1],lw=3)
        ax.scatter(median_rank[order],y,color=COLORS[4],edgecolor=EDGE,s=48,zorder=3)
        ax.set_yticks(y,[compact_sector_name(names[i], 27) for i in order])
        ax.invert_yaxis()
        style_axis(ax,xlabel="Stress-test rank")
        ax.set_xlim(max(0,q05[order].min()-5),min(len(a.sectors),q95[order].max()+5))
        plt.setp(ax.get_yticklabels(), fontsize=10.5)

    def p4(ax):
        ax.set_xlim(0,2); ax.set_ylim(0,2)
        colors=[COLORS[2],COLORS[1],COLORS[6],COLORS[0]]
        positions=[(0,0),(1,0),(0,1),(1,1)]
        for (label,n,gt),color,(x,y) in zip(quadrants,colors,positions):
            ax.add_patch(Rectangle((x,y),1,1,facecolor=color,alpha=0.88,edgecolor="white",lw=2))
            ax.text(x+0.5,y+0.58,label,ha="center",va="center",fontsize=11,fontweight="bold")
            ax.text(x+0.5,y+0.35,f"n={n}; {gt:.2f} Gt",ha="center",va="center",fontsize=10.5)
        ax.set_xticks([0.5,1.5],["Lower","Higher"])
        ax.set_yticks([0.5,1.5],["Lower","Higher"])
        ax.set_xlabel("Annual mean emissions", fontweight="bold")
        ax.set_ylabel("Allocation fragility", fontweight="bold")
        ax.tick_params(length=0)
        for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_fontweight("bold")
        for spine in ax.spines.values(): spine.set_visible(False)

    def p5(ax):
        order = top10[::-1]
        y = np.arange(len(order))
        ax.hlines(y, emission_pct[order], fragility_pct[order], color=LIGHT, lw=3)
        ax.scatter(emission_pct[order], y, s=58, color=COLORS[1], edgecolor=EDGE,
                   zorder=3, label="Emission rank")
        ax.scatter(fragility_pct[order], y, s=58, color=COLORS[0], edgecolor=EDGE,
                   zorder=3, label="Fragility rank")
        ax.set_yticks(y, [compact_sector_name(names[i], 26) for i in order])
        style_axis(ax, xlabel="Rank percentile")
        ax.set_xlim(0, 1.03)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
                  columnspacing=0.8, handletextpad=0.4)
        plt.setp(ax.get_yticklabels(), fontsize=9.7)

    def p6(ax):
        x = np.arange(len(top50_summary))
        bars = ax.bar(
            x,
            top50_summary["count"],
            width=THREE_CATEGORY_BAR_WIDTH,
            color=COLORS[:len(top50_summary)],
            edgecolor=EDGE,
        )
        ax.set_xticks(x, [f"A{c}" for c in top50_summary.index])
        style_axis(ax, xlabel="Consensus archetype", ylabel="Top-50 sector count")
        ax.set_ylim(0, max(10, float(top50_summary["count"].max()) * 1.25))
        for bar, count, gt in zip(bars, top50_summary["count"], top50_summary["Gt"]):
            ax.text(bar.get_x() + bar.get_width()/2, count + 0.8,
                    f"n={int(count)}\n{gt:.2f} Gt", ha="center", fontsize=10,
                    fontweight="bold", color=INK)

    return [p1,p2,p3,p4,p5,p6]


def build_figure_7(a: Analysis) -> list[Callable]:
    summary = a.retention_sensitivity.sort_values("source_retention")
    levels = summary["source_retention"].to_numpy()
    x = 100 * levels
    ordered_colors = get_palette("thermal_gradient_8", len(levels))
    stability_pivot = a.retention_sector_stability.pivot(
        index="Sector", columns="source_retention", values="PartitionStability"
    ).reindex(columns=levels)
    sort_columns = [levels[0], levels[1]] if len(levels) > 1 else [levels[0]]
    stability_pivot = stability_pivot.sort_values(sort_columns, kind="stable")
    stability_pivot = stability_pivot.head(min(80, len(stability_pivot)))
    stability_quantiles = a.retention_sector_stability.groupby("source_retention")[
        "PartitionStability"
    ].quantile([0.0, 0.05, 0.25]).unstack().reindex(levels)

    def add_current_setting(ax):
        ax.axvline(100 * SOURCE_ATTENUATION, color=EDGE, ls="--", lw=1.2, alpha=0.75)
        ax.set_xticks(x, [f"{v:.0f}" for v in x])

    def p1(ax):
        ax.plot(x, summary["median_source_ARI"], marker="o", ms=7, lw=2.4,
                color=COLORS[1], label="Median ARI")
        ax.plot(x, summary["minimum_source_ARI"], marker="s", ms=6.5, lw=2.4,
                color=COLORS[0], label="Minimum ARI")
        add_current_setting(ax)
        style_axis(ax, xlabel="Lineage retention (%)", ylabel="ARI vs full SACC partition")
        ax.set_ylim(0, 1.04)
        ax.legend(loc="lower right")

    def p2(ax):
        minimum = 1.0
        for retention, color in zip(levels, ordered_colors):
            values = a.retention_sector_stability.loc[
                a.retention_sector_stability["source_retention"] == retention,
                "PartitionStability",
            ].to_numpy()
            values = np.sort(values)
            minimum = min(minimum, float(values.min()))
            cumulative = np.arange(1, len(values) + 1) / len(values)
            ax.plot(values, cumulative, lw=2.0, color=color, label=f"{100*retention:.0f}%")
        style_axis(ax, xlabel="Sector partition stability", ylabel="Cumulative sector fraction")
        ax.set_xlim(max(0, minimum - 0.025), 1.002); ax.set_ylim(0, 1.01)
        ax.legend(loc="upper left", ncol=2, title="Retention",
                  fontsize=9.5, title_fontsize=10.5)

    def p3(ax):
        ax.plot(x, 1.0 - summary["PAC"], marker="D", ms=6.5, lw=2.4,
                color=COLORS[4], label="1 - PAC")
        ax.plot(x, summary["fraction_stability_ge_0_95"], marker="^", ms=7, lw=2.4,
                color=COLORS[2], label="Sectors with stability >= 0.95")
        add_current_setting(ax)
        style_axis(ax, xlabel="Lineage retention (%)", ylabel="Robustness fraction")
        lower = min(float((1.0-summary["PAC"]).min()), float(summary["fraction_stability_ge_0_95"].min()))
        ax.set_ylim(max(0, lower - 0.035), 1.01)
        ax.legend(loc="lower right")

    def p4(ax):
        ax.plot(x, summary["priority_rank_spearman"], marker="o", ms=7, lw=2.4,
                color=COLORS[3], label="Priority-rank Spearman")
        ax.plot(x, summary["top50_priority_overlap"], marker="s", ms=6.5, lw=2.4,
                color=COLORS[6], label="Top-50 overlap")
        add_current_setting(ax)
        style_axis(ax, xlabel="Lineage retention (%)", ylabel="Priority robustness")
        lower = min(float(summary["priority_rank_spearman"].min()), float(summary["top50_priority_overlap"].min()))
        ax.set_ylim(max(0, lower - 0.035), 1.01)
        ax.legend(loc="lower right")

    def p5(ax):
        for q, label, color, marker in [
            (0.0, "Minimum", COLORS[0], "s"),
            (0.05, "5th percentile", COLORS[6], "D"),
            (0.25, "25th percentile", COLORS[2], "o"),
        ]:
            ax.plot(x, stability_quantiles[q], marker=marker, ms=6.5, lw=2.3,
                    color=color, label=label)
        add_current_setting(ax)
        style_axis(ax, xlabel="Lineage retention (%)",
                   ylabel="Lower-tail sector stability")
        ax.set_ylim(max(0, float(stability_quantiles.min().min()) - 0.04), 1.01)
        ax.legend(loc="lower right")

    def p6(ax):
        cmap = LinearSegmentedColormap.from_list("stability_atlas", [COLORS[0], COLORS[6], "#F7F7F7", COLORS[2]])
        atlas_vmin = max(0.0, float(stability_pivot.to_numpy().min()) - 0.02)
        im = ax.imshow(stability_pivot.to_numpy(), aspect="auto", cmap=cmap, vmin=atlas_vmin, vmax=1)
        ax.set_xticks(np.arange(len(levels)), [f"{100*v:.0f}" for v in levels])
        ax.set_yticks([0, len(stability_pivot)-1], ["least stable", f"rank {len(stability_pivot)}"])
        for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_fontweight("bold")
        ax.set_xlabel("Lineage retention (%)", fontweight="bold")
        ax.set_ylabel("Stability-ranked sectors", fontweight="bold")
        ax.tick_params(axis="x", direction="out", bottom=True, length=6, width=1.4)
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Partition stability", fontweight="bold")

    return [p1, p2, p3, p4, p5, p6]


def build_figure_8(a: Analysis) -> list[Callable]:
    summary = a.loo_summary.sort_values("omitted_year")
    years = summary["omitted_year"].to_numpy(int)
    matches = np.sum(a.loo_assignments == a.labels[None, :], axis=0)
    match_matrix = np.zeros((a.selected_k, len(YEARS) + 1), dtype=float)
    for cluster in sorted(np.unique(a.labels)):
        idx = a.labels == cluster
        for n_match in range(len(YEARS) + 1):
            match_matrix[cluster - 1, n_match] = 100 * np.mean(matches[idx] == n_match)
    changed_by_cluster = np.zeros((a.selected_k, len(YEARS)), dtype=int)
    for j, _year in enumerate(YEARS):
        for cluster in sorted(np.unique(a.labels)):
            idx = a.labels == cluster
            changed_by_cluster[cluster - 1, j] = int(np.sum(a.loo_assignments[j, idx] != a.labels[idx]))

    def p1(ax):
        ax.plot(years, summary["ARI_vs_full"], marker="o", lw=2.4, ms=7,
                color=COLORS[1], label="ARI")
        ax.plot(years, summary["assignment_agreement"], marker="s", lw=2.4, ms=6.5,
                color=COLORS[0], label="Sector agreement")
        style_axis(ax, xlabel="Omitted inventory year", ylabel="Full-solution agreement")
        lower = min(float(summary["ARI_vs_full"].min()), float(summary["assignment_agreement"].min()))
        pad = max((1.0 - lower) * 0.15, 0.001)
        ax.set_xticks(years)
        ax.set_ylim(max(0.0, lower - pad), 1.002)
        numeric_ticks(ax, x=False, y=True, n=5)
        ax.legend(loc="lower right")

    def p2(ax):
        im = ax.imshow(match_matrix, aspect="auto",
                       cmap=LinearSegmentedColormap.from_list("loo_match", ["#F7F7F7", COLORS[1], COLORS[4]]),
                       vmin=0, vmax=100)
        for i in range(match_matrix.shape[0]):
            for j in range(match_matrix.shape[1]):
                value = match_matrix[i, j]
                if value >= 0.1:
                    ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                            fontsize=9.5, fontweight="bold", color="white" if value >= 55 else INK)
        style_heatmap(
            ax, im, list(range(len(YEARS) + 1)), [f"A{c}" for c in range(1, a.selected_k + 1)],
            xlabel="Full-label matches (years)", colorbar=True,
        )
        ax.figure.axes[-1].set_ylabel("Sectors within archetype (%)", fontweight="bold")

    def p3(ax):
        for archetype, color in zip(sorted(a.loo_cluster_emissions["Archetype"].unique()), COLORS[:a.selected_k]):
            d = a.loo_cluster_emissions[a.loo_cluster_emissions["Archetype"] == archetype]
            ax.plot(d["omitted_year"], d["change_vs_full_pct"], marker="o", ms=7,
                    lw=2.4, color=color, label=archetype)
        ax.axhline(0, color=EDGE, ls="--", lw=1.2)
        style_axis(ax, xlabel="Omitted inventory year", ylabel="Emission change vs full (%)")
        ax.set_xticks(years)
        ax.legend(loc="best", ncol=3)
        numeric_ticks(ax, x=False, y=True, n=5)
        pce = a.loo_cluster_emissions[
            (a.loo_cluster_emissions["omitted_year"] == 2019)
            & (a.loo_cluster_emissions["Archetype"] == "A3")
        ]
        if not pce.empty:
            value = float(pce.iloc[0]["change_vs_full_pct"])
            ax.annotate("PCE shifts A1 -> A3", (2019, value), xytext=(18, -24),
                        textcoords="offset points", fontsize=9.5, fontweight="bold",
                        arrowprops={"arrowstyle": "->", "color": EDGE, "lw": 1.1})

    def p4(ax):
        ax.plot(years, summary["priority_rank_spearman"], marker="o", ms=7, lw=2.4,
                color=COLORS[3], label="Priority-rank Spearman")
        ax.plot(years, summary["top50_priority_overlap"], marker="s", ms=6.5, lw=2.4,
                color=COLORS[6], label="Top-50 overlap")
        style_axis(ax, xlabel="Omitted inventory year", ylabel="Priority robustness")
        lower = min(float(summary["priority_rank_spearman"].min()), float(summary["top50_priority_overlap"].min()))
        pad = max((1.0 - lower) * 0.15, 0.001)
        ax.set_xticks(years)
        ax.set_ylim(max(0.0, lower - pad), 1.002)
        numeric_ticks(ax, x=False, y=True, n=5)
        ax.legend(loc="lower right")

    def p5(ax):
        ax.plot(years, summary["silhouette"], marker="o", ms=7, lw=2.4,
                color=COLORS[1], label="Silhouette")
        ax.plot(years, 1.0 - summary["PAC"], marker="s", ms=6.5, lw=2.4,
                color=COLORS[4], label="1 - PAC")
        style_axis(ax, xlabel="Omitted inventory year", ylabel="Structural diagnostic")
        ax.set_xticks(years)
        lower = min(float(summary["silhouette"].min()), float((1-summary["PAC"]).min()))
        ax.set_ylim(max(0, lower - 0.04), 1.02)
        ax.legend(loc="lower right")

    def p6(ax):
        vmax = max(1, int(changed_by_cluster.max()))
        cmap = LinearSegmentedColormap.from_list("loo_changes", ["#F7F7F7", COLORS[6], COLORS[0]])
        im = ax.imshow(changed_by_cluster, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        for i in range(changed_by_cluster.shape[0]):
            for j in range(changed_by_cluster.shape[1]):
                value = int(changed_by_cluster[i, j])
                ax.text(j, i, str(value), ha="center", va="center", fontsize=11,
                        fontweight="bold", color="white" if value >= max(2, 0.65*vmax) else INK)
        style_heatmap(ax, im, years, [f"Full A{c}" for c in range(1, a.selected_k+1)],
                      xlabel="Omitted inventory year", colorbar=True)
        ax.figure.axes[-1].set_ylabel("Reassigned sectors", fontweight="bold")

    return [p1, p2, p3, p4, p5, p6]


def build_figure_9(a: Analysis) -> list[Callable]:
    summary = a.family_summary.copy()
    share_order = summary.sort_values("CO2e_share")
    impact_order = summary.sort_values(["ARI_vs_full", "changed_sectors"], ascending=[True, False])
    transition_pct = np.divide(
        a.family_transition,
        a.family_transition.sum(axis=1, keepdims=True),
        out=np.zeros_like(a.family_transition, dtype=float),
        where=a.family_transition.sum(axis=1, keepdims=True) > 0,
    ) * 100
    origin = a.family_origin_change.pivot(
        index="AttributionSource", columns="Archetype", values="changed_fraction"
    ) * 100
    origin = origin.reindex(impact_order["AttributionSource"]).fillna(0)

    def p1(ax):
        values = 100 * share_order["CO2e_share"].to_numpy()
        bars = ax.barh(np.arange(len(values)), values, color=COLORS[3], edgecolor=EDGE)
        ax.set_yticks(np.arange(len(values)), [compact_attribution(x) for x in share_order["AttributionSource"]])
        style_axis(ax, xlabel="Five-year CO$_2$e share (%)")
        ax.set_xlim(0, max(values) * 1.16)
        for bar, value in zip(bars, values):
            ax.text(value + 0.5, bar.get_y() + bar.get_height()/2, f"{value:.1f}%",
                    va="center", fontsize=10, fontweight="bold", color=INK)

    def p2(ax):
        y = np.arange(len(impact_order))
        values = impact_order["ARI_vs_full"].to_numpy()
        ax.hlines(y, 0, values, color=COLORS[1], lw=3)
        ax.scatter(values, y, s=70, color=COLORS[4], edgecolor=EDGE, zorder=3)
        ax.set_yticks(y, [compact_attribution(x) for x in impact_order["AttributionSource"]])
        ax.invert_yaxis()
        style_axis(ax, xlabel="ARI vs full partition")
        ax.set_xlim(max(0, float(values.min()) - 0.08), 1.04)
        for yi, (value, changed) in enumerate(zip(values, impact_order["changed_sectors"])):
            ax.text(min(1.02, value + 0.02), yi, f"n={int(changed)}", va="center",
                    fontsize=9.5, fontweight="bold", color=INK)

    def p3(ax):
        plotted = summary[(summary["CO2e_share"] >= 0.001) | (summary["changed_sectors"] > 0)].copy()
        x = 100 * plotted["CO2e_share"].to_numpy()
        y = plotted["median_Aitchison_shift"].to_numpy()
        sizes = 60 + 5 * plotted["changed_sectors"].to_numpy()
        ax.scatter(x, y, s=sizes, c=plotted["ARI_vs_full"],
                   cmap=LinearSegmentedColormap.from_list("family_impact", [COLORS[0], COLORS[6], COLORS[2]]),
                   vmin=summary["ARI_vs_full"].min(), vmax=1.0, alpha=0.82,
                   edgecolors=EDGE, linewidths=0.6)
        annotation_offsets = {"Direct": (8, 22), "BEA_Detail_Use_SUT": (8, 14)}
        for _, row in plotted.iterrows():
            family_name = row["AttributionSource"]
            if family_name not in annotation_offsets:
                continue
            ax.annotate(compact_attribution(family_name),
                        (100 * row["CO2e_share"], row["median_Aitchison_shift"]),
                        xytext=annotation_offsets[family_name], textcoords="offset points",
                        fontsize=8.7, fontweight="bold",
                        arrowprops={"arrowstyle": "-", "color": EDGE, "lw": 0.7})
        zero_impact = summary[(summary["CO2e_share"] < 0.001) & (summary["changed_sectors"] <= 0)]
        if not zero_impact.empty:
            labels = "\n".join(compact_attribution(x) for x in zero_impact["AttributionSource"])
            ax.text(0.04, 0.96, f"Zero impact:\n{labels}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=9.2, fontweight="bold", color=MUTED)
        style_axis(ax, xlabel="Family CO$_2$e share (%)",
                   ylabel="Median Aitchison shift")
        numeric_ticks(ax, n=5)

    def p4(ax):
        cmap = LinearSegmentedColormap.from_list("family_transition", ["#F7F7F7", COLORS[1], COLORS[4]])
        im = ax.imshow(transition_pct, cmap=cmap, vmin=0, vmax=100)
        for i in range(a.selected_k):
            for j in range(a.selected_k):
                value = transition_pct[i, j]
                count = a.family_transition[i, j]
                ax.text(j, i, f"{count}\n({value:.1f}%)", ha="center", va="center",
                        fontsize=11, fontweight="bold", color="white" if value >= 55 else INK)
        family = humanize_attribution(a.most_disruptive_family)
        style_heatmap(
            ax, im, [f"A{c}" for c in range(1, a.selected_k + 1)],
            [f"A{c}" for c in range(1, a.selected_k + 1)],
            xlabel="After BEA-use stress", ylabel="Full archetype", colorbar=True,
            xrotation=0,
        )
        ax.figure.axes[-1].set_ylabel("Within-row transition (%)", fontweight="bold")

    def p5(ax):
        ordered = summary.sort_values("p95_Aitchison_shift")
        y = np.arange(len(ordered))
        med = ordered["median_Aitchison_shift"].to_numpy()
        p95 = ordered["p95_Aitchison_shift"].to_numpy()
        ax.hlines(y, med, p95, color=LIGHT, lw=3)
        ax.scatter(med, y, s=58, color=COLORS[3], edgecolor=EDGE, zorder=3, label="Median")
        ax.scatter(p95, y, s=58, color=COLORS[0], edgecolor=EDGE, zorder=3, label="95th percentile")
        ax.set_yticks(y, [compact_attribution(x) for x in ordered["AttributionSource"]])
        style_axis(ax, xlabel="Sector Aitchison shift")
        ax.legend(loc="lower right")

    def p6(ax):
        cmap = LinearSegmentedColormap.from_list("family_origin", ["#F7F7F7", COLORS[6], COLORS[0]])
        im = ax.imshow(origin.to_numpy(), aspect="auto", cmap=cmap, vmin=0,
                       vmax=max(1.0, float(origin.to_numpy().max())))
        for i in range(origin.shape[0]):
            for j in range(origin.shape[1]):
                value = origin.iloc[i, j]
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=9.2,
                        fontweight="bold", color="white" if value >= 35 else INK)
        style_heatmap(
            ax,
            im,
            origin.columns.tolist(),
            [compact_attribution(x) for x in origin.index],
            colorbar=True,
            xrotation=0,
        )
        ax.figure.axes[-1].set_ylabel("Reassigned sectors (%)", fontweight="bold")

    return [p1, p2, p3, p4, p5, p6]


def export_ordered_figure(number: int, stem: str, builders: list[Callable], a: Analysis) -> None:
    folder = FIGURE_OUTPUT_ROOT / f"Fig{number:02d}_{stem}"
    folder.mkdir(parents=True, exist_ok=True)
    n_panels = len(builders)
    # Use the same two-column geometry as the final manuscript figure pages.
    # Keeping at most two panels per row prevents publication-size text from
    # being compressed into an unreadable three-column strip.
    ncols = 2
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(PANEL_WIDTH_IN * ncols, PANEL_HEIGHT_IN * nrows),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).ravel()
    for ax, builder in zip(axes_array, builders):
        builder(ax)
    for ax in axes_array[n_panels:]:
        ax.remove()
    enforce_readable_text(fig)
    save_figure_with_subfigures(fig, folder, f"Fig{number:02d}_{stem}", dpi=600, close=True)

    # Independently render every subfigure from the same panel builder. This
    # intentionally replaces crop-based helper exports with standalone canvases.
    subdir = folder / "subfigures"
    subdir.mkdir(parents=True, exist_ok=True)
    for old in subdir.glob(f"Fig{number:02d}_{stem}_*.*"):
        if old.suffix.lower() in {".png", ".pdf"}:
            old.unlink()
    records = []
    for letter, builder in zip(string.ascii_lowercase, builders):
        panel, ax = plt.subplots(
            figsize=(PANEL_WIDTH_IN, PANEL_HEIGHT_IN),
            constrained_layout=True,
        )
        builder(ax)
        enforce_readable_text(panel)
        png = subdir / f"Fig{number:02d}_{stem}_{letter}.png"
        pdf = subdir / f"Fig{number:02d}_{stem}_{letter}.pdf"
        # Tight bounding boxes preserve long labels and external legends.  The
        # assembler normalises the resulting panel extents into equal cells.
        panel.savefig(png, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.08)
        panel.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.08)
        plt.close(panel)
        records.append((letter, png.name, pdf.name))
    lines = ["# Independently rendered subfigures", "",
             "Each panel was rebuilt on a standalone canvas from the same plotting function; no panel is a crop.", ""]
    lines.extend([f"- {letter}: `{png}`, `{pdf}`" for letter,png,pdf in records])
    (subdir / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def export_tables(a: Analysis) -> dict:
    a.k_table.to_csv(TABLE_ROOT / "sacc_k_selection.csv", index=False, encoding="utf-8-sig")
    a.model_comparison.to_csv(TABLE_ROOT / "sacc_model_comparison.csv", index=False, encoding="utf-8-sig")
    a.closure.to_csv(TABLE_ROOT / "accounting_closure.csv", index=False, encoding="utf-8-sig")
    a.retention_sensitivity.to_csv(
        TABLE_ROOT / "source_retention_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    a.retention_sector_stability.to_csv(
        TABLE_ROOT / "source_retention_sector_stability.csv", index=False, encoding="utf-8-sig"
    )
    a.loo_summary.to_csv(
        TABLE_ROOT / "leave_one_year_out_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    a.loo_cluster_emissions.to_csv(
        TABLE_ROOT / "leave_one_year_out_archetype_emissions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        a.loo_assignments.T,
        index=a.sectors,
        columns=[f"Omit_{year}" for year in YEARS],
    ).rename_axis("Sector").to_csv(
        TABLE_ROOT / "leave_one_year_out_assignments.csv", encoding="utf-8-sig"
    )
    a.family_summary.to_csv(
        TABLE_ROOT / "attribution_family_stress.csv", index=False, encoding="utf-8-sig"
    )
    a.family_origin_change.to_csv(
        TABLE_ROOT / "attribution_family_archetype_change.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        a.family_transition,
        index=[f"Full_A{c}" for c in range(1, a.selected_k + 1)],
        columns=[f"Stressed_A{c}" for c in range(1, a.selected_k + 1)],
    ).to_csv(TABLE_ROOT / "attribution_family_transition.csv", encoding="utf-8-sig")

    totals = a.base.sum(axis=1)
    q05=np.percentile(a.rank_runs,5,axis=0); q50=np.percentile(a.rank_runs,50,axis=0); q95=np.percentile(a.rank_runs,95,axis=0)
    assignments = pd.DataFrame({
        "Sector": a.sectors,
        "SectorName": [a.sector_names.get(s,s) for s in a.sectors],
        "Archetype": a.labels,
        "AnnualMean_kgCO2e": totals,
        "PartitionStability": a.stability,
        "SourceHHI": a.source_hhi,
        "AllocationFragility": a.fragility,
        "MeasurementPriority": a.priority,
        "Rank_p05": q05,
        "Rank_median": q50,
        "Rank_p95": q95,
    })
    assignments.to_csv(TABLE_ROOT / "sacc_sector_assignments.csv", index=False, encoding="utf-8-sig")
    assignments.sort_values("MeasurementPriority", ascending=False).to_csv(
        TABLE_ROOT / "measurement_priority.csv", index=False, encoding="utf-8-sig"
    )

    group_comp, _ = grouped_composition(a)
    profiles = group_comp.assign(Archetype=a.labels).groupby("Archetype").median()
    profiles.to_csv(TABLE_ROOT / "sacc_cluster_profiles.csv", encoding="utf-8-sig")
    edges,_=source_cluster_tables(a)
    edges.to_csv(TABLE_ROOT / "source_cluster_edges.csv",index=False,encoding="utf-8-sig")

    summary = {
        "years": YEARS,
        "n_detailed_sectors": len(a.sectors),
        "n_gases": len(a.gases),
        "lineage_level": LINEAGE_LEVEL,
        "n_lineages": len(a.source_names),
        "n_recorded_meta_sources": int(a.m2["MetaSources"].nunique()),
        "annual_mean_national_GtCO2e": float(a.base.sum()/1e12),
        "selected_K": a.selected_k,
        "source_attenuation_factor": SOURCE_ATTENUATION,
        "selected_K_metrics": a.k_table[a.k_table.K==a.selected_k].iloc[0].to_dict(),
        "median_sector_stability": float(np.median(a.stability)),
        "median_source_HHI": float(np.median(a.source_hhi)),
        "median_allocation_fragility": float(np.median(a.fragility)),
        "retention_sensitivity_levels": RETENTION_LEVELS,
        "leave_one_year_out_min_ARI": float(a.loo_summary["ARI_vs_full"].min()),
        "leave_one_year_out_min_priority_spearman": float(a.loo_summary["priority_rank_spearman"].min()),
        "most_disruptive_attribution_family": a.most_disruptive_family,
        "most_disruptive_family_ARI": float(a.family_summary.iloc[0]["ARI_vs_full"]),
        "most_disruptive_family_changed_sectors": int(a.family_summary.iloc[0]["changed_sectors"]),
        "closure_median_relative_error": float(a.closure.relative_error.median()),
        "closure_max_relative_error": float(a.closure.relative_error.max()),
        "claim_boundary": "Allocation robustness and measurement priority; not causal mitigation or calibrated uncertainty.",
    }
    (TABLE_ROOT / "analysis_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    return summary


def main() -> int:
    a = run_analysis()
    summary = export_tables(a)
    systems = [
        (1,"data_provenance_audit",build_figure_1(a)),
        (2,"accounting_identifiability",build_figure_2(a)),
        (3,"SACC_validation",build_figure_3(a)),
        (4,"robust_archetypes",build_figure_4(a)),
        (5,"attribution_network",build_figure_5(a)),
        (6,"measurement_priorities",build_figure_6(a)),
        (7,"perturbation_severity",build_figure_7(a)),
        (8,"leave_one_year_out",build_figure_8(a)),
        (9,"source_family_stress",build_figure_9(a)),
    ]
    only_raw = os.environ.get("SACC_ONLY_FIGURES", "").strip()
    if only_raw:
        requested = {int(value.strip()) for value in only_raw.split(",") if value.strip()}
        systems = [item for item in systems if item[0] in requested]
    for number,stem,builders in systems:
        export_ordered_figure(number,stem,builders,a)
        print(f"Exported Fig{number:02d}_{stem}")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
