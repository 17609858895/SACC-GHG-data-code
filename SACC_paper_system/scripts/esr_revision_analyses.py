"""Reviewer-requested analyses for the Environmental Systems Research revision.

The script reuses the data handling, zero replacement, constrained perturbation
and consensus functions of ``generate_sacc_paper_figures.py`` and writes every
result table to ``tables/esr_revision``.  Run the main pipeline first; this
script does not modify its outputs.

Sections (reviewer comment in brackets):
  S01 data reconciliation (R2.2)              S09 archetype heterogeneity (R2.9)
  S02 closure discrepancies (R2.3)            S10 continuous shift diagnostics (R2.10)
  S03 zero diagnostics (R1.3, R2.4)           S11 single-year profiles and PCE (R2.11)
  S04 zero-treatment sensitivity (R1.4, R2.4) S12 family severity (R2.12)
  S05 lineage definition (R2.5)               S13 ranking weights and sizes (R2.13)
  S06 single-lineage gases, redistribution    S14 release-revision check (R2.14)
      tests (R1.1, R2.6)                      S15 synthetic allocation errors (R2.14)
  S07 stratified and weighted ensembles (R2.7) S16 lineage-level review leads (R1.2, R2.1)
  S08 K = 2, penalties, size rule, seeds (R2.8)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("SACC_FAST_REPLOT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "3")  # must precede the numpy/scikit-learn imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ARI  # noqa: E402
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score  # noqa: E402

import generate_sacc_paper_figures as G  # noqa: E402

OUT = G.SYSTEM_ROOT / "tables" / "esr_revision"
OUT.mkdir(parents=True, exist_ok=True)
SUMMARY: dict = {}
K = 3
TOP_N = 50
SECTORS_OF_INTEREST = {
    "484": "Truck Transportation",
    "562212": "Solid Waste Landfill",
    "F01000": "Personal Consumption Expenditures",
    "32411": "Petroleum Refineries",
    "2211": "Electric Power Generation, Transmission and Distribution",
    "333415": "Air-conditioning and refrigeration equipment manufacturing",
    "334413": "Semiconductor manufacturing",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(df: pd.DataFrame, name: str, index: bool = False) -> None:
    df.to_csv(OUT / name, index=index, encoding="utf-8-sig")


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    return value


# ---------------------------------------------------------------------------
# Generic building blocks
# ---------------------------------------------------------------------------

def closure_rows(amounts: np.ndarray) -> np.ndarray:
    comp = np.clip(np.asarray(amounts, float), 0.0, None)
    sums = comp.sum(axis=1, keepdims=True)
    return np.divide(comp, sums, out=np.full_like(comp, 1.0 / comp.shape[1]), where=sums > 0)


def zero_replace(amounts: np.ndarray, factor: float = 0.65, cap: float = 0.5,
                 mode: str = "row_min", const: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Multiplicative replacement with configurable delta rule.

    Returns the replaced composition and the replaced mass of each row.
    mode = row_min  : delta = factor x smallest positive part of the row (main rule)
    mode = col_min  : delta_j = factor x smallest positive share of part j across rows
    mode = const    : delta = const (fixed share)
    The total replacement mass of a row is limited to ``cap``.
    """
    comp = closure_rows(amounts)
    col_min = np.array([
        comp[comp[:, j] > 0, j].min() if np.any(comp[:, j] > 0) else 1e-12
        for j in range(comp.shape[1])
    ])
    mass = np.zeros(comp.shape[0])
    for i in range(comp.shape[0]):
        pos = comp[i] > 0
        if pos.all():
            continue
        if not pos.any():
            comp[i] = 1.0 / comp.shape[1]
            continue
        if mode == "row_min":
            delta = np.full((~pos).sum(), factor * comp[i, pos].min())
        elif mode == "col_min":
            delta = factor * col_min[~pos]
        elif mode == "const":
            delta = np.full((~pos).sum(), float(const))
        else:
            raise ValueError(mode)
        total = float(delta.sum())
        if total >= cap:
            delta = delta * cap / total
            total = cap
        comp[i, ~pos] = delta
        comp[i, pos] *= (1.0 - total) / comp[i, pos].sum()
        mass[i] = total
    return comp / comp.sum(axis=1, keepdims=True), mass


def clr_of(comp: np.ndarray) -> np.ndarray:
    lg = np.log(comp)
    return lg - lg.mean(axis=1, keepdims=True)


def make_transform(factor=0.65, cap=0.5, mode="row_min", const=None, amalgamate=None):
    def transform(amounts: np.ndarray) -> np.ndarray:
        x = amounts if amalgamate is None else amounts @ amalgamate
        return clr_of(zero_replace(x, factor, cap, mode, const)[0])
    return transform


BASE_TRANSFORM = make_transform()


def kmeans(x: np.ndarray, k: int = K, seed: int = G.RANDOM_STATE, n_init: int = 10) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(x)


def weighted_consensus(runs: list[np.ndarray], k: int, weights=None):
    n = len(runs[0])
    w = np.ones(len(runs)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    cons = np.zeros((n, n))
    for wi, lab in zip(w, runs):
        cons += wi * (lab[:, None] == lab[None, :])
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    return model.fit_predict(np.clip(1.0 - cons, 0, 1)), cons


def pac(cons: np.ndarray) -> float:
    up = np.triu_indices(cons.shape[0], 1)
    v = cons[up]
    return float(np.mean((v > 0.10) & (v < 0.90)))


def sacc(base: np.ndarray, blocks: list[np.ndarray], transform=BASE_TRANSFORM, k: int = K,
         retention: float = G.SOURCE_ATTENUATION, seed: int = G.RANDOM_STATE, n_init: int = 10,
         weights=None, perturbed: list[np.ndarray] | None = None) -> dict:
    """Rebuild the SACC consensus for a given transform / ensemble."""
    if perturbed is None:
        perturbed = []
        for b in blocks:
            perturbed.append(G.constrained_perturb(base, b, retention))
    xs = [transform(p) for p in perturbed]
    runs = [kmeans(x, k, seed, n_init) for x in xs]
    final0, cons = weighted_consensus(runs, k, weights)
    totals = base.sum(axis=1)
    labels = G.relabel_by_emissions(final0, totals)
    aligned = [G.align_to_reference(r, labels) for r in runs]
    w = np.ones(len(runs)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    agree = np.vstack(aligned) == labels[None, :]
    stability = agree.mean(axis=0) if weights is None else (w[:, None] * agree).sum(axis=0)
    stability = np.where(np.abs(stability - 1.0) < 1e-12, 1.0, stability)
    aris = np.array([ARI(labels, r) for r in runs])
    x0 = transform(base)
    return dict(labels=labels, runs=aligned, stability=stability, consensus=cons, PAC=pac(cons),
                aris=aris, clr=x0, silhouette=float(silhouette_score(x0, labels)))


def pct(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(pct=True).to_numpy()


def top_set(score: np.ndarray, n: int = TOP_N, codes=None) -> set:
    # Descending score, ascending sector code resolves exact ties.
    codes = np.arange(len(score)) if codes is None else np.asarray(codes)
    order = sorted(range(len(score)), key=lambda i: (-score[i], str(codes[i])))
    return set(order[:n])


def lineage_hhi(blocks: list[np.ndarray]) -> np.ndarray:
    ss = np.stack([b.sum(axis=1) for b in blocks], axis=1)
    sh = np.divide(ss, ss.sum(axis=1, keepdims=True), out=np.zeros_like(ss),
                   where=ss.sum(axis=1, keepdims=True) > 0)
    return np.sum(sh ** 2, axis=1)


def blocks_by(m2: pd.DataFrame, key: str, sectors, gases, n_years: int = 5):
    long = m2.groupby([key, "SectorProducedBy", "Flowable"], as_index=False)["CO2e_kg"].sum()
    long["CO2e_kg"] /= n_years
    names = sorted(long[key].unique())
    return names, [G.matrix_from_long(long[long[key] == nm], sectors, gases) for nm in names]


def aitchison_shift(x_ref: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x_new - x_ref, axis=1)


# ---------------------------------------------------------------------------
# Load the primary analysis once
# ---------------------------------------------------------------------------

log(f"Loading primary analysis (lineage level = {G.LINEAGE_LEVEL})")
A = G.run_analysis()
SEC = np.array(A.sectors)
NAMES = A.sector_names
GASES = A.gases
X = A.base
E = X.sum(axis=1)
TOTAL = E.sum()
LAB = A.labels
CLR0 = A.clr
STAB = A.stability
HHI = A.source_hhi
FRAG = A.fragility
PRIO = A.priority
BLOCKS = A.source_matrices
LNAMES = A.source_names
L = len(BLOCKS)
E_PCT = pct(E)
IDX = {s: i for i, s in enumerate(SEC)}
GAS_GROUP = np.array([G.gas_group(g) for g in GASES])
GROUPS = ["CO$_2$", "CH$_4$", "N$_2$O", "F-gases"]
AMALG = np.stack([(GAS_GROUP == g).astype(float) for g in GROUPS], axis=1)  # 17 x 4
FAM_NAMES, FAM_BLOCKS = blocks_by(A.m2, "AttributionSources", A.sectors, GASES)
assert np.allclose(sum(FAM_BLOCKS), X, rtol=1e-9, atol=1e-3)
lineage_family = (A.m2.groupby("Lineage")["AttributionSources"].agg(lambda s: ";".join(sorted(set(s))))
                  .reindex(LNAMES))
assert not lineage_family.str.contains(";").any(), "A lineage maps to several families"

# Per-lineage attenuation results reused below.
PERTURBED = A.perturbed
PCLR = A.perturbed_clr
RUNS = A.source_label_runs
SHIFT = np.stack([aitchison_shift(CLR0, x) for x in PCLR])            # L x n
EREL = np.stack([(p.sum(axis=1) - E) / np.where(E > 0, E, 1) for p in PERTURBED])  # L x n
LIN_ARI = np.array([ARI(LAB, r) for r in RUNS])
LIN_CHANGED = np.array([(r != LAB).sum() for r in RUNS])
SUMMARY["primary"] = dict(lineages=L, recorded_meta_sources=int(A.m2["MetaSources"].nunique()),
                          median_stability=float(np.median(STAB)), median_hhi=float(np.median(HHI)),
                          median_fragility=float(np.median(FRAG)),
                          sectors_stability_lt1=int((STAB < 1).sum()))


def screen_metrics(score: np.ndarray, n: int = TOP_N, ref: set | None = None) -> dict:
    t = top_set(score, n, SEC)
    out = dict(emissions_Gt=float(E[list(t)].sum() / 1e12), share_pct=float(100 * E[list(t)].sum() / TOTAL),
               median_fragility=float(np.median(FRAG[list(t)])))
    if ref is not None:
        out["overlap"] = len(t & ref)
    return out


# ---------------------------------------------------------------------------
# S01 Data reconciliation (R2.2)
# ---------------------------------------------------------------------------

def s01_reconciliation() -> None:
    log("S01 data reconciliation")
    det = pd.read_excel(G.DATA_ROOT / "GHGs_by_Detailed_Sector_US_2019-2023_v2025.1.0.xlsx",
                        sheet_name="Main", dtype={"Sector": str})
    agg = pd.read_excel(G.DATA_ROOT / "GHGs_by_Aggregate_Sector_US_2019-2023_v2025.1.0.xlsx",
                        sheet_name="Main", dtype={"Sector": str})
    m2, m1 = A.m2, A.m1
    rows = []
    for label, excel, parquet in [("detailed", det, m2), ("aggregate", agg, m1)]:
        g = parquet.groupby(["Year", "SectorProducedBy", "Flowable"], as_index=False)["FlowAmount"].sum()
        g = g.rename(columns={"SectorProducedBy": "Sector"})
        mg = excel.merge(g, on=["Year", "Sector", "Flowable"], how="outer", suffixes=("_xlsx", "_parquet"),
                         indicator=True)
        both = mg[mg["_merge"] == "both"]
        rel = np.abs(both["FlowAmount_xlsx"] - both["FlowAmount_parquet"]) / np.maximum(
            np.abs(both["FlowAmount_xlsx"]), 1.0)
        bad = mg[(mg["_merge"] != "both")].copy()
        bad = pd.concat([bad, both[rel > 1e-9]])
        bad["gwp"] = bad["Flowable"].map(G.GWP_AR5)
        bad["CO2e_parquet_kg"] = bad["FlowAmount_parquet"].fillna(0) * bad["gwp"]
        bad["CO2e_xlsx_kg"] = bad["FlowAmount_xlsx"].fillna(0) * bad["gwp"]
        save(bad.drop(columns=["_merge"]), f"S01_{label}_inconsistent_keys.csv")
        rows.append(dict(product=label, excel_records=len(excel), parquet_records=len(parquet),
                         parquet_keys_after_collapsing_sources=len(g), matched_keys=len(both),
                         only_excel=int((mg["_merge"] == "left_only").sum()),
                         only_parquet=int((mg["_merge"] == "right_only").sum()),
                         max_relative_difference=float(rel.max()),
                         keys_differing_gt_1e_9=int((rel > 1e-9).sum()),
                         inconsistent_keys=len(bad),
                         inconsistent_gases=";".join(sorted(bad["Flowable"].unique())),
                         inconsistent_sectors=";".join(sorted(bad["Sector"].unique())),
                         inconsistent_CO2e_parquet_kg=float(bad["CO2e_parquet_kg"].sum()),
                         inconsistent_abs_diff_CO2e_kg=float(np.abs(bad["CO2e_parquet_kg"] - bad["CO2e_xlsx_kg"]).sum()),
                         sectors=int(excel["Sector"].nunique()), gases=int(excel["Flowable"].nunique()),
                         years=int(excel["Year"].nunique())))
    rec = pd.DataFrame(rows)
    save(rec, "S01_record_reconciliation.csv")

    dup = m2.duplicated(["Year", "SectorProducedBy", "Flowable", "MetaSources"]).sum()
    n_sec, n_gas, n_year = len(SEC), len(GASES), len(G.YEARS)
    full_grid = n_sec * n_gas * n_year
    observed = m2.groupby(["Year", "SectorProducedBy", "Flowable"]).size().shape[0]
    yearly = (m2.groupby(["SectorProducedBy", "Flowable", "Year"])["CO2e_kg"].sum() > 0).groupby(
        ["SectorProducedBy", "Flowable"]).sum()
    pos_cells = int((X > 0).sum())
    intermittent = int(((yearly > 0) & (yearly < n_year)).sum())
    steps = pd.DataFrame([
        ("Excel workbooks (detailed + aggregate)", int(rec["excel_records"].sum())),
        ("Detailed Excel records (sector-gas-year totals)", int(rec.loc[0, "excel_records"])),
        ("Aggregate Excel records", int(rec.loc[1, "excel_records"])),
        ("m2 Parquet records (sector-gas-year-lineage)", len(m2)),
        ("m1 Parquet records (aggregate sector-gas-year-lineage)", len(m1)),
        ("Duplicated m2 keys (sector-gas-year-MetaSources)", int(dup)),
        ("m2 sector-gas-year keys after summing lineages", observed),
        ("Complete sector-gas-year grid (527 x 17 x 5)", full_grid),
        ("Absent sector-gas-year combinations (no record; set to 0)", full_grid - observed),
        ("Sector-gas cells in the five-year mean matrix", n_sec * n_gas),
        ("Positive sector-gas cells", pos_cells),
        ("Zero sector-gas cells (absent in all five years)", n_sec * n_gas - pos_cells),
        ("Cells positive in only 1-4 of the five years", intermittent),
        ("Recorded MetaSources entries", int(m2["MetaSources"].nunique())),
        ("Time-consistent lineage series (primary unit)", int(m2["Lineage"].nunique())),
        ("MetaSources entries per inventory year", int(m2.groupby("Year")["MetaSources"].nunique().max())),
    ], columns=["step", "count"])
    save(steps, "S01_processing_counts.csv")
    SUMMARY["S01"] = dict(reconciliation=rec.to_dict("records"), counts=dict(zip(steps.step, steps["count"])))


# ---------------------------------------------------------------------------
# S02 Closure discrepancies (R2.3)
# ---------------------------------------------------------------------------

def s02_closure() -> None:
    log("S02 closure discrepancies")
    c = A.closure.copy()
    c["abs_discrepancy_kgCO2e"] = np.abs(c["m2_reaggregated_kg"] - c["m1_kg"])
    nat_gas = A.m2.groupby(["Year", "Flowable"])["CO2e_kg"].sum()
    nat_year = A.m2.groupby("Year")["CO2e_kg"].sum()
    c["share_of_national_gas_total"] = c["abs_discrepancy_kgCO2e"] / c.set_index(["Year", "Flowable"]).index.map(nat_gas).to_numpy()
    c["share_of_national_total"] = c["abs_discrepancy_kgCO2e"] / c["Year"].map(nat_year).to_numpy()
    flagged = c[c["relative_error"] > 1e-6].sort_values("relative_error", ascending=False)
    names = A.m1.drop_duplicates("SectorProducedBy")
    save(flagged, "S02_closure_flagged.csv")
    by_sector = flagged.groupby(["AggregateSector", "Flowable"]).agg(
        years=("Year", "nunique"), max_relative=("relative_error", "max"),
        mean_abs_kgCO2e=("abs_discrepancy_kgCO2e", "mean"), m1_mean_kgCO2e=("m1_kg", "mean"),
        max_share_national_total=("share_of_national_total", "max")).reset_index()
    save(by_sector, "S02_closure_flagged_by_sector.csv")
    c["signed"] = c["m2_reaggregated_kg"] - c["m1_kg"]
    net = c.groupby(["Year", "Flowable"])["signed"].sum()
    unmatched_m1 = A.m1.groupby(["Year", "SectorProducedBy", "Flowable"]).size().shape[0] - len(c)
    SUMMARY["S02"] = dict(
        combinations=len(c), above_1e_9=int((c.relative_error > 1e-9).sum()),
        above_1e_6=len(flagged), above_1e_3=int((c.relative_error > 1e-3).sum()),
        max_relative=float(c.relative_error.max()),
        max_abs_kg=float(c.abs_discrepancy_kgCO2e.max()),
        sum_abs_flagged_kg=float(flagged.abs_discrepancy_kgCO2e.sum()),
        max_share_national_total=float(c.share_of_national_total.max()),
        max_share_national_gas=float(c.share_of_national_gas_total.max()),
        flagged_by_sector=by_sector.to_dict("records"),
        net_signed_by_gas_year_max_abs_kg=float(net.abs().max()),
        unmatched_m1_keys=int(unmatched_m1),
    )


# ---------------------------------------------------------------------------
# S03 Zero diagnostics (R1.3, R2.4)
# ---------------------------------------------------------------------------

def s03_zeros() -> None:
    log("S03 zero diagnostics")
    comp, mass = zero_replace(X)
    nz = (X <= 0).sum(axis=1)
    rows_zero = nz > 0
    pre_cap = np.array([0.65 * closure_rows(X)[i, X[i] > 0].min() * nz[i] if nz[i] else 0 for i in range(len(X))])
    capped = pre_cap >= 0.5
    smallest = np.array([closure_rows(X)[i, X[i] > 0].min() for i in range(len(X))])
    by_gas = pd.DataFrame({
        "gas": GASES, "group": GAS_GROUP,
        "zero_sectors": (X <= 0).sum(axis=0),
        "positive_below_1t": ((X > 0) & (X < 1e3)).sum(axis=0),
        "national_share_pct": 100 * X.sum(axis=0) / TOTAL,
    })
    save(by_gas, "S03_zeros_by_gas.csv")
    per_sector = pd.DataFrame({"Sector": SEC, "Archetype": LAB, "zero_parts": nz, "replaced_mass": mass,
                               "pre_cap_mass": pre_cap, "cap_reached": capped, "smallest_positive_share": smallest})
    save(per_sector, "S03_zero_replacement_by_sector.csv")
    m = mass[rows_zero]
    SUMMARY["S03"] = dict(
        zero_cells=int((X <= 0).sum()), cells=int(X.size), zero_cell_pct=float(100 * (X <= 0).mean()),
        sectors_with_zero=int(rows_zero.sum()), median_zero_parts=float(np.median(nz)),
        zero_parts_by_archetype={f"A{c}": float(np.median(nz[LAB == c])) for c in range(1, K + 1)},
        cap_reached=int(capped.sum()), cap_reached_sectors=[f"{SEC[i]} {NAMES.get(SEC[i], '')}" for i in np.where(capped)[0]],
        replaced_mass_median=float(np.median(m)), replaced_mass_p90=float(np.percentile(m, 90)),
        replaced_mass_max=float(m.max()), replaced_mass_lt_1e_3=int((m < 1e-3).sum()),
        replaced_mass_lt_1e_2=int((m < 1e-2).sum()), positive_cells_below_1t=int(((X > 0) & (X < 1e3)).sum()),
        fgas_zero_cells=int((X[:, GAS_GROUP == "F-gases"] <= 0).sum()),
        fgas_cells=int((GAS_GROUP == "F-gases").sum() * len(X)),
    )


# ---------------------------------------------------------------------------
# S04 Zero-treatment sensitivity (R1.4, R2.4)
# ---------------------------------------------------------------------------

def k_selection(base: np.ndarray, blocks: list[np.ndarray], transform, ks=range(2, 9),
                perturbed=None, seed=G.RANDOM_STATE) -> pd.DataFrame:
    if perturbed is None:
        perturbed = [G.constrained_perturb(base, b, G.SOURCE_ATTENUATION) for b in blocks]
    xs = [transform(p) for p in perturbed]
    x0 = transform(base)
    rows = []
    for k in ks:
        runs = [kmeans(x, k, seed) for x in xs]
        final, cons = weighted_consensus(runs, k)
        sizes = np.bincount(final, minlength=k)
        aris = [ARI(final, r) for r in runs]
        rows.append(dict(K=k, silhouette=float(silhouette_score(x0, final)),
                         davies_bouldin=float(davies_bouldin_score(x0, final)), PAC=pac(cons),
                         median_source_ARI=float(np.median(aris)), minimum_source_ARI=float(np.min(aris)),
                         minimum_cluster_size=int(sizes.min()),
                         cluster_sizes=";".join(map(str, sorted(sizes.tolist(), reverse=True)))))
    t = pd.DataFrame(rows)
    t["selection_score"] = t["silhouette"] - 0.5 * t["PAC"] - 0.02 * (t["K"] - 3)
    return t


def select_k(t: pd.DataFrame, alpha=0.5, beta=0.02, min_size=20, kmin=3) -> int | None:
    q = t["silhouette"] - alpha * t["PAC"] - beta * (t["K"] - 3)
    ok = (t["minimum_cluster_size"] >= min_size) & (t["K"] >= kmin)
    if not ok.any():
        return None
    return int(t.loc[q[ok].idxmax(), "K"])


def family_stress(base, transform, labels):
    x0 = transform(base)
    out = {}
    for nm, b in zip(FAM_NAMES, FAM_BLOCKS):
        cand = G.constrained_perturb(base, b, G.SOURCE_ATTENUATION)
        lab = G.align_to_reference(kmeans(transform(cand)), labels)
        out[nm] = dict(ARI=float(ARI(labels, lab)), changed=int((lab != labels).sum()),
                       median_shift=float(np.median(aitchison_shift(x0, transform(cand)))))
    return out


def s04_zero_sensitivity() -> None:
    log("S04 zero-treatment sensitivity")
    variants = [
        ("Main rule: 0.65 x row minimum, cap 0.50", dict()),
        ("Factor 0.10", dict(factor=0.10)),
        ("Factor 0.35", dict(factor=0.35)),
        ("Factor 0.90", dict(factor=0.90)),
        ("Cap 0.05", dict(cap=0.05)),
        ("Cap 0.10", dict(cap=0.10)),
        ("Cap 0.25", dict(cap=0.25)),
        ("Cap 0.90", dict(cap=0.90)),
        ("Column detection limit (0.65 x gas minimum)", dict(mode="col_min")),
        ("Fixed share 1e-6", dict(mode="const", const=1e-6)),
        ("Fixed share 1e-4", dict(mode="const", const=1e-4)),
        ("Four-part amalgamation (CO2, CH4, N2O, F-gases)", dict(amalgamate=AMALG)),
    ]
    rows, krows = [], []
    base_top = top_set(PRIO, TOP_N, SEC)
    base_fam = family_stress(X, BASE_TRANSFORM, LAB)
    for name, kw in variants:
        tr = make_transform(**kw)
        r = sacc(X, BLOCKS, tr, perturbed=PERTURBED)
        aligned = G.align_to_reference(r["labels"], LAB)
        frag = 1 - r["stability"] * (1 - HHI)
        prio = np.sqrt(E_PCT * pct(frag))
        fam = family_stress(X, tr, r["labels"])
        kt = k_selection(X, BLOCKS, tr, perturbed=PERTURBED)
        kt.insert(0, "variant", name)
        krows.append(kt)
        mass = zero_replace(X @ kw["amalgamate"] if "amalgamate" in kw else X,
                            kw.get("factor", 0.65), kw.get("cap", 0.5), kw.get("mode", "row_min"),
                            kw.get("const"))[1]
        rows.append(dict(
            variant=name, ARI_vs_main=float(ARI(LAB, r["labels"])), changed_vs_main=int((aligned != LAB).sum()),
            selected_K_3to8=select_k(kt), best_K_2to8=select_k(kt, kmin=2),
            silhouette=r["silhouette"], PAC=r["PAC"], median_lineage_ARI=float(np.median(r["aris"])),
            minimum_lineage_ARI=float(np.min(r["aris"])), sectors_stability_lt1=int((r["stability"] < 1).sum()),
            top50_overlap_with_main=len(top_set(prio, TOP_N, SEC) & base_top),
            priority_spearman=float(spearmanr(prio, PRIO).statistic),
            most_disruptive_family=min(fam, key=lambda f: (fam[f]["ARI"], -fam[f]["changed"])),
            BEA_use_changed=fam["BEA_Detail_Use_SUT"]["changed"], EIA_MECS_changed=fam["EIA_MECS_Energy"]["changed"],
            Direct_changed=fam["Direct"]["changed"],
            other_families_changed=int(sum(v["changed"] for f, v in fam.items()
                                           if f not in {"BEA_Detail_Use_SUT", "EIA_MECS_Energy", "Direct"})),
            truck_landfill_same=bool(aligned[IDX["484"]] == aligned[IDX["562212"]]),
            median_replaced_mass=float(np.median(mass[mass > 0])) if np.any(mass > 0) else 0.0,
            cap_reached=int(np.sum(mass >= kw.get("cap", 0.5) - 1e-12)) if kw.get("mode", "row_min") != "const" else 0,
        ))
        log(f"   {name}: ARI={rows[-1]['ARI_vs_main']:.3f} changed={rows[-1]['changed_vs_main']}")
    t = pd.DataFrame(rows)
    save(t, "S04_zero_sensitivity.csv")
    save(pd.concat(krows), "S04_zero_sensitivity_k_selection.csv")
    SUMMARY["S04"] = dict(table=t.to_dict("records"), main_family=base_fam)


# ---------------------------------------------------------------------------
# S05 Lineage definition (R2.5)
# ---------------------------------------------------------------------------

def table_key(meta: str) -> str:
    s = G.lineage_series(meta)
    return s.split(".")[0]


def s05_lineage_definition() -> None:
    log("S05 lineage definition")
    m2 = A.m2.copy()
    m2["GHGITable"] = m2["MetaSources"].map(table_key)
    levels = [("Recorded MetaSources entries", "MetaSources"), ("Time-consistent series (primary)", "Lineage"),
              ("GHGI source table", "GHGITable"), ("Attribution family", "AttributionSources")]
    base_top = top_set(PRIO, TOP_N, SEC)
    emis_top = top_set(E, TOP_N, SEC)
    hh = {}
    rows = []
    for label, key in levels:
        names, blocks = blocks_by(m2, key, A.sectors, GASES)
        h = lineage_hhi(blocks)
        hh[label] = h
        ph = np.sqrt(E_PCT * pct(h))
        t = top_set(ph, TOP_N, SEC)
        rows.append(dict(level=label, units=len(names), median_HHI=float(np.median(h)),
                         p90_HHI=float(np.percentile(h, 90)),
                         spearman_HHI_vs_primary=float(spearmanr(h, HHI).statistic),
                         top50_emissions_HHI_overlap_with_primary_joint=len(t & base_top),
                         top50_overlap_with_emissions_only=len(t & emis_top),
                         top50_emissions_share_pct=float(100 * E[list(t)].sum() / TOTAL),
                         petroleum_refineries_HHI=float(h[IDX["32411"]]),
                         petroleum_refineries_joint_rank=int((ph > ph[IDX["32411"]]).sum() + 1)))
    # Recorded-entry SACC (original submission), for comparison of stability and screen.
    rec_names, rec_blocks = blocks_by(m2, "MetaSources", A.sectors, GASES)
    r = sacc(X, rec_blocks)
    frag_rec = 1 - r["stability"] * (1 - hh["Recorded MetaSources entries"])
    prio_rec = np.sqrt(E_PCT * pct(frag_rec))
    # GHGI-table-level SACC (perturb all entries of a table jointly).
    tab_names, tab_blocks = blocks_by(m2, "GHGITable", A.sectors, GASES)
    rt = sacc(X, tab_blocks)
    frag_tab = 1 - rt["stability"] * (1 - hh["GHGI source table"])
    prio_tab = np.sqrt(E_PCT * pct(frag_tab))
    t = pd.DataFrame(rows)
    save(t, "S05_lineage_definition.csv")
    # Annex series detail.
    annex = (m2[m2.MetaSources != m2.Lineage].groupby(["Lineage", "MetaSources"])["Year"]
             .agg(lambda s: ",".join(map(str, sorted(set(s))))).reset_index())
    save(annex, "S05_merged_year_specific_entries.csv")
    SUMMARY["S05"] = dict(
        table=t.to_dict("records"),
        recorded=dict(units=len(rec_names), ARI_vs_primary=float(ARI(LAB, r["labels"])),
                      median_lineage_ARI=float(np.median(r["aris"])), min_lineage_ARI=float(np.min(r["aris"])),
                      stability_lt1=int((r["stability"] < 1).sum()),
                      top50_overlap_with_primary=len(top_set(prio_rec, TOP_N, SEC) & base_top),
                      top50_share_pct=float(100 * E[list(top_set(prio_rec, TOP_N, SEC))].sum() / TOTAL)),
        table_level=dict(units=len(tab_names), ARI_vs_primary=float(ARI(LAB, rt["labels"])),
                         stability_lt1=int((rt["stability"] < 1).sum()),
                         top50_overlap_with_primary=len(top_set(prio_tab, TOP_N, SEC) & base_top)),
        merged_entries=len(annex), merged_series=int(annex.Lineage.nunique()),
    )
    np.save(OUT / "S05_hhi_levels.npy", np.stack([hh[l] for l, _ in levels]))


# ---------------------------------------------------------------------------
# S06 Single-lineage gases and within-lineage redistribution (R1.1, R2.6)
# ---------------------------------------------------------------------------

def redistribute(block: np.ndarray, how: str, lam: float = 0.5) -> np.ndarray:
    out = block.copy()
    for g in range(block.shape[1]):
        col = block[:, g]
        tot = col.sum()
        sup = col > 0
        if tot <= 0 or sup.sum() < 2:
            continue
        s = col[sup] / tot
        if how == "flatten":
            s_new = (1 - lam) * s + lam / sup.sum()
        elif how == "sharpen":
            s_new = s ** 2 / np.sum(s ** 2)
        else:
            raise ValueError(how)
        out[sup, g] = s_new * tot
    return out


def s06_single_lineage_and_redistribution() -> None:
    log("S06 single-lineage gases and redistribution")
    gas_tot = X.sum(axis=0)
    lin_gas = np.stack([b.sum(axis=0) for b in BLOCKS])           # L x 17
    n_lin = (lin_gas > 0).sum(axis=0)
    gas_rows = pd.DataFrame({"gas": GASES, "lineages_with_positive_mass": n_lin,
                             "largest_lineage_share": lin_gas.max(axis=0) / gas_tot,
                             "largest_lineage": [G.humanize_meta_source(LNAMES[i]) for i in lin_gas.argmax(axis=0)]})
    save(gas_rows, "S06_gas_lineage_coverage.csv")
    sole = gas_rows[gas_rows.lineages_with_positive_mass == 1]
    # Fraction of each lineage's mass that is exactly reversed by column rescaling.
    reversed_frac = np.array([lin_gas[l, n_lin == 1].sum() / max(lin_gas[l].sum(), 1e-30) for l in range(L)])
    noop = SHIFT.max(axis=1) < 1e-9
    support_one = np.array([all(((b[:, g] > 0).sum() <= 1) for g in range(b.shape[1]) if b[:, g].sum() > 0)
                            for b in BLOCKS])

    def run_redistribution(how: str):
        pert = [np.clip(X - b + redistribute(b, how), 0, None) for b in BLOCKS]
        for p in pert:
            assert np.allclose(p.sum(axis=0), X.sum(axis=0), rtol=1e-10, atol=1e-2)
        xs = [BASE_TRANSFORM(p) for p in pert]
        runs = [G.align_to_reference(kmeans(x), LAB) for x in xs]
        shift = np.stack([aitchison_shift(CLR0, x) for x in xs])
        erel = np.stack([(p.sum(axis=1) - E) / E for p in pert])
        return pert, runs, shift, erel

    res = {}
    lin_rows = []
    for how in ["flatten", "sharpen"]:
        pert, runs, shift, erel = run_redistribution(how)
        aris = np.array([ARI(LAB, r) for r in runs])
        changed = np.array([(r != LAB).sum() for r in runs])
        stab = np.mean(np.vstack(runs) == LAB[None, :], axis=0)
        res[how] = dict(median_ARI=float(np.median(aris)), min_ARI=float(aris.min()),
                        tests_with_label_change=int((changed > 0).sum()),
                        noop_tests=int((shift.max(axis=1) < 1e-9).sum()),
                        sectors_stability_lt1=int((stab < 1).sum()),
                        median_sector_max_shift=float(np.median(shift.max(axis=0))),
                        sectors_max_abs_erel_gt_10pct=int((np.abs(erel).max(axis=0) > 0.10).sum()),
                        informative_where_attenuation_noop=int(np.sum(noop & (shift.max(axis=1) > 1e-6))))
        for l in range(L):
            lin_rows.append(dict(lineage=LNAMES[l], test=how, ARI=aris[l], changed=int(changed[l]),
                                 max_shift=float(shift[l].max()), max_abs_erel=float(np.abs(erel[l]).max())))
        res[how + "_stability"] = stab
    att = dict(median_ARI=float(np.median(LIN_ARI)), min_ARI=float(LIN_ARI.min()),
               tests_with_label_change=int((LIN_CHANGED > 0).sum()), noop_tests=int(noop.sum()),
               sectors_stability_lt1=int((STAB < 1).sum()),
               median_sector_max_shift=float(np.median(SHIFT.max(axis=0))),
               sectors_max_abs_erel_gt_10pct=int((np.abs(EREL).max(axis=0) > 0.10).sum()))
    # Combined ensemble (attenuation + flatten + sharpen): stability and screen.
    comb = np.mean(np.vstack([STAB, res["flatten_stability"], res["sharpen_stability"]]), axis=0)
    frag_c = 1 - comb * (1 - HHI)
    prio_c = np.sqrt(E_PCT * pct(frag_c))
    save(pd.DataFrame(lin_rows), "S06_redistribution_by_lineage.csv")
    per_lin = pd.DataFrame({"lineage": LNAMES, "family": lineage_family.values,
                            "share_pct": 100 * lin_gas.sum(axis=1) / TOTAL,
                            "mass_reversed_by_rescaling": reversed_frac, "attenuation_noop": noop,
                            "single_sector_support": support_one})
    save(per_lin, "S06_lineage_reversal.csv")
    SUMMARY["S06"] = dict(
        sole_lineage_gases=sole[["gas", "largest_lineage"]].to_dict("records"),
        lineages_fully_reversed=int(np.sum(reversed_frac > 0.999)),
        lineages_partly_reversed=int(np.sum((reversed_frac > 0) & (reversed_frac <= 0.999))),
        attenuation_noop_tests=int(noop.sum()),
        noop_lineages=[G.humanize_meta_source(LNAMES[l]) for l in np.where(noop)[0]],
        single_sector_support_lineages=int(support_one.sum()),
        attenuation=att, flatten=res["flatten"], sharpen=res["sharpen"],
        combined_stability_lt1=int((comb < 1).sum()),
        combined_top50_overlap=len(top_set(prio_c, TOP_N, SEC) & top_set(PRIO, TOP_N, SEC)),
    )


# ---------------------------------------------------------------------------
# S07 Stratified summaries and ensemble weights (R2.7)
# ---------------------------------------------------------------------------

def s07_stratified_weighted() -> None:
    log("S07 stratified and weighted ensembles")
    lin_mass = np.array([b.sum() for b in BLOCKS])
    share = lin_mass / TOTAL
    lin_gas = np.stack([[b[:, GAS_GROUP == g].sum() for g in GROUPS] for b in BLOCKS])
    dom = np.array(GROUPS)[lin_gas.argmax(axis=1)]
    noop = SHIFT.max(axis=1) < 1e-9
    tab = pd.DataFrame({"lineage": LNAMES, "family": lineage_family.values, "share_pct": 100 * share,
                        "dominant_gas_group": [d.replace("$", "").replace("_", "") for d in dom],
                        "sectors_covered": [(b.sum(axis=1) > 0).sum() for b in BLOCKS],
                        "ARI": LIN_ARI, "changed_labels": LIN_CHANGED,
                        "median_shift": np.median(SHIFT, axis=1), "max_shift": SHIFT.max(axis=1),
                        "max_abs_emission_change": np.abs(EREL).max(axis=1), "noop": noop})
    bins = pd.cut(tab.share_pct, [-1, 0.01, 0.1, 1, 101], labels=["<0.01%", "0.01-0.1%", "0.1-1%", ">=1%"])
    tab["mass_class"] = bins
    save(tab, "S07_lineage_tests.csv")

    def strat(col):
        g = tab.groupby(col, observed=True)
        return pd.DataFrame({"tests": g.size(), "noop": g.noop.sum(), "median_ARI": g.ARI.median(),
                             "minimum_ARI": g.ARI.min(), "tests_with_label_change": g.changed_labels.apply(lambda s: int((s > 0).sum())),
                             "mean_changed_labels": g.changed_labels.mean(), "median_max_shift": g.max_shift.median()})
    s_mass = strat("mass_class")
    s_gas = strat("dominant_gas_group")
    s_fam = strat("family")
    save(s_mass, "S07_by_mass_class.csv", index=True)
    save(s_gas, "S07_by_gas_group.csv", index=True)
    save(s_fam, "S07_by_family.csv", index=True)
    # Gas-stratified sector stability: tests whose dominant gas is in each group.
    gas_stab = {}
    for g in tab.dominant_gas_group.unique():
        sel = np.where(tab.dominant_gas_group == g)[0]
        st = np.mean(np.vstack([RUNS[i] for i in sel]) == LAB[None, :], axis=0)
        gas_stab[g] = dict(tests=len(sel), sectors_lt1=int((st < 1).sum()), min_stability=float(st.min()))
    schemes = {
        "Equal (main)": np.ones(L),
        "Equal, excluding no-op tests": (~noop).astype(float),
        "Mass-weighted": share,
        "Square-root mass": np.sqrt(share),
        "Gas-group balanced": np.array([1.0 / np.sum(dom == d) for d in dom]),
    }
    rows = []
    base_top = top_set(PRIO, TOP_N, SEC)
    for nm, w in schemes.items():
        lab0, cons = weighted_consensus([np.asarray(r) for r in RUNS], K, w)
        lab = G.align_to_reference(G.relabel_by_emissions(lab0, E), LAB)
        ww = w / w.sum()
        st = (ww[:, None] * (np.vstack(RUNS) == lab[None, :])).sum(axis=0)
        frag = 1 - st * (1 - HHI)
        prio = np.sqrt(E_PCT * pct(frag))
        rows.append(dict(weights=nm, ARI_vs_main=float(ARI(LAB, lab)), changed=int((lab != LAB).sum()), PAC=pac(cons),
                         sectors_stability_lt1=int((st < 0.999999).sum()), min_stability=float(st.min()),
                         top50_overlap=len(top_set(prio, TOP_N, SEC) & base_top),
                         priority_spearman=float(spearmanr(prio, PRIO).statistic)))
    w = pd.DataFrame(rows)
    save(w, "S07_weighted_ensembles.csv")
    SUMMARY["S07"] = dict(by_mass=s_mass.reset_index().to_dict("records"), by_gas=s_gas.reset_index().to_dict("records"),
                          by_family=s_fam.reset_index().to_dict("records"), gas_stratified_stability=gas_stab,
                          weights=w.to_dict("records"),
                          share_of_tests_lt_0_1pct=float(np.mean(share < 0.001)))


# ---------------------------------------------------------------------------
# S08 K = 2, selection penalties, minimum size and random seeds (R2.8)
# ---------------------------------------------------------------------------

def s08_k_and_seeds() -> None:
    log("S08 K selection, penalties and seeds")
    kt = k_selection(X, BLOCKS, BASE_TRANSFORM, perturbed=PERTURBED)
    save(kt, "S08_k_selection_2to8.csv")
    grid = []
    for alpha in [0, 0.25, 0.5, 1.0, 2.0]:
        for beta in [0, 0.01, 0.02, 0.05, 0.10]:
            for ms in [1, 10, 20, 30, 50]:
                for kmin in [2, 3]:
                    grid.append(dict(alpha=alpha, beta=beta, min_size=ms, kmin=kmin,
                                     selected=select_k(kt, alpha, beta, ms, kmin)))
    grid = pd.DataFrame(grid)
    save(grid, "S08_selection_grid.csv")
    # K = 2 partition versus the K = 3 solution.
    r2 = sacc(X, BLOCKS, k=2, perturbed=PERTURBED)
    ct = pd.crosstab(pd.Series(LAB, name="K3"), pd.Series(r2["labels"], name="K2"))
    save(ct, "S08_k2_vs_k3.csv", index=True)
    # Seed variability of the full SACC consensus and of single-run CLR K-means.
    seed_rows = []
    for seed in range(20):
        r = sacc(X, BLOCKS, seed=seed, perturbed=PERTURBED)
        lab = G.align_to_reference(r["labels"], LAB)
        frag = 1 - r["stability"] * (1 - HHI)
        prio = np.sqrt(E_PCT * pct(frag))
        seed_rows.append(dict(seed=seed, ARI_vs_main=float(ARI(LAB, r["labels"])), changed=int((lab != LAB).sum()),
                              PAC=r["PAC"], min_lineage_ARI=float(r["aris"].min()),
                              sectors_stability_lt1=int((r["stability"] < 1).sum()),
                              top50_overlap=len(top_set(prio, TOP_N, SEC) & top_set(PRIO, TOP_N, SEC))))
    seeds = pd.DataFrame(seed_rows)
    save(seeds, "S08_seed_sacc.csv")
    opt = []
    for n_init in [1, 10]:
        for seed in range(100 if n_init == 1 else 20):
            lab = kmeans(CLR0, K, seed, n_init)
            opt.append(dict(n_init=n_init, seed=seed, ARI_vs_main=float(ARI(LAB, lab))))
    opt = pd.DataFrame(opt)
    save(opt, "S08_seed_unperturbed_kmeans.csv")
    SUMMARY["S08"] = dict(
        k_table=kt.to_dict("records"),
        selected_main=select_k(kt), selected_including_k2=select_k(kt, kmin=2),
        grid_frequency_kmin3=grid[grid.kmin == 3].selected.value_counts().to_dict(),
        grid_frequency_kmin2=grid[grid.kmin == 2].selected.value_counts().to_dict(),
        k2_crosstab=ct.to_dict(), k2_silhouette=r2["silhouette"], k2_PAC=r2["PAC"],
        seed_sacc=dict(min_ARI=float(seeds.ARI_vs_main.min()), max_changed=int(seeds.changed.max()),
                       min_top50=int(seeds.top50_overlap.min()), PAC_range=[float(seeds.PAC.min()), float(seeds.PAC.max())]),
        seed_unperturbed={str(n): dict(min_ARI=float(g.ARI_vs_main.min()), frac_ARI_1=float((g.ARI_vs_main > 0.999999).mean()))
                          for n, g in opt.groupby("n_init")},
    )


# ---------------------------------------------------------------------------
# S09 Archetype heterogeneity, truck transportation and landfill (R2.9)
# ---------------------------------------------------------------------------

def s09_heterogeneity() -> None:
    log("S09 archetype heterogeneity")
    comp = zero_replace(X)[0]
    cent = np.stack([CLR0[LAB == c].mean(axis=0) for c in range(1, K + 1)])
    d_own = np.linalg.norm(CLR0 - cent[LAB - 1], axis=1)
    d_all = np.linalg.norm(CLR0[:, None, :] - cent[None, :, :], axis=2)
    sst = ((CLR0 - CLR0.mean(axis=0)) ** 2).sum()
    ssw = ((CLR0 - cent[LAB - 1]) ** 2).sum()
    eta = []
    for j, g in enumerate(GASES):
        v = CLR0[:, j]
        sw = sum(((v[LAB == c] - v[LAB == c].mean()) ** 2).sum() for c in range(1, K + 1))
        eta.append(1 - sw / ((v - v.mean()) ** 2).sum())
    eta = pd.DataFrame({"gas": GASES, "eta2": eta}).sort_values("eta2", ascending=False)
    save(eta, "S09_clr_coordinate_eta2.csv")
    shares = closure_rows(X)
    grp = shares @ AMALG
    dom = np.array(["CO2", "CH4", "N2O", "F-gases"])[grp.argmax(axis=1)]
    dom_tab = pd.crosstab(pd.Series(LAB, name="Archetype"), pd.Series(dom, name="dominant_gas_group"))
    save(dom_tab, "S09_dominant_gas_by_archetype.csv", index=True)
    het = pd.DataFrame({
        "Archetype": [f"A{c}" for c in range(1, K + 1)],
        "n": [int((LAB == c).sum()) for c in range(1, K + 1)],
        "median_distance_to_centroid": [float(np.median(d_own[LAB == c])) for c in range(1, K + 1)],
        "p90_distance_to_centroid": [float(np.percentile(d_own[LAB == c], 90)) for c in range(1, K + 1)],
        "median_zero_parts": [float(np.median((X[LAB == c] <= 0).sum(axis=1))) for c in range(1, K + 1)],
        "median_CO2_share": [float(np.median(grp[LAB == c, 0])) for c in range(1, K + 1)],
        "median_silhouette": [float(np.median(silhouette_samples(CLR0, LAB)[LAB == c])) for c in range(1, K + 1)],
    })
    save(het, "S09_within_cluster_heterogeneity.csv")
    centroid_dist = pd.DataFrame(np.linalg.norm(cent[:, None] - cent[None], axis=2),
                                 index=[f"A{c}" for c in range(1, 4)], columns=[f"A{c}" for c in range(1, 4)])
    save(centroid_dist, "S09_centroid_distances.csv", index=True)
    # Zero-pattern only and dense-gas only clusterings.
    presence = (X > 0).astype(float)
    pres_lab = kmeans(presence)
    dense = (X > 0).mean(axis=0) >= 0.95
    dense_lab = kmeans(clr_of(zero_replace(X[:, dense])[0])) if dense.sum() >= 2 else None
    nz = (X <= 0).sum(axis=1)
    # Truck and landfill decomposition.
    t, lf = IDX["484"], IDX["562212"]
    D = len(GASES)
    pair_rows = []
    lr = lambda v: v[:, None] - v[None, :]
    for name, i in [("Truck Transportation", t), ("Solid Waste Landfill", lf)]:
        for c in range(K):
            diff = lr(CLR0[i]) - lr(cent[c])
            contrib = (diff ** 2) / D / 2  # each unordered pair counted once
            iu = np.triu_indices(D, 1)
            order = np.argsort(contrib[iu])[::-1][:5]
            pair_rows.append(dict(sector=name, centroid=f"A{c + 1}", distance=float(d_all[i, c]),
                                  top_pairs="; ".join(f"{GASES[iu[0][o]]}/{GASES[iu[1][o]]}:{contrib[iu][o]:.2f}" for o in order)))
    pairs = pd.DataFrame(pair_rows)
    save(pairs, "S09_truck_landfill_centroid_distances.csv")
    coord = pd.DataFrame({"gas": GASES, "truck_share": shares[t], "landfill_share": shares[lf],
                          "truck_clr": CLR0[t], "landfill_clr": CLR0[lf], "A1_centroid": cent[0],
                          "A2_centroid": cent[1], "A3_centroid": cent[2],
                          "truck_zero": X[t] <= 0, "landfill_zero": X[lf] <= 0})
    # Contribution of each coordinate to the squared distance to A1 versus A2.
    for nm, i in [("truck", t), ("landfill", lf)]:
        coord[f"{nm}_sq_to_A1"] = (CLR0[i] - cent[0]) ** 2
        coord[f"{nm}_sq_to_A2"] = (CLR0[i] - cent[1]) ** 2
        coord[f"{nm}_sq_to_A3"] = (CLR0[i] - cent[2]) ** 2
    save(coord, "S09_truck_landfill_coordinates.csv")
    same_zero = int(np.sum((X[t] <= 0) == (X[lf] <= 0)))
    SUMMARY["S09"] = dict(
        R2_between=float(1 - ssw / sst), eta2_top=eta.head(6).to_dict("records"), eta2_bottom=eta.tail(4).to_dict("records"),
        heterogeneity=het.to_dict("records"), dominant_gas=dom_tab.to_dict(), centroid_distances=centroid_dist.to_dict(),
        presence_only_ARI=float(ARI(LAB, pres_lab)),
        dense_gases=[GASES[j] for j in np.where(dense)[0]],
        dense_only_ARI=float(ARI(LAB, dense_lab)) if dense_lab is not None else None,
        zero_count_by_archetype={f"A{c}": [int(np.percentile(nz[LAB == c], q)) for q in (10, 50, 90)] for c in range(1, 4)},
        truck=dict(distance_to=d_all[t].tolist(), zero_parts=int((X[t] <= 0).sum())),
        landfill=dict(distance_to=d_all[lf].tolist(), zero_parts=int((X[lf] <= 0).sum())),
        truck_landfill_distance=float(np.linalg.norm(CLR0[t] - CLR0[lf])),
        truck_landfill_same_zero_pattern_parts=same_zero,
        median_within_A1_pair_distance=float(np.median([np.linalg.norm(CLR0[i] - CLR0[j]) for i in np.where(LAB == 1)[0]
                                                         for j in np.where(LAB == 1)[0] if i < j])),
        pairs=pairs.to_dict("records"),
    )


# ---------------------------------------------------------------------------
# S10 Continuous diagnostics for individual-lineage tests (R2.10)
# ---------------------------------------------------------------------------

def s10_continuous() -> None:
    log("S10 continuous shift diagnostics")
    d_w = float(np.median(np.linalg.norm(CLR0 - np.stack([CLR0[LAB == c].mean(axis=0) for c in range(1, 4)])[LAB - 1], axis=1)))
    max_shift = SHIFT.max(axis=0)
    arg_shift = SHIFT.argmax(axis=0)
    max_erel = np.abs(EREL).max(axis=0)
    arg_erel = np.abs(EREL).argmax(axis=0)
    stable = STAB >= 1 - 1e-12
    big_shift = max_shift >= d_w
    big_e = max_erel >= 0.10
    tab = pd.DataFrame({
        "Sector": SEC, "SectorName": [NAMES.get(s, s) for s in SEC], "Archetype": LAB,
        "AnnualMean_MtCO2e": E / 1e9, "Stability": STAB, "HHI": HHI,
        "max_Aitchison_shift": max_shift, "lineage_max_shift": [G.humanize_meta_source(LNAMES[i]) for i in arg_shift],
        "lineage_max_shift_share_in_sector": [BLOCKS[l][i].sum() / E[i] for i, l in enumerate(arg_shift)],
        "max_abs_emission_change": max_erel, "lineage_max_emission_change": [G.humanize_meta_source(LNAMES[i]) for i in arg_erel],
        "stable_label_large_shift": stable & big_shift, "stable_label_large_emission_change": stable & big_e,
    })
    save(tab.sort_values("max_Aitchison_shift", ascending=False), "S10_sector_continuous_diagnostics.csv")
    hid = tab[(tab.stable_label_large_shift | tab.stable_label_large_emission_change)]
    shift_pct = pct(max_shift)
    prio_shift = np.sqrt(E_PCT * shift_pct)
    SUMMARY["S10"] = dict(
        within_cluster_median_distance=d_w,
        stable_sectors=int(stable.sum()), stable_large_shift=int((stable & big_shift).sum()),
        stable_large_emission_change=int((stable & big_e).sum()), stable_either=int(len(hid)),
        stable_either_emissions_share_pct=float(100 * hid.AnnualMean_MtCO2e.sum() * 1e9 / TOTAL),
        spearman_maxshift_HHI=float(spearmanr(max_shift, HHI).statistic),
        spearman_maxshift_fragility=float(spearmanr(max_shift, FRAG).statistic),
        top50_overlap_emissions_shift_vs_primary=len(top_set(prio_shift, TOP_N, SEC) & top_set(PRIO, TOP_N, SEC)),
        examples=hid.sort_values("AnnualMean_MtCO2e", ascending=False).head(12).to_dict("records"),
        lineage_max_shift_low_share=int(np.sum((tab.lineage_max_shift_share_in_sector < 0.05) & big_shift)),
    )


# ---------------------------------------------------------------------------
# S11 Single-year profiles and Personal Consumption Expenditures (R2.11)
# ---------------------------------------------------------------------------

def s11_years() -> None:
    log("S11 single-year profiles and PCE")
    cent = np.stack([CLR0[LAB == c].mean(axis=0) for c in range(1, 4)])
    pce = IDX["F01000"]
    rows, pce_rows, lineage_rows = [], [], []
    year_labels = {}
    for year in G.YEARS:
        sub = A.m2[A.m2.Year == year]
        mean_long = sub.groupby(["SectorProducedBy", "Flowable"], as_index=False)["CO2e_kg"].sum()
        Xt = G.matrix_from_long(mean_long, A.sectors, GASES)
        names, blocks = blocks_by(sub, "Lineage", A.sectors, GASES, n_years=1)
        r = sacc(Xt, blocks)
        lab = G.align_to_reference(r["labels"], LAB)
        year_labels[year] = lab
        ct = BASE_TRANSFORM(Xt)
        nearest = np.argmin(np.linalg.norm(ct[:, None] - cent[None], axis=2), axis=1) + 1
        rows.append(dict(year=year, SACC_ARI_vs_reference=float(ARI(LAB, lab)), SACC_changed=int((lab != LAB).sum()),
                         nearest_centroid_changed=int((nearest != LAB).sum()), PAC=r["PAC"],
                         silhouette=r["silhouette"], PCE_label=int(lab[pce]), PCE_nearest=int(nearest[pce]),
                         profile_shift_vs_mean_median=float(np.median(np.linalg.norm(ct - CLR0, axis=1)))))
        grp = closure_rows(Xt) @ AMALG
        d = np.linalg.norm(ct[pce] - cent, axis=1)
        pce_rows.append(dict(year=year, total_MtCO2e=float(Xt[pce].sum() / 1e9), CO2_pct=100 * grp[pce, 0],
                             CH4_pct=100 * grp[pce, 1], N2O_pct=100 * grp[pce, 2], Fgas_pct=100 * grp[pce, 3],
                             dist_A1=float(d[0]), dist_A2=float(d[1]), dist_A3=float(d[2]),
                             margin_A3_minus_A1=float(d[2] - d[0]), zero_parts=int((Xt[pce] <= 0).sum())))
        for nm, b in zip(names, blocks):
            v = b[pce].sum()
            if v > 0:
                lineage_rows.append(dict(year=year, lineage=G.humanize_meta_source(nm), share=v / Xt[pce].sum()))
    yr = pd.DataFrame(rows)
    pc = pd.DataFrame(pce_rows)
    ln = pd.DataFrame(lineage_rows).pivot_table(index="lineage", columns="year", values="share", fill_value=0)
    ln = ln.loc[ln.max(axis=1).sort_values(ascending=False).index]
    save(yr, "S11_single_year_partitions.csv")
    save(pc, "S11_PCE_by_year.csv")
    save(ln, "S11_PCE_lineage_shares.csv", index=True)
    pair = [dict(a=y1, b=y2, ARI=float(ARI(year_labels[y1], year_labels[y2]))) for y1 in G.YEARS for y2 in G.YEARS if y1 < y2]
    sil = silhouette_samples(CLR0, LAB)
    d_pce = np.linalg.norm(CLR0[pce] - cent, axis=1)
    pce_fgas_by_gas = pd.DataFrame({y: closure_rows(G.matrix_from_long(
        A.m2[A.m2.Year == y].groupby(["SectorProducedBy", "Flowable"], as_index=False)["CO2e_kg"].sum(),
        A.sectors, GASES))[pce] for y in G.YEARS}, index=GASES)
    save(pce_fgas_by_gas, "S11_PCE_gas_shares_by_year.csv", index=True)
    loo = pd.read_csv(G.TABLE_ROOT / "leave_one_year_out_assignments.csv", dtype={"Sector": str}).set_index("Sector")
    changed_loo = {c: [f"{s} {NAMES.get(s, '')}" for s in loo.index[loo[c].to_numpy() != pd.Series(LAB, index=SEC).reindex(loo.index).to_numpy()]]
                   for c in loo.columns}
    SUMMARY["S11"] = dict(years=yr.to_dict("records"), pce=pc.to_dict("records"), pairwise_year_ARI=pair,
                          pce_silhouette=float(sil[pce]), pce_dist_full=d_pce.tolist(),
                          pce_silhouette_rank_pct=float(100 * (sil < sil[pce]).mean()),
                          pce_top_lineages=ln.head(6).reset_index().to_dict("records"),
                          loo_changed=changed_loo)


# ---------------------------------------------------------------------------
# S12 Family severity (R2.12)
# ---------------------------------------------------------------------------

def s12_family_severity() -> None:
    log("S12 family severity")
    lin_fam = lineage_family.to_numpy()
    rows = []
    sets10 = {}
    for nm, b in zip(FAM_NAMES, FAM_BLOCKS):
        for r_ in [0.0, 0.10, 0.25, 0.50, 0.75]:
            try:
                cand = G.constrained_perturb(X, b, r_)
            except AssertionError:
                rows.append(dict(family=nm, retention=r_, feasible=False))
                continue
            ct = BASE_TRANSFORM(cand)
            lab = G.align_to_reference(kmeans(ct), LAB)
            changed = set(np.where(lab != LAB)[0])
            if abs(r_ - 0.10) < 1e-9:
                sets10[nm] = changed
            # Priority recomputed from the stressed matrix: stressed emissions and stressed lineage shares.
            scale = np.divide(X.sum(axis=0), np.clip(X - (1 - r_) * b, 0, None).sum(axis=0),
                              out=np.ones(X.shape[1]), where=np.clip(X - (1 - r_) * b, 0, None).sum(axis=0) > 0)
            lin_blocks = [(blk * (r_ if lin_fam[i] == nm else 1.0)) * scale[None, :] for i, blk in enumerate(BLOCKS)]
            h2 = lineage_hhi(lin_blocks)
            e2 = cand.sum(axis=1)
            frag2 = 1 - STAB * (1 - h2)
            prio2 = np.sqrt(pct(e2) * pct(frag2))
            rows.append(dict(family=nm, retention=r_, feasible=True, ARI=float(ARI(LAB, lab)), changed=len(changed),
                             changed_set=";".join(SEC[sorted(changed)]),
                             median_shift=float(np.median(aitchison_shift(CLR0, ct))),
                             priority_spearman=float(spearmanr(prio2, PRIO).statistic),
                             top50_overlap=len(top_set(prio2, TOP_N, SEC) & top_set(PRIO, TOP_N, SEC))))
    t = pd.DataFrame(rows)

    def jaccard(a: set, b: set) -> float:
        return 1.0 if not a and not b else len(a & b) / len(a | b)

    t["jaccard_with_10pct"] = [
        jaccard(set(filter(None, str(s).split(";"))), set(SEC[sorted(sets10.get(f, set()))])) if feas else np.nan
        for f, s, feas in zip(t.family, t["changed_set"].fillna(""), t.feasible)
    ]
    save(t, "S12_family_severity.csv")
    # Union of sectors reassigned by BEA use / EIA MECS across levels, with HHI and screen membership.
    focus = t[t.family.isin(["BEA_Detail_Use_SUT", "EIA_MECS_Energy"]) & t.feasible]
    SUMMARY["S12"] = dict(table=t.drop(columns=["changed_set"]).to_dict("records"),
                          focus=focus[["family", "retention", "ARI", "changed", "jaccard_with_10pct", "top50_overlap", "priority_spearman"]].to_dict("records"))


# ---------------------------------------------------------------------------
# S13 Ranking weights and shortlist sizes (R2.13)
# ---------------------------------------------------------------------------

def s13_weights() -> None:
    log("S13 ranking weights and shortlist sizes")
    F_PCT = pct(FRAG)
    H_PCT = pct(HHI)
    rows = []
    ref = {n: top_set(PRIO, n, SEC) for n in [25, 50, 75, 100]}
    emis = {n: top_set(E, n, SEC) for n in [25, 50, 75, 100]}
    ref_i = IDX["32411"]
    for w in [0.25, 1 / 3, 0.5, 2 / 3, 0.75, 0.9]:
        p = E_PCT ** w * F_PCT ** (1 - w)
        ph = E_PCT ** w * H_PCT ** (1 - w)
        for n in [25, 50, 75, 100]:
            t = top_set(p, n, SEC)
            rows.append(dict(weight_emissions=w, shortlist=n, overlap_with_main_rule=len(t & ref[n]),
                             overlap_with_emissions_only=len(t & emis[n]),
                             overlap_emissions_HHI=len(top_set(ph, n, SEC) & t),
                             emissions_share_pct=float(100 * E[list(t)].sum() / TOTAL),
                             median_fragility=float(np.median(FRAG[list(t)])),
                             petroleum_refineries_rank=int((p > p[ref_i]).sum() + 1)))
    # Arithmetic mean variant
    pa = 0.5 * (E_PCT + F_PCT)
    for n in [25, 50, 75, 100]:
        t = top_set(pa, n, SEC)
        rows.append(dict(weight_emissions="arithmetic 0.5", shortlist=n, overlap_with_main_rule=len(t & ref[n]),
                         overlap_with_emissions_only=len(t & emis[n]),
                         overlap_emissions_HHI=len(top_set(0.5 * (E_PCT + H_PCT), n, SEC) & t),
                         emissions_share_pct=float(100 * E[list(t)].sum() / TOTAL),
                         median_fragility=float(np.median(FRAG[list(t)])),
                         petroleum_refineries_rank=int((pa > pa[ref_i]).sum() + 1)))
    t = pd.DataFrame(rows)
    save(t, "S13_weights_and_sizes.csv")
    entry = None
    for w in np.arange(0.50, 1.001, 0.01):
        p = E_PCT ** w * F_PCT ** (1 - w)
        if (p > p[ref_i]).sum() + 1 <= TOP_N:
            entry = float(round(w, 2))
            break
    SUMMARY["S13"] = dict(table=t.to_dict("records"), refineries_entry_weight=entry,
                          refineries=dict(emission_rank=int((E > E[ref_i]).sum() + 1), HHI=float(HHI[ref_i]),
                                          fragility=float(FRAG[ref_i]), joint_rank=int((PRIO > PRIO[ref_i]).sum() + 1),
                                          stability=float(STAB[ref_i])))


# ---------------------------------------------------------------------------
# S14 Release-revision check against the previous dataset release (R2.14)
# ---------------------------------------------------------------------------

GWP_EXTRA = {"Perfluorobutane": 9200, "Perfluorohexane": 7910}


def s14_release_revision() -> None:
    log("S14 release-revision check")
    book = pd.read_excel(G.DATA_ROOT / "GHG_v2025.1.0_validation.xlsx", sheet_name=None, dtype={"Sector": str})
    frames = []
    for y in [2019, 2020, 2021, 2022]:
        d = book[f"{y} Sector"].copy()
        d["Year"] = y
        frames.append(d)
    v = pd.concat(frames)
    gwp = {**G.GWP_AR5, **GWP_EXTRA}
    v = v[v.Flowable.isin(gwp)].copy()
    for c in ["national_new", "national_old"]:
        v[c] = pd.to_numeric(v[c], errors="coerce").fillna(0.0) * v.Flowable.map(gwp)
    mean = v.groupby(["Sector", "Flowable"])[["national_new", "national_old"]].sum() / 4
    new = mean["national_new"].unstack(fill_value=0).reindex(index=SEC, fill_value=0)
    old = mean["national_old"].unstack(fill_value=0).reindex(index=SEC, fill_value=0)
    gas_cols = [g for g in GASES if g in new.columns]
    extra = [g for g in new.columns if g not in GASES]
    new17 = new.reindex(columns=GASES, fill_value=0).to_numpy()
    old17 = old.reindex(columns=GASES, fill_value=0).to_numpy()
    Tn, To = new17.sum(axis=0), old17.sum(axis=0)
    s_new = np.divide(new17, Tn, out=np.zeros_like(new17), where=Tn > 0)
    s_old = np.divide(old17, To, out=np.zeros_like(old17), where=To > 0)
    En = new17.sum(axis=1)
    # Allocation-driven revision: change in sector share of each national gas total, valued at new totals.
    alloc_rev = np.divide(np.abs(s_new - s_old) @ Tn, En, out=np.zeros_like(En), where=En > 0)
    total_rev = np.divide(np.abs(new17.sum(axis=1) - old17.sum(axis=1)), En, out=np.zeros_like(En), where=En > 0)
    prof_rev = np.linalg.norm(BASE_TRANSFORM(new17) - BASE_TRANSFORM(old17 + 0.0), axis=1)
    # Grouping all fluorinated categories removes the reclassification of the unspecified
    # HFC/PFC aggregate into individual HFC species between the two releases.
    prof_rev_grouped = np.linalg.norm(clr_of(zero_replace(new17 @ AMALG)[0]) - clr_of(zero_replace(old17 @ AMALG)[0]), axis=1)
    valid = (En > 0) & (old17.sum(axis=1) > 0)
    fam_changed = np.zeros(len(SEC), bool)
    fam_tab = pd.read_csv(OUT / "S12_family_severity.csv").fillna("")
    for s in fam_tab[(fam_tab.retention.round(2) == 0.10)].changed_set:
        for code in str(s).split(";"):
            if code in IDX:
                fam_changed[IDX[code]] = True
    cont = pd.read_csv(OUT / "S10_sector_continuous_diagnostics.csv", dtype={"Sector": str}).set_index("Sector").reindex(SEC)
    diag = {
        "Emissions": E, "HHI": HHI, "Fragility": FRAG, "Emissions-fragility priority": PRIO,
        "Emissions-HHI priority": np.sqrt(E_PCT * pct(HHI)),
        "Maximum lineage Aitchison shift": cont.max_Aitchison_shift.to_numpy(),
        "Maximum lineage emission change": cont.max_abs_emission_change.to_numpy(),
    }
    target = {"allocation_revision": alloc_rev, "profile_revision_grouped_F": prof_rev_grouped,
              "profile_revision_17": prof_rev}
    rows = []
    rng = np.random.default_rng(7)
    for tname, tv in target.items():
        q75 = np.quantile(tv[valid], 0.75)
        hi = valid & (tv >= q75)
        base_rate = hi[valid].mean()
        for dname, dv in diag.items():
            rho = spearmanr(dv[valid], tv[valid]).statistic
            boots = []
            idx = np.where(valid)[0]
            for _ in range(500):
                b = rng.choice(idx, len(idx), replace=True)
                boots.append(spearmanr(dv[b], tv[b]).statistic)
            top = list(top_set(np.where(valid, dv, -np.inf), TOP_N, SEC))
            rows.append(dict(target=tname, diagnostic=dname, spearman=float(rho),
                             ci_low=float(np.percentile(boots, 2.5)), ci_high=float(np.percentile(boots, 97.5)),
                             top50_fraction_high_revision=float(hi[top].mean()), base_rate=float(base_rate)))
        rows.append(dict(target=tname, diagnostic="Family-stress reassigned (10%)",
                         spearman=float(spearmanr(fam_changed[valid].astype(float), tv[valid]).statistic),
                         top50_fraction_high_revision=float(hi[fam_changed].mean()) if fam_changed.any() else np.nan,
                         base_rate=float(base_rate)))
    from scipy.stats import rankdata
    nz = (X <= 0).sum(axis=1).astype(float)

    def partial_rank(a, b, z):
        ra, rb, rz = rankdata(a), rankdata(b), rankdata(z)
        ea = ra - np.polyval(np.polyfit(rz, ra, 1), rz)
        eb = rb - np.polyval(np.polyfit(rz, rb, 1), rz)
        return float(np.corrcoef(ea, eb)[0, 1])
    t = pd.DataFrame(rows)
    t["partial_given_zero_parts"] = [
        partial_rank(diag[dn][valid], target[tn][valid], nz[valid]) if dn in diag else np.nan
        for tn, dn in zip(t["target"], t["diagnostic"])]
    save(t, "S14_release_revision_association.csv")
    sec_tab = pd.DataFrame({"Sector": SEC, "SectorName": [NAMES.get(s, s) for s in SEC], "allocation_revision": alloc_rev,
                            "total_revision": total_rev, "profile_revision_17": prof_rev,
                            "profile_revision_grouped_F": prof_rev_grouped, "valid": valid})
    save(sec_tab.sort_values("allocation_revision", ascending=False), "S14_sector_revisions.csv")
    nat = pd.DataFrame({"gas": GASES, "new_kgCO2e": Tn, "old_kgCO2e": To, "ratio": np.divide(Tn, To, out=np.full_like(Tn, np.nan), where=To > 0)})
    save(nat, "S14_national_gas_revision.csv")
    SUMMARY["S14"] = dict(sectors_valid=int(valid.sum()), extra_flowables=extra,
                          median_allocation_revision=float(np.median(alloc_rev[valid])),
                          p90_allocation_revision=float(np.percentile(alloc_rev[valid], 90)),
                          median_profile_revision=float(np.median(prof_rev[valid])),
                          median_profile_revision_grouped=float(np.median(prof_rev_grouped[valid])),
                          national_total_ratio=float(Tn.sum() / To.sum()), association=t.to_dict("records"),
                          top_revised=sec_tab[valid].sort_values("allocation_revision", ascending=False).head(10).to_dict("records"))


# ---------------------------------------------------------------------------
# S15 Synthetic allocation errors with known truth (R2.14)
# ---------------------------------------------------------------------------

def inject(block: np.ndarray, kind: str, rng: np.random.Generator, frac: float = 0.3, n_wrong: int = 3) -> np.ndarray:
    out = block.copy()
    if kind == "key_permutation":
        for g in range(block.shape[1]):
            sup = np.where(block[:, g] > 0)[0]
            if len(sup) >= 2:
                perm = rng.permutation(sup)
                while len(sup) > 1 and np.all(perm == sup):
                    perm = rng.permutation(sup)
                out[sup, g] = block[perm, g]
        return out
    if kind == "crosswalk_leakage":
        covered = block.sum(axis=1) > 0
        outside = np.where(~covered)[0]
        if len(outside) == 0:
            return out
        targets = rng.choice(outside, size=min(n_wrong, len(outside)), replace=False)
        moved = frac * block
        out = block - moved
        out[targets] += moved.sum(axis=0)[None, :] / len(targets)
        return out
    raise ValueError(kind)


def s15_synthetic() -> None:
    log("S15 synthetic allocation-error test")
    rng_master = np.random.default_rng(20261121)
    lin_mass = np.array([b.sum() for b in BLOCKS])
    eligible = [l for l in range(L) if lin_mass[l] / TOTAL >= 1e-4]  # >= 0.01% of national CO2e
    rows, sec_rows = [], []
    for kind in ["key_permutation", "crosswalk_leakage", "scale_error"]:
        for l in eligible:
            rng = np.random.default_rng(rng_master.integers(1e9))
            if kind == "scale_error":
                Xc = G.constrained_perturb(X, BLOCKS[l], 2.0)  # lineage overstated by 100%, totals restored
                scale = np.divide(X.sum(axis=0), (X + BLOCKS[l]).sum(axis=0))
                blocks_c = [(b * (2.0 if i == l else 1.0)) * scale[None, :] for i, b in enumerate(BLOCKS)]
            else:
                bad = inject(BLOCKS[l], kind, rng)
                if np.allclose(bad, BLOCKS[l]):
                    continue
                Xc = X - BLOCKS[l] + bad
                blocks_c = [bad if i == l else b for i, b in enumerate(BLOCKS)]
            assert np.allclose(Xc.sum(axis=0), X.sum(axis=0), rtol=1e-9, atol=1e-2)
            clr_c = BASE_TRANSFORM(Xc)
            prof = aitchison_shift(CLR0, clr_c)
            Ec = Xc.sum(axis=1)
            mag = np.abs(Ec - E) / np.maximum(E, 1.0)
            affected_prof = prof >= 0.5
            affected_mag = mag >= 0.10
            affected = affected_prof | affected_mag
            if not affected.any():
                continue
            # Diagnostics an analyst would compute from the corrupted release.
            hc = lineage_hhi(blocks_c)
            pert = [G.constrained_perturb(Xc, b, G.SOURCE_ATTENUATION) for b in blocks_c]
            pclr = [BASE_TRANSFORM(p) for p in pert]
            lab_c = G.relabel_by_emissions(weighted_consensus([kmeans(x) for x in pclr], K)[0], Ec)
            runs_c = [G.align_to_reference(kmeans(x), lab_c) for x in pclr]
            st_c = np.mean(np.vstack(runs_c) == lab_c[None, :], axis=0)
            fr_c = 1 - st_c * (1 - hc)
            ep = pct(Ec)
            screens = {"Emissions only": Ec, "HHI only": hc, "Emissions-HHI": np.sqrt(ep * pct(hc)),
                       "Emissions-fragility": np.sqrt(ep * pct(fr_c))}
            tops = {k: top_set(v, TOP_N, SEC) for k, v in screens.items()}
            resp = np.stack([aitchison_shift(clr_c, x) for x in pclr])             # L x n
            shares_c = np.stack([b.sum(axis=1) for b in blocks_c]) / np.maximum(Ec, 1e-30)  # L x n
            flips = np.stack([r != lab_c for r in runs_c])                          # L x n
            for i in np.where(affected)[0]:
                share_rank = int((shares_c[:, i] > shares_c[l, i]).sum() + 1)
                resp_rank = int((resp[:, i] > resp[l, i]).sum() + 1)
                sec_rows.append(dict(kind=kind, lineage=LNAMES[l], sector=SEC[i], profile_affected=bool(affected_prof[i]),
                                     magnitude_affected=bool(affected_mag[i]), shift=float(prof[i]), rel_change=float(mag[i]),
                                     erroneous_share=float(shares_c[l, i]), share_rank=share_rank, response_rank=resp_rank,
                                     label_flip_on_erroneous=bool(flips[l, i]),
                                     label_changed_by_error=bool(G.align_to_reference(lab_c, LAB)[i] != LAB[i]),
                                     **{f"in_{k}": bool(i in v) for k, v in tops.items()}))
            rows.append(dict(kind=kind, lineage=LNAMES[l], share_pct=100 * lin_mass[l] / TOTAL, affected=int(affected.sum()),
                             profile_affected=int(affected_prof.sum()), magnitude_affected=int(affected_mag.sum())))
        log(f"   {kind}: {sum(r['kind'] == kind for r in rows)} scenarios")
    scen = pd.DataFrame(rows)
    sec = pd.DataFrame(sec_rows)
    save(scen, "S15_synthetic_scenarios.csv")
    save(sec, "S15_synthetic_affected_sectors.csv")
    summ = []
    for kind, g in sec.groupby("kind"):
        for subset, h in [("all affected", g), ("profile-affected", g[g.profile_affected]),
                          ("magnitude-only", g[g.magnitude_affected & ~g.profile_affected])]:
            if h.empty:
                continue
            summ.append(dict(kind=kind, subset=subset, n=len(h), scenarios=int(h.lineage.nunique()),
                             recall_emissions=h["in_Emissions only"].mean(), recall_HHI=h["in_HHI only"].mean(),
                             recall_emissions_HHI=h["in_Emissions-HHI"].mean(),
                             recall_emissions_fragility=h["in_Emissions-fragility"].mean(),
                             pointer_share_hit1=(h.share_rank == 1).mean(), pointer_response_hit1=(h.response_rank == 1).mean(),
                             pointer_share_hit3=(h.share_rank <= 3).mean(), pointer_response_hit3=(h.response_rank <= 3).mean(),
                             either_pointer_hit1=((h.share_rank == 1) | (h.response_rank == 1)).mean(),
                             label_changed=h.label_changed_by_error.mean(),
                             minor_lineage=(h.erroneous_share < 0.05).mean()))
    s = pd.DataFrame(summ)
    save(s, "S15_synthetic_summary.csv")
    minor = sec[sec.erroneous_share < 0.05]
    SUMMARY["S15"] = dict(eligible_lineages=len(eligible), scenarios=scen.groupby("kind").size().to_dict(),
                          summary=s.to_dict("records"),
                          minor_lineage=dict(n=len(minor), share_hit1=float((minor.share_rank == 1).mean()),
                                             response_hit1=float((minor.response_rank == 1).mean()),
                                             response_hit3=float((minor.response_rank <= 3).mean())))


# ---------------------------------------------------------------------------
# S16 Lineage-level review leads in the released account (R1.2, R2.1)
# ---------------------------------------------------------------------------

def s16_review_leads() -> None:
    log("S16 lineage-level review leads")
    shares = np.stack([b.sum(axis=1) for b in BLOCKS]) / np.maximum(E, 1e-30)  # L x n
    dom = shares.argmax(axis=0)
    flips = np.stack([r != LAB for r in RUNS])  # L x n
    rows = []
    for i in np.where(flips.any(axis=0))[0]:
        fl = np.where(flips[:, i])[0]
        rows.append(dict(Sector=SEC[i], SectorName=NAMES.get(SEC[i], SEC[i]), Archetype=int(LAB[i]),
                         AnnualMean_MtCO2e=float(E[i] / 1e9), Stability=float(STAB[i]), HHI=float(HHI[i]),
                         dominant_lineage=G.humanize_meta_source(LNAMES[dom[i]]), dominant_share=float(shares[dom[i], i]),
                         flipping_lineages="; ".join(G.humanize_meta_source(LNAMES[l]) for l in fl),
                         max_flipping_share=float(shares[fl, i].max()),
                         dominant_lineage_flips=bool(dom[i] in fl),
                         flipping_families="; ".join(sorted(set(lineage_family.iloc[fl]))),
                         in_emissions_HHI_top50=bool(i in top_set(np.sqrt(E_PCT * pct(HHI)), TOP_N, SEC)),
                         emission_rank=int((E > E[i]).sum() + 1)))
    t = pd.DataFrame(rows).sort_values("AnnualMean_MtCO2e", ascending=False)
    save(t, "S16_label_flip_review_leads.csv")
    # Family-stress reassigned sectors outside the emissions-HHI shortlist.
    fam = pd.read_csv(G.TABLE_ROOT / "attribution_family_stress.csv")
    sev = pd.read_csv(OUT / "S12_family_severity.csv").fillna("")
    eh_top = top_set(np.sqrt(E_PCT * pct(HHI)), TOP_N, SEC)
    fam_rows = []
    for f in ["BEA_Detail_Use_SUT", "EIA_MECS_Energy", "Direct"]:
        s = sev[(sev.family == f) & (sev.retention.round(2) == 0.10)].changed_set.iloc[0]
        for code in [c for c in str(s).split(";") if c]:
            i = IDX[code]
            fam_rows.append(dict(family=f, Sector=code, SectorName=NAMES.get(code, code), Archetype=int(LAB[i]),
                                 AnnualMean_MtCO2e=float(E[i] / 1e9), emission_rank=int((E > E[i]).sum() + 1),
                                 HHI=float(HHI[i]), family_share=float(sum(BLOCKS[l][i].sum() for l in range(L)
                                                                           if lineage_family.iloc[l] == f) / E[i]),
                                 in_emissions_HHI_top50=bool(i in eh_top)))
    ft = pd.DataFrame(fam_rows)
    save(ft, "S16_family_reassigned_sectors.csv")
    outside = ft[~ft.in_emissions_HHI_top50]
    SUMMARY["S16"] = dict(
        sectors_with_flips=len(t), flips_not_by_dominant=int((~t.dominant_lineage_flips).sum()),
        flips_only_minor_lineages=int((t.max_flipping_share < 0.05).sum()),
        flip_emissions_share_pct=float(100 * t.AnnualMean_MtCO2e.sum() * 1e9 / TOTAL),
        flip_cases=t.head(15).to_dict("records"),
        family_reassigned=int(ft.Sector.nunique()), family_reassigned_outside_EH_top50=int(outside.Sector.nunique()),
        family_reassigned_outside_EH_top50_emissions_pct=float(100 * E[[IDX[s] for s in outside.Sector.unique()]].sum() / TOTAL),
        family_outside_high_emitters=outside[outside.emission_rank <= 100].sort_values("emission_rank").to_dict("records"),
    )


SECTIONS = {
    "01": s01_reconciliation, "02": s02_closure, "03": s03_zeros, "04": s04_zero_sensitivity,
    "05": s05_lineage_definition, "06": s06_single_lineage_and_redistribution, "07": s07_stratified_weighted,
    "08": s08_k_and_seeds, "09": s09_heterogeneity, "10": s10_continuous, "11": s11_years,
    "12": s12_family_severity, "13": s13_weights, "14": s14_release_revision, "15": s15_synthetic,
    "16": s16_review_leads,
}


def main() -> int:
    only = os.environ.get("ESR_SECTIONS", "").strip()
    wanted = [s.strip().zfill(2) for s in only.split(",") if s.strip()] if only else list(SECTIONS)
    summary_path = OUT / "revision_summary.json"
    if summary_path.exists() and only:
        SUMMARY.update(json.loads(summary_path.read_text(encoding="utf-8")))
    for key in wanted:
        t0 = time.time()
        SECTIONS[key]()
        log(f"S{key} done in {time.time() - t0:.1f} s")
        summary_path.write_text(json.dumps(jsonable(SUMMARY), indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
